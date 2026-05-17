"""Layer 4 — RAG-Augmented Risk Inference: Batch inference over all graphs."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from shared.utils import save_json

from layer4.config import RISK_THRESHOLD
from layer4.graph_model import ClinicalGraphSAGE, get_patient_embedding
from layer4.risk_heads import MultiTaskRiskModel
from layer4.faiss_indexer import load_index
from layer4.rag_retriever import retrieve_similar_patients, format_rag_context

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> MultiTaskRiskModel:
    """Load trained MultiTaskRiskModel from checkpoint.

    Reconstructs ClinicalGraphSAGE encoder + MultiTaskRiskModel and
    loads saved state_dict from ``models/risk_heads/best_model.pt``.

    Returns
    -------
    MultiTaskRiskModel
        The loaded model on DEVICE, in eval mode.
    """
    encoder = ClinicalGraphSAGE()
    model = MultiTaskRiskModel(graph_model=encoder)

    checkpoint_path = "models/risk_heads/best_model.pt"
    if not Path(checkpoint_path).exists():
        logger.warning(
            "Checkpoint %s not found — using untrained model. "
            "Run `python run_layer4.py --mode train` first.",
            checkpoint_path,
        )
        return model.to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    logger.info("Loaded model from %s", checkpoint_path)
    return model


def run_pipeline(input_folder: str, output_folder: str) -> dict:
    """Batch inference over ALL *_graph.pt files in input_folder.

    Parameters
    ----------
    input_folder : str
        Path to folder containing ``*_graph.pt`` files from Layer 3.
    output_folder : str
        Path to folder where prediction JSONs will be saved.

    Returns
    -------
    dict
        Summary: {"processed": N, "failed": M, "errors": [...]}.
    """
    # ── Load model ──
    model = _load_model()

    # ── Load FAISS index ──
    index, patient_ids = load_index()

    # ── Create output folder ──
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # ── Iterate over all graph files ──
    graph_files = sorted(Path(input_folder).glob("*_graph.pt"))
    logger.info("Found %d graph files in %s", len(graph_files), input_folder)

    processed = 0
    failed = 0
    errors: list[str] = []

    for graph_path in graph_files:
        try:
            # ⚠️ weights_only=False required: HeteroData is not a plain tensor dict
            graph = torch.load(str(graph_path), weights_only=False)

            # Derive patient_id from graph metadata or filename
            patient_id = getattr(graph["patient"], "patient_id", None)
            if patient_id is None:
                stem = graph_path.stem
                if stem.endswith("_graph"):
                    patient_id = stem[:-len("_graph")]
                else:
                    patient_id = stem

            # ── Move graph to device FIRST (before embedding extraction) ──
            graph = graph.to(DEVICE)

            # ── Get patient embedding (256-dim) ──
            emb = get_patient_embedding(graph, model.encoder)

            # ── RAG retrieval ──
            retrieved = retrieve_similar_patients(emb, index, patient_ids)

            # ── Format RAG context (768-dim) ──
            rag_ctx = format_rag_context(retrieved).unsqueeze(0).to(DEVICE)  # [1, 768]

            # ── Forward pass → logits ──
            with torch.no_grad():
                preds = model(graph, rag_ctx)

            # ── Apply sigmoid to get probabilities ──
            readmission_prob = torch.sigmoid(preds["readmission"]).item()
            deterioration_prob = torch.sigmoid(preds["deterioration"]).item()
            medication_prob = torch.sigmoid(preds["medication"]).item()

            # ── Apply RISK_THRESHOLD → binary flags ──
            readmission_flag = int(readmission_prob >= RISK_THRESHOLD["readmission"])
            deterioration_flag = int(deterioration_prob >= RISK_THRESHOLD["deterioration"])
            medication_flag = int(medication_prob >= RISK_THRESHOLD["medication"])

            # ── Build prediction dict ──
            prediction = {
                "patient_id": patient_id,
                "source_graph": graph_path.name,
                "readmission": {
                    "probability": round(readmission_prob, 4),
                    "flag": readmission_flag,
                    "threshold": RISK_THRESHOLD["readmission"],
                },
                "deterioration": {
                    "probability": round(deterioration_prob, 4),
                    "flag": deterioration_flag,
                    "threshold": RISK_THRESHOLD["deterioration"],
                },
                "medication": {
                    "probability": round(medication_prob, 4),
                    "flag": medication_flag,
                    "threshold": RISK_THRESHOLD["medication"],
                },
                "similar_patients": [
                    {
                        "patient_id": r["patient_id"],
                        "distance": round(r["distance"], 4),
                    }
                    for r in retrieved
                ],
            }

            # ── Save prediction JSON ──
            output_path = f"{output_folder}/{patient_id}_predictions.json"
            save_json(prediction, output_path)

            processed += 1
            logger.info("Processed %s → %s", patient_id, output_path)

        except Exception as e:
            failed += 1
            error_msg = f"{graph_path.name}: {str(e)}"
            errors.append(error_msg)
            logger.error("Failed to process %s: %s", graph_path.name, e)

    summary = {
        "processed": processed,
        "failed": failed,
        "errors": errors,
    }

    logger.info(
        "Pipeline complete: %d processed, %d failed",
        processed, failed,
    )

    return summary