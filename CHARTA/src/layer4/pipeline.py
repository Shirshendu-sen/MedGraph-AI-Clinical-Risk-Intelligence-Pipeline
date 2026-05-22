"""Layer 4 — Batch inference over all graphs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from layer4.config import (
    RISK_THRESHOLD,
    GRAPHSAGE_IN_DIM,
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_OUT_DIM,
    GRAPHSAGE_NUM_LAYERS,
    GRAPHSAGE_DROPOUT,
    DEFAULT_GRAPH_FOLDERS,
)
from layer4.graph_model import ClinicalGraphSAGE
from layer4.readmission_head import ReadmissionRiskModel
from shared.utils import save_json

logger = logging.getLogger(__name__)


def _load_model(checkpoint_path: str, device: torch.device) -> ReadmissionRiskModel:
    """Load a trained ReadmissionRiskModel from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``best_readmission_model.pt`` checkpoint file.
    device : torch.device
        Target device (CPU or CUDA).

    Returns
    -------
    ReadmissionRiskModel
        The loaded model in eval mode.
    """
    # Build model architecture first
    graph_encoder = ClinicalGraphSAGE(
        in_dim=GRAPHSAGE_IN_DIM,
        hidden_dim=GRAPHSAGE_HIDDEN_DIM,
        out_dim=GRAPHSAGE_OUT_DIM,
        num_layers=GRAPHSAGE_NUM_LAYERS,
        dropout=GRAPHSAGE_DROPOUT,
    )
    model = ReadmissionRiskModel(graph_encoder)

    # Load state dict from checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model from %s (epoch %d, val_AUROC %.4f)",
        checkpoint_path, checkpoint.get("epoch", -1), checkpoint.get("val_auroc", 0.0),
    )
    return model


def _discover_graph_files(input_folder: str | list[str]) -> list[Path]:
    """Discover all *_graph.pt files across one or more input folders.

    Parameters
    ----------
    input_folder : str | list[str]
        Single path or list of paths to graph directories.

    Returns
    -------
    list[Path]
        Sorted list of graph file paths.
    """
    if isinstance(input_folder, (list, tuple)):
        folders = [Path(p) for p in input_folder]
    else:
        folders = [Path(input_folder)]

    graph_files: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            graph_files.extend(sorted(folder.glob("*_graph.pt")))
        else:
            logger.warning("Graph folder not found: %s", folder)

    return graph_files


def run_pipeline(
    input_folder: str | list[str] | None = None,
    output_folder: str = "data/predictions",
    checkpoint_path: str = "models/best_readmission_model.pt",
) -> dict:
    """Batch inference over ALL *_graph.pt files in input_folder(s).

    Parameters
    ----------
    input_folder : str | list[str] | None
        Path(s) to folder containing ``*_graph.pt`` files from Layer 3.
        If None, defaults to DEFAULT_GRAPH_FOLDERS.
    output_folder : str
        Path to folder where prediction JSON files will be saved.
    checkpoint_path : str
        Path to the trained model checkpoint.

    Returns
    -------
    dict
        Summary with keys: processed, failed, errors.
    """
    if input_folder is None:
        input_folder = DEFAULT_GRAPH_FOLDERS

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────
    model = _load_model(checkpoint_path, device)

    # ── Discover graph files ─────────────────────────────────────────
    graph_files = _discover_graph_files(input_folder)
    if not graph_files:
        logger.warning("No *_graph.pt files found in %s", input_folder)
        return {"processed": 0, "failed": 0, "errors": []}

    logger.info("Found %d graph file(s) for inference.", len(graph_files))

    # ── Batch inference ──────────────────────────────────────────────
    processed = 0
    failed = 0
    errors: list[str] = []

    for graph_path in graph_files:
        try:
            # Load graph — weights_only=False required: HeteroData is
            # not a plain tensor dict
            graph: HeteroData = torch.load(str(graph_path), weights_only=False)
            graph = graph.to(device)

            # Get patient_id from graph metadata
            try:
                patient_id = graph["patient"].patient_id
            except (AttributeError, KeyError):
                # Fallback: derive from filename
                patient_id = graph_path.stem.replace("_graph", "")

            # Forward pass — raw logit output (NO sigmoid in model)
            # Single-graph inference: batch_dict is None (defaults to
            # zero vector for all entity nodes — correct for 1 graph).
            logit = model(graph.x_dict, graph.edge_index_dict)  # [1, 1] raw logit

            # ⚠️ BUG-N4 FIX: apply sigmoid HERE (not in the model)
            # to convert logit → probability
            pred = torch.sigmoid(logit)
            risk_score = float(pred.squeeze().cpu())

            # Determine risk level
            risk_level = "HIGH" if risk_score >= RISK_THRESHOLD else "LOW"

            # ── Build output JSON ────────────────────────────────────
            prediction = {
                "metadata": {
                    "patient_id": patient_id,
                    "layer": "layer4_readmission_risk_prediction",
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                },
                "readmission_risk": round(risk_score, 4),
                "risk_level": risk_level,
            }

            # ── Save prediction JSON ─────────────────────────────────
            output_path = output_dir / f"{patient_id}_predictions.json"
            save_json(prediction, str(output_path))

            logger.info(
                "Patient %s — risk=%.4f, level=%s → saved to %s",
                patient_id, risk_score, risk_level, output_path,
            )
            processed += 1

        except Exception as e:
            failed += 1
            error_msg = f"{graph_path.name}: {str(e)}"
            errors.append(error_msg)
            logger.error("Failed to process %s: %s", graph_path.name, e)

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "processed": processed,
        "failed": failed,
        "errors": errors,
    }
    logger.info("Inference complete — processed=%d, failed=%d", processed, failed)
    return summary