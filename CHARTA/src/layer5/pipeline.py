"""
Layer 5 – Explainable Clinical Report Generation
Orchestrate explainability and report generation.
"""

import logging
from pathlib import Path

import torch

from shared.utils import load_json

from layer5.feature_explainer import get_top_features
from layer5.report_builder import build_report, save_report

logger = logging.getLogger(__name__)


def run_pipeline(predictions_folder: str, graphs_folder: str, output_folder: str) -> dict:
    """Generate explainable clinical reports from Layer 4 predictions and Layer 3 graphs.

    For each ``*_predictions.json`` in predictions_folder, the corresponding
    ``*_graph.pt`` and ``*_graph_meta.json`` are loaded from graphs_folder. Top
    contributing entities are extracted via embedding-norm importance, and a
    plain-English clinical summary report is saved as JSON.

    Parameters
    ----------
    predictions_folder : str
        Path to folder containing ``*_predictions.json`` files from Layer 4.
    graphs_folder : str
        Path to folder containing ``*_graph.pt`` and ``*_graph_meta.json`` from Layer 3.
    output_folder : str
        Path where ``*_report.json`` files will be saved.

    Returns
    -------
    dict
        Summary with keys: processed, failed, output_files, errors.
    """
    pred_dir = Path(predictions_folder)
    graph_dir = Path(graphs_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(pred_dir.glob("*_predictions.json"))
    if not prediction_files:
        logger.warning("No *_predictions.json files found in %s", predictions_folder)
        return {"processed": 0, "failed": 0, "output_files": [], "errors": []}

    logger.info("Found %d prediction file(s).", len(prediction_files))

    processed = 0
    failed = 0
    output_files: list[str] = []
    errors: list[str] = []

    for pred_path in prediction_files:
        try:
            # Derive patient_id from filename (e.g. "mtsamples_0001_predictions.json")
            stem = pred_path.stem  # "mtsamples_0001_predictions"
            if stem.endswith("_predictions"):
                patient_id = stem[: -len("_predictions")]
            else:
                patient_id = stem

            logger.info("Generating report for patient: %s", patient_id)

            # ── Load prediction ─────────────────────────────────────────
            prediction = load_json(str(pred_path))

            # ── Load graph ──────────────────────────────────────────────
            graph_path = graph_dir / f"{patient_id}_graph.pt"
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")
            graph = torch.load(str(graph_path), weights_only=False)

            # ── Load graph metadata ─────────────────────────────────────
            meta_path = graph_dir / f"{patient_id}_graph_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Graph metadata file not found: {meta_path}")
            meta = load_json(str(meta_path))

            # ⚠️ BUG-N2 FIX: pass both entity_index AND name_index from meta
            name_index = meta.get("name_index", {})
            if not name_index:
                # Fallback: create name_index from entity_index keys (concept_ids)
                # mapping each concept_id to itself (concept_id as name)
                name_index = {cid: cid for cid in meta["entity_index"]}
            top_features = get_top_features(
                graph=graph,
                entity_index=meta["entity_index"],
                name_index=name_index,
            )

            # ── Build and save report ───────────────────────────────────
            report = build_report(patient_id, prediction, top_features)
            output_path = str(out_dir / f"{patient_id}_report.json")
            save_report(report, output_path)

            output_files.append(output_path)
            processed += 1
            logger.info("  ✓ Saved report: %s", output_path)

        except Exception as exc:
            failed += 1
            err_msg = f"Failed for {pred_path.name}: {exc}"
            logger.error("  ✗ %s", err_msg)
            errors.append(err_msg)

    summary = {
        "processed": processed,
        "failed": failed,
        "output_files": output_files,
        "errors": errors,
    }
    logger.info(
        "Layer 5 complete — processed: %d, failed: %d",
        processed,
        failed,
    )
    return summary