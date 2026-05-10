"""Layer 3 — Temporal Document Graph: Group by patient, build all graphs."""

import logging
from collections import defaultdict
from pathlib import Path

import torch

from shared.utils import load_json, save_json

from layer3.node_encoder import load_encoder
from layer3.graph_builder import build_patient_graph, validate_graph

logger = logging.getLogger(__name__)


def _group_by_patient(file_paths: list[Path]) -> dict[str, list[dict]]:
    """Group extracted JSON files by patient_id.

    Patient_id is derived from the filename prefix — everything before the
    last underscore-delimited segment.  For example:
        "mtsamples_0001_visit1_extracted.json" → "mtsamples_0001_visit1"
        "sample_cardiology_extracted.json"     → "sample_cardiology"

    Parameters
    ----------
    file_paths : list[Path]
        Paths to ``*_extracted.json`` files.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from patient_id to list of document dicts.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for fp in file_paths:
        # Derive patient_id from filename prefix
        stem = fp.stem  # e.g. "mtsamples_0001_extracted"
        if stem.endswith("_extracted"):
            patient_id = stem[: -len("_extracted")]
        else:
            patient_id = stem
        doc = load_json(str(fp))
        doc["filename"] = fp.name
        groups[patient_id].append(doc)
    return groups


def run_pipeline(input_folder: str, output_folder: str) -> dict:
    """Run the full Layer 3 graph-building pipeline.

    Parameters
    ----------
    input_folder : str
        Path to folder containing ``*_extracted.json`` files from Layer 2.
    output_folder : str
        Path to folder where patient graph files will be saved.

    Returns
    -------
    dict
        Summary with keys: patients_processed, patients_failed, output_files.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder ONCE ─────────────────────────────────────────────
    logger.info("Loading ClinicalBERT encoder …")
    tokenizer, encoder_model = load_encoder()

    # ── Discover input files ──────────────────────────────────────────
    extracted_files = sorted(input_dir.glob("*_extracted.json"))
    if not extracted_files:
        logger.warning("No *_extracted.json files found in %s", input_folder)
        return {"patients_processed": 0, "patients_failed": 0, "output_files": []}

    logger.info("Found %d extracted JSON file(s).", len(extracted_files))

    # ── Group by patient ──────────────────────────────────────────────
    patient_groups = _group_by_patient(extracted_files)
    logger.info("Grouped into %d patient(s).", len(patient_groups))

    # ── Build graphs per patient ──────────────────────────────────────
    patients_processed = 0
    patients_failed = 0
    output_files: list[str] = []

    for patient_id, doc_list in patient_groups.items():
        try:
            logger.info("Building graph for patient: %s (%d doc(s))", patient_id, len(doc_list))

            graph = build_patient_graph(doc_list, patient_id, tokenizer, encoder_model)
            validate_graph(graph)

            # Save graph
            graph_path = output_dir / f"{patient_id}_graph.pt"
            torch.save(graph, str(graph_path))
            output_files.append(str(graph_path))

            # Compute total edges across all edge types
            edge_types = [
                ("entity", "occurs_in", "visit"),
                ("visit", "before", "visit"),
                ("entity", "relates_to", "entity"),
                ("entity", "co_occurs_with", "entity"),
            ]
            num_edges = sum(graph[et].edge_index.shape[1] for et in edge_types)

            # Derive source_dataset from patient_id prefix (e.g. "mtsamples" from "mtsamples_0001")
            source_dataset = patient_id.rsplit("_", 1)[0] if "_" in patient_id else patient_id

            # Save metadata
            graph_filename = f"{patient_id}_graph.pt"
            meta = {
                "patient_id": patient_id,
                "num_entities": graph["entity"].x.shape[0],
                "num_visits": graph["visit"].x.shape[0],
                "num_edges": num_edges,
                "entity_index": graph.entity_index,
                "visit_dates": graph.visit_dates,
                "graph_file": graph_filename,
                "source_dataset": source_dataset,
            }
            meta_path = output_dir / f"{patient_id}_graph_meta.json"
            save_json(meta, str(meta_path))
            output_files.append(str(meta_path))

            patients_processed += 1
            logger.info("  ✓ Saved graph: %s", graph_path.name)

        except Exception as exc:
            patients_failed += 1
            logger.error("  ✗ Failed for patient %s: %s", patient_id, exc)

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "patients_processed": patients_processed,
        "patients_failed": patients_failed,
        "output_files": output_files,
    }
    logger.info(
        "Layer 3 complete — processed: %d, failed: %d",
        patients_processed,
        patients_failed,
    )
    return summary
