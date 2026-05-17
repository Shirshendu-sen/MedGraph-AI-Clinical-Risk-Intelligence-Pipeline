"""Layer 4 — RAG-Augmented Risk Inference: Build + query FAISS index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import torch

from layer4.config import (
    FAISS_INDEX_DIM,
    FAISS_INDEX_PATH,
    FAISS_IDS_PATH,
    FAISS_TOP_K,
    GRAPHSAGE_INPUT_DIM,
)

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray, patient_ids: list[str]) -> None:
    """Build a FAISS L2 index from patient graph embeddings and save to disk.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (N, FAISS_INDEX_DIM) — one embedding per patient.
    patient_ids : list[str]
        List of N patient IDs corresponding to each row in embeddings.
    """
    assert embeddings.shape[0] == len(patient_ids), (
        f"Embeddings count ({embeddings.shape[0]}) != patient_ids count ({len(patient_ids)})"
    )
    assert embeddings.shape[1] == FAISS_INDEX_DIM, (
        f"Embedding dim ({embeddings.shape[1]}) != FAISS_INDEX_DIM ({FAISS_INDEX_DIM})"
    )

    # Create output directory if needed
    index_dir = Path(FAISS_INDEX_PATH).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    # Build L2 flat index
    index = faiss.IndexFlatL2(FAISS_INDEX_DIM)
    index.add(embeddings.astype(np.float32))

    logger.info(
        "FAISS index built: %d vectors, dim=%d", index.ntotal, FAISS_INDEX_DIM
    )

    # Save index and patient IDs
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(patient_ids, f)

    logger.info("FAISS index saved to %s", FAISS_INDEX_PATH)
    logger.info("Patient IDs saved to %s", FAISS_IDS_PATH)


def load_index() -> tuple[faiss.Index, list[str]]:
    """Load a previously saved FAISS index and patient ID mapping.

    Returns
    -------
    tuple[faiss.Index, list[str]]
        The FAISS index and the list of patient IDs.
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "r", encoding="utf-8") as f:
        patient_ids = json.load(f)

    logger.info(
        "FAISS index loaded: %d vectors, dim=%d", index.ntotal, FAISS_INDEX_DIM
    )
    return index, patient_ids


def query_index(
    index: faiss.Index,
    patient_ids: list[str],
    query: np.ndarray,
    top_k: int = FAISS_TOP_K,
) -> list[dict]:
    """Query the FAISS index for the top-K most similar patients.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index to search.
    patient_ids : list[str]
        Patient IDs corresponding to index vectors.
    query : np.ndarray
        Query embedding of shape (FAISS_INDEX_DIM,).
    top_k : int
        Number of nearest neighbors to retrieve.

    Returns
    -------
    list[dict]
        List of dicts with keys: patient_id, distance, rank.
    """
    D, I = index.search(query.reshape(1, -1).astype(np.float32), top_k)

    results: list[dict] = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            # FAISS returns -1 when fewer than top_k vectors exist
            continue
        results.append({
            "patient_id": patient_ids[idx],
            "distance": float(D[0][rank]),
            "rank": rank + 1,
        })

    return results


def build_index_from_folders(
    graph_folders: list[str],
    output_folder: str,
) -> None:
    """Load all patient graphs from folders, extract embeddings, and build FAISS index.

    Uses a fixed random projection (768 → 256) to create initial embeddings
    from the patient_x mean-pooled ClinicalBERT vectors in each graph.
    The index can be rebuilt later with learned GraphSAGE embeddings.

    Parameters
    ----------
    graph_folders : list[str]
        Paths to folders containing ``*_graph.pt`` files from Layer 3.
    output_folder : str
        Path to folder where FAISS index and patient IDs will be saved.
    """
    # ── Fixed random projection matrix (768 → 256) ──
    # Seeded for reproducibility; same projection used for all patients
    rng = np.random.RandomState(42)
    projection = rng.randn(GRAPHSAGE_INPUT_DIM, FAISS_INDEX_DIM).astype(np.float32)
    # Normalize rows so projection preserves relative distances (Johnson-Lindenstrauss)
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)

    embeddings_list: list[np.ndarray] = []
    patient_ids_list: list[str] = []

    for folder in graph_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning("Graph folder %s does not exist — skipping", folder)
            continue

        graph_files = sorted(folder_path.glob("*_graph.pt"))
        logger.info("Loading %d graphs from %s", len(graph_files), folder)

        for gf in graph_files:
            graph = torch.load(str(gf), weights_only=False)

            # Extract patient_id from graph metadata or filename
            patient_id = getattr(graph["patient"], "patient_id", None)
            if patient_id is None:
                stem = gf.stem
                if stem.endswith("_graph"):
                    patient_id = stem[:-len("_graph")]
                else:
                    patient_id = stem

            # Extract patient_x: [1, 768] → project to [1, 256] → flatten to [256]
            patient_x = graph["patient"].x.numpy()  # [1, 768]
            projected = (patient_x @ projection).squeeze(0)  # [256]

            embeddings_list.append(projected)
            patient_ids_list.append(patient_id)

    if len(embeddings_list) == 0:
        logger.error("No graphs found in any folder — cannot build FAISS index")
        return

    embeddings = np.stack(embeddings_list)  # [N, 256]
    logger.info("Extracted %d patient embeddings, shape=%s", len(embeddings_list), embeddings.shape)

    # ── Override config paths with output_folder ──
    import layer4.config as cfg
    original_index_path = cfg.FAISS_INDEX_PATH
    original_ids_path = cfg.FAISS_IDS_PATH

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    cfg.FAISS_INDEX_PATH = str(output_path / "faiss.index")
    cfg.FAISS_IDS_PATH = str(output_path / "patient_ids.json")

    build_index(embeddings, patient_ids_list)

    # Restore original config paths
    cfg.FAISS_INDEX_PATH = original_index_path
    cfg.FAISS_IDS_PATH = original_ids_path