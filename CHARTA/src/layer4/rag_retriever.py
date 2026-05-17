"""Layer 4 — RAG-Augmented Risk Inference: Retrieve similar patients + format context."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch

from layer3.node_encoder import load_encoder, encode_text
from layer4.faiss_indexer import query_index
from layer4.config import FAISS_TOP_K, LABELS_CSV_PATH

logger = logging.getLogger(__name__)

# ── Lazy-loaded encoder (loaded once, reused across calls) ──
_encoder_tokenizer = None
_encoder_model = None


def _get_encoder():
    """Lazy-load ClinicalBERT encoder (heavy — only load when first needed)."""
    global _encoder_tokenizer, _encoder_model
    if _encoder_tokenizer is None:
        _encoder_tokenizer, _encoder_model = load_encoder()
    return _encoder_tokenizer, _encoder_model


# ── Labels cache ──
_labels_cache: dict[str, dict] | None = None


def _load_labels_cache() -> dict[str, dict]:
    """Load risk labels CSV into a cache dict."""
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache

    _labels_cache = {}
    csv_path = Path(LABELS_CSV_PATH)
    if not csv_path.exists():
        logger.warning("Labels CSV not found at %s", LABELS_CSV_PATH)
        return _labels_cache

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            _labels_cache[pid] = {
                "readmission": int(row.get("readmission", 0)),
                "deterioration": int(row.get("deterioration", 0)),
                "medication": int(row.get("medication", 0)),
            }
    return _labels_cache


def _find_extracted_jsons(patient_id: str, extracted_folder: str) -> list[Path]:
    """Find all extracted JSON files for a given patient_id.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    extracted_folder : str
        Path to folder containing ``*_extracted.json`` files.

    Returns
    -------
    list[Path]
        Matching extracted JSON file paths.
    """
    folder = Path(extracted_folder)
    if not folder.exists():
        return []

    # Match files like: {patient_id}_extracted.json or {patient_id}_*_extracted.json
    matches = list(folder.glob(f"{patient_id}_extracted.json"))
    if not matches:
        matches = list(folder.glob(f"{patient_id}*_extracted.json"))
    return sorted(matches)


def retrieve_similar_patients(
    embedding: np.ndarray,
    index,
    patient_ids: list[str],
    top_k: int = FAISS_TOP_K,
    extracted_folder: str = "data/extracted",
) -> list[dict]:
    """Retrieve top-K similar patients from FAISS index with entity context.

    Parameters
    ----------
    embedding : np.ndarray
        Query patient embedding, shape (FAISS_INDEX_DIM,).
    index : faiss.Index
        FAISS index to search.
    patient_ids : list[str]
        Patient IDs corresponding to index vectors.
    top_k : int
        Number of nearest neighbors to retrieve.
    extracted_folder : str
        Path to folder with ``*_extracted.json`` files from Layer 2.

    Returns
    -------
    list[dict]
        Each dict has: patient_id, top_entities (list of dicts), risk_label (dict), distance.
    """
    results = query_index(index, patient_ids, embedding, top_k)
    labels_cache = _load_labels_cache()

    enriched: list[dict] = []
    for result in results:
        pid = result["patient_id"]

        # Load extracted JSON(s) for this patient
        json_paths = _find_extracted_jsons(pid, extracted_folder)
        all_entities: list[dict] = []
        for jp in json_paths:
            from shared.utils import load_json
            doc = load_json(str(jp))
            all_entities.extend(doc.get("entities", []))

        # Sort by link_score descending, take top-5
        sorted_entities = sorted(
            all_entities,
            key=lambda e: float(e.get("link_score", 0.0)),
            reverse=True,
        )
        top_entities = sorted_entities[:5]

        # Get risk labels
        risk_label = labels_cache.get(pid, {
            "readmission": 0, "deterioration": 0, "medication": 0,
        })

        enriched.append({
            "patient_id": pid,
            "top_entities": top_entities,
            "risk_label": risk_label,
            "distance": result["distance"],
        })

    return enriched


def format_rag_context(retrieved: list[dict]) -> torch.Tensor:
    """Format RAG context from retrieved similar patients as a 768-dim tensor.

    For each retrieved patient, encode their top entity texts with ClinicalBERT,
    mean-pool the entity embeddings → [768] per patient. Then mean-pool across
    all K retrieved patients → final [768] tensor.

    The 768→256 projection happens INSIDE MultiTaskRiskModel.forward(),
    not here.

    Parameters
    ----------
    retrieved : list[dict]
        Output of retrieve_similar_patients(), each with top_entities list.

    Returns
    -------
    torch.Tensor
        Shape (768,) — mean-pooled RAG context embedding.
    """
    tokenizer, model = _get_encoder()
    device = next(model.parameters()).device

    patient_embeddings: list[np.ndarray] = []

    for entry in retrieved:
        entities = entry.get("top_entities", [])
        if len(entities) == 0:
            # No entities — use zero vector
            patient_embeddings.append(np.zeros(768, dtype=np.float32))
            continue

        # Encode each entity text → 768-dim
        entity_embs: list[np.ndarray] = []
        for ent in entities:
            text = ent.get("concept_name") or ent.get("text", "")
            if text:
                emb = encode_text(text, tokenizer, model)
                entity_embs.append(emb)

        if len(entity_embs) == 0:
            patient_embeddings.append(np.zeros(768, dtype=np.float32))
        else:
            # Mean-pool entity embeddings for this patient → [768]
            patient_emb = np.mean(np.stack(entity_embs), axis=0)
            patient_embeddings.append(patient_emb)

    if len(patient_embeddings) == 0:
        return torch.zeros(768, dtype=torch.float32)

    # Stack K patient vectors → [K, 768], mean over K → [768]
    stacked = np.stack(patient_embeddings)  # [K, 768]
    rag_context = np.mean(stacked, axis=0)  # [768]

    return torch.tensor(rag_context, dtype=torch.float32)