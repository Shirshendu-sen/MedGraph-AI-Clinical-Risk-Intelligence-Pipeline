"""Layer 3 — Temporal Document Graph: ClinicalBERT embeddings for entity nodes."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from shared.constants import CLINICALBERT_MODEL
from layer3.config import EMBEDDING_DIM


def load_encoder() -> tuple[AutoTokenizer, AutoModel]:
    """Load ClinicalBERT tokenizer and model.

    Returns
    -------
    tuple[AutoTokenizer, AutoModel]
        The tokenizer and model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
    model = AutoModel.from_pretrained(CLINICALBERT_MODEL)
    model.eval()
    model.to(device)
    return tokenizer, model


def encode_text(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """Encode a single text string into a 768-dim [CLS] embedding.

    Parameters
    ----------
    text : str
        The entity text to encode.
    tokenizer : AutoTokenizer
        ClinicalBERT tokenizer.
    model : AutoModel
        ClinicalBERT model.

    Returns
    -------
    np.ndarray
        Shape (768,) — the [CLS] token embedding.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, max_length=64, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()


def encode_entity_nodes(
    entities: list[dict],
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> dict[str, np.ndarray]:
    """Encode all unique entity nodes into ClinicalBERT embeddings.

    Parameters
    ----------
    entities : list[dict]
        List of entity dicts, each with at least "text" and optionally
        "concept_id" and "concept_name".
    tokenizer : AutoTokenizer
        ClinicalBERT tokenizer.
    model : AutoModel
        ClinicalBERT model.

    Returns
    -------
    dict[str, np.ndarray]
        Key: concept_id (or entity text if concept_id is None);
        Value: 768-dim numpy array.
    """
    embeddings: dict[str, np.ndarray] = {}
    for entity in entities:
        concept_id = entity.get("concept_id")
        if concept_id in (None, "UNKNOWN", ""):
            concept_id = entity["text"]
        if concept_id in embeddings:
            continue
        text = entity.get("concept_name") or entity["text"]
        embeddings[concept_id] = encode_text(text, tokenizer, model)
    return embeddings
