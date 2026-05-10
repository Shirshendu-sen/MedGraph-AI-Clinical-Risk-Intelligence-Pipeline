# Layer 2 — Clinical NLP Extraction: Relation Extractor

"""ClinicalBERT CID relation extraction module."""

import logging
from collections import defaultdict
from itertools import combinations

from layer2.config import (
    CLINICALBERT_MODEL,
    RELATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Path to fine-tuned LoRA weights (populated after Step 34)
LORA_WEIGHTS_PATH = "models/lora_weights/clinicalbert_rel/"


def load_relation_model():
    """Load the relation extraction model and tokenizer.

    Before Step 34 (fine-tuning), this returns a placeholder.
    After Step 34, loads from models/lora_weights/clinicalbert_rel/.

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForSequenceClassification]
        Tokenizer and model for relation classification.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        LORA_WEIGHTS_PATH,  # after fine-tuning
    )
    return tokenizer, model


def extract_relations(
    sentences: list[str],
    entities: list[dict],
    tokenizer=None,
    model=None,
) -> list[dict]:
    """Extract CID relations between entity pairs within the same sentence.

    Before fine-tuning: returns [] — placeholder, Layer 3 still builds graphs.

    Parameters
    ----------
    sentences : list[str]
        List of sentence strings (indexed by sentence_idx).
    entities : list[dict]
        Entity dicts from the NER + linking stage.
    tokenizer : AutoTokenizer, optional
        Pre-loaded tokenizer. If None, function returns [].
    model : AutoModelForSequenceClassification, optional
        Pre-loaded model. If None, function returns [].

    Returns
    -------
    list[dict]
        List of relation dicts with keys:
        entity_1, entity_2, relation_type, confidence, sentence_idx.
    """
    # Before fine-tuning: return [] — placeholder
    if tokenizer is None or model is None:
        logger.info("Relation model not loaded — returning empty relations (placeholder).")
        return []

    # Group entities by sentence index
    entities_by_sent = defaultdict(list)
    for ent in entities:
        entities_by_sent[ent["sentence_idx"]].append(ent)

    relations = []

    for sent_idx, sent_entities in entities_by_sent.items():
        # Only consider sentences with 2+ entities
        if len(sent_entities) < 2:
            continue

        sentence = sentences[sent_idx]

        # For each entity pair in the same sentence
        for e1, e2 in combinations(sent_entities, 2):
            input_text = (
                f"[E1] {e1['text']} [/E1] {sentence} "
                f"[E2] {e2['text']} [/E2]"
            )

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            import torch

            with torch.no_grad():
                logits = model(**inputs).logits

            label = torch.argmax(logits, dim=-1).item()  # 0=None, 1=CID
            probs = torch.softmax(logits, dim=-1)
            prob = probs[0][label].item()

            if prob >= RELATION_THRESHOLD and label == 1:
                relations.append({
                    "entity_1":     e1["text"],
                    "entity_2":     e2["text"],
                    "relation_type": "CID",
                    "confidence":   prob,
                    "sentence_idx": e1["sentence_idx"],
                })

    return relations
