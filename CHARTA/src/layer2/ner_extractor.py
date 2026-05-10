# Layer 2 — Clinical NLP Extraction: NER Extractor

"""ScispaCy NER extraction module."""

import spacy

from layer2.config import SCISPACY_MODEL, NER_ENTITY_TYPES


def load_ner_model() -> spacy.Language:
    """Load the scispaCy NER model.

    Returns
    -------
    spacy.Language
        Loaded scispaCy pipeline with NER component.
    """
    nlp = spacy.load(SCISPACY_MODEL)
    return nlp


def extract_entities(doc: spacy.tokens.Doc, sentence_idx: int) -> list[dict]:
    """Extract named entities from a processed spaCy Doc.

    IMPORTANT: takes a PROCESSED spaCy Doc object, not raw text.
    sentence_idx is the 0-based position of this sentence in the document.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A spaCy Doc that has already been processed through the NLP pipeline.
    sentence_idx : int
        0-based index of this sentence within the source document.

    Returns
    -------
    list[dict]
        List of entity dicts, each with keys:
        text, label, start_char, end_char, sentence_idx.
    """
    entities = []
    for ent in doc.ents:
        if ent.label_ in NER_ENTITY_TYPES:
            entities.append({
                "text":         ent.text,
                "label":        ent.label_,      # "DISEASE" or "CHEMICAL"
                "start_char":   ent.start_char,
                "end_char":     ent.end_char,
                "sentence_idx": sentence_idx,
            })
    return entities
