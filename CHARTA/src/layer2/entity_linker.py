# Layer 2 — Clinical NLP Extraction: Entity Linker

"""MeSH + RxNorm concept linking via scispaCy EntityLinker.

MedCAT removed (archived Jul 28 2025, all models require NIH login).
Replacement: scispaCy built-in EntityLinker — zero registration, auto-downloads.
"""

import logging

import spacy
import scispacy.linking  # registers the "scispacy_linker" factory

from layer2.config import DISEASE_LINKER, DRUG_LINKER

logger = logging.getLogger(__name__)


def add_entity_linkers(nlp: spacy.Language) -> spacy.Language:
    """Add MeSH and RxNorm entity linkers to the spaCy pipeline.

    Call ONCE when building the pipeline, not per-sentence.
    Gracefully skips pipes when linker data cannot be downloaded (offline).

    Parameters
    ----------
    nlp : spacy.Language
        A spaCy Language object (already loaded with NER model).

    Returns
    -------
    spacy.Language
        The same nlp object with mesh_linker and rxnorm_linker pipes added
        (when network is available), or without them (when offline).
    """
    linker_configs = [
        ("mesh_linker", DISEASE_LINKER, {}),
        ("rxnorm_linker", DRUG_LINKER, {"last": True}),
    ]
    for pipe_name, linker_name, extra_kwargs in linker_configs:
        try:
            nlp.add_pipe(
                "scispacy_linker",
                name=pipe_name,
                config={
                    "linker_name": linker_name,
                    "resolve_abbreviations": True,
                },
                **extra_kwargs,
            )
        except Exception as exc:
            # Catches OSError, ConnectionError, requests.exceptions.ConnectionError,
            # urllib3.exceptions.MaxRetryError, etc. when linker data can't be downloaded.
            logger.warning(
                "Could not add %s pipe (%s): %s — continuing without it.",
                pipe_name, linker_name, exc,
            )
    return nlp


def link_entities(
    doc: spacy.tokens.Doc,
    entities: list[dict],
    nlp: spacy.Language,
) -> list[dict]:
    """Link extracted entities to MeSH / RxNorm concepts.

    Runs AFTER nlp(sentence) — linker already applied inside the spaCy pipeline.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A processed spaCy Doc (linker pipes already executed).
    entities : list[dict]
        Entity dicts from extract_entities(), each with start_char / end_char.
    nlp : spacy.Language
        The spaCy Language object (needed to access linker pipes for KB lookups).

    Returns
    -------
    list[dict]
        Same entity dicts updated with concept_id, concept_name, kb_source, link_score.
    """
    # Grab linker objects from the pipeline for KB lookups (may be absent offline)
    has_mesh = "mesh_linker" in nlp.pipe_names
    has_rxnorm = "rxnorm_linker" in nlp.pipe_names
    mesh_linker = nlp.get_pipe("mesh_linker") if has_mesh else None
    rxnorm_linker = nlp.get_pipe("rxnorm_linker") if has_rxnorm else None

    for entity in entities:
        # Determine linker and kb_source based on entity label
        if entity["label"] == "DISEASE":
            linker = mesh_linker
            linker_name = "mesh"
        else:
            linker = rxnorm_linker
            linker_name = "rxnorm"

        # If the required linker pipe is missing (offline), use fallback values
        if linker is None:
            entity.update({
                "concept_id":   None,
                "concept_name": entity["text"],
                "kb_source":    linker_name,
                "link_score":   0.0,
            })
            continue

        # ⚠️ CORRECT: use doc.char_span() for character offsets, NOT doc[start:end]
        # doc[start:end] uses TOKEN indices — passing char offsets gives wrong spans
        span = doc.char_span(
            entity["start_char"],
            entity["end_char"],
            alignment_mode="expand",  # handles tokenisation edge cases
        )

        if span is None:
            logger.warning(
                "char_span returned None for entity '%s' [%d:%d]",
                entity["text"],
                entity["start_char"],
                entity["end_char"],
            )
            entity.update({
                "concept_id":   None,
                "concept_name": entity["text"],
                "kb_source":    linker_name,
                "link_score":   0.0,
            })
            continue

        try:
            kb_ents = span._.kb_ents  # list of (concept_id, score) tuples

            if kb_ents:
                concept_id, score = kb_ents[0]  # take top-1 candidate
                concept_name = linker.kb.cui_to_entity[concept_id].canonical_name
            else:
                concept_id, concept_name, score = None, entity["text"], 0.0

            entity.update({
                "concept_id":   concept_id if concept_id else "UNKNOWN",
                "concept_name": concept_name,
                "kb_source":    linker_name,
                "link_score":   float(score),
            })
        except Exception as e:
            logger.warning("Skipping linker error for entity '%s': %s", entity["text"], e)
            entity.update({
                "concept_id":   "UNKNOWN",
                "concept_name": entity["text"],
                "kb_source":    linker_name,
                "link_score":   0.0,
            })

    return entities
