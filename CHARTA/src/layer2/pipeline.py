# Layer 2 — Clinical NLP Extraction: Pipeline

"""Orchestrate all NLP steps for Layer 2."""

import logging
from pathlib import Path

from shared.utils import load_json, save_json

from layer2.ner_extractor import load_ner_model, extract_entities
from layer2.entity_linker import add_entity_linkers, link_entities
from layer2.relation_extractor import load_relation_model, extract_relations
from layer2.temporal_normalizer import (
    extract_temporal_expressions,
    attach_timestamps_to_entities,
)

logger = logging.getLogger(__name__)


def run_pipeline(input_folder: str, output_folder: str) -> dict:
    """Run the full Layer 2 NLP pipeline on all Layer 1 processed JSONs.

    Parameters
    ----------
    input_folder : str
        Path to folder containing *_processed.json files from Layer 1.
    output_folder : str
        Path to folder where Layer 2 output JSONs will be saved.

    Returns
    -------
    dict
        Summary with keys: processed, failed, errors.
    """
    # ── Load models once ──────────────────────────────────────────────
    logger.info("Loading NER model …")
    nlp = load_ner_model()

    logger.info("Adding entity linkers (MeSH + RxNorm) …")
    nlp = add_entity_linkers(nlp)

    logger.info("Loading relation model …")
    try:
        tokenizer, rel_model = load_relation_model()
    except Exception as exc:
        logger.warning("Relation model not available (%s) — using placeholder.", exc)
        tokenizer, rel_model = None, None

    # ── Process each file ─────────────────────────────────────────────
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    errors = []

    json_files = sorted(input_path.glob("*_processed.json"))
    logger.info("Found %d processed JSON file(s) in %s", len(json_files), input_folder)

    for json_file in json_files:
        try:
            logger.info("Processing: %s", json_file.name)
            doc = load_json(str(json_file))

            sentences = doc["content"]["sentences"]
            all_entities = []

            # ── Per-sentence NER + linking ────────────────────────────
            for idx, sentence in enumerate(sentences):
                spacy_doc = nlp(sentence)
                entities = extract_entities(spacy_doc, idx)
                entities = link_entities(spacy_doc, entities, nlp)
                all_entities.extend(entities)

            # ── Relation extraction ───────────────────────────────────
            relations = extract_relations(sentences, all_entities, tokenizer, rel_model)

            # ── Temporal normalisation ────────────────────────────────
            full_text = " ".join(sentences)
            temp_exprs = extract_temporal_expressions(full_text)
            all_entities = attach_timestamps_to_entities(all_entities, temp_exprs)

            # ── Build output ──────────────────────────────────────────
            output = {
                "metadata": doc.get("metadata", {}),
                "entities": all_entities,
                "relations": relations,
                "temporal_expressions": temp_exprs,
            }

            out_file = output_path / json_file.name.replace("_processed.json", "_extracted.json")
            save_json(output, str(out_file))

            processed += 1
            logger.info("  → %d entities, %d relations saved to %s",
                        len(all_entities), len(relations), out_file.name)

        except Exception as exc:
            failed += 1
            errors.append({"file": json_file.name, "error": str(exc)})
            logger.error("Failed on %s: %s", json_file.name, exc)

    summary = {"processed": processed, "failed": failed, "errors": errors}
    logger.info("Pipeline complete: %d processed, %d failed", processed, failed)
    return summary
