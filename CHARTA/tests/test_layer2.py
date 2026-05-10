"""Tests for Layer 2 — Clinical NLP Extraction."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import spacy

from layer2.config import (
    BC5CDR_NER_DATASET,
    BC5CDR_RE_CONFIG,
    BC5CDR_RE_DATASET,
    DISEASE_LINKER,
    DRUG_LINKER,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    NCBI_DISEASE_DATASET,
    NER_ENTITY_TYPES,
    RELATION_THRESHOLD,
    SCISPACY_MODEL,
)
from layer2.ner_extractor import extract_entities, load_ner_model
from layer2.temporal_normalizer import (
    TEMPORAL_PATTERNS,
    attach_timestamps_to_entities,
    extract_temporal_expressions,
    normalize_to_iso,
)


SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


# ─── config.py tests ────────────────────────────────────────────────

class TestConfig:
    def test_scispacy_model_name(self):
        assert SCISPACY_MODEL == "en_ner_bc5cdr_md"

    def test_disease_linker_is_mesh(self):
        assert DISEASE_LINKER == "mesh"

    def test_drug_linker_is_rxnorm(self):
        assert DRUG_LINKER == "rxnorm"

    def test_ner_entity_types(self):
        assert "DISEASE" in NER_ENTITY_TYPES
        assert "CHEMICAL" in NER_ENTITY_TYPES

    def test_bc5cdr_ner_dataset(self):
        assert BC5CDR_NER_DATASET == "tner/bc5cdr"

    def test_bc5cdr_re_dataset(self):
        assert BC5CDR_RE_DATASET == "bigbio/bc5cdr"

    def test_bc5cdr_re_config(self):
        assert BC5CDR_RE_CONFIG == "bc5cdr_bigbio_kb"

    def test_ncbi_disease_dataset(self):
        assert NCBI_DISEASE_DATASET == "ncbi/ncbi_disease"

    def test_relation_threshold(self):
        assert 0 < RELATION_THRESHOLD <= 1.0

    def test_lora_params(self):
        assert LORA_RANK > 0
        assert LORA_ALPHA > 0
        assert len(LORA_TARGET_MODULES) > 0

    def test_clinicalbert_imported_from_shared(self):
        """CLINICALBERT_MODEL should come from shared.constants, not re-defined."""
        from layer2.config import CLINICALBERT_MODEL
        from shared.constants import CLINICALBERT_MODEL as SHARED_MODEL
        assert CLINICALBERT_MODEL == SHARED_MODEL


# ─── ner_extractor.py tests ─────────────────────────────────────────

class TestNERExtractor:
    @pytest.fixture(scope="class")
    def nlp(self):
        return load_ner_model()

    def test_load_ner_model_returns_language(self, nlp):
        assert isinstance(nlp, spacy.Language)

    def test_model_has_ner_pipe(self, nlp):
        assert "ner" in nlp.pipe_names

    def test_extract_entities_disease(self, nlp):
        doc = nlp("Patient has hypertension and diabetes.")
        entities = extract_entities(doc, 0)
        disease_ents = [e for e in entities if e["label"] == "DISEASE"]
        assert len(disease_ents) >= 1
        for ent in disease_ents:
            assert ent["label"] == "DISEASE"
            assert "text" in ent
            assert "start_char" in ent
            assert "end_char" in ent
            assert ent["sentence_idx"] == 0

    def test_extract_entities_chemical(self, nlp):
        doc = nlp("Patient was treated with aspirin and metformin.")
        entities = extract_entities(doc, 0)
        chem_ents = [e for e in entities if e["label"] == "CHEMICAL"]
        assert len(chem_ents) >= 1

    def test_extract_entities_sentence_idx(self, nlp):
        doc = nlp("Patient has hypertension.")
        entities = extract_entities(doc, 5)
        for ent in entities:
            assert ent["sentence_idx"] == 5

    def test_extract_entities_only_filtered_types(self, nlp):
        """Only DISEASE and CHEMICAL labels should be returned."""
        doc = nlp("Patient has hypertension and was given aspirin.")
        entities = extract_entities(doc, 0)
        for ent in entities:
            assert ent["label"] in NER_ENTITY_TYPES

    def test_extract_entities_returns_list(self, nlp):
        doc = nlp("Hello world.")
        entities = extract_entities(doc, 0)
        assert isinstance(entities, list)


# ─── temporal_normalizer.py tests ───────────────────────────────────

class TestTemporalNormalizer:
    def test_extract_iso_date(self):
        text = "Admitted on 2023-09-12 for chest pain."
        exprs = extract_temporal_expressions(text)
        raw_texts = [e["raw_text"] for e in exprs]
        assert "2023-09-12" in raw_texts

    def test_extract_us_date(self):
        text = "Visit on 09/12/2023."
        exprs = extract_temporal_expressions(text)
        assert len(exprs) >= 1

    def test_extract_named_month_date(self):
        text = "Admitted on September 12, 2023."
        exprs = extract_temporal_expressions(text)
        assert len(exprs) >= 1

    def test_extract_relative_date(self):
        text = "Symptoms began 2 days ago."
        exprs = extract_temporal_expressions(text)
        raw_texts = [e["raw_text"] for e in exprs]
        assert any("ago" in r for r in raw_texts)

    def test_extract_returns_char_offsets(self):
        text = "Date: 2023-09-12"
        exprs = extract_temporal_expressions(text)
        assert len(exprs) >= 1
        for expr in exprs:
            assert "start_char" in expr
            assert "end_char" in expr
            assert expr["start_char"] < expr["end_char"]

    def test_normalize_iso_date(self):
        assert normalize_to_iso("2023-09-12") == "2023-09-12"

    def test_normalize_named_month(self):
        result = normalize_to_iso("September 12, 2023")
        assert result == "2023-09-12"

    def test_normalize_us_date(self):
        result = normalize_to_iso("09/12/2023")
        assert result == "2023-09-12"

    def test_normalize_invalid_returns_raw(self):
        result = normalize_to_iso("not_a_date")
        assert result == "not_a_date"

    def test_attach_timestamps_to_entities(self):
        entities = [
            {"text": "hypertension", "start_char": 11, "end_char": 23, "sentence_idx": 0},
        ]
        temp_exprs = [
            {"raw_text": "2023-09-12", "start_char": 0, "end_char": 10},
        ]
        result = attach_timestamps_to_entities(entities, temp_exprs)
        assert len(result) == 1
        assert result[0]["timestamp"] == "2023-09-12"

    def test_attach_timestamps_no_temporal(self):
        entities = [
            {"text": "hypertension", "start_char": 0, "end_char": 12, "sentence_idx": 0},
        ]
        result = attach_timestamps_to_entities(entities, [])
        assert result[0]["timestamp"] is None

    def test_attach_timestamps_nearest(self):
        entities = [
            {"text": "hypertension", "start_char": 5, "end_char": 17, "sentence_idx": 0},
            {"text": "diabetes", "start_char": 80, "end_char": 88, "sentence_idx": 0},
        ]
        temp_exprs = [
            {"raw_text": "2023-01-01", "start_char": 0, "end_char": 10},
            {"raw_text": "2024-06-15", "start_char": 70, "end_char": 80},
        ]
        result = attach_timestamps_to_entities(entities, temp_exprs)
        # First entity is closer to first date
        assert result[0]["timestamp"] == "2023-01-01"
        # Second entity is closer to second date
        assert result[1]["timestamp"] == "2024-06-15"


# ─── relation_extractor.py tests ────────────────────────────────────

class TestRelationExtractor:
    def test_extract_relations_placeholder(self):
        """Before fine-tuning, extract_relations returns []."""
        from layer2.relation_extractor import extract_relations
        sentences = ["Patient has hypertension and takes aspirin."]
        entities = [
            {"text": "hypertension", "label": "DISEASE", "sentence_idx": 0,
             "start_char": 12, "end_char": 24},
            {"text": "aspirin", "label": "CHEMICAL", "sentence_idx": 0,
             "start_char": 35, "end_char": 42},
        ]
        result = extract_relations(sentences, entities)
        assert result == []

    def test_extract_relations_none_model(self):
        """Explicitly passing None for model/tokenizer returns []."""
        from layer2.relation_extractor import extract_relations
        sentences = ["Test sentence."]
        entities = [{"text": "test", "sentence_idx": 0}]
        result = extract_relations(sentences, entities, tokenizer=None, model=None)
        assert result == []


# ─── entity_linker.py tests ─────────────────────────────────────────

class TestEntityLinker:
    def test_add_entity_linkers_adds_pipes(self):
        """add_entity_linkers should add mesh_linker and rxnorm_linker pipes."""
        from layer2.entity_linker import add_entity_linkers
        nlp = spacy.load(SCISPACY_MODEL)
        nlp = add_entity_linkers(nlp)
        assert "mesh_linker" in nlp.pipe_names
        assert "rxnorm_linker" in nlp.pipe_names

    def test_link_entities_adds_concept_fields(self):
        """link_entities should add concept_id, concept_name, kb_source, link_score."""
        from layer2.entity_linker import add_entity_linkers, link_entities
        nlp = spacy.load(SCISPACY_MODEL)
        nlp = add_entity_linkers(nlp)

        doc = nlp("Patient has hypertension.")
        entities = extract_entities(doc, 0)
        if entities:
            linked = link_entities(doc, entities, nlp)
            for ent in linked:
                assert "concept_id" in ent
                assert "concept_name" in ent
                assert "kb_source" in ent
                assert "link_score" in ent

    def test_link_entities_disease_uses_mesh(self):
        """DISEASE entities should have kb_source='mesh'."""
        from layer2.entity_linker import add_entity_linkers, link_entities
        nlp = spacy.load(SCISPACY_MODEL)
        nlp = add_entity_linkers(nlp)

        doc = nlp("Patient has hypertension.")
        entities = extract_entities(doc, 0)
        disease_ents = [e for e in entities if e["label"] == "DISEASE"]
        if disease_ents:
            linked = link_entities(doc, disease_ents, nlp)
            for ent in linked:
                assert ent["kb_source"] == "mesh"

    def test_link_entities_chemical_uses_rxnorm(self):
        """CHEMICAL entities should have kb_source='rxnorm'."""
        from layer2.entity_linker import add_entity_linkers, link_entities
        nlp = spacy.load(SCISPACY_MODEL)
        nlp = add_entity_linkers(nlp)

        doc = nlp("Patient was given aspirin.")
        entities = extract_entities(doc, 0)
        chem_ents = [e for e in entities if e["label"] == "CHEMICAL"]
        if chem_ents:
            linked = link_entities(doc, chem_ents, nlp)
            for ent in linked:
                assert ent["kb_source"] == "rxnorm"

    def test_no_cui_key_in_output(self):
        """Output should use 'concept_id', NOT 'cui'."""
        from layer2.entity_linker import add_entity_linkers, link_entities
        nlp = spacy.load(SCISPACY_MODEL)
        nlp = add_entity_linkers(nlp)

        doc = nlp("Patient has hypertension.")
        entities = extract_entities(doc, 0)
        if entities:
            linked = link_entities(doc, entities, nlp)
            for ent in linked:
                assert "cui" not in ent
                # concept_id must always be present (None when offline)
                assert "concept_id" in ent


# ─── pipeline.py integration test ───────────────────────────────────

class TestPipeline:
    def test_run_pipeline_on_sample(self):
        """Run the full pipeline on a single processed JSON."""
        from layer2.pipeline import run_pipeline

        # Create a minimal input file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create a minimal processed JSON matching Layer 1 output format
            sample_doc = {
                "metadata": {"source_file": "test.txt", "file_type": ".txt"},
                "content": {
                    "sentences": [
                        "Patient has hypertension and was given aspirin.",
                    ],
                    "sentence_count": 1,
                },
            }
            input_file = input_dir / "test_processed.json"
            with open(input_file, "w") as f:
                json.dump(sample_doc, f)

            summary = run_pipeline(str(input_dir), str(output_dir))

            assert summary["processed"] == 1
            assert summary["failed"] == 0

            # Check output file exists
            output_file = output_dir / "test_processed.json"
            assert output_file.exists()

            # Validate output structure
            with open(output_file) as f:
                result = json.load(f)

            assert "metadata" in result
            assert "entities" in result
            assert "relations" in result
            assert "temporal_expressions" in result
            assert isinstance(result["entities"], list)
            assert isinstance(result["relations"], list)

            # If entities found, validate they have concept_id (not cui)
            if result["entities"]:
                for ent in result["entities"]:
                    assert "concept_id" in ent
                    assert "cui" not in ent
