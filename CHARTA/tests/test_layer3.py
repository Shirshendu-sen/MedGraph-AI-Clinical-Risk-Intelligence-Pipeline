"""Tests for Layer 3 — Temporal Document Graph."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from layer3.config import (
    CO_OCCUR_WINDOW,
    EMBEDDING_DIM,
    GRAPH_SAVE_FORMAT,
    MIN_ENTITIES_PER_GRAPH,
)
from layer3.edge_typer import (
    build_cooccurrence_edges,
    build_relation_edges,
    build_temporal_edges,
)
from layer3.graph_builder import build_patient_graph, validate_graph
from layer3.node_encoder import encode_entity_nodes, encode_text
from layer3.pipeline import _group_by_patient, run_pipeline


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_entities():
    """Minimal entity list for edge tests."""
    return [
        {"text": "hypertension", "concept_id": "D006973", "concept_name": "Hypertension", "sentence_idx": 0, "node_idx": 0},
        {"text": "diabetes", "concept_id": "D003920", "concept_name": "Diabetes Mellitus", "sentence_idx": 1, "node_idx": 1},
        {"text": "chest pain", "concept_id": "D002637", "concept_name": "Chest Pain", "sentence_idx": 2, "node_idx": 2},
        {"text": "aspirin", "concept_id": "D001241", "concept_name": "Aspirin", "sentence_idx": 3, "node_idx": 3},
    ]


@pytest.fixture
def sample_visits():
    """Minimal visit list for temporal edge tests."""
    return [
        {"visit_idx": 0, "earliest_ts": "2023-01-10"},
        {"visit_idx": 1, "earliest_ts": "2023-03-15"},
        {"visit_idx": 2, "earliest_ts": "2023-06-20"},
    ]


@pytest.fixture
def sample_relations():
    """Minimal relation list."""
    return [
        {"entity_1": "D006973", "entity_2": "D001241", "relation_type": "treated_with"},
    ]


@pytest.fixture
def entity_index():
    return {"D006973": 0, "D003920": 1, "D002637": 2, "D001241": 3}


@pytest.fixture
def mock_tokenizer_model():
    """Mock ClinicalBERT tokenizer and model that return deterministic embeddings."""
    tokenizer = MagicMock()

    # Mock tokenizer __call__ to return dict of tensors
    def mock_tokenize(text, **kwargs):
        return {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

    tokenizer.side_effect = mock_tokenize

    # Build a proper mock model
    model = MagicMock()

    # model.parameters() must return a fresh iterator each call
    def mock_parameters():
        param = MagicMock()
        param.device = torch.device("cpu")
        return iter([param])

    model.parameters = mock_parameters

    # model(**inputs) must return object with last_hidden_state
    def mock_forward(**kwargs):
        output = MagicMock()
        output.last_hidden_state = torch.randn(1, 10, EMBEDDING_DIM)
        return output

    model.side_effect = mock_forward

    return tokenizer, model


@pytest.fixture
def sample_extracted_doc():
    """A single extracted document dict mimicking Layer 2 output."""
    return {
        "filename": "sample_cardiology_extracted.json",
        "metadata": {"source_file": "sample_cardiology.txt"},
        "entities": [
            {"text": "hypertension", "label": "DISEASE", "sentence_idx": 0, "concept_id": "D006973", "concept_name": "Hypertension", "timestamp": "2023-09-12"},
            {"text": "diabetes", "label": "DISEASE", "sentence_idx": 1, "concept_id": "D003920", "concept_name": "Diabetes Mellitus", "timestamp": "2023-09-12"},
            {"text": "aspirin", "label": "CHEMICAL", "sentence_idx": 2, "concept_id": "D001241", "concept_name": "Aspirin", "timestamp": "2023-09-12"},
        ],
        "relations": [
            {"entity_1": "D006973", "entity_2": "D001241", "relation_type": "treated_with"},
        ],
        "temporal_expressions": [],
    }


@pytest.fixture
def sample_extracted_doc2():
    """A second extracted document for the same patient (different visit)."""
    return {
        "filename": "sample_cardiology_visit2_extracted.json",
        "metadata": {"source_file": "sample_cardiology_visit2.txt"},
        "entities": [
            {"text": "chest pain", "label": "DISEASE", "sentence_idx": 0, "concept_id": "D002637", "concept_name": "Chest Pain", "timestamp": "2023-10-05"},
            {"text": "hypertension", "label": "DISEASE", "sentence_idx": 1, "concept_id": "D006973", "concept_name": "Hypertension", "timestamp": "2023-10-05"},
        ],
        "relations": [],
        "temporal_expressions": [],
    }


# ─── config.py tests ────────────────────────────────────────────────

class TestConfig:
    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768

    def test_co_occur_window(self):
        assert CO_OCCUR_WINDOW == 3

    def test_min_entities_per_graph(self):
        assert MIN_ENTITIES_PER_GRAPH == 2

    def test_graph_save_format(self):
        assert GRAPH_SAVE_FORMAT == "pt"

    def test_clinicalbert_imported_from_shared(self):
        """CLINICALBERT_MODEL should come from shared.constants, not re-defined."""
        from layer3.config import CLINICALBERT_MODEL
        from shared.constants import CLINICALBERT_MODEL as SHARED_MODEL
        assert CLINICALBERT_MODEL == SHARED_MODEL


# ─── edge_typer.py tests ────────────────────────────────────────────

class TestTemporalEdges:
    def test_consecutive_visits(self, sample_visits):
        edges = build_temporal_edges(sample_visits)
        assert len(edges) == 2
        assert edges[0] == (0, 1, "before")
        assert edges[1] == (1, 2, "before")

    def test_single_visit_no_edges(self):
        edges = build_temporal_edges([{"visit_idx": 0, "earliest_ts": "2023-01-01"}])
        assert edges == []

    def test_empty_visits(self):
        edges = build_temporal_edges([])
        assert edges == []

    def test_visits_sorted_by_timestamp(self):
        visits = [
            {"visit_idx": 2, "earliest_ts": "2023-06-20"},
            {"visit_idx": 0, "earliest_ts": "2023-01-10"},
            {"visit_idx": 1, "earliest_ts": "2023-03-15"},
        ]
        edges = build_temporal_edges(visits)
        assert len(edges) == 2
        # Should be sorted: 0→1, 1→2
        assert edges[0] == (0, 1, "before")
        assert edges[1] == (1, 2, "before")


class TestRelationEdges:
    def test_basic_relation(self, sample_relations, entity_index):
        edges = build_relation_edges(sample_relations, entity_index)
        assert len(edges) == 1
        assert edges[0] == (0, 3, "treated_with")

    def test_missing_entity_skipped(self, entity_index):
        relations = [{"entity_1": "D006973", "entity_2": "UNKNOWN", "relation_type": "treated_with"}]
        edges = build_relation_edges(relations, entity_index)
        assert edges == []

    def test_empty_relations(self, entity_index):
        edges = build_relation_edges([], entity_index)
        assert edges == []

    def test_both_entities_missing(self, entity_index):
        relations = [{"entity_1": "X", "entity_2": "Y", "relation_type": "test"}]
        edges = build_relation_edges(relations, entity_index)
        assert edges == []


class TestCooccurrenceEdges:
    def test_within_window(self, sample_entities):
        edges = build_cooccurrence_edges(sample_entities, window=3)
        # All pairs within window=3 should be linked
        assert len(edges) > 0
        for src, dst, rel in edges:
            assert rel == "co_occurs_with"

    def test_window_zero_same_sentence_only(self):
        entities = [
            {"text": "a", "sentence_idx": 0, "node_idx": 0},
            {"text": "b", "sentence_idx": 0, "node_idx": 1},
            {"text": "c", "sentence_idx": 1, "node_idx": 2},
        ]
        edges = build_cooccurrence_edges(entities, window=0)
        # Only a-b (same sentence_idx=0)
        assert len(edges) == 1
        assert edges[0] == (0, 1, "co_occurs_with")

    def test_excludes_relation_edges(self, sample_entities):
        relation_edge_set = {(0, 3)}
        edges = build_cooccurrence_edges(sample_entities, window=10, relation_edge_set=relation_edge_set)
        for src, dst, _ in edges:
            assert (src, dst) != (0, 3)

    def test_no_self_loops(self):
        entities = [
            {"text": "a", "sentence_idx": 0, "node_idx": 0},
        ]
        edges = build_cooccurrence_edges(entities, window=5)
        assert edges == []

    def test_no_duplicate_edges(self):
        entities = [
            {"text": "a", "sentence_idx": 0, "node_idx": 0},
            {"text": "b", "sentence_idx": 0, "node_idx": 1},
        ]
        edges = build_cooccurrence_edges(entities, window=5)
        assert len(edges) == 1


# ─── node_encoder.py tests ──────────────────────────────────────────

class TestNodeEncoder:
    def test_encode_text_returns_correct_shape(self, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        result = encode_text("hypertension", tokenizer, model)
        assert isinstance(result, np.ndarray)
        assert result.shape == (EMBEDDING_DIM,)

    def test_encode_entity_nodes_keys(self, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        entities = [
            {"text": "hypertension", "concept_id": "D006973", "concept_name": "Hypertension"},
            {"text": "diabetes", "concept_id": "D003920", "concept_name": "Diabetes Mellitus"},
        ]
        result = encode_entity_nodes(entities, tokenizer, model)
        assert "D006973" in result
        assert "D003920" in result
        assert result["D006973"].shape == (EMBEDDING_DIM,)

    def test_encode_entity_nodes_fallback_to_text(self, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        entities = [
            {"text": "chest pain", "concept_id": None, "concept_name": "Chest Pain"},
        ]
        result = encode_entity_nodes(entities, tokenizer, model)
        # Key should be the text since concept_id is None
        assert "chest pain" in result

    def test_encode_entity_nodes_deduplication(self, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        entities = [
            {"text": "hypertension", "concept_id": "D006973", "concept_name": "Hypertension"},
            {"text": "hypertension", "concept_id": "D006973", "concept_name": "Hypertension"},
        ]
        result = encode_entity_nodes(entities, tokenizer, model)
        # Should only encode once for duplicate concept_id
        assert len(result) == 1
        assert "D006973" in result


# ─── graph_builder.py tests ─────────────────────────────────────────

class TestBuildPatientGraph:
    def test_single_doc_graph_structure(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)

        assert isinstance(graph, HeteroData)
        # Node features
        assert graph["entity"].x.shape[1] == EMBEDDING_DIM
        assert graph["visit"].x.shape[1] == EMBEDDING_DIM
        assert graph["patient"].x.shape == (1, EMBEDDING_DIM)

        # 3 unique entities
        assert graph["entity"].x.shape[0] == 3
        # 1 visit
        assert graph["visit"].x.shape[0] == 1

    def test_multi_doc_graph(self, sample_extracted_doc, sample_extracted_doc2, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph(
            [sample_extracted_doc, sample_extracted_doc2], "test_patient", tokenizer, model
        )

        # 4 unique concept_ids: D006973, D003920, D001241, D002637
        assert graph["entity"].x.shape[0] == 4
        # 2 visits
        assert graph["visit"].x.shape[0] == 2

    def test_edge_types_present(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)

        # occurs_in edges should exist
        assert graph["entity", "occurs_in", "visit"].edge_index.shape[0] == 2
        assert graph["entity", "occurs_in", "visit"].edge_index.shape[1] > 0

        # relates_to edges (1 relation in sample)
        assert graph["entity", "relates_to", "entity"].edge_index.shape[0] == 2

    def test_temporal_edges_multi_visit(self, sample_extracted_doc, sample_extracted_doc2, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph(
            [sample_extracted_doc, sample_extracted_doc2], "test_patient", tokenizer, model
        )

        # Should have before edge between visits
        before_ei = graph["visit", "before", "visit"].edge_index
        assert before_ei.shape[0] == 2
        assert before_ei.shape[1] == 1

    def test_entity_index_stored(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)

        assert hasattr(graph, "entity_index")
        assert isinstance(graph.entity_index, dict)
        assert "D006973" in graph.entity_index

    def test_visit_dates_stored(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)

        assert hasattr(graph, "visit_dates")
        assert isinstance(graph.visit_dates, list)
        assert len(graph.visit_dates) == 1

    def test_patient_id_stored(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)

        assert graph["patient"].patient_id == "test_patient"


class TestValidateGraph:
    def test_valid_graph_passes(self, sample_extracted_doc, mock_tokenizer_model):
        tokenizer, model = mock_tokenizer_model
        graph = build_patient_graph([sample_extracted_doc], "test_patient", tokenizer, model)
        assert validate_graph(graph) is True

    def test_too_few_entities_fails(self):
        graph = HeteroData()
        graph["entity"].x = torch.randn(1, EMBEDDING_DIM)  # Only 1 entity
        graph["visit"].x = torch.randn(1, EMBEDDING_DIM)
        graph["patient"].x = torch.randn(1, EMBEDDING_DIM)
        graph["entity", "occurs_in", "visit"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)

        with pytest.raises(ValueError, match="minimum is 2"):
            validate_graph(graph)

    def test_nan_features_fails(self):
        graph = HeteroData()
        graph["entity"].x = torch.full((3, EMBEDDING_DIM), float("nan"))
        graph["visit"].x = torch.randn(1, EMBEDDING_DIM)
        graph["patient"].x = torch.randn(1, EMBEDDING_DIM)
        graph["entity", "occurs_in", "visit"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
        graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)

        with pytest.raises(ValueError, match="NaN"):
            validate_graph(graph)

    def test_no_edges_fails(self):
        graph = HeteroData()
        graph["entity"].x = torch.randn(3, EMBEDDING_DIM)
        graph["visit"].x = torch.randn(1, EMBEDDING_DIM)
        graph["patient"].x = torch.randn(1, EMBEDDING_DIM)
        graph["entity", "occurs_in", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)

        with pytest.raises(ValueError, match="no edges"):
            validate_graph(graph)


# ─── pipeline.py tests ──────────────────────────────────────────────

class TestGroupByPatient:
    def test_group_by_patient_basic(self, tmp_path):
        doc1 = {"metadata": {}, "entities": [], "relations": []}
        doc2 = {"metadata": {}, "entities": [], "relations": []}

        f1 = tmp_path / "mtsamples_0001_extracted.json"
        f2 = tmp_path / "mtsamples_0002_extracted.json"
        f1.write_text(json.dumps(doc1))
        f2.write_text(json.dumps(doc2))

        groups = _group_by_patient([f1, f2])
        assert "mtsamples_0001" in groups
        assert "mtsamples_0002" in groups
        assert len(groups) == 2

    def test_group_by_patient_same_patient(self, tmp_path):
        doc = {"metadata": {}, "entities": [], "relations": []}

        f1 = tmp_path / "mtsamples_0001_visit1_extracted.json"
        f2 = tmp_path / "mtsamples_0001_visit2_extracted.json"
        f1.write_text(json.dumps(doc))
        f2.write_text(json.dumps(doc))

        groups = _group_by_patient([f1, f2])
        # Both files share prefix "mtsamples_0001_visit1" and "mtsamples_0001_visit2"
        # They will be different patient_ids since _extracted is stripped
        assert len(groups) == 2


class TestRunPipeline:
    @patch("layer3.pipeline.load_encoder")
    def test_pipeline_creates_output_files(self, mock_load_encoder, mock_tokenizer_model, tmp_path):
        mock_load_encoder.return_value = mock_tokenizer_model

        # Create input extracted JSON
        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        output_dir = tmp_path / "graphs"

        doc = {
            "metadata": {"source_file": "test.txt"},
            "entities": [
                {"text": "hypertension", "label": "DISEASE", "sentence_idx": 0, "concept_id": "D006973", "concept_name": "Hypertension", "timestamp": "2023-09-12"},
                {"text": "diabetes", "label": "DISEASE", "sentence_idx": 1, "concept_id": "D003920", "concept_name": "Diabetes Mellitus", "timestamp": "2023-09-12"},
                {"text": "aspirin", "label": "CHEMICAL", "sentence_idx": 2, "concept_id": "D001241", "concept_name": "Aspirin", "timestamp": "2023-09-12"},
            ],
            "relations": [
                {"entity_1": "D006973", "entity_2": "D001241", "relation_type": "treated_with"},
            ],
            "temporal_expressions": [],
        }

        input_file = input_dir / "sample_test_extracted.json"
        input_file.write_text(json.dumps(doc))

        result = run_pipeline(str(input_dir), str(output_dir))

        assert result["patients_processed"] == 1
        assert result["patients_failed"] == 0
        assert len(result["output_files"]) == 2  # .pt + _meta.json

        # Check graph file exists
        graph_file = output_dir / "sample_test_graph.pt"
        assert graph_file.exists()

        # Check meta file exists and has correct schema
        meta_file = output_dir / "sample_test_graph_meta.json"
        assert meta_file.exists()

        meta = json.loads(meta_file.read_text())
        assert meta["patient_id"] == "sample_test"
        assert meta["num_entities"] == 3
        assert meta["num_visits"] == 1
        assert "num_edges" in meta
        assert isinstance(meta["num_edges"], int)
        assert "entity_index" in meta
        assert isinstance(meta["entity_index"], dict)
        assert "visit_dates" in meta
        assert isinstance(meta["visit_dates"], list)
        assert meta["graph_file"] == "sample_test_graph.pt"
        assert "source_dataset" in meta

    @patch("layer3.pipeline.load_encoder")
    def test_pipeline_empty_input(self, mock_load_encoder, mock_tokenizer_model, tmp_path):
        mock_load_encoder.return_value = mock_tokenizer_model

        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "graphs"

        result = run_pipeline(str(input_dir), str(output_dir))

        assert result["patients_processed"] == 0
        assert result["patients_failed"] == 0

    @patch("layer3.pipeline.load_encoder")
    def test_pipeline_meta_json_schema(self, mock_load_encoder, mock_tokenizer_model, tmp_path):
        mock_load_encoder.return_value = mock_tokenizer_model

        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        output_dir = tmp_path / "graphs"

        doc = {
            "metadata": {"source_file": "test.txt"},
            "entities": [
                {"text": "hypertension", "label": "DISEASE", "sentence_idx": 0, "concept_id": "D006973", "concept_name": "Hypertension", "timestamp": "2023-09-12"},
                {"text": "diabetes", "label": "DISEASE", "sentence_idx": 1, "concept_id": "D003920", "concept_name": "Diabetes Mellitus", "timestamp": "2023-09-12"},
            ],
            "relations": [],
            "temporal_expressions": [],
        }

        input_file = input_dir / "mtsamples_0001_extracted.json"
        input_file.write_text(json.dumps(doc))

        run_pipeline(str(input_dir), str(output_dir))

        meta_file = output_dir / "mtsamples_0001_graph_meta.json"
        meta = json.loads(meta_file.read_text())

        # Verify exact schema
        required_keys = {"patient_id", "num_entities", "num_visits", "num_edges", "entity_index", "visit_dates", "graph_file", "source_dataset"}
        assert set(meta.keys()) == required_keys
        assert meta["patient_id"] == "mtsamples_0001"
        assert meta["source_dataset"] == "mtsamples"
        assert meta["graph_file"] == "mtsamples_0001_graph.pt"
