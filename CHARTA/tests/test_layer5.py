"""Tests for Layer 5 — Explainable Clinical Report Generation."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layer5.config import NUM_TOP_FEATURES
from layer5.feature_explainer import get_top_features
from layer5.report_builder import (
    build_plain_english_summary,
    build_report,
    format_risk_level,
    save_report,
)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_graph():
    """Create a minimal HeteroData graph with entity embeddings."""
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        pytest.skip("torch_geometric not available, skipping graph tests")

    graph = HeteroData()
    # 5 entities with random 768-dim embeddings
    graph["entity"].x = torch.randn(5, 768)
    graph["entity", "occurs_in", "visit"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]], dtype=torch.long
    )
    graph["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph["visit"].x = torch.randn(1, 768)
    graph["patient"].x = torch.randn(1, 768)
    return graph


@pytest.fixture
def entity_index():
    return {
        "UNKNOWN": 0,
        "D006973": 1,  # Hypertension
        "D003920": 2,  # Diabetes Mellitus
        "D002637": 3,  # Chest Pain
        "D001241": 4,  # Aspirin
    }


@pytest.fixture
def name_index():
    return {
        "UNKNOWN": "Unknown",
        "D006973": "Hypertension",
        "D003920": "Diabetes Mellitus",
        "D002637": "Chest Pain",
        "D001241": "Aspirin",
    }


@pytest.fixture
def sample_prediction():
    return {
        "metadata": {
            "patient_id": "test_patient",
            "layer": "layer4_readmission_risk_prediction",
        },
        "readmission_risk": 0.84,
        "risk_level": "HIGH",
    }


@pytest.fixture
def sample_top_features():
    return [
        {"entity_name": "Hypertension", "concept_id": "D006973", "importance": 2.5},
        {"entity_name": "Diabetes Mellitus", "concept_id": "D003920", "importance": 1.8},
        {"entity_name": "Aspirin", "concept_id": "D001241", "importance": 1.2},
    ]


# ─── config.py tests ────────────────────────────────────────────────

class TestConfig:
    def test_num_top_features_is_positive(self):
        assert NUM_TOP_FEATURES > 0

    def test_num_top_features_default(self):
        assert NUM_TOP_FEATURES == 3


# ─── feature_explainer.py tests ─────────────────────────────────────

class TestGetTopFeatures:
    def test_returns_correct_number(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=3)
        assert len(features) == 3

    def test_returns_requested_count(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=2)
        assert len(features) == 2

    def test_uses_name_index(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=5)
        for f in features:
            # entity_name should come from name_index, not be a CUI
            assert f["entity_name"] in name_index.values() or f["entity_name"].startswith("entity_")

    def test_output_has_expected_keys(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=2)
        for f in features:
            assert "entity_name" in f
            assert "concept_id" in f
            assert "importance" in f
            # MUST NOT have 'cui' key
            assert "cui" not in f

    def test_concept_id_not_cui(self, sample_graph, entity_index, name_index):
        """Verify concept_id is used, NOT cui."""
        features = get_top_features(sample_graph, entity_index, name_index, n=2)
        for f in features:
            assert "cui" not in f
            assert f["concept_id"] in entity_index

    def test_importance_scores_are_positive(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=5)
        for f in features:
            assert f["importance"] >= 0

    def test_sorted_by_importance_descending(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=5)
        scores = [f["importance"] for f in features]
        assert scores == sorted(scores, reverse=True)

    def test_fallback_to_entity_name(self, sample_graph, entity_index):
        """When name_index is empty, should fall back to concept_id or entity_N."""
        features = get_top_features(sample_graph, entity_index, {}, n=5)
        for f in features:
            assert f["entity_name"] in entity_index.values() or f["entity_name"].startswith("entity_")

    def test_n_greater_than_entities(self, sample_graph, entity_index, name_index):
        features = get_top_features(sample_graph, entity_index, name_index, n=100)
        assert len(features) == 5  # Only 5 entities in graph

    def test_empty_graph_returns_empty(self):
        """An empty-ish graph should return empty list."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not available")
        graph = HeteroData()
        graph["entity"].x = torch.randn(0, 768)
        result = get_top_features(
            graph,
            entity_index={"UNKNOWN": 0},
            name_index={"UNKNOWN": "Unknown"},
            n=3,
        )
        assert result == []


# ─── report_builder.py tests ────────────────────────────────────────

class TestFormatRiskLevel:
    def test_high_risk(self):
        assert format_risk_level(0.5) == "HIGH"
        assert format_risk_level(0.9) == "HIGH"

    def test_low_risk(self):
        assert format_risk_level(0.0) == "LOW"
        assert format_risk_level(0.49) == "LOW"

    def test_boundary(self):
        assert format_risk_level(0.5) == "HIGH"
        assert format_risk_level(0.499) == "LOW"


class TestBuildPlainEnglishSummary:
    def test_high_risk_summary(self, sample_top_features):
        summary = build_plain_english_summary(0.84, sample_top_features)
        assert "High" in summary
        assert "84%" in summary
        assert "Hypertension" in summary
        assert "Diabetes Mellitus" in summary
        assert "Aspirin" in summary

    def test_low_risk_summary(self, sample_top_features):
        summary = build_plain_english_summary(0.12, sample_top_features)
        assert "Low" in summary
        assert "12%" in summary

    def test_single_factor(self):
        features = [{"entity_name": "Hypertension", "concept_id": "D006973", "importance": 2.5}]
        summary = build_plain_english_summary(0.75, features)
        assert "due to Hypertension" in summary

    def test_empty_factors(self):
        summary = build_plain_english_summary(0.5, [])
        assert "due to" in summary

    def test_risk_score_zero(self):
        summary = build_plain_english_summary(0.0, [{"entity_name": "None", "concept_id": "", "importance": 0}])
        assert "0%" in summary
        assert "Low" in summary

    def test_risk_score_one(self):
        summary = build_plain_english_summary(1.0, [{"entity_name": "Critical", "concept_id": "", "importance": 5}])
        assert "100%" in summary
        assert "High" in summary


class TestBuildReport:
    def test_report_structure(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        assert "metadata" in report
        assert "risk_summary" in report
        assert "explanation" in report
        assert "disclaimer" in report

    def test_metadata_content(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        assert report["metadata"]["patient_id"] == "test_patient"
        assert report["metadata"]["layer"] == "layer5_explainable_clinical_report"

    def test_risk_summary(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        assert report["risk_summary"]["readmission_risk"] == 0.84
        assert report["risk_summary"]["risk_level"] == "HIGH"

    def test_explanation_plain_english(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        assert isinstance(report["explanation"]["plain_english"], str)
        assert len(report["explanation"]["plain_english"]) > 0
        assert "Hypertension" in report["explanation"]["plain_english"]

    def test_explanation_top_factors(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        factors = report["explanation"]["top_factors"]
        assert len(factors) == 3
        for f in factors:
            assert "concept_id" in f
            assert "cui" not in f  # Must NOT use cui
            assert "entity_name" in f
            assert "importance" in f

    def test_disclaimer_present(self, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        assert "disclaimer" in report
        assert "research prototype" in report["disclaimer"].lower()


class TestSaveReport:
    def test_saves_json(self, tmp_path, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        output_path = str(tmp_path / "test_report.json")
        save_report(report, output_path)

        # Verify file exists
        assert Path(output_path).exists()

        # Verify contents
        with open(output_path, "r") as f:
            loaded = json.load(f)
        assert loaded["metadata"]["patient_id"] == "test_patient"
        assert loaded["explanation"]["plain_english"] != ""
        assert "concept_id" in loaded["explanation"]["top_factors"][0]
        assert "cui" not in loaded["explanation"]["top_factors"][0]

    def test_saves_indented_json(self, tmp_path, sample_prediction, sample_top_features):
        report = build_report("test_patient", sample_prediction, sample_top_features)
        output_path = str(tmp_path / "test_report.json")
        save_report(report, output_path)

        with open(output_path, "r") as f:
            content = f.read()
        # Check it's pretty-printed (indentation)
        assert '  ' in content

    def test_overwrites_existing(self, tmp_path, sample_prediction, sample_top_features):
        output_path = tmp_path / "test_report.json"
        # Write initial
        report1 = build_report("patient1", sample_prediction, sample_top_features)
        save_report(report1, str(output_path))
        # Overwrite
        report2 = build_report("patient2", sample_prediction, sample_top_features)
        save_report(report2, str(output_path))
        # Verify overwritten
        with open(output_path, "r") as f:
            loaded = json.load(f)
        assert loaded["metadata"]["patient_id"] == "patient2"


# ─── pipeline.py tests ──────────────────────────────────────────────

class TestPipeline:
    def test_run_pipeline_on_sample_predictions(self, tmp_path):
        """Integration test: run full pipeline on sample data."""
        from layer5.pipeline import run_pipeline

        # Create minimal prediction file
        pred_dir = tmp_path / "predictions"
        graph_dir = tmp_path / "graphs"
        out_dir = tmp_path / "explanations"
        pred_dir.mkdir()
        graph_dir.mkdir()

        prediction = {
            "metadata": {"patient_id": "test", "layer": "layer4_readmission_risk_prediction"},
            "readmission_risk": 0.72,
            "risk_level": "HIGH",
        }
        pred_file = pred_dir / "test_predictions.json"
        with open(pred_file, "w") as f:
            json.dump(prediction, f)

        # Check if torch_geometric is available
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            # Create a minimal graph manually without torch_geometric
            pytest.skip("torch_geometric not available, skipping pipeline integration test")

        # Create a fake HeteroData graph and meta
        graph = HeteroData()
        graph["entity"].x = torch.randn(3, 768)
        graph["entity", "occurs_in", "visit"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 0, 0]], dtype=torch.long
        )
        graph["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["visit"].x = torch.randn(1, 768)
        graph["patient"].x = torch.randn(1, 768)

        graph_file = graph_dir / "test_graph.pt"
        torch.save(graph, str(graph_file))

        meta = {
            "patient_id": "test",
            "num_entities": 3,
            "num_visits": 1,
            "num_edges": 3,
            "entity_index": {"UNKNOWN": 0, "D006973": 1, "D001241": 2},
            "name_index": {"UNKNOWN": "Unknown", "D006973": "Hypertension", "D001241": "Aspirin"},
            "visit_dates": [""],
            "graph_file": "test_graph.pt",
            "source_dataset": "test",
        }
        meta_file = graph_dir / "test_graph_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f)

        result = run_pipeline(str(pred_dir), str(graph_dir), str(out_dir))

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["output_files"]) == 1

        # Verify output
        out_file = Path(result["output_files"][0])
        assert out_file.exists()
        with open(out_file) as f:
            report = json.load(f)
        assert report["explanation"]["plain_english"] != ""
        assert "concept_id" in report["explanation"]["top_factors"][0]
        assert "cui" not in report["explanation"]["top_factors"][0]

    def test_pipeline_no_predictions(self, tmp_path):
        from layer5.pipeline import run_pipeline

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "out"

        result = run_pipeline(str(empty_dir), str(empty_dir), str(out_dir))
        assert result["processed"] == 0
        assert result["failed"] == 0

    def test_pipeline_graph_not_found(self, tmp_path):
        from layer5.pipeline import run_pipeline

        pred_dir = tmp_path / "predictions"
        graph_dir = tmp_path / "graphs"
        out_dir = tmp_path / "out"
        pred_dir.mkdir()
        graph_dir.mkdir()

        prediction = {
            "metadata": {"patient_id": "test"},
            "readmission_risk": 0.5,
            "risk_level": "LOW",
        }
        pred_file = pred_dir / "test_predictions.json"
        with open(pred_file, "w") as f:
            json.dump(prediction, f)

        result = run_pipeline(str(pred_dir), str(graph_dir), str(out_dir))
        assert result["processed"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert "Graph file not found" in result["errors"][0]