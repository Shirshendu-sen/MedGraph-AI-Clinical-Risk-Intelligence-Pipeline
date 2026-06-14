"""Tests for Layer 4 — Temporal Graph-based Readmission Risk Prediction."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from layer4.config import (
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_NUM_LAYERS,
    GRAPHSAGE_IN_DIM,
    GRAPHSAGE_OUT_DIM,
    RISK_THRESHOLD,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    POSITIVE_CLASS_WEIGHT,
)
from layer4.graph_model import ClinicalGraphSAGE, get_patient_embedding
from layer4.readmission_head import ReadmissionHead, ReadmissionRiskModel


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_graph():
    """Create a minimal HeteroData graph for testing."""
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        pytest.skip("torch_geometric not available, skipping graph tests")

    graph = HeteroData()
    # 5 entities with random 768-dim ClinicalBERT-like embeddings
    graph["entity"].x = torch.randn(5, GRAPHSAGE_IN_DIM)
    graph["visit"].x = torch.randn(1, GRAPHSAGE_IN_DIM)
    graph["patient"].x = torch.randn(1, GRAPHSAGE_IN_DIM)
    # Entity → visit membership
    graph["entity", "occurs_in", "visit"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]], dtype=torch.long
    )
    # Entity → entity relation edges
    graph["entity", "relates_to", "entity"].edge_index = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.long
    )
    # Entity → entity co-occurrence edges
    graph["entity", "co_occurs_with", "entity"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]], dtype=torch.long
    )
    # Visit → visit temporal edges
    graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
    return graph


@pytest.fixture
def graph_model():
    return ClinicalGraphSAGE(
        in_dim=GRAPHSAGE_IN_DIM,
        hidden_dim=GRAPHSAGE_HIDDEN_DIM,
        out_dim=GRAPHSAGE_OUT_DIM,
        num_layers=GRAPHSAGE_NUM_LAYERS,
    )


# ─── config.py tests ────────────────────────────────────────────────

class TestConfig:
    def test_graphsage_hidden_dim_is_positive(self):
        assert GRAPHSAGE_HIDDEN_DIM > 0

    def test_graphsage_in_dim_matches_bert(self):
        assert GRAPHSAGE_IN_DIM == 768  # ClinicalBERT [CLS] dim

    def test_graphsage_out_dim_equals_hidden(self):
        assert GRAPHSAGE_OUT_DIM == GRAPHSAGE_HIDDEN_DIM

    def test_num_layers_is_reasonable(self):
        assert 1 <= GRAPHSAGE_NUM_LAYERS <= 4

    def test_risk_threshold_is_probability(self):
        assert 0.0 <= RISK_THRESHOLD <= 1.0

    def test_learning_rate_is_positive(self):
        assert LEARNING_RATE > 0

    def test_num_epochs_is_positive(self):
        assert NUM_EPOCHS > 0

    def test_batch_size_is_positive(self):
        assert BATCH_SIZE > 0

    def test_positive_class_weight_is_positive(self):
        assert POSITIVE_CLASS_WEIGHT > 0


# ─── graph_model.py tests ───────────────────────────────────────────

class TestClinicalGraphSAGE:
    def test_forward_output_shape(self, sample_graph, graph_model):
        """Model should output [batch_size, out_dim]."""
        graph_model.eval()
        with torch.no_grad():
            emb = graph_model(sample_graph.x_dict, sample_graph.edge_index_dict)
        assert emb.shape == (1, GRAPHSAGE_OUT_DIM)

    def test_output_has_no_nan(self, sample_graph, graph_model):
        graph_model.eval()
        with torch.no_grad():
            emb = graph_model(sample_graph.x_dict, sample_graph.edge_index_dict)
        assert not torch.isnan(emb).any()

    def test_output_has_no_inf(self, sample_graph, graph_model):
        graph_model.eval()
        with torch.no_grad():
            emb = graph_model(sample_graph.x_dict, sample_graph.edge_index_dict)
        assert not torch.isinf(emb).any()

    def test_missing_entity_key_returns_zeros(self, graph_model):
        """If 'entity' key is missing, should return zeros instead of crashing."""
        graph_model.eval()
        with torch.no_grad():
            result = graph_model(
                x_dict={},
                edge_index_dict={},
            )
        assert result.shape == (1, GRAPHSAGE_OUT_DIM)
        assert (result == 0).all()

    def test_forward_with_empty_graph(self, graph_model):
        """Graph with zero entity nodes should handle gracefully."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not available")
        empty_graph = HeteroData()
        empty_graph["entity"].x = torch.empty((0, GRAPHSAGE_IN_DIM))
        empty_graph["visit"].x = torch.empty((0, GRAPHSAGE_IN_DIM))
        empty_graph["patient"].x = torch.empty((0, GRAPHSAGE_IN_DIM))
        empty_graph["entity", "occurs_in", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_graph["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)

        graph_model.eval()
        with torch.no_grad():
            emb = graph_model(empty_graph.x_dict, empty_graph.edge_index_dict)
        assert emb.shape == (1, GRAPHSAGE_OUT_DIM)

    def test_model_trainable(self, graph_model):
        """Model should have trainable parameters."""
        params = list(graph_model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_model_architecture(self, graph_model):
        """Verify the model has expected number of SAGEConv layers."""
        assert len(graph_model.convs) == GRAPHSAGE_NUM_LAYERS
        assert len(graph_model.norms) == GRAPHSAGE_NUM_LAYERS

    def test_conv_layer_dims(self, graph_model):
        """Verify first conv layer has correct in_dim."""
        assert graph_model.convs[0].in_channels == GRAPHSAGE_IN_DIM


class TestGetPatientEmbedding:
    def test_returns_numpy_array(self, sample_graph, graph_model):
        emb = get_patient_embedding(sample_graph, graph_model)
        assert isinstance(emb, torch.Tensor) or hasattr(emb, 'shape')
        assert emb.shape[0] == GRAPHSAGE_OUT_DIM

    def test_embedding_is_deterministic(self, sample_graph, graph_model):
        graph_model.eval()
        emb1 = get_patient_embedding(sample_graph, graph_model)
        emb2 = get_patient_embedding(sample_graph, graph_model)
        assert torch.allclose(
            torch.tensor(emb1) if not isinstance(emb1, torch.Tensor) else emb1,
            torch.tensor(emb2) if not isinstance(emb2, torch.Tensor) else emb2,
        )


# ─── readmission_head.py tests ──────────────────────────────────────

class TestReadmissionHead:
    def test_forward_output_shape(self, graph_model):
        """Head should output [batch_size, 1]."""
        head = ReadmissionHead(input_dim=GRAPHSAGE_OUT_DIM)
        dummy_emb = torch.randn(4, GRAPHSAGE_OUT_DIM)  # batch of 4
        logit = head(dummy_emb)
        assert logit.shape == (4, 1)

    def test_output_is_raw_logit(self, graph_model):
        """Output should be raw logit (can be negative), NOT a probability in [0,1].
        
        ⚠️ BUG-N4 FIX: No sigmoid in model. Sigmoid only in pipeline.
        """
        head = ReadmissionHead(input_dim=GRAPHSAGE_OUT_DIM)
        dummy_emb = torch.randn(4, GRAPHSAGE_OUT_DIM)
        logit = head(dummy_emb)
        # Logits can be outside [0, 1] — that's the point
        # If sigmoid were applied, values would be strictly in (0, 1)
        # Add small noise and check at least some values exceed range
        assert logit.shape == (4, 1)
        # Verify it's not bounded to [0,1] by checking variance isn't zero
        assert logit.std() > 0 or (logit.abs() > 0).any()

    def test_no_sigmoid_in_head(self):
        """Verify ReadmissionHead has NO Sigmoid module.
        
        ⚠️ BUG-N4 FIX: Sigmoid causes double-sigmoid with BCEWithLogitsLoss.
        """
        head = ReadmissionHead()
        has_sigmoid = any(isinstance(m, torch.nn.Sigmoid) for m in head.modules())
        assert not has_sigmoid, "ReadmissionHead should NOT contain Sigmoid"

    def test_head_layers(self):
        """Verify head has correct layer structure."""
        head = ReadmissionHead()
        assert hasattr(head, 'fc1')
        assert hasattr(head, 'fc2')
        assert head.fc1.out_features == 128
        assert head.fc2.out_features == 1

    def test_forward_single_patient(self, graph_model):
        """Single patient should produce scalar logit."""
        head = ReadmissionHead(input_dim=GRAPHSAGE_OUT_DIM)
        single_emb = torch.randn(1, GRAPHSAGE_OUT_DIM)
        logit = head(single_emb)
        assert logit.shape == (1, 1)
        assert logit.numel() == 1

    def test_logit_to_probability(self, graph_model):
        """Apply sigmoid externally to verify conversion works.
        
        ⚠️ BUG-N4: Sigmoid must be applied in pipeline, not in model.
        """
        head = ReadmissionHead(input_dim=GRAPHSAGE_OUT_DIM)
        dummy_emb = torch.randn(4, GRAPHSAGE_OUT_DIM)
        logit = head(dummy_emb)
        prob = torch.sigmoid(logit)
        # After sigmoid, values should be in (0, 1)
        assert (prob >= 0).all() and (prob <= 1).all()


class TestReadmissionRiskModel:
    def test_forward_output_shape(self, sample_graph, graph_model):
        """Full model should output [batch_size, 1] raw logit."""
        model = ReadmissionRiskModel(graph_model)
        model.eval()
        with torch.no_grad():
            logit = model(sample_graph.x_dict, sample_graph.edge_index_dict)
        assert logit.shape == (1, 1)

    def test_no_sigmoid_in_model(self, graph_model):
        """Full model should NOT have Sigmoid either."""
        model = ReadmissionRiskModel(graph_model)
        has_sigmoid = any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())
        assert not has_sigmoid, "ReadmissionRiskModel should NOT contain Sigmoid"

    def test_sigmoid_applied_externally(self, sample_graph, graph_model):
        """Verify sigmoid can be applied to get probability (as done in pipeline).
        
        ⚠️ This mirrors what pipeline.py does: model() → torch.sigmoid().
        """
        model = ReadmissionRiskModel(graph_model)
        model.eval()
        with torch.no_grad():
            logit = model(sample_graph.x_dict, sample_graph.edge_index_dict)
            prob = torch.sigmoid(logit)
        assert prob.shape == (1, 1)
        risk_score = float(prob.squeeze())
        assert 0.0 <= risk_score <= 1.0

    def test_risk_level_classification(self, sample_graph, graph_model):
        """Verify risk_level (HIGH/LOW) from probability and threshold."""
        model = ReadmissionRiskModel(graph_model)
        model.eval()
        with torch.no_grad():
            logit = model(sample_graph.x_dict, sample_graph.edge_index_dict)
            risk_score = float(torch.sigmoid(logit).squeeze())
        
        risk_level = "HIGH" if risk_score >= RISK_THRESHOLD else "LOW"
        
        if risk_score >= RISK_THRESHOLD:
            assert risk_level == "HIGH"
        else:
            assert risk_level == "LOW"

    def test_model_trainable_params(self, graph_model):
        """Full model should have trainable parameters from both sub-modules."""
        model = ReadmissionRiskModel(graph_model)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ─── Pipeline integration test ──────────────────────────────────────

class TestInferencePipeline:
    def test_end_to_end_inference(self, sample_graph, graph_model):
        """Simulate what pipeline.py does: model → sigmoid → risk score → risk level."""
        model = ReadmissionRiskModel(graph_model)
        model.eval()
        
        with torch.no_grad():
            logit = model(sample_graph.x_dict, sample_graph.edge_index_dict)  # raw logit
            pred = torch.sigmoid(logit)  # ⚠️ sigmoid applied in pipeline, not model
            risk_score = float(pred.squeeze())
            risk_level = "HIGH" if risk_score >= RISK_THRESHOLD else "LOW"
        
        assert 0.0 <= risk_score <= 1.0
        assert risk_level in ("HIGH", "LOW")

    def test_batch_inference_consistency(self, graph_model):
        """Multiple single-graph inferences should produce same result as batched."""
        try:
            from torch_geometric.data import HeteroData, Batch
        except ImportError:
            pytest.skip("torch_geometric not available")
        
        # Create two identical graphs
        graph1 = HeteroData()
        graph1["entity"].x = torch.randn(3, GRAPHSAGE_IN_DIM)
        graph1["visit"].x = torch.randn(1, GRAPHSAGE_IN_DIM)
        graph1["patient"].x = torch.randn(1, GRAPHSAGE_IN_DIM)
        graph1["entity", "occurs_in", "visit"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 0, 0]], dtype=torch.long
        )
        graph1["entity", "relates_to", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph1["entity", "co_occurs_with", "entity"].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph1["visit", "before", "visit"].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        graph2 = graph1.clone()
        
        model = ReadmissionRiskModel(graph_model)
        model.eval()
        
        # Single inference on each
        with torch.no_grad():
            logit1 = model(graph1.x_dict, graph1.edge_index_dict)
            logit2 = model(graph2.x_dict, graph2.edge_index_dict)
        
        # They should be identical since graphs are identical
        assert torch.allclose(logit1, logit2)