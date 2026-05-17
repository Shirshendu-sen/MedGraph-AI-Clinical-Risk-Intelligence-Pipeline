"""Layer 4 — RAG-Augmented Risk Inference: MultiTaskRiskModel with 3 risk heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from layer4.graph_model import ClinicalGraphSAGE
from layer4.config import GRAPHSAGE_INPUT_DIM, GRAPHSAGE_HIDDEN_DIM


class RiskHead(nn.Module):
    """Single binary risk prediction head.

    Takes a 512-dim concatenated vector (256 GNN + 256 RAG) and
    produces a single raw logit (no sigmoid — BCEWithLogitsLoss
    applies sigmoid internally).

    Parameters
    ----------
    input_dim : int
        Input dimension (default: 512 = 256 GNN + 256 RAG).
    """

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict risk logit.

        Parameters
        ----------
        x : torch.Tensor
            Shape [batch_size, input_dim].

        Returns
        -------
        torch.Tensor
            Shape [batch_size, 1] — raw logit (sigmoid applied externally).
        """
        return self.net(x)


class MultiTaskRiskModel(nn.Module):
    """Multi-task risk prediction model combining GNN encoder + RAG context.

    Architecture:
    - ClinicalGraphSAGE encoder produces 256-dim graph embedding
    - RAG context (768-dim raw entity mean embeddings) projected to 256-dim
    - Concatenated [256 GNN + 256 RAG] = 512-dim fed to 3 RiskHeads
    - Outputs: readmission, deterioration, medication logits

    Parameters
    ----------
    graph_model : ClinicalGraphSAGE
        The GraphSAGE encoder instance.
    """

    def __init__(self, graph_model: ClinicalGraphSAGE):
        super().__init__()
        self.encoder = graph_model

        # ⚠️ Fixed: input is 768 (raw entity mean embeddings), NOT 256
        # Projects RAG context 768→256 to match GNN output dim
        self.rag_projection = nn.Linear(GRAPHSAGE_INPUT_DIM, GRAPHSAGE_HIDDEN_DIM)

        # 256 GNN + 256 RAG = 512 input to each risk head
        self.readmission_head = RiskHead(512)
        self.deterioration_head = RiskHead(512)
        self.medication_head = RiskHead(512)

    def forward(
        self,
        graph,
        rag_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode graph + RAG context → 3 risk predictions.

        Parameters
        ----------
        graph : HeteroData or Batch
            Patient heterogeneous graph (or batched graph).
        rag_context : torch.Tensor
            RAG context embedding, shape [batch_size, 768].
            Raw mean-pooled entity embeddings from similar patients.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: "readmission", "deterioration", "medication".
            Each value: shape [batch_size, 1] — raw logit.
        """
        # Graph embedding: [batch_size, 256]
        # Use batch_dict if available (from PyG Batch)
        batch_dict = getattr(graph, "batch_dict", None)
        graph_emb = self.encoder(graph.x_dict, graph.edge_index_dict, batch_dict)

        # RAG projection: [batch_size, 768] → [batch_size, 256]
        rag_emb = self.rag_projection(rag_context)

        # Concatenate: [batch_size, 512]
        combined = torch.cat([graph_emb, rag_emb], dim=-1)

        return {
            "readmission": self.readmission_head(combined),
            "deterioration": self.deterioration_head(combined),
            "medication": self.medication_head(combined),
        }