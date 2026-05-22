"""Layer 4 — Single readmission prediction head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU

from layer4.config import (
    GRAPHSAGE_OUT_DIM,
    READMISSION_HEAD_DIM,
    GRAPHSAGE_DROPOUT,
)
from layer4.graph_model import ClinicalGraphSAGE


class ReadmissionHead(torch.nn.Module):
    """Prediction head that maps graph embeddings to 30-day readmission risk.

    ⚠️ BUG-N4 FIX: NO Sigmoid here — BCEWithLogitsLoss in trainer.py
    applies sigmoid internally during training. Adding Sigmoid here
    causes double-sigmoid, which squashes gradients and prevents
    convergence. Apply ``torch.sigmoid()`` only at inference time in
    ``pipeline.py`` AFTER the model forward pass.

    Architecture
    ────────────
    Linear(256 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1)

    Output is a **raw logit**, NOT a probability.
    """

    def __init__(
        self,
        input_dim: int = GRAPHSAGE_OUT_DIM,
        hidden_dim: int = READMISSION_HEAD_DIM,
        dropout: float = GRAPHSAGE_DROPOUT,
    ):
        """Initialise ReadmissionHead.

        Parameters
        ----------
        input_dim : int
            Input dimension (matches GNN output dim, default: 256).
        hidden_dim : int
            Intermediate hidden dimension (default: 128).
        dropout : float
            Dropout probability (default: 0.3).
        """
        super().__init__()

        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.drop = Dropout(dropout)
        self.fc2 = Linear(hidden_dim, 1)  # raw logit output — NO sigmoid

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: graph embedding → readmission logit.

        Parameters
        ----------
        graph_emb : torch.Tensor
            Patient-level graph embedding of shape ``[batch_size, input_dim]``.

        Returns
        -------
        torch.Tensor
            Raw logit of shape ``[batch_size, 1]`` — NOT a probability.
            Apply ``torch.sigmoid()`` externally at inference time.
        """
        x = self.fc1(graph_emb)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x  # raw logit — NO sigmoid


class ReadmissionRiskModel(torch.nn.Module):
    """Combined model: ClinicalGraphSAGE encoder + ReadmissionHead.

    This is the full end-to-end model used for training and inference.
    """

    def __init__(self, graph_model: ClinicalGraphSAGE):
        """Initialise ReadmissionRiskModel.

        Parameters
        ----------
        graph_model : ClinicalGraphSAGE
            The GNN encoder that produces patient-level embeddings.
        """
        super().__init__()

        self.encoder = graph_model
        self.readmission_head = ReadmissionHead(
            input_dim=graph_model.out_dim,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass: HeteroData → readmission logit.

        Parameters
        ----------
        x_dict : dict[str, torch.Tensor]
            Node feature dicts from HeteroData.
        edge_index_dict : dict
            Edge index dicts from HeteroData.
        batch_dict : dict[str, torch.Tensor] | None
            Batch assignment vectors from PyG Batch._batch_dict.
            Propagated to ClinicalGraphSAGE.forward() for correct
            per-graph pooling in batched training.

        Returns
        -------
        torch.Tensor
            Raw logit of shape ``[batch_size, 1]`` — NOT a probability.
        """
        graph_emb = self.encoder(x_dict, edge_index_dict, batch_dict)  # [batch_size, 256]
        return self.readmission_head(graph_emb)                        # [batch_size, 1] — raw logit