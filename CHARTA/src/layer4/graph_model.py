"""Layer 4 — ClinicalGraphSAGE model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData

from layer4.config import (
    GRAPHSAGE_IN_DIM,
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_OUT_DIM,
    GRAPHSAGE_NUM_LAYERS,
    GRAPHSAGE_DROPOUT,
)


class ClinicalGraphSAGE(torch.nn.Module):
    """GraphSAGE GNN for encoding clinical patient graphs.

    Operates on the ``entity`` node type of a HeteroData graph.
    After message-passing, applies ``global_mean_pool`` over entity
    nodes to produce a patient-level embedding of shape ``[batch_size, out_dim]``.

    Architecture
    ────────────
    - 2 SAGEConv layers: in_dim → hidden_dim → out_dim
    - LayerNorm after each conv (safe for any node count, unlike BatchNorm1d)
    - ReLU activation
    - Dropout (0.3)
    - global_mean_pool on entity node features
    """

    def __init__(
        self,
        in_dim: int = GRAPHSAGE_IN_DIM,
        hidden_dim: int = GRAPHSAGE_HIDDEN_DIM,
        out_dim: int = GRAPHSAGE_OUT_DIM,
        num_layers: int = GRAPHSAGE_NUM_LAYERS,
        dropout: float = GRAPHSAGE_DROPOUT,
    ):
        """Initialise ClinicalGraphSAGE.

        Parameters
        ----------
        in_dim : int
            Input feature dimension (768 = ClinicalBERT [CLS] dim).
        hidden_dim : int
            Hidden layer dimension.
        out_dim : int
            Output embedding dimension.
        num_layers : int
            Number of SAGEConv layers.
        dropout : float
            Dropout probability.
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        # ── SAGEConv layers ──────────────────────────────────────────
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

        # ── LayerNorm layers (replaces BatchNorm1d — safe for N=1) ──
        # BatchNorm1d computes variance across nodes (dim 0); if only 1
        # entity node exists, variance=0 → NaN.  LayerNorm normalizes
        # across features (dim 1) per node, which is always safe.
        self.norms = ModuleList([
            LayerNorm(dims[i + 1]) for i in range(num_layers)
        ])

        # ── Dropout ──────────────────────────────────────────────────
        self.drop = Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE on entity nodes.

        Parameters
        ----------
        x_dict : dict[str, torch.Tensor]
            Node feature dicts from HeteroData.x_dict.
            Must contain ``"entity"`` key with shape ``[N_entities, in_dim]``.
        edge_index_dict : dict
            Edge index dicts from HeteroData.edge_index_dict.
        batch_dict : dict[str, torch.Tensor] | None
            Batch assignment vectors from PyG Batch._batch_dict.
            Maps each node to its graph index in the batch.
            If None, assumes single-graph inference (all nodes → graph 0).

        Returns
        -------
        torch.Tensor
            Patient-level embedding of shape ``[batch_size, out_dim]``.
        """
        # ── BUG-FIX: Gracefully handle missing "entity" key ───────────
        # If a graph has zero entity nodes or the key is absent
        # (e.g. after collation of heterogeneous batches), return a
        # zero embedding instead of crashing with KeyError.
        if "entity" not in x_dict or x_dict["entity"].numel() == 0:
            # Determine batch size from other node types or default to 1
            batch_size = 1
            if batch_dict is not None and "entity" in batch_dict:
                batch_size = int(batch_dict["entity"].max().item()) + 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.out_dim, device=device)

        # Extract entity node features
        entity_x = x_dict["entity"]  # [N_entities, in_dim]

        # ── NaN/Inf guard: sanitize input features ────────────────────
        if torch.isnan(entity_x).any() or torch.isinf(entity_x).any():
            entity_x = torch.nan_to_num(entity_x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Message-passing on entity-entity edges ───────────────────
        # We use the homogeneous entity-to-entity edge types:
        #   ("entity", "relates_to", "entity") and ("entity", "co_occurs_with", "entity")
        # Merge them into a single edge_index for SAGEConv (which is homogeneous).
        entity_edge_indices = []

        for edge_type in [
            ("entity", "relates_to", "entity"),
            ("entity", "co_occurs_with", "entity"),
        ]:
            if edge_type in edge_index_dict:
                ei = edge_index_dict[edge_type]
                # Validate edge indices are in-bounds for num nodes
                if ei.numel() > 0:
                    max_idx = ei.max().item() if ei.numel() > 0 else -1
                    if max_idx >= entity_x.size(0):
                        # Out-of-bounds edge indices — filter them
                        mask = (ei[0] < entity_x.size(0)) & (ei[1] < entity_x.size(0))
                        ei = ei[:, mask]
                    if ei.numel() > 0:
                        entity_edge_indices.append(ei)

        if entity_edge_indices:
            # Concatenate all entity-to-entity edges
            edge_index = torch.cat(entity_edge_indices, dim=1)  # [2, E_total]
        else:
            # No entity edges — use empty edge_index (self-loops only via SAGEConv)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=entity_x.device)

        # ── Apply SAGEConv layers ────────────────────────────────────
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            entity_x = conv(entity_x, edge_index)
            entity_x = norm(entity_x)
            entity_x = F.relu(entity_x)
            entity_x = self.drop(entity_x)

            # NaN guard after each layer
            if torch.isnan(entity_x).any():
                entity_x = torch.nan_to_num(entity_x, nan=0.0)

        # ── Global pooling → patient embedding ───────────────────────
        # BUG-FIX: The batch vector MUST come from PyG's Batch object,
        # not from x_dict (which only contains node features).
        # For batched data: batch_dict["entity"] maps each entity node
        #   to its graph index in the batch → correct per-graph pooling.
        # For single graph: all nodes belong to graph 0 → zeros vector.
        if batch_dict is not None and "entity" in batch_dict:
            batch_vector = batch_dict["entity"]  # [N_entities]
        else:
            # Single-graph inference: all entity nodes → one patient
            batch_vector = torch.zeros(entity_x.size(0), dtype=torch.long, device=entity_x.device)

        patient_emb = global_mean_pool(entity_x, batch_vector)  # [batch_size, out_dim]

        # Final NaN/Inf guard on output embedding
        if torch.isnan(patient_emb).any() or torch.isinf(patient_emb).any():
            patient_emb = torch.nan_to_num(patient_emb, nan=0.0, posinf=0.0, neginf=0.0)

        return patient_emb


def get_patient_embedding(graph: HeteroData, model: ClinicalGraphSAGE) -> np.ndarray:
    """Extract a patient-level embedding from a single graph.

    Parameters
    ----------
    graph : HeteroData
        A single patient graph from Layer 3.
    model : ClinicalGraphSAGE
        The trained GNN encoder.

    Returns
    -------
    np.ndarray
        Patient embedding of shape ``[out_dim]`` (default: [256]).
    """
    model.eval()
    device = next(model.parameters()).device

    # Move graph to model device
    graph = graph.to(device)

    with torch.no_grad():
        # Single graph: no batch_dict needed (defaults to zeros vector)
        embedding = model(graph.x_dict, graph.edge_index_dict)  # [1, out_dim]

    return embedding.squeeze(0).cpu().numpy()  # [out_dim]