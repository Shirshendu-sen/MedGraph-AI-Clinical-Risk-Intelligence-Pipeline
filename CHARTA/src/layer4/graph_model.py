"""Layer 4 — RAG-Augmented Risk Inference: ClinicalGraphSAGE model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, global_mean_pool

from layer4.config import (
    GRAPHSAGE_INPUT_DIM,
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_NUM_LAYERS,
)


class ClinicalGraphSAGE(torch.nn.Module):
    """GraphSAGE encoder for heterogeneous patient graphs.

    Applies SAGEConv layers on entity node features, with BatchNorm,
    ReLU activation, and Dropout. Produces a 256-dim graph-level
    embedding via global mean pooling.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (default: 768, matching ClinicalBERT).
    hidden_dim : int
        Hidden layer dimension (default: 256).
    out_dim : int
        Output dimension (default: 256).
    num_layers : int
        Number of SAGEConv layers (default: 2).
    metadata : tuple, optional
        HeteroData metadata (node_types, edge_types). Accepted for
        compatibility with trainer but not used in this homogeneous
        entity-only architecture.
    """

    def __init__(
        self,
        in_dim: int = GRAPHSAGE_INPUT_DIM,
        hidden_dim: int = GRAPHSAGE_HIDDEN_DIM,
        out_dim: int = GRAPHSAGE_HIDDEN_DIM,
        num_layers: int = GRAPHSAGE_NUM_LAYERS,
        metadata=None,
    ):
        super().__init__()

        # Store metadata for reference (not used in forward — entity-only)
        self.metadata = metadata

        # Build SAGEConv layers: in_dim → hidden_dim → out_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])

        # BatchNorm for each hidden/output layer
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(dims[i + 1])
            for i in range(num_layers)
        ])

        self.drop = torch.nn.Dropout(0.3)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        batch_dict: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE layers on entity nodes.

        Parameters
        ----------
        x_dict : dict
            Node feature dicts from HeteroData.x_dict.
            Must contain "entity" key with shape [N_entities, in_dim].
        edge_index_dict : dict
            Edge index dicts from HeteroData.edge_index_dict.
        batch_dict : dict or None
            Batch assignment dicts from PyG Batch.batch_dict.
            Used for global_mean_pool on batched graphs.

        Returns
        -------
        torch.Tensor
            Graph-level embedding, shape [batch_size, out_dim].
        """
        # We operate on entity nodes only — they have the richest features
        x = x_dict["entity"]  # [N_entities, in_dim]

        # Build homogeneous edge_index from all entity↔entity edge types
        entity_edge_types = [
            ("entity", "relates_to", "entity"),
            ("entity", "co_occurs_with", "entity"),
        ]
        edge_indices = []
        for et in entity_edge_types:
            if et in edge_index_dict:
                ei = edge_index_dict[et]
                if ei.numel() > 0:
                    edge_indices.append(ei)

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)  # [2, total_entity_edges]
        else:
            # No entity edges — create empty edge_index
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Apply SAGEConv layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)

        # Graph-level pooling: mean-pool all entity node embeddings
        # Use batch_dict if available (batched graphs from DataLoader)
        if batch_dict is not None and "entity" in batch_dict:
            batch = batch_dict["entity"]
        elif hasattr(x_dict.get("entity", None), "batch") and x_dict["entity"].batch is not None:
            batch = x_dict["entity"].batch
        else:
            # Single graph — all nodes belong to graph 0
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_emb = global_mean_pool(x, batch)  # [batch_size, out_dim]
        return graph_emb


def get_patient_embedding(graph: HeteroData, model: ClinicalGraphSAGE) -> np.ndarray:
    """Extract a 256-dim patient embedding from a graph using the trained model.

    Parameters
    ----------
    graph : HeteroData
        A single patient heterogeneous graph.
    model : ClinicalGraphSAGE
        The trained GraphSAGE model.

    Returns
    -------
    np.ndarray
        Shape (256,) — the patient graph embedding.
    """
    model.eval()
    with torch.no_grad():
        emb = model(graph.x_dict, graph.edge_index_dict)
    return emb.cpu().numpy().flatten()  # [256]