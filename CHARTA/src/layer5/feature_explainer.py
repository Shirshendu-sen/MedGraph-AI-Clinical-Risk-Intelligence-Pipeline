"""
Layer 5 – Explainable Clinical Report Generation
Lightweight feature importance from GNN embeddings.
"""

import torch
from torch_geometric.data import HeteroData

from layer5.config import NUM_TOP_FEATURES


def get_top_features(
    graph: HeteroData,
    entity_index: dict,
    name_index: dict,
    n: int = NUM_TOP_FEATURES,
) -> list[dict]:
    """Extracts the top-n entities by their embedding L2-norm magnitude
    as a lightweight proxy for contribution — no SHAP background corpus needed.

    ⚠️ BUG-N2 FIX: name_index must be passed in from graph_meta.json.
      entity_index maps {concept_id_or_text → node_idx}.
      reverse_index maps {node_idx → concept_id_or_text} — this is the concept ID, NOT the name.
      name_index maps {concept_id_or_text → human_readable_name} (e.g. "D006973" → "Hypertension").
      Using reverse_index for entity_name would store "D006973" in entity_name, which is wrong.

    Parameters
    ----------
    graph : HeteroData
        The heterogeneous patient graph from Layer 3.
    entity_index : dict
        Maps {concept_id_or_text → node_idx}.
    name_index : dict
        Maps {concept_id_or_text → human_readable_name}.
    n : int
        Number of top contributing entities to return (default NUM_TOP_FEATURES).

    Returns
    -------
    list[dict]
        Each dict: {"entity_name": str, "concept_id": str, "importance": float}
    """
    entity_embeddings = graph["entity"].x                       # [N_entities, 768]
    importance_scores = entity_embeddings.norm(dim=1)           # L2 norm → [N]

    top_indices = torch.argsort(importance_scores, descending=True)[:n]

    reverse_index = {v: k for k, v in entity_index.items()}     # {node_idx → concept_id_or_text}

    results = []
    for idx in top_indices:
        idx_val = idx.item()
        concept_id = reverse_index.get(idx_val, "")
        entity_name = name_index.get(concept_id, f"entity_{idx_val}")
        results.append(
            {
                "entity_name": entity_name,
                "concept_id": concept_id,
                "importance": float(importance_scores[idx_val]),
            }
        )

    return results