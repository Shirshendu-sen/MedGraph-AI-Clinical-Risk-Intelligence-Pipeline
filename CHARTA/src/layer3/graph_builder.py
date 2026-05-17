"""Layer 3 — Temporal Document Graph: Assemble HeteroData graph."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from layer3.config import CO_OCCUR_WINDOW, EMBEDDING_DIM, MIN_ENTITIES_PER_GRAPH
from layer3.edge_typer import (
    build_cooccurrence_edges,
    build_relation_edges,
    build_temporal_edges,
)
from layer3.node_encoder import encode_entity_nodes


def build_patient_graph(
    extracted_docs: list[dict],
    patient_id: str,
    tokenizer,        # loaded ONCE in pipeline.py — do NOT call load_encoder() here
    encoder_model,    # same — calling it per-patient reloads 1.3 GB repeatedly
) -> HeteroData:
    """Build a heterogeneous knowledge graph for a single patient.

    Parameters
    ----------
    extracted_docs : list[dict]
        Layer 2 output documents belonging to this patient.
    patient_id : str
        Patient identifier.
    tokenizer : AutoTokenizer
        Pre-loaded ClinicalBERT tokenizer.
    encoder_model : AutoModel
        Pre-loaded ClinicalBERT model.

    Returns
    -------
    HeteroData
        The assembled heterogeneous graph.
    """
    # ── Collect all entities from all docs into a flat list ──
    all_entities: list[dict] = []
    all_relations: list[dict] = []
    doc_filenames: list[str] = []

    for doc in extracted_docs:
        filename = doc.get("filename", doc.get("doc_id", ""))
        doc_filenames.append(filename)
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])
        for ent in entities:
            ent["_doc_filename"] = filename
        for rel in relations:
            rel["_doc_filename"] = filename
        all_entities.extend(entities)
        all_relations.extend(relations)

    # ── Build entity_index: {concept_id → integer_node_idx} ──
    # When concept_id is "UNKNOWN" or None, fall back to entity text
    # so each unique entity gets its own node (not collapsed into one).
    entity_index: dict[str, int] = {}
    for ent in all_entities:
        cid = ent.get("concept_id")
        if cid in (None, "UNKNOWN", ""):
            cid = ent["text"]
        if cid not in entity_index:
            entity_index[cid] = len(entity_index)
        ent["concept_id"] = cid
        ent["node_idx"] = entity_index[cid]

    # ── Build visit_index: {doc_filename → integer_visit_idx} ──
    visit_index: dict[str, int] = {}
    for filename in doc_filenames:
        if filename not in visit_index:
            visit_index[filename] = len(visit_index)

    # ── Node embeddings ──
    node_embeddings = encode_entity_nodes(all_entities, tokenizer, encoder_model)

    # entity_x: [N_unique_entities, 768] — one row per unique concept_id
    sorted_concept_ids = sorted(entity_index.keys(), key=lambda cid: entity_index[cid])
    entity_x = torch.stack(
        [torch.tensor(node_embeddings[cid], dtype=torch.float32) for cid in sorted_concept_ids]
    )

    # visit_x: mean-pool entity_x per visit group → [N_visits, 768]
    visit_entity_indices: dict[int, list[int]] = {}
    for ent in all_entities:
        vidx = visit_index[ent["_doc_filename"]]
        visit_entity_indices.setdefault(vidx, []).append(ent["node_idx"])

    visit_x = torch.zeros(len(visit_index), EMBEDDING_DIM, dtype=torch.float32)
    for vidx, ent_indices in visit_entity_indices.items():
        visit_x[vidx] = entity_x[ent_indices].mean(dim=0)

    # patient_x: [1, 768]
    patient_x = entity_x.mean(dim=0).unsqueeze(0)

    # ── Edge: entity → visit membership (occurs_in) ──
    occurs_in_src: list[int] = []
    occurs_in_dst: list[int] = []
    for ent in all_entities:
        occurs_in_src.append(ent["node_idx"])
        occurs_in_dst.append(visit_index[ent["_doc_filename"]])
    edge_index_occurs_in = torch.tensor([occurs_in_src, occurs_in_dst], dtype=torch.long)

    # ── Edge: visit → visit temporal (before) ──
    visits: list[dict] = []
    for filename, vidx in visit_index.items():
        # Find earliest timestamp among entities in this visit
        ents_in_visit = [e for e in all_entities if e["_doc_filename"] == filename]
        timestamps = [
            e.get("timestamp") or e.get("normalized_date") or ""
            for e in ents_in_visit
        ]
        # Filter out empty strings for min comparison, fallback to ""
        non_empty = [t for t in timestamps if t]
        earliest_ts = min(non_empty) if non_empty else ""
        visits.append({"visit_idx": vidx, "earliest_ts": earliest_ts})

    temporal_edges = build_temporal_edges(visits)
    if temporal_edges:
        before_src = [e[0] for e in temporal_edges]
        before_dst = [e[1] for e in temporal_edges]
        edge_index_before = torch.tensor([before_src, before_dst], dtype=torch.long)
    else:
        edge_index_before = torch.empty((2, 0), dtype=torch.long)

    # ── Edge: entity → entity relations ──
    relation_edges = build_relation_edges(all_relations, entity_index)
    if relation_edges:
        rel_src = [e[0] for e in relation_edges]
        rel_dst = [e[1] for e in relation_edges]
        edge_index_relates = torch.tensor([rel_src, rel_dst], dtype=torch.long)
    else:
        edge_index_relates = torch.empty((2, 0), dtype=torch.long)

    # ── Edge: entity → entity co-occurrence ──
    relation_edge_set: set[tuple[int, int]] = set()
    for e in relation_edges:
        relation_edge_set.add((e[0], e[1]))

    cooccurrence_edges = build_cooccurrence_edges(
        all_entities, CO_OCCUR_WINDOW, relation_edge_set
    )
    if cooccurrence_edges:
        co_src = [e[0] for e in cooccurrence_edges]
        co_dst = [e[1] for e in cooccurrence_edges]
        edge_index_cooccurs = torch.tensor([co_src, co_dst], dtype=torch.long)
    else:
        edge_index_cooccurs = torch.empty((2, 0), dtype=torch.long)

    # ── Collect visit dates (sorted by visit_idx) ──
    visit_dates: list[str] = []
    for vidx in range(len(visit_index)):
        visit_info = next(v for v in visits if v["visit_idx"] == vidx)
        visit_dates.append(visit_info["earliest_ts"])

    # ── Assemble HeteroData ──
    graph = HeteroData()
    graph["entity"].x = entity_x
    graph["visit"].x = visit_x
    graph["patient"].x = patient_x

    graph["entity", "occurs_in", "visit"].edge_index = edge_index_occurs_in
    graph["visit", "before", "visit"].edge_index = edge_index_before
    graph["entity", "relates_to", "entity"].edge_index = edge_index_relates
    graph["entity", "co_occurs_with", "entity"].edge_index = edge_index_cooccurs

    # Store metadata on graph for downstream meta JSON generation
    graph["patient"].patient_id = patient_id
    graph["entity"].num_nodes = entity_x.shape[0]
    graph["visit"].num_nodes = visit_x.shape[0]
    graph.entity_index = entity_index
    graph.visit_dates = visit_dates

    return graph


def validate_graph(graph: HeteroData) -> bool:
    """Validate a patient heterogeneous graph.

    Parameters
    ----------
    graph : HeteroData
        The graph to validate.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # Must have at least MIN_ENTITIES_PER_GRAPH entities
    n_entities = graph["entity"].x.shape[0]
    if n_entities < MIN_ENTITIES_PER_GRAPH:
        raise ValueError(
            f"Graph has {n_entities} entities, minimum is {MIN_ENTITIES_PER_GRAPH}"
        )

    # No NaN values in entity features
    if torch.isnan(graph["entity"].x).any():
        raise ValueError("Entity node features contain NaN values")

    # At least 1 edge type has non-empty edge_index
    edge_types = [
        ("entity", "occurs_in", "visit"),
        ("visit", "before", "visit"),
        ("entity", "relates_to", "entity"),
        ("entity", "co_occurs_with", "entity"),
    ]
    has_edges = any(
        graph[et].edge_index.numel() > 0 for et in edge_types
    )
    if not has_edges:
        raise ValueError("Graph has no edges in any edge type")

    return True
