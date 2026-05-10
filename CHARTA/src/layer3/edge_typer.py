"""Layer 3 — Temporal Document Graph: Build temporal, relation, co-occurrence edges."""

from __future__ import annotations

from layer3.config import CO_OCCUR_WINDOW


def build_temporal_edges(visits: list[dict]) -> list[tuple[int, int, str]]:
    """Build temporal edges between consecutive visits.

    Visits are sorted by their earliest entity timestamp and consecutive
    pairs are linked with a "before" relation.

    Parameters
    ----------
    visits : list[dict]
        Each visit dict must have an "earliest_ts" key (sortable) and a
        "visit_idx" key (integer node index).

    Returns
    -------
    list[tuple[int, int, str]]
        List of (src_idx, dst_idx, "before") tuples.
    """
    if len(visits) < 2:
        return []

    sorted_visits = sorted(visits, key=lambda v: v.get("earliest_ts", ""))
    edges: list[tuple[int, int, str]] = []
    for i in range(len(sorted_visits) - 1):
        src_idx = sorted_visits[i]["visit_idx"]
        dst_idx = sorted_visits[i + 1]["visit_idx"]
        edges.append((src_idx, dst_idx, "before"))
    return edges


def build_relation_edges(
    relations: list[dict],
    entity_index: dict[str, int],
) -> list[tuple[int, int, str]]:
    """Build edges from extracted relations between entities.

    Parameters
    ----------
    relations : list[dict]
        Each relation dict must have "entity_1", "entity_2" (concept_ids),
        and "relation_type" keys.
    entity_index : dict[str, int]
        Mapping from concept_id to integer node index.

    Returns
    -------
    list[tuple[int, int, str]]
        List of (src_idx, dst_idx, relation_type) tuples.
    """
    edges: list[tuple[int, int, str]] = []
    for rel in relations:
        idx1 = entity_index.get(rel["entity_1"])
        idx2 = entity_index.get(rel["entity_2"])
        if idx1 is not None and idx2 is not None:
            edges.append((idx1, idx2, rel["relation_type"]))
    return edges


def build_cooccurrence_edges(
    entities: list[dict],
    window: int = CO_OCCUR_WINDOW,
    relation_edge_set: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, str]]:
    """Build co-occurrence edges between entities within a sentence window.

    Two entities are linked if their sentence indices differ by at most
    ``window``. Edges that already exist in ``relation_edge_set`` are
    skipped to avoid duplication.

    Parameters
    ----------
    entities : list[dict]
        Each entity dict must have "sentence_idx" and "node_idx" keys.
    window : int
        Maximum sentence distance for co-occurrence (default: CO_OCCUR_WINDOW).
    relation_edge_set : set[tuple[int, int]] | None
        Set of (src_idx, dst_idx) pairs already present as relation edges.
        These pairs are excluded to avoid duplicates.

    Returns
    -------
    list[tuple[int, int, str]]
        List of (src_idx, dst_idx, "co_occurs_with") tuples.
    """
    if relation_edge_set is None:
        relation_edge_set = set()

    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            if abs(a["sentence_idx"] - b["sentence_idx"]) <= window:
                pair = (a["node_idx"], b["node_idx"])
                rev_pair = (b["node_idx"], a["node_idx"])
                if pair not in seen and pair not in relation_edge_set and rev_pair not in relation_edge_set:
                    edges.append((a["node_idx"], b["node_idx"], "co_occurs_with"))
                    seen.add(pair)

    return edges
