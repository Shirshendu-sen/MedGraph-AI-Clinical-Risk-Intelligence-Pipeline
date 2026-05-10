"""Layer 3 — Temporal Document Graph: Graph constants."""

from shared.constants import CLINICALBERT_MODEL  # do NOT re-define here

EMBEDDING_DIM          = 768
CO_OCCUR_WINDOW        = 3     # sentences within 3 of each other = co-occurrence edge
MIN_ENTITIES_PER_GRAPH = 2     # skip patient if fewer than 2 entities found
GRAPH_SAVE_FORMAT      = "pt"
