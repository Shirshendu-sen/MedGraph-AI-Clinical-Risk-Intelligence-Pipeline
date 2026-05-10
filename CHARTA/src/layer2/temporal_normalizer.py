# Layer 2 — Clinical NLP Extraction: Temporal Normalizer

"""Date extraction + ISO normalisation module."""

import logging
import re
from datetime import datetime

import dateparser
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

TEMPORAL_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                        # ISO: 2023-09-12
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                  # 09/12/2023
    r"\b\d{1,2} (?:days?|weeks?|months?) ago\b",
]


def extract_temporal_expressions(text: str) -> list[dict]:
    """Find all temporal expressions in text using regex patterns.

    Parameters
    ----------
    text : str
        Input text to search for temporal expressions.

    Returns
    -------
    list[dict]
        List of dicts with keys: raw_text, start_char, end_char.
    """
    expressions = []
    for pattern in TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            expressions.append({
                "raw_text":   match.group(),
                "start_char": match.start(),
                "end_char":   match.end(),
            })
    return expressions


def normalize_to_iso(raw_date: str) -> str:
    """Normalise a raw date string to ISO 8601 format (YYYY-MM-DD).

    Parameters
    ----------
    raw_date : str
        Raw date string extracted from text.

    Returns
    -------
    str
        ISO-formatted date string, or the original raw_date on failure.
    """
    # Try dateparser first — handles relative dates like "2 months ago", "3 weeks ago"
    parsed = dateparser.parse(raw_date)
    if parsed:
        return parsed.strftime("%Y-%m-%d")

    # Fallback to dateutil for standard date formats
    try:
        dt = dateutil_parser.parse(raw_date, default=datetime(2000, 1, 1))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        logger.warning("Could not normalise date: '%s'", raw_date)
        return raw_date


def attach_timestamps_to_entities(
    entities: list[dict],
    temporal_exprs: list[dict],
) -> list[dict]:
    """Attach the nearest temporal expression (ISO-normalised) to each entity.

    Finds the nearest temporal_expr in the same sentence context for each
    entity based on character proximity.

    Parameters
    ----------
    entities : list[dict]
        Entity dicts with start_char, end_char, sentence_idx.
    temporal_exprs : list[dict]
        Temporal expression dicts with raw_text, start_char, end_char.

    Returns
    -------
    list[dict]
        Same entity dicts with an added "timestamp" key (ISO string or None).
    """
    if not temporal_exprs:
        for entity in entities:
            entity["timestamp"] = None
        return entities

    for entity in entities:
        # Find nearest temporal expression by character distance
        best_dist = float("inf")
        best_expr = None

        for expr in temporal_exprs:
            # Distance from entity to temporal expression
            dist = min(
                abs(entity["start_char"] - expr["end_char"]),
                abs(entity["end_char"] - expr["start_char"]),
            )
            if dist < best_dist:
                best_dist = dist
                best_expr = expr

        if best_expr is not None:
            entity["timestamp"] = normalize_to_iso(best_expr["raw_text"])
        else:
            entity["timestamp"] = None

    return entities
