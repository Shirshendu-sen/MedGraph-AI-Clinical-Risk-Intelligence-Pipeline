import re

import ftfy


# Common medical abbreviations for expansion
MEDICAL_ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "esi": "emergency severity index",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "nvd": "nausea vomiting diarrhea",
    "fhd": "family history of diabetes",
    "fh": "family history",
}


def clean_text(raw_text: str, expand_abbreviations: bool = True) -> dict:
    """Clean and normalise raw extracted text.

    Returns:
        {cleaned_text, sentences, sentence_count, original_length,
         cleaned_length, placeholder_count}
    """
    original_length = len(raw_text)
    text = raw_text

    # Fix encoding issues
    text = ftfy.fix_text(text)

    # Replace MIMIC-style placeholders
    text = re.sub(r"\[\*\*.*?\*\*\]", "[REDACTED]", text)

    # Count placeholders
    placeholder_count = text.count("[REDACTED]")

    # Remove non-printable control characters (keep \n, \t, space)
    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)

    # Collapse multiple spaces/tabs per line
    text = re.sub(r"[ \t]+", " ", text)

    # Remove lines that are page numbers, separator lines, or single-char lines
    lines = text.split("\n")
    lines = [line for line in lines if not _should_remove_line(line)]
    text = "\n".join(lines)

    # Collapse >1 consecutive blank lines to exactly 1
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Expand abbreviations
    if expand_abbreviations:
        text = _expand_abbreviations(text)

    # Segment into sentences
    sentences = _segment_sentences(text)

    return {
        "cleaned_text": text,
        "sentences": sentences,
        "sentence_count": len(sentences),
        "original_length": original_length,
        "cleaned_length": len(text),
        "placeholder_count": placeholder_count,
    }


def _should_remove_line(line: str) -> bool:
    """Return True for page numbers, separator lines, single-char lines."""
    stripped = line.strip()

    if not stripped:
        return False  # blank lines handled separately

    # Single character line
    if len(stripped) == 1:
        return True

    # Page number line (e.g., "Page 3", "3", "--- Page 3 ---")
    if re.match(r"^[-=\s]*page\s*\d+[-=\s]*$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^\d+$", stripped):
        return True

    # Separator line (e.g., "---", "===", "___")
    if re.match(r"^[-=_]{3,}$", stripped):
        return True

    return False


def _expand_abbreviations(text: str) -> str:
    """Expand common medical abbreviations in text."""
    for abbrev, expansion in MEDICAL_ABBREVIATIONS.items():
        # Match abbreviation as a whole word, case-insensitive
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def _segment_sentences(text: str) -> list:
    """Split text into sentences on [.!?] followed by a capital letter."""
    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
