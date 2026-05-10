import json
import os
from datetime import datetime
from pathlib import Path

from src.layer1.config import SUPPORTED_EXTENSIONS
from src.layer1.pdf_extractor import extract_text_from_pdf
from src.layer1.image_extractor import extract_text_from_image
from src.layer1.text_cleaner import clean_text


def run_pipeline(input_folder: str, output_folder: str) -> dict:
    """Batch-process all supported files in a folder.

    Returns:
        {processed, failed, empty, skipped, errors, output_files}
    """
    summary = {
        "processed": 0,
        "failed": 0,
        "empty": 0,
        "skipped": 0,
        "errors": [],
        "output_files": [],
    }

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover all supported files
    files = _discover_files(input_path)

    for file_path in files:
        result = _process_single_file(file_path, output_path)

        status = result.get("status", "failed")
        if status == "processed":
            summary["processed"] += 1
            summary["output_files"].append(str(result.get("output_path", "")))
        elif status == "empty":
            summary["empty"] += 1
        elif status == "skipped":
            summary["skipped"] += 1
        elif status == "failed":
            summary["failed"] += 1
            summary["errors"].append({
                "file": str(file_path),
                "error": result.get("error", "Unknown error"),
            })

    # Save pipeline summary
    _save_summary(summary, output_path)
    _print_summary(summary)

    return summary


def _discover_files(input_path: Path) -> list:
    """Recursively discover all files with supported extensions."""
    all_extensions = []
    for ext_list in SUPPORTED_EXTENSIONS.values():
        all_extensions.extend(ext_list)

    files = []
    for ext in all_extensions:
        files.extend(input_path.rglob(f"*{ext}"))

    return sorted(files)


def _process_single_file(file_path: Path, output_path: Path) -> dict:
    """Process a single file: extract → clean → save JSON.

    Returns:
        {status, output_path, error}
    """
    # Guard: 0-byte → status:"empty", skip
    if file_path.stat().st_size == 0:
        return {"status": "empty", "output_path": None, "error": None}

    suffix = file_path.suffix.lower()
    extraction_result = None

    try:
        # Extract based on file type
        if suffix in SUPPORTED_EXTENSIONS["pdf"]:
            extraction_result = extract_text_from_pdf(str(file_path))
        elif suffix in SUPPORTED_EXTENSIONS["image"]:
            extraction_result = extract_text_from_image(str(file_path))
        elif suffix in SUPPORTED_EXTENSIONS["text"]:
            extraction_result = _read_plain_text(str(file_path))
        else:
            return {
                "status": "skipped",
                "output_path": None,
                "error": f"Unsupported extension: {suffix}",
            }

        # Check for extraction errors
        if extraction_result.get("error"):
            return {
                "status": "failed",
                "output_path": None,
                "error": extraction_result["error"],
            }

        # Get raw text
        raw_text = extraction_result.get("full_text", "") or extraction_result.get("text", "")
        if not raw_text.strip():
            return {"status": "empty", "output_path": None, "error": None}

        # Clean text
        cleaning_result = clean_text(raw_text)

        # Build output document
        output_doc = {
            "metadata": {
                "source_file": str(file_path),
                "file_name": file_path.name,
                "file_type": suffix,
                "file_size_bytes": file_path.stat().st_size,
                "processed_at": datetime.now().isoformat(),
            },
            "extraction": extraction_result,
            "cleaning": cleaning_result,
            "content": {
                "raw_text": raw_text,
                "cleaned_text": cleaning_result["cleaned_text"],
                "sentences": cleaning_result["sentences"],
                "sentence_count": cleaning_result["sentence_count"],
            },
        }

        # Save JSON: {stem}_processed.json
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{file_path.stem}_processed.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_doc, f, indent=2, ensure_ascii=False)

        return {"status": "processed", "output_path": str(out_file), "error": None}

    except Exception as e:
        return {"status": "failed", "output_path": None, "error": str(e)}


def _read_plain_text(file_path: str) -> dict:
    """Read a plain text file trying multiple encodings.

    Tries utf-8, latin-1, cp1252 in order.
    """
    if not os.path.isfile(file_path):
        return {
            "file_name": os.path.basename(file_path),
            "text": "",
            "char_count": 0,
            "encoding_used": None,
            "error": f"File not found: {file_path}",
        }

    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            return {
                "file_name": os.path.basename(file_path),
                "text": text,
                "char_count": len(text),
                "encoding_used": encoding,
                "error": None,
            }
        except (UnicodeDecodeError, UnicodeError):
            continue

    return {
        "file_name": os.path.basename(file_path),
        "text": "",
        "char_count": 0,
        "encoding_used": None,
        "error": "Failed to decode file with any supported encoding",
    }


def _save_summary(summary: dict, output_path: Path) -> None:
    """Save pipeline summary to _pipeline_summary.json."""
    summary_file = output_path / "_pipeline_summary.json"
    summary["timestamp"] = datetime.now().isoformat()
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_summary(summary: dict) -> None:
    """Print a human-readable pipeline summary."""
    print("\n" + "=" * 50)
    print("  Layer 1 — Document Ingestion Pipeline Summary")
    print("=" * 50)
    print(f"  Processed : {summary['processed']}")
    print(f"  Empty     : {summary['empty']}")
    print(f"  Skipped   : {summary['skipped']}")
    print(f"  Failed    : {summary['failed']}")
    if summary["errors"]:
        print("\n  Errors:")
        for err in summary["errors"]:
            print(f"    - {err['file']}: {err['error']}")
    print("=" * 50 + "\n")
