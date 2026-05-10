"""Tests for Layer 1 — Document Ingestion."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from layer1.config import (
    MIN_NATIVE_CHARS,
    PDF_RENDER_DPI,
    SUPPORTED_EXTENSIONS,
    TESSERACT_CONFIG,
)
from layer1.text_cleaner import (
    MEDICAL_ABBREVIATIONS,
    _expand_abbreviations,
    _segment_sentences,
    _should_remove_line,
    clean_text,
)
from layer1.pipeline import (
    _discover_files,
    _process_single_file,
    _read_plain_text,
    run_pipeline,
)


SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


# ─── config.py tests ────────────────────────────────────────────────

class TestConfig:
    def test_min_native_chars_is_positive(self):
        assert MIN_NATIVE_CHARS > 0

    def test_pdf_render_dpi_is_reasonable(self):
        assert 150 <= PDF_RENDER_DPI <= 600

    def test_tesseract_config_is_string(self):
        assert isinstance(TESSERACT_CONFIG, str)
        assert "--psm" in TESSERACT_CONFIG

    def test_supported_extensions_has_expected_keys(self):
        assert "pdf" in SUPPORTED_EXTENSIONS
        assert "image" in SUPPORTED_EXTENSIONS
        assert "text" in SUPPORTED_EXTENSIONS

    def test_supported_extensions_contains_common_formats(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS["pdf"]
        assert ".jpg" in SUPPORTED_EXTENSIONS["image"]
        assert ".png" in SUPPORTED_EXTENSIONS["image"]
        assert ".txt" in SUPPORTED_EXTENSIONS["text"]


# ─── text_cleaner.py tests ──────────────────────────────────────────

class TestCleanText:
    def test_basic_cleaning(self):
        raw = "Hello   World\n\n\n\nFoo"
        result = clean_text(raw, expand_abbreviations=False)
        assert "Hello" in result["cleaned_text"]
        assert result["cleaned_length"] > 0
        # Multiple blank lines collapsed
        assert "\n\n\n" not in result["cleaned_text"]

    def test_mimic_placeholder_replacement(self):
        raw = "Patient [**Last Name**] presented with [**Date**] symptoms."
        result = clean_text(raw, expand_abbreviations=False)
        assert "[REDACTED]" in result["cleaned_text"]
        assert "[**" not in result["cleaned_text"]
        assert result["placeholder_count"] == 2

    def test_control_character_removal(self):
        raw = "Hello\x00World\x07Test\nNewline"
        result = clean_text(raw, expand_abbreviations=False)
        assert "\x00" not in result["cleaned_text"]
        assert "\x07" not in result["cleaned_text"]
        assert "Newline" in result["cleaned_text"]

    def test_space_collapse(self):
        raw = "Hello     World\t\tTab"
        result = clean_text(raw, expand_abbreviations=False)
        assert "     " not in result["cleaned_text"]
        assert "\t\t" not in result["cleaned_text"]

    def test_abbreviation_expansion(self):
        raw = "Patient has HTN and DM."
        result = clean_text(raw, expand_abbreviations=True)
        assert "hypertension" in result["cleaned_text"].lower()
        assert "diabetes mellitus" in result["cleaned_text"].lower()

    def test_abbreviation_expansion_disabled(self):
        raw = "Patient has HTN and DM."
        result = clean_text(raw, expand_abbreviations=False)
        assert "HTN" in result["cleaned_text"]
        assert "DM" in result["cleaned_text"]

    def test_sentence_segmentation(self):
        raw = "This is sentence one. This is sentence two! Is this sentence three?"
        result = clean_text(raw, expand_abbreviations=False)
        assert result["sentence_count"] >= 2
        assert len(result["sentences"]) == result["sentence_count"]

    def test_original_and_cleaned_length(self):
        raw = "Hello   World\n\n\n\nFoo"
        result = clean_text(raw, expand_abbreviations=False)
        assert result["original_length"] == len(raw)
        assert result["cleaned_length"] <= result["original_length"]

    def test_empty_string(self):
        result = clean_text("", expand_abbreviations=False)
        assert result["cleaned_text"] == ""
        assert result["sentence_count"] == 0

    def test_encoding_fix(self):
        raw = "Patient has f\u00c3\u00b6o symptoms"
        result = clean_text(raw, expand_abbreviations=False)
        assert result["cleaned_length"] > 0


class TestShouldRemoveLine:
    def test_page_number_line(self):
        assert _should_remove_line("Page 3") is True
        assert _should_remove_line("--- Page 3 ---") is True
        assert _should_remove_line("page 1") is True

    def test_numeric_only_line(self):
        assert _should_remove_line("42") is True

    def test_separator_line(self):
        assert _should_remove_line("---") is True
        assert _should_remove_line("===") is True
        assert _should_remove_line("___") is True

    def test_single_char_line(self):
        assert _should_remove_line("X") is True

    def test_normal_line_not_removed(self):
        assert _should_remove_line("Patient presents with chest pain.") is False

    def test_blank_line_not_removed(self):
        assert _should_remove_line("") is False


class TestExpandAbbreviations:
    def test_htn_expansion(self):
        result = _expand_abbreviations("Patient has HTN")
        assert "hypertension" in result.lower()

    def test_dm_expansion(self):
        result = _expand_abbreviations("History of DM")
        assert "diabetes mellitus" in result.lower()

    def test_case_insensitive(self):
        result = _expand_abbreviations("Patient has htn and dm")
        assert "hypertension" in result.lower()
        assert "diabetes mellitus" in result.lower()

    def test_no_expansion_in_middle_of_word(self):
        result = _expand_abbreviations("The administration")
        # "dm" inside "administration" should NOT be expanded
        assert "diabetes mellitus" not in result.lower() or "administration" in result


class TestSegmentSentences:
    def test_basic_segmentation(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = _segment_sentences(text)
        assert len(sentences) >= 2

    def test_no_sentences(self):
        text = "no punctuation here"
        sentences = _segment_sentences(text)
        assert len(sentences) == 1

    def test_empty_string(self):
        sentences = _segment_sentences("")
        assert len(sentences) == 0


# ─── pipeline.py tests ──────────────────────────────────────────────

class TestReadPlainText:
    def test_utf8_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello UTF-8 world", encoding="utf-8")
        result = _read_plain_text(str(f))
        assert result["text"] == "Hello UTF-8 world"
        assert result["encoding_used"] == "utf-8"
        assert result["error"] is None

    def test_latin1_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes("Héllo Wörld".encode("latin-1"))
        result = _read_plain_text(str(f))
        assert result["error"] is None
        assert result["char_count"] > 0

    def test_nonexistent_file(self):
        result = _read_plain_text("/nonexistent/path/file.txt")
        assert result["error"] is not None


class TestDiscoverFiles:
    def test_discovers_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        files = _discover_files(tmp_path)
        assert len(files) == 2

    def test_discovers_nested_files(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        files = _discover_files(tmp_path)
        assert len(files) == 1

    def test_ignores_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "readme.md").write_text("# Hello")
        files = _discover_files(tmp_path)
        assert len(files) == 0


class TestProcessSingleFile:
    def test_process_txt_file(self, tmp_path):
        # Create input file
        input_file = tmp_path / "input" / "test.txt"
        input_file.parent.mkdir(parents=True)
        input_file.write_text(
            "Patient [**Name**] has HTN and DM. This is a test.",
            encoding="utf-8",
        )
        output_dir = tmp_path / "output"

        result = _process_single_file(input_file, output_dir)
        assert result["status"] == "processed"
        assert result["output_path"] is not None

        # Verify output JSON
        out_path = Path(result["output_path"])
        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        assert "metadata" in doc
        assert "extraction" in doc
        assert "cleaning" in doc
        assert "content" in doc
        assert "[REDACTED]" in doc["content"]["cleaned_text"]
        assert "hypertension" in doc["content"]["cleaned_text"].lower()

    def test_process_empty_file(self, tmp_path):
        input_file = tmp_path / "input" / "empty.txt"
        input_file.parent.mkdir(parents=True)
        input_file.write_text("", encoding="utf-8")
        output_dir = tmp_path / "output"

        result = _process_single_file(input_file, output_dir)
        assert result["status"] == "empty"

    def test_process_zero_byte_file(self, tmp_path):
        input_file = tmp_path / "input" / "zero.txt"
        input_file.parent.mkdir(parents=True)
        input_file.write_bytes(b"")
        output_dir = tmp_path / "output"

        result = _process_single_file(input_file, output_dir)
        assert result["status"] == "empty"


class TestRunPipeline:
    def test_pipeline_with_sample_data(self, tmp_path):
        # Copy sample data to temp input dir
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for f in SAMPLE_DATA_DIR.glob("*.txt"):
            shutil.copy(f, input_dir / f.name)

        output_dir = tmp_path / "output"

        summary = run_pipeline(str(input_dir), str(output_dir))

        assert summary["processed"] == 3
        assert summary["failed"] == 0
        assert len(summary["output_files"]) == 3

        # Verify summary file exists
        summary_file = output_dir / "_pipeline_summary.json"
        assert summary_file.exists()

    def test_pipeline_empty_input_folder(self, tmp_path):
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        summary = run_pipeline(str(input_dir), str(output_dir))

        assert summary["processed"] == 0
        assert summary["failed"] == 0

    def test_pipeline_mixed_files(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Valid file
        (input_dir / "valid.txt").write_text(
            "Patient has HTN. Follow up in one week.", encoding="utf-8"
        )
        # Empty file
        (input_dir / "empty.txt").write_bytes(b"")
        # Unsupported file
        (input_dir / "data.csv").write_text("a,b,c", encoding="utf-8")

        output_dir = tmp_path / "output"
        summary = run_pipeline(str(input_dir), str(output_dir))

        assert summary["processed"] == 1
        assert summary["empty"] == 1


# ─── Integration: sample_data files ─────────────────────────────────

class TestSampleDataFiles:
    def test_sample_data_dir_exists(self):
        assert SAMPLE_DATA_DIR.exists()

    def test_sample_data_has_three_txt_files(self):
        txt_files = list(SAMPLE_DATA_DIR.glob("*.txt"))
        assert len(txt_files) == 3

    def test_sample_files_are_non_empty(self):
        for f in SAMPLE_DATA_DIR.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            assert len(content) > 0

    def test_sample_files_contain_mimic_placeholders(self):
        total_placeholders = 0
        for f in SAMPLE_DATA_DIR.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            total_placeholders += content.count("[**")
        assert total_placeholders > 0

    def test_sample_files_contain_medical_abbreviations(self):
        found_abbrevs = set()
        for f in SAMPLE_DATA_DIR.glob("*.txt"):
            content = f.read_text(encoding="utf-8")
            for abbrev in MEDICAL_ABBREVIATIONS:
                if abbrev.upper() in content.upper():
                    found_abbrevs.add(abbrev)
        assert len(found_abbrevs) > 0
