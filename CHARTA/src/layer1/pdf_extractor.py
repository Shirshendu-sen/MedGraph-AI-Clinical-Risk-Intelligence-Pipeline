import os

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from src.layer1.config import (
    MIN_NATIVE_CHARS,
    PDF_RENDER_DPI,
    TESSERACT_CONFIG,
)


def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract text from a PDF file using native extraction with OCR fallback.

    Returns:
        {file_name, total_pages, pages: list[dict], full_text, error}
    """
    result = {
        "file_name": os.path.basename(pdf_path),
        "total_pages": 0,
        "pages": [],
        "full_text": "",
        "error": None,
    }

    # Guard: file must exist and size > 0 bytes
    if not os.path.isfile(pdf_path):
        result["error"] = f"File not found: {pdf_path}"
        return result

    if os.path.getsize(pdf_path) == 0:
        result["error"] = f"File is empty: {pdf_path}"
        return result

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["total_pages"] = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_data = _extract_single_page(page, page_num, pdf_path)
                result["pages"].append(page_data)

        # Join pages with "\n\n" → full_text
        result["full_text"] = "\n\n".join(
            p["text"] for p in result["pages"]
        )
    except Exception as e:
        result["error"] = str(e)

    return result


def _extract_single_page(page, page_num: int, pdf_path: str) -> dict:
    """Extract text from a single PDF page.

    Returns:
        {page_num, method, text, char_count}
    """
    native_text = page.extract_text() or ""

    if len(native_text) >= MIN_NATIVE_CHARS:
        return {
            "page_num": page_num,
            "method": "native",
            "text": native_text,
            "char_count": len(native_text),
        }
    else:
        return _ocr_pdf_page_with_fitz(pdf_path, page_num)


def _ocr_pdf_page_with_fitz(pdf_path: str, page_num: int) -> dict:
    """OCR a single PDF page using PyMuPDF (fitz) rendering + Tesseract.

    Uses PyMuPDF (fitz) — NO Poppler dependency.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # fitz is 0-indexed
    mat = fitz.Matrix(PDF_RENDER_DPI / 72, PDF_RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

    return {
        "page_num": page_num,
        "method": "ocr",
        "text": ocr_text,
        "char_count": len(ocr_text),
    }
