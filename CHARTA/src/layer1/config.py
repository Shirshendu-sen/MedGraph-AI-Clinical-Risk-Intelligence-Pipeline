import sys

# OS-aware Tesseract path — do NOT hardcode for Linux/Mac
if sys.platform == "win32":
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    TESSERACT_CMD = "tesseract"  # on PATH for Linux/Mac

MIN_NATIVE_CHARS = 20       # chars below this → treat page as scanned
PDF_RENDER_DPI   = 300      # DPI for PyMuPDF page rendering
TESSERACT_CONFIG = "--psm 3 --oem 3"
SUPPORTED_EXTENSIONS = {
    "pdf":   [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
    "text":  [".txt"],
}
