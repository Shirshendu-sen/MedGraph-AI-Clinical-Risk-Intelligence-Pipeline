import os

import cv2
import numpy as np
from PIL import Image
import pytesseract

from src.layer1.config import TESSERACT_CONFIG


def extract_text_from_image(image_path: str) -> dict:
    """Extract text from an image file using OCR with preprocessing.

    Returns:
        {file_name, text, char_count, preprocessing, error}
    """
    result = {
        "file_name": os.path.basename(image_path),
        "text": "",
        "char_count": 0,
        "preprocessing": [],
        "error": None,
    }

    if not os.path.isfile(image_path):
        result["error"] = f"File not found: {image_path}"
        return result

    if os.path.getsize(image_path) == 0:
        result["error"] = f"File is empty: {image_path}"
        return result

    try:
        preprocessed_img, preprocessing_steps = _preprocess_image(
            cv2.imread(image_path)
        )

        if preprocessed_img is None:
            result["error"] = f"Failed to read image: {image_path}"
            return result

        result["preprocessing"] = preprocessing_steps

        pil_img = Image.fromarray(preprocessed_img)
        ocr_text = pytesseract.image_to_string(
            pil_img, config=TESSERACT_CONFIG
        )

        result["text"] = ocr_text
        result["char_count"] = len(ocr_text)
    except Exception as e:
        result["error"] = str(e)

    return result


def _preprocess_image(img: np.ndarray) -> tuple:
    """Preprocess image for better OCR results.

    Steps:
        1. Convert to grayscale
        2. Upscale if width < 1000
        3. Gaussian blur denoise
        4. Adaptive threshold

    Returns:
        (preprocessed_img, preprocessing_steps_list)
    """
    steps = []

    if img is None:
        return None, steps

    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale")

    # Upscale if width < 1000
    height, width = img.shape
    if width < 1000:
        scale = 2.0
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        steps.append(f"upscale_{scale}x")

    # Denoise
    img = cv2.GaussianBlur(img, (1, 1), 0)
    steps.append("gaussian_blur")

    # Adaptive threshold
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    steps.append("adaptive_threshold")

    return img, steps
