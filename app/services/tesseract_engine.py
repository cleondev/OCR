from __future__ import annotations

import statistics
from typing import Optional

import pytesseract
from pytesseract import Output
from PIL import Image

from ..config import settings
from .ocr_base import OCREngine, OcrOutput


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: Optional[str] = None) -> None:
        initial = (lang or settings.tess_lang).strip()
        self.lang = initial or settings.tess_lang

    def set_language(self, lang: Optional[str]) -> None:
        candidate = (lang or settings.tess_lang).strip()
        self.lang = candidate or settings.tess_lang

    def run(self, image: Image.Image) -> OcrOutput:
        text = pytesseract.image_to_string(image, lang=self.lang)
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=Output.DICT)
        confidences = [float(conf) for conf in data.get("conf", []) if conf not in {"-1", "-1.0"}]
        confidence = statistics.mean(confidences) if confidences else None
        return OcrOutput(text=text.strip(), confidence=confidence)
