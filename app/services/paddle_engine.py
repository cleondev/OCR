from __future__ import annotations

import statistics
from typing import Optional

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from ..config import settings
from .ocr_base import OCREngine, OcrOutput


class PaddleOCREngine:
    name = "paddle"

    def __init__(self, lang: Optional[str] = None) -> None:
        self.lang = lang or settings.paddle_lang
        self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    def run(self, image: Image.Image) -> OcrOutput:
        np_image = np.array(image.convert("RGB"))
        results = self._ocr.ocr(np_image, cls=True)
        texts = []
        confidences = []
        for res in results:
            for line in res:
                text, confidence = line[1][0], float(line[1][1])
                texts.append(text)
                confidences.append(confidence)
        aggregated_text = "\n".join(texts)
        confidence = statistics.mean(confidences) if confidences else None
        return OcrOutput(text=aggregated_text.strip(), confidence=confidence)
