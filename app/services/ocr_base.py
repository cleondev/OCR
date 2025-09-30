from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass
class OcrOutput:
    text: str
    confidence: float | None


class OCREngine(Protocol):
    name: str

    def run(self, image: Image.Image) -> OcrOutput:
        ...
