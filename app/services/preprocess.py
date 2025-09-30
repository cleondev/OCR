from __future__ import annotations

from typing import List, Tuple

from PIL import Image, ImageFilter, ImageOps


class ImagePreprocessor:
    """Apply a simple but effective preprocessing pipeline."""

    def __init__(self) -> None:
        self.steps = (
            ("original", self._identity),
            ("grayscale", self._to_grayscale),
            ("contrast", self._enhance_contrast),
            ("median_filter", self._median_filter),
            ("threshold", self._threshold),
        )

    def generate(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        outputs: List[Tuple[str, Image.Image]] = []
        current = image
        for label, fn in self.steps:
            current = fn(current if label != "original" else image)
            outputs.append((label, current))
        return outputs

    @staticmethod
    def _identity(image: Image.Image) -> Image.Image:
        return image.copy()

    @staticmethod
    def _to_grayscale(image: Image.Image) -> Image.Image:
        return ImageOps.grayscale(image)

    @staticmethod
    def _enhance_contrast(image: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(image)

    @staticmethod
    def _median_filter(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def _threshold(image: Image.Image) -> Image.Image:
        if image.mode != "L":
            image = ImageOps.grayscale(image)
        return image.point(lambda p: 255 if p > 160 else 0, mode="1").convert("L")
