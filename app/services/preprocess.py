from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

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

    def generate(
        self,
        image: Image.Image,
        *,
        allowed_labels: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, Image.Image]]:
        """Return the available image variants.

        Some OCR engines hoạt động tốt hơn khi bỏ qua các bước xử lý ảnh
        nhất định. Để hỗ trợ điều này, cho phép truyền vào ``allowed_labels``
        nhằm giới hạn các biến thể trả về mà vẫn duy trì chuỗi xử lý cho các
        bước tiếp theo.
        """

        outputs: List[Tuple[str, Image.Image]] = []
        current = image
        allowed: Optional[Set[str]] = set(allowed_labels) if allowed_labels is not None else None
        for label, fn in self.steps:
            base_image = image if label == "original" else current
            next_image = fn(base_image)
            if allowed is None or label in allowed:
                outputs.append((label, next_image))
            current = next_image
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
