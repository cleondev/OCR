from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from ..config import settings
from .ocr_base import OCREngine, OcrOutput


PADDLE_VI_DICT_PATH = (
    Path(__file__).resolve().parent.parent / "resources" / "paddle_vi_dict.txt"
)
# PaddleOCR recognition heads for the latin model are trained with a
# dictionary of exactly 185 characters.  Supplying a longer list shifts the
# internal blank index and corrupts all decoded text (observed as spurious
# ``Ă`` characters between words).  Keep a hard limit here to avoid
# accidentally loading an incompatible dictionary file.
MAX_CUSTOM_DICT_CHARS = 185


logger = logging.getLogger(__name__)
# Đường dẫn tới từ điển tiếng Việt mở rộng cho PaddleOCR.


class PaddleOCREngine:
    name = "paddle"

    def __init__(self, lang: Optional[str] = None) -> None:
        initial = (lang or settings.paddle_lang).strip()
        self.lang = initial or settings.paddle_lang
        self._ocr: PaddleOCR | None = None

    def _ensure_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            ocr_kwargs = {"use_angle_cls": True, "lang": self.lang, "show_log": False}
            if self.lang.lower().startswith("vi"):
                dict_path = self._resolve_custom_dict()
                if dict_path is not None:
                    ocr_kwargs["rec_char_dict_path"] = dict_path
            self._ocr = PaddleOCR(**ocr_kwargs)
        return self._ocr

    def _resolve_custom_dict(self) -> Optional[str]:
        if not PADDLE_VI_DICT_PATH.exists():
            return None
        try:
            with PADDLE_VI_DICT_PATH.open("r", encoding="utf-8") as handle:
                # ``splitlines`` preserves the leading space entry that Paddle expects.
                entries = handle.read().splitlines()
        except OSError as exc:  # pragma: no cover - defensive guard
            logger.warning("Unable to read Paddle dictionary %s: %s", PADDLE_VI_DICT_PATH, exc)
            return None
        if len(entries) > MAX_CUSTOM_DICT_CHARS:
            logger.warning(
                "Ignoring custom Paddle dictionary %s: contains %d entries but the Latin "
                "recognition model only supports %d.",
                PADDLE_VI_DICT_PATH,
                len(entries),
                MAX_CUSTOM_DICT_CHARS,
            )
            return None
        return str(PADDLE_VI_DICT_PATH)

    def set_language(self, lang: Optional[str]) -> None:
        candidate = (lang or settings.paddle_lang).strip()
        new_lang = candidate or settings.paddle_lang
        if new_lang == self.lang and self._ocr is not None:
            return
        self.lang = new_lang
        # Khởi tạo lại PaddleOCR ở lần chạy kế tiếp để áp dụng ngôn ngữ mới.
        self._ocr = None

    def preferred_variants(self) -> tuple[str, ...]:
        """Các bước tiền xử lý phù hợp nhất cho PaddleOCR.

        PaddleOCR hoạt động tốt nhất khi giữ nguyên chi tiết và màu sắc của
        dấu tiếng Việt. Các bước làm nổi bật như ``threshold`` có xu hướng
        làm mất dấu, dẫn đến kết quả sai lệch. Vì vậy chỉ sử dụng các biến thể
        giữ nguyên thông tin quan trọng.
        """

        return ("original", "grayscale", "contrast")

    def run(self, image: Image.Image) -> OcrOutput:
        np_image = np.array(image.convert("RGB"))
        ocr = self._ensure_ocr()
        results = ocr.ocr(np_image, cls=True)
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
