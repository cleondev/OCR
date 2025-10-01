from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..config import settings
from .ocr_base import OCREngine, OcrOutput


logger = logging.getLogger(__name__)


class TrOCREngine:
    """OCR engine sử dụng mô hình ``microsoft/trocr`` từ Hugging Face."""

    name = "trocr"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = (model_name or settings.trocr_model_name).strip() or settings.trocr_model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def preferred_variants(self) -> tuple[str, ...]:
        """Ưu tiên giữ nguyên chi tiết và độ tương phản tự nhiên."""

        return ("original", "grayscale", "contrast")

    def _ensure_components(self) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        if self._processor is None:
            try:
                self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            except OSError as exc:  # pragma: no cover - chỉ log lỗi tải model
                logger.error("Không thể tải TrOCR processor %s: %s", self.model_name, exc)
                raise
        if self._model is None:
            try:
                model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            except OSError as exc:  # pragma: no cover - chỉ log lỗi tải model
                logger.error("Không thể tải TrOCR model %s: %s", self.model_name, exc)
                raise
            self._model = model.to(self.device)
            self._model.eval()
            self._ensure_generation_tokens()
        return self._processor, self._model

    def _ensure_generation_tokens(self) -> None:
        """Bổ sung các mã đặc biệt cần thiết cho quá trình sinh chuỗi."""

        if self._processor is None or self._model is None:  # pragma: no cover - defensive
            return

        tokenizer = self._processor.tokenizer
        generation_config = self._model.generation_config

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None and generation_config.pad_token_id is None:
            generation_config.pad_token_id = pad_token_id
        if pad_token_id is not None and getattr(self._model.config, "pad_token_id", None) is None:
            self._model.config.pad_token_id = pad_token_id

        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        cls_token_id = getattr(tokenizer, "cls_token_id", None)
        start_token_id = bos_token_id if bos_token_id is not None else cls_token_id
        if start_token_id is not None and generation_config.decoder_start_token_id is None:
            generation_config.decoder_start_token_id = start_token_id
        if start_token_id is not None and getattr(self._model.config, "decoder_start_token_id", None) is None:
            self._model.config.decoder_start_token_id = start_token_id

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and generation_config.eos_token_id is None:
            generation_config.eos_token_id = eos_token_id
        if eos_token_id is not None and getattr(self._model.config, "eos_token_id", None) is None:
            self._model.config.eos_token_id = eos_token_id

    def set_model(self, model_name: Optional[str]) -> None:
        candidate = (model_name or settings.trocr_model_name).strip()
        new_name = candidate or settings.trocr_model_name
        if new_name == self.model_name and self._model is not None and self._processor is not None:
            return
        self.model_name = new_name
        self._processor = None
        self._model = None

    def run(self, image: Image.Image) -> OcrOutput:
        processor, model = self._ensure_components()
        pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequence = generated.sequences[0]
        text = processor.batch_decode(sequence.unsqueeze(0), skip_special_tokens=True)[0].strip()
        confidence = None
        scores = generated.scores
        if scores:
            probabilities = []
            for step_scores, token_id in zip(scores, sequence[1:]):  # bỏ token BOS
                probs = step_scores.softmax(dim=-1)
                token_index = int(token_id)
                if probs.dim() == 2:
                    probabilities.append(probs[0, token_index].item())
                else:
                    probabilities.append(probs[token_index].item())
            if probabilities:
                confidence = float(sum(probabilities) / len(probabilities))
        return OcrOutput(text=text, confidence=confidence)
