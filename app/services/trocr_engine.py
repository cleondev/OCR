from __future__ import annotations

import logging
from pathlib import Path
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

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_path: Optional[Path | str] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> None:
        self.model_name = (model_name or settings.trocr_model_name).strip() or settings.trocr_model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved_path = model_path if model_path is not None else settings.trocr_model_path
        self.model_path: Path | None
        if resolved_path:
            candidate = Path(resolved_path).expanduser()
            if str(candidate).strip():
                self.model_path = candidate
            else:
                self.model_path = None
        else:
            self.model_path = None
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None
        defaults = {
            "max_new_tokens": 128,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }
        if generation_kwargs:
            defaults.update(generation_kwargs)
        self._generation_kwargs = defaults

    def preferred_variants(self) -> tuple[str, ...]:
        """Ưu tiên giữ nguyên chi tiết và độ tương phản tự nhiên."""

        return ("original", "grayscale", "contrast")

    def _ensure_components(self) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        if self._processor is None:
            try:
                self._processor = self._load_processor()
            except OSError as exc:  # pragma: no cover - chỉ log lỗi tải model
                logger.error(
                    "Không thể tải TrOCR processor từ %s: %s",
                    self._describe_source(),
                    exc,
                )
                raise self._translate_os_error(exc) from exc
        if self._model is None:
            try:
                model = self._load_model()
            except OSError as exc:  # pragma: no cover - chỉ log lỗi tải model
                logger.error(
                    "Không thể tải TrOCR model từ %s: %s",
                    self._describe_source(),
                    exc,
                )
                raise self._translate_os_error(exc) from exc
            self._model = model.to(self.device)
            self._model.eval()
            self._ensure_generation_tokens()
        return self._processor, self._model

    def _describe_source(self) -> str:
        if self.model_path is not None:
            return f"đường dẫn {self.model_path}"
        return f"mô hình Hugging Face '{self.model_name}'"

    def _translate_os_error(self, exc: OSError) -> RuntimeError:
        hint = (
            "Đảm bảo máy chủ có thể truy cập huggingface.co hoặc tải sẵn mô hình và "
            "đặt biến môi trường OCR_TROCR_MODEL_PATH trỏ tới thư mục đó."
        )
        if self.model_path is not None:
            hint = (
                "Kiểm tra lại đường dẫn OCR_TROCR_MODEL_PATH. Thư mục cần chứa các "
                "tệp cấu hình và trọng số do Hugging Face phát hành."
            )
        message = (
            f"Không thể tải thành phần TrOCR từ {self._describe_source()}: {exc}. {hint}"
        )
        return RuntimeError(message)

    def _load_processor(self) -> TrOCRProcessor:
        if self.model_path is not None:
            return TrOCRProcessor.from_pretrained(str(self.model_path))
        return TrOCRProcessor.from_pretrained(self.model_name)

    def _load_model(self) -> VisionEncoderDecoderModel:
        if self.model_path is not None:
            return VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
        return VisionEncoderDecoderModel.from_pretrained(self.model_name)

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
        generation_options = dict(self._generation_kwargs)
        generation_options.update(
            {
                "output_scores": True,
                "return_dict_in_generate": True,
            }
        )
        with torch.no_grad():
            generated = model.generate(pixel_values, **generation_options)
        sequence = generated.sequences[0]
        text = processor.batch_decode(sequence.unsqueeze(0), skip_special_tokens=True)[0].strip()
        confidence = None
        if getattr(generated, "sequences_scores", None) is not None:
            confidence = float(generated.sequences_scores[0].exp().item())
        else:
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
