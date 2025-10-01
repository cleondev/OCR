from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import UploadFile
from sqlalchemy.orm import selectinload
from PIL import Image

from ..config import settings
from ..database import session_scope
from ..models import OCRImage, OCRRun, OCRTextResult
from .file_processing import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    ensure_pdf,
    load_image,
    pdf_to_images,
    save_upload_file,
)
from .ocr_base import OCREngine
from .paddle_engine import PaddleOCREngine
from .preprocess import ImagePreprocessor
from .tesseract_engine import TesseractEngine

class OCRService:
    def __init__(self) -> None:
        self.preprocessor = ImagePreprocessor()
        self._engine_factories: Dict[str, Callable[[Optional[str]], OCREngine]] = {
            "tesseract": lambda lang=None: TesseractEngine(lang=lang),
            "paddle": lambda lang=None: PaddleOCREngine(lang=lang),
        }

    def list_engines(self) -> List[str]:
        return list(self._engine_factories.keys())

    def get_engine(self, name: str, *, lang: Optional[str] = None) -> OCREngine:
        try:
            factory = self._engine_factories[name]
        except KeyError as exc:  # pragma: no cover - guard rails
            raise ValueError(f"Unsupported OCR engine: {name}") from exc
        engine = factory(lang)
        if lang is not None and hasattr(engine, "set_language"):
            engine.set_language(lang)
        return engine

    def default_language_for(self, name: str) -> Optional[str]:
        if name == "tesseract":
            return settings.tess_lang
        if name == "paddle":
            return settings.paddle_lang
        return None

    def process(self, file: UploadFile, engine_name: str, *, lang: Optional[str] = None) -> OCRRun:
        resolved_lang = (lang.strip() if lang else None) or self.default_language_for(engine_name)
        engine = self.get_engine(engine_name, lang=resolved_lang)
        run_dir = settings.data_dir / uuid.uuid4().hex
        original_dir = run_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_dir / file.filename
        save_upload_file(file, original_path)

        run = OCRRun(
            engine=engine.name,
            original_file_path=str(original_path),
            language=resolved_lang,
        )

        images: List[Tuple[Path, Image.Image]] = []
        converted_pdf_path: Path | None = None

        suffix = original_path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            image = load_image(original_path)
            images.append((original_path, image))
        elif suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
            converted_dir = run_dir / "converted"
            converted_dir.mkdir(parents=True, exist_ok=True)
            if suffix == ".pdf":
                converted_pdf_path = original_path
            else:
                converted_pdf_path = ensure_pdf(original_path, converted_dir)
            run.converted_file_path = str(converted_pdf_path)
            images_paths = pdf_to_images(converted_pdf_path, run_dir / "pages")
            for path in images_paths:
                images.append((path, load_image(path)))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        with session_scope() as session:
            session.add(run)
            session.flush()

            for index, (image_path, image) in enumerate(images, start=1):
                db_image = OCRImage(
                    run_id=run.id,
                    path=str(image_path),
                    kind="source",
                    label=f"page_{index}",
                    sequence=index,
                )
                session.add(db_image)
                session.flush()

                preferred_variants = None
                if hasattr(engine, "preferred_variants"):
                    preferred_callable = getattr(engine, "preferred_variants")
                    if callable(preferred_callable):
                        preferred_variants = preferred_callable()
                    else:
                        preferred_variants = preferred_callable  # pragma: no cover - defensive
                variants = self.preprocessor.generate(
                    image,
                    allowed_labels=preferred_variants,
                )
                for order, (label, variant_image) in enumerate(variants):
                    variant_path = run_dir / "preprocessed" / f"{db_image.label}_{label}.png"
                    variant_path.parent.mkdir(parents=True, exist_ok=True)
                    variant_image.save(variant_path, format="PNG")
                    db_variant_image = OCRImage(
                        run_id=run.id,
                        path=str(variant_path),
                        kind="preprocessed",
                        label=f"{db_image.label}_{label}",
                        sequence=order,
                    )
                    session.add(db_variant_image)
                    session.flush()

                    result = engine.run(variant_image)
                    text_result = OCRTextResult(
                        run_id=run.id,
                        image_id=db_variant_image.id,
                        variant_label=db_variant_image.label,
                        text=result.text,
                        confidence=result.confidence,
                    )
                    session.add(text_result)

            session.flush()
            best = self._select_best_result(run)
            if best:
                run.summary_text = best.text
                run.best_confidence = best.confidence
            session.add(run)
            session.commit()
            refreshed_run = (
                session.query(OCRRun)
                .options(selectinload(OCRRun.images), selectinload(OCRRun.text_results))
                .get(run.id)
            )
            if refreshed_run is not None:
                for image in refreshed_run.images:
                    session.expunge(image)
                for result in refreshed_run.text_results:
                    session.expunge(result)
                session.expunge(refreshed_run)
        return refreshed_run

    def _select_best_result(self, run: OCRRun) -> OCRTextResult | None:
        if not run.text_results:
            return None
        sorted_results = sorted(
            run.text_results,
            key=lambda r: (r.confidence if r.confidence is not None else 0.0, len(r.text)),
            reverse=True,
        )
        return sorted_results[0]
