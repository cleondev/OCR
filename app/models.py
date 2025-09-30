from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class OCRRun(Base):
    __tablename__ = "ocr_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    engine: Mapped[str] = mapped_column(String(50))
    original_file_path: Mapped[str] = mapped_column(String)
    converted_file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    best_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    images: Mapped[List["OCRImage"]] = relationship("OCRImage", back_populates="run", cascade="all, delete-orphan")
    text_results: Mapped[List["OCRTextResult"]] = relationship("OCRTextResult", back_populates="run", cascade="all, delete-orphan")


class OCRImage(Base):
    __tablename__ = "ocr_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ocr_runs.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String(20))  # source or preprocessed
    label: Mapped[str] = mapped_column(String(100))
    sequence: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[OCRRun] = relationship("OCRRun", back_populates="images")
    text_results: Mapped[List["OCRTextResult"]] = relationship("OCRTextResult", back_populates="image")


class OCRTextResult(Base):
    __tablename__ = "ocr_text_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ocr_runs.id", ondelete="CASCADE"))
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ocr_images.id", ondelete="SET NULL"))
    variant_label: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped[OCRRun] = relationship("OCRRun", back_populates="text_results")
    image: Mapped[Optional[OCRImage]] = relationship("OCRImage", back_populates="text_results")
