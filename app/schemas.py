from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class OCRImageSchema(BaseModel):
    id: int
    path: Path
    kind: str
    label: str
    sequence: int

    class Config:
        orm_mode = True


class OCRTextResultSchema(BaseModel):
    id: int
    variant_label: str
    text: str
    confidence: Optional[float]
    image_id: Optional[int]

    class Config:
        orm_mode = True


class OCRRunSchema(BaseModel):
    id: int
    created_at: datetime
    engine: str
    language: Optional[str]
    original_file_path: Path
    converted_file_path: Optional[Path]
    summary_text: Optional[str]
    best_confidence: Optional[float]
    images: List[OCRImageSchema]
    text_results: List[OCRTextResultSchema]

    class Config:
        orm_mode = True


class OCRRequestSchema(BaseModel):
    engine: str = "tesseract"


class OCRResponseSchema(BaseModel):
    run: OCRRunSchema
