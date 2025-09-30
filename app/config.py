from __future__ import annotations

from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration."""

    data_dir: Path = Field(default=Path("storage"), env="OCR_DATA_DIR")
    database_url: str = Field(default="sqlite:///storage/ocr.db", env="OCR_DATABASE_URL")
    pdf_dpi: int = Field(default=300, env="OCR_PDF_DPI")
    tess_lang: str = Field(default="eng", env="OCR_TESS_LANG")
    paddle_lang: str = Field(default="en", env="OCR_PADDLE_LANG")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
