from __future__ import annotations

from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration."""

    data_dir: Path = Field(default=Path("storage"), env="OCR_DATA_DIR")
    database_url: str = Field(default="sqlite:///storage/ocr.db", env="OCR_DATABASE_URL")
    pdf_dpi: int = Field(default=300, env="OCR_PDF_DPI")
    # Tesseract hỗ trợ truyền nhiều mã ngôn ngữ dạng "vie+eng" để nhận diện đa ngôn ngữ.
    # Mặc định ưu tiên tiếng Việt nhưng vẫn bao phủ một phần tiếng Anh.
    tess_lang: str = Field(default="vie+eng", env="OCR_TESS_LANG")
    # PaddleOCR sử dụng mã "vi" cho tiếng Việt.
    paddle_lang: str = Field(default="vi", env="OCR_PADDLE_LANG")
    # TrOCR sử dụng tên mô hình trên Hugging Face, ví dụ "microsoft/trocr-base-printed".
    trocr_model_name: str = Field(
        default="microsoft/trocr-base-printed", env="OCR_TROCR_MODEL"
    )
    # Có thể trỏ đến thư mục mô hình đã tải sẵn (offline).
    trocr_model_path: Path | None = Field(default=None, env="OCR_TROCR_MODEL_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
