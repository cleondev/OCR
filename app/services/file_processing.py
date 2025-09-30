from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image

from ..config import settings

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}


def save_upload_file(uploaded_file, destination: Path) -> Path:
    uploaded_file.file.seek(0)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    uploaded_file.file.close()
    return destination


def ensure_pdf(path: Path, output_dir: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path
    if suffix in {".doc", ".docx"}:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{path.stem}.pdf"
        command = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            str(path),
            "--outdir",
            str(output_dir),
        ]
        subprocess.run(command, check=True)
        return pdf_path
    raise ValueError(f"Unsupported document format: {path.suffix}")


def pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=settings.pdf_dpi)
    paths: List[Path] = []
    for index, image in enumerate(images, start=1):
        path = output_dir / f"page_{index:03d}.png"
        image.save(path, format="PNG")
        paths.append(path)
    return paths


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")
