from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import inspect, text
from sqlalchemy.orm import selectinload

from PIL import Image
import pytesseract
from pytesseract import Output

from .config import settings
from .database import Base, engine, session_scope
from .models import OCRImage, OCRRun
from .schemas import OCRResponseSchema, OCRRunSchema
from .services.ocr_service import OCRService

app = FastAPI(title="OCR Service", version="1.0.0")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def format_confidence(value: float | None) -> str | None:
    if value is None:
        return None
    normalized = value
    if normalized > 1:
        normalized = min(normalized, 100.0)
    else:
        normalized *= 100
    return f"{normalized:.2f}%"


templates.env.filters["format_confidence"] = format_confidence


def _ensure_language_column() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("ocr_runs")}
    if "language" not in columns:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE ocr_runs ADD COLUMN language VARCHAR(50)"))
            connection.commit()


Base.metadata.create_all(bind=engine)
_ensure_language_column()
ocr_service = OCRService()


def _default_lang_for(engine: str) -> str:
    return ocr_service.default_language_for(engine) or ""


def _detach_run(session, run: OCRRun) -> OCRRun:
    """Detach ORM objects so they can be safely returned/used after the session."""

    for image in run.images:
        session.expunge(image)
    for result in run.text_results:
        session.expunge(result)
    session.expunge(run)
    return run


def _load_runs() -> list[OCRRun]:
    with session_scope() as session:
        runs = (
            session.query(OCRRun)
            .options(selectinload(OCRRun.images), selectinload(OCRRun.text_results))
            .order_by(OCRRun.created_at.desc())
            .all()
        )
        return [_detach_run(session, run) for run in runs]


def _load_run(run_id: int) -> OCRRun | None:
    with session_scope() as session:
        run = session.get(
            OCRRun,
            run_id,
            options=(selectinload(OCRRun.images), selectinload(OCRRun.text_results)),
        )
        if not run:
            return None
        return _detach_run(session, run)


@app.post("/api/v1/ocr", response_model=OCRResponseSchema)
async def run_ocr(engine: str = "tesseract", file: UploadFile = File(...), lang: str | None = None):
    try:
        run = ocr_service.process(file=file, engine_name=engine, lang=lang)
    except Exception as exc:  # pragma: no cover - guard rails
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run": run}


@app.get("/api/v1/ocr/engines", response_model=list[str])
async def list_engines():
    return ocr_service.list_engines()


@app.get("/api/v1/ocr", response_model=list[OCRRunSchema])
async def list_runs():
    return _load_runs()


@app.get("/api/v1/ocr/{run_id}", response_model=OCRRunSchema)
async def get_run(run_id: int):
    run = _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    runs = _load_runs()
    engines = ocr_service.list_engines()
    selected_engine = "tesseract" if "tesseract" in engines else (engines[0] if engines else "")
    selected_lang = _default_lang_for(selected_engine) if selected_engine else ""
    default_langs = {engine: _default_lang_for(engine) for engine in engines}
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "runs": runs,
            "engines": engines,
            "selected_engine": selected_engine,
            "selected_lang": selected_lang,
            "engine_default_langs": default_langs,
            "error": None,
            "now": datetime.utcnow(),
            "active_page": "dashboard",
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    engine: str = Form("tesseract"),
    lang: str | None = Form(None),
    file: UploadFile = File(...),
):
    engines = ocr_service.list_engines()
    try:
        run = ocr_service.process(file=file, engine_name=engine, lang=lang)
    except Exception as exc:  # pragma: no cover - guard rails
        runs = _load_runs()
        selected_lang = lang or _default_lang_for(engine)
        default_langs = {name: _default_lang_for(name) for name in engines}
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "runs": runs,
                "engines": engines,
                "selected_engine": engine,
                "selected_lang": selected_lang,
                "engine_default_langs": default_langs,
                "error": str(exc),
                "now": datetime.utcnow(),
                "active_page": "dashboard",
            },
            status_code=400,
        )
    return RedirectResponse(url=f"/runs/{run.id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: int):
    run = _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    source_images = sorted((img for img in run.images if img.kind == "source"), key=lambda img: img.sequence)

    preprocessed_map: dict[str, list[OCRImage]] = {}
    for image in (img for img in run.images if img.kind == "preprocessed"):
        parts = image.label.split("_")
        base_label = "_".join(parts[:2]) if len(parts) >= 2 else image.label
        preprocessed_map.setdefault(base_label, []).append(image)

    preprocessed_groups = []
    for source in source_images:
        variants = sorted(preprocessed_map.get(source.label, []), key=lambda img: img.sequence)
        if variants:
            preprocessed_groups.append({"source": source, "variants": variants})

    remaining_variants = []
    for label, images in preprocessed_map.items():
        if all(group["source"] is None or label != group["source"].label for group in preprocessed_groups):
            remaining_variants.extend(images)

    if remaining_variants:
        preprocessed_groups.append({"source": None, "variants": sorted(remaining_variants, key=lambda img: (img.sequence, img.label))})
    results = sorted(
        run.text_results,
        key=lambda r: (r.confidence if r.confidence is not None else 0.0, len(r.text)),
        reverse=True,
    )

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "source_images": list(source_images),
            "preprocessed_groups": preprocessed_groups,
            "results": results,
            "now": datetime.utcnow(),
            "active_page": "dashboard",
        },
    )


def _render_labeling_template(
    request: Request,
    *,
    image_data: str | None,
    boxes: list[dict[str, object]],
    image_width: int | None,
    image_height: int | None,
    filename: str | None,
    error: str | None = None,
    language: str | None = None,
    status_code: int = 200,
):
    return templates.TemplateResponse(
        "label_text.html",
        {
            "request": request,
            "image_data": image_data,
            "boxes": boxes,
            "image_width": image_width,
            "image_height": image_height,
            "filename": filename,
            "error": error,
            "language": language,
            "now": datetime.utcnow(),
            "active_page": "labeling",
            "settings": settings,
        },
        status_code=status_code,
    )


@app.get("/labeling", response_class=HTMLResponse)
async def labeling_home(request: Request):
    return _render_labeling_template(
        request,
        image_data=None,
        boxes=[],
        image_width=None,
        image_height=None,
        filename=None,
    )


@app.post("/labeling", response_class=HTMLResponse)
async def labeling_detect(request: Request, file: UploadFile = File(...), lang: str | None = Form(None)):
    raw_bytes = await file.read()
    if not raw_bytes:
        return _render_labeling_template(
            request,
            image_data=None,
            boxes=[],
            image_width=None,
            image_height=None,
            filename=file.filename,
            error="Không thể đọc dữ liệu từ tệp tải lên.",
            status_code=400,
        )

    try:
        image = Image.open(BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - guard rails for invalid uploads
        return _render_labeling_template(
            request,
            image_data=None,
            boxes=[],
            image_width=None,
            image_height=None,
            filename=file.filename,
            error=f"Tệp không phải là ảnh hợp lệ: {exc}",
            status_code=400,
        )

    image = image.convert("RGB")
    width, height = image.size
    language = (lang.strip() if lang else None) or settings.tess_lang

    try:
        data = pytesseract.image_to_data(image, lang=language, output_type=Output.DICT)
    except Exception as exc:  # pragma: no cover - guard rails when OCR backend fails
        return _render_labeling_template(
            request,
            image_data=None,
            boxes=[],
            image_width=None,
            image_height=None,
            filename=file.filename,
            error=f"Không thể nhận diện văn bản: {exc}",
            language=language,
            status_code=500,
        )

    boxes: list[dict[str, object]] = []
    total = len(data.get("text", []))
    for index in range(total):
        word = (data.get("text", [""])[index] or "").strip()
        if not word:
            continue

        confidence_value = None
        confidence_raw = data.get("conf", [None])[index]
        if confidence_raw not in (None, ""):
            try:
                confidence_value = float(confidence_raw)
            except (TypeError, ValueError):
                confidence_value = None

        if confidence_value is not None and confidence_value < 0:
            continue

        left = int(data.get("left", [0])[index] or 0)
        top = int(data.get("top", [0])[index] or 0)
        box_width = int(data.get("width", [0])[index] or 0)
        box_height = int(data.get("height", [0])[index] or 0)

        boxes.append(
            {
                "text": word,
                "confidence": confidence_value,
                "confidence_display": f"{confidence_value:.1f}" if confidence_value is not None else None,
                "left": left,
                "top": top,
                "width": box_width,
                "height": box_height,
                "left_pct": (left / width * 100) if width else 0,
                "top_pct": (top / height * 100) if height else 0,
                "width_pct": (box_width / width * 100) if width else 0,
                "height_pct": (box_height / height * 100) if height else 0,
            }
        )

    boxes.sort(key=lambda item: (item["top"], item["left"]))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_data = f"data:image/png;base64,{encoded_image}"

    return _render_labeling_template(
        request,
        image_data=image_data,
        boxes=boxes,
        image_width=width,
        image_height=height,
        filename=file.filename,
        language=language,
    )


@app.get("/runs/{run_id}/images/{image_id}")
async def get_run_image(run_id: int, image_id: int):
    with session_scope() as session:
        image = (
            session.query(OCRImage)
            .filter(OCRImage.run_id == run_id, OCRImage.id == image_id)
            .one_or_none()
        )
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        session.expunge(image)

    image_path = Path(image.path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on server")
    return FileResponse(image_path)
