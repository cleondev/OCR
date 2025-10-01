from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import inspect, text
from sqlalchemy.orm import selectinload

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
        },
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
