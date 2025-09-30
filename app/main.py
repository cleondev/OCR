from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from sqlalchemy.orm import selectinload

from .database import Base, engine, session_scope
from .models import OCRRun
from .schemas import OCRResponseSchema, OCRRunSchema
from .services.ocr_service import OCRService

app = FastAPI(title="OCR Service", version="1.0.0")

Base.metadata.create_all(bind=engine)
ocr_service = OCRService()


@app.post("/api/v1/ocr", response_model=OCRResponseSchema)
async def run_ocr(engine: str = "tesseract", file: UploadFile = File(...)):
    try:
        run = ocr_service.process(file=file, engine_name=engine)
    except Exception as exc:  # pragma: no cover - guard rails
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run": run}


@app.get("/api/v1/ocr", response_model=list[OCRRunSchema])
async def list_runs():
    with session_scope() as session:
        runs = (
            session.query(OCRRun)
            .options(selectinload(OCRRun.images), selectinload(OCRRun.text_results))
            .order_by(OCRRun.created_at.desc())
            .all()
        )
        for run in runs:
            for image in run.images:
                session.expunge(image)
            for result in run.text_results:
                session.expunge(result)
            session.expunge(run)
    return runs


@app.get("/api/v1/ocr/{run_id}", response_model=OCRRunSchema)
async def get_run(run_id: int):
    with session_scope() as session:
        run = session.get(
            OCRRun,
            run_id,
            options=(selectinload(OCRRun.images), selectinload(OCRRun.text_results)),
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        for image in run.images:
            session.expunge(image)
        for result in run.text_results:
            session.expunge(result)
        session.expunge(run)
    return run
