"""
Ingestion API endpoints.

POST /ingest  → upload file + trigger ingestion pipeline
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.exceptions import IngestionError, UnsupportedFileTypeError
from app.dependencies import get_ingestion_pipeline
from app.services.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestionResponse(BaseModel):
    file_name: str
    pages_extracted: int
    chunks_produced: int
    chunks_upserted: int
    duplicates_skipped: int
    status: str


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    department: str = Form(...),
    source_name: str = Form(...),
    source_type: str = Form(...),
    supplier: str | None = Form(default=None),
    milestone: str | None = Form(default=None),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestionResponse:
    """
    Upload a file and run the ingestion pipeline.

    Accepts: PDF, XLSX, CSV, TXT, email files.
    Stores chunks as dense + sparse vectors in Qdrant.

    Args:
        file:         Uploaded file (multipart/form-data).
        project_name: Manufacturing project name.
        department:   Originating department.
        source_name:  Human-readable source label.
        source_type:  File category (pdf, spreadsheet, text etc).
        supplier:     Optional supplier name.
        milestone:    Optional project milestone.

    Returns:
        IngestionResponse with pipeline stats.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    logger.info(
        "Ingestion request received",
        extra={
            "file": file.filename,
            "project": project_name,
            "department": department,
        },
    )

    # [WHY] Write upload to temp file — pipeline expects a Path on disk.
    # UploadFile is a stream; extractors need seekable file access.
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        stats = await pipeline.ingest(
            file_path=tmp_path,
            project_name=project_name,
            department=department,
            source_name=source_name,
            source_type=source_type,
            supplier=supplier,
            milestone=milestone,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{exc.message}: {exc.detail}",
        )
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{exc.message}: {exc.detail}",
        )
    except Exception as exc:
        logger.error("Unexpected ingestion error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{exc.message}: {exc.detail}",
        )
    finally:
        # [WHY] Always delete temp file — never leak disk space.
        tmp_path.unlink(missing_ok=True)

    return IngestionResponse(
        file_name=file.filename,
        status="success",
        **stats,
    )