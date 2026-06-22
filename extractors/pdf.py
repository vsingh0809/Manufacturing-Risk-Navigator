"""
PDF file extractor using PyMuPDF (fitz).

Produces one RawDocument per page so page_number metadata
is accurate for citation traceability.
"""

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import ExtractionError
from app.models.document import DocumentMetadata, RawDocument
from extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class PdfExtractor:
    """
    Extracts text from PDF files page by page.

    One RawDocument per page — preserves page_number for citations.
    Empty pages are skipped with a warning (not an error).

    Satisfies BaseExtractor Protocol without explicit inheritance.
    """

    # [WHY] Minimum character threshold to consider a page non-empty.
    # Scanned PDFs often return a few whitespace chars per page.
    _MIN_PAGE_CONTENT_LENGTH: int = 10

    def supported_mime_types(self) -> list[str]:
        """Return MIME types this extractor handles."""
        return ["application/pdf"]

    async def extract(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> list[RawDocument]:
        """
        Extract text from each page of a PDF file.

        Args:
            file_path: Absolute path to the PDF file.
            metadata:  Base DocumentMetadata — page_number and
                       content_hash are set per page inside this method.

        Returns:
            List of RawDocument, one per non-empty page.

        Raises:
            ExtractionError: If the PDF cannot be opened or read.
        """
        # [WHY] fitz.open is synchronous but CPU-bound not I/O-bound.
        # PyMuPDF reads the whole file into memory on open —
        # wrapping in asyncio.to_thread would add overhead for no gain
        # at this scale. Flag for revisit if files exceed 500 pages.
        try:
            pdf_document = fitz.open(str(file_path))
        except fitz.FileDataError as exc:
            raise ExtractionError(
                message=f"Cannot open PDF: {file_path.name}",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise ExtractionError(
                message=f"Unexpected error opening PDF: {file_path.name}",
                detail=str(exc),
            ) from exc

        total_pages = len(pdf_document)
        logger.info(
            "Extracting PDF",
            extra={"file": file_path.name, "total_pages": total_pages},
        )

        raw_documents: list[RawDocument] = []

        try:
            for page_index in range(total_pages):
                page_number = page_index + 1  # 1-indexed for humans

                try:
                    page = pdf_document[page_index]
                    content = page.get_text("text").strip()
                except Exception as exc:
                    raise ExtractionError(
                        message=f"Failed to extract page {page_number} from {file_path.name}",
                        detail=str(exc),
                    ) from exc

                if len(content) < self._MIN_PAGE_CONTENT_LENGTH:
                    logger.warning(
                        "Skipping near-empty page",
                        extra={
                            "file": file_path.name,
                            "page_number": page_number,
                            "content_length": len(content),
                        },
                    )
                    continue

                # [WHY] content_hash is computed per page, not per file.
                # Deduplicator needs to detect duplicate pages across
                # different uploads of the same document (next sprint).
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                # [WHY] model_copy(update=...) on frozen Pydantic model.
                # metadata is frozen=True so we cannot mutate it.
                # model_copy creates a new instance with only the
                # specified fields changed — all other fields inherited.
                page_metadata = metadata.model_copy(
                    update={
                        "page_number": page_number,
                        "total_pages": total_pages,
                        "content_hash": content_hash,
                    }
                )

                raw_documents.append(
                    RawDocument(
                        metadata=page_metadata,
                        content=content,
                    )
                )

        finally:
            # [WHY] Always close the PDF document even if extraction
            # fails midway. PyMuPDF holds a file handle — not closing
            # causes resource leaks under concurrent ingestion.
            pdf_document.close()

        if not raw_documents:
            raise ExtractionError(
                message=f"No extractable content found in PDF: {file_path.name}",
                detail="All pages were empty or below minimum content threshold",
            )

        logger.info(
            "PDF extraction complete",
            extra={
                "file": file_path.name,
                "pages_extracted": len(raw_documents),
                "pages_skipped": total_pages - len(raw_documents),
            },
        )

        return raw_documents