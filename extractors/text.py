"""
Plain text extractor.

Handles: .txt, .md, email dumps, meeting transcripts, RFQ documents.
One RawDocument per file — no pagination needed for text files.
"""

import hashlib
import logging
from pathlib import Path

import aiofiles

from app.core.exceptions import ExtractionError
from app.models.document import DocumentMetadata, RawDocument
from extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

# [WHY] Minimum content length before we consider extraction successful.
# Prevents empty .txt files from entering the chunker silently.
_MIN_CONTENT_LENGTH: int = 20


class TextExtractor:
    """
    Extracts plain text from unstructured text files.

    Handles UTF-8 with graceful fallback to latin-1 for legacy files —
    manufacturing companies often have old Windows-encoded documents.

    Satisfies BaseExtractor Protocol without explicit inheritance.
    """

    def supported_mime_types(self) -> list[str]:
        """Return MIME types this extractor handles."""
        return [
            "text/plain",
            "text/markdown",
            "message/rfc822",   # .eml email files
        ]

    async def extract(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> list[RawDocument]:
        """
        Read and return the full text content of a file.

        Attempts UTF-8 first, falls back to latin-1 for legacy encoding.

        Args:
            file_path: Absolute path to the text file.
            metadata:  Base DocumentMetadata to attach.

        Returns:
            Single-element list containing one RawDocument.

        Raises:
            ExtractionError: If the file cannot be read or is empty.
        """
        content = await self._read_with_encoding_fallback(file_path)
        content = content.strip()

        if len(content) < _MIN_CONTENT_LENGTH:
            raise ExtractionError(
                message=f"File content too short to process: {file_path.name}",
                detail=f"Content length {len(content)} below minimum {_MIN_CONTENT_LENGTH}",
            )

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        text_metadata = metadata.model_copy(
            update={
                "content_hash": content_hash,
                "metadata": {
                    **metadata.metadata,
                    "char_count": len(content),
                    "line_count": content.count("\n") + 1,
                },
            }
        )

        logger.info(
            "Text extraction complete",
            extra={
                "file": file_path.name,
                "char_count": len(content),
            },
        )

        return [RawDocument(metadata=text_metadata, content=content)]

    async def _read_with_encoding_fallback(self, file_path: Path) -> str:
        """
        Read file content attempting UTF-8 then falling back to latin-1.

        Args:
            file_path: Path to the file to read.

        Returns:
            File content as string.

        Raises:
            ExtractionError: If the file cannot be read with either encoding.
        """
        # [WHY] Try UTF-8 first — it is the correct encoding for modern files.
        # Fall back to latin-1 for legacy manufacturing documents that were
        # created on Windows systems in the 1990s-2000s (common in this domain).
        for encoding in ("utf-8", "latin-1"):
            try:
                async with aiofiles.open(file_path, encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                logger.warning(
                    "Encoding failed, trying fallback",
                    extra={"file": file_path.name, "failed_encoding": encoding},
                )
                continue
            except OSError as exc:
                raise ExtractionError(
                    message=f"Cannot read file: {file_path.name}",
                    detail=str(exc),
                ) from exc

        raise ExtractionError(
            message=f"Cannot decode file with any supported encoding: {file_path.name}",
            detail="Tried: utf-8, latin-1",
        )