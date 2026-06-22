"""
ExtractorRegistry.

Maps MIME type strings to BaseExtractor instances.
Follows the Registry pattern — new extractors are registered
without modifying any existing code.
"""

import logging
from pathlib import Path

import magic

from app.core.exceptions import ExtractionError, UnsupportedFileTypeError
from app.models.document import DocumentMetadata, RawDocument
from extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """
    Registry that maps MIME types to extractor instances.

    Usage:
        registry = ExtractorRegistry()
        registry.register(PdfExtractor())
        registry.register(SpreadsheetExtractor())

        documents = await registry.extract(file_path, metadata)

    Design:
        - One registry instance created at app startup (in dependencies.py)
        - Injected into the ingestion pipeline via constructor
        - Never instantiated inside service functions
    """

    def __init__(self) -> None:
        # [WHY] Dict not list — O(1) lookup by MIME type vs O(n) scan.
        # With 10 extractors this is trivial. With 50 it matters.
        self._registry: dict[str, BaseExtractor] = {}

    def register(self, extractor: BaseExtractor) -> None:
        """
        Register an extractor for all MIME types it declares.

        One extractor can handle multiple MIME types
        (e.g. SpreadsheetExtractor handles both xlsx and csv).

        Args:
            extractor: Any object satisfying the BaseExtractor Protocol.

        Raises:
            TypeError: If extractor does not satisfy BaseExtractor Protocol.
        """
        # [WHY] Runtime Protocol check before registration.
        # Catches mistakes at startup, not at request time.
        if not isinstance(extractor, BaseExtractor):
            raise TypeError(
                f"{type(extractor).__name__} does not satisfy BaseExtractor Protocol. "
                f"Must implement supported_mime_types() and extract()."
            )

        for mime_type in extractor.supported_mime_types():
            if mime_type in self._registry:
                logger.warning(
                    "Overwriting existing extractor for MIME type",
                    extra={"mime_type": mime_type,
                           "extractor": type(extractor).__name__},
                )
            self._registry[mime_type] = extractor
            logger.info(
                "Registered extractor",
                extra={"mime_type": mime_type,
                       "extractor": type(extractor).__name__},
            )

    def get(self, mime_type: str) -> BaseExtractor:
        """
        Retrieve the extractor registered for a given MIME type.

        Args:
            mime_type: MIME type string to look up.

        Returns:
            BaseExtractor instance for that MIME type.

        Raises:
            UnsupportedFileTypeError: If no extractor is registered.
        """
        extractor = self._registry.get(mime_type)
        if extractor is None:
            raise UnsupportedFileTypeError(mime_type=mime_type)
        return extractor

    def registered_types(self) -> list[str]:
        """Return all currently registered MIME types."""
        return list(self._registry.keys())

    async def extract(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> list[RawDocument]:
        """
        Detect MIME type from file bytes and route to correct extractor.

        MIME type is detected from file content (not filename extension)
        to prevent spoofing — a renamed .exe as .pdf would be caught.

        Args:
            file_path: Absolute path to the file on disk.
            metadata:  DocumentMetadata to attach to extracted documents.

        Returns:
            List of RawDocument instances from the matched extractor.

        Raises:
            UnsupportedFileTypeError: If no extractor handles the detected type.
            ExtractionError: If extraction fails after routing.
        """
        # [WHY] Detect from file bytes, not extension.
        # Extension is user-controlled. Bytes are not.
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
        except Exception as exc:
            raise ExtractionError(
                message=f"Failed to detect MIME type for {file_path.name}",
                detail=str(exc),
            ) from exc

        logger.info(
            "Detected MIME type",
            extra={"file": file_path.name, "mime_type": mime_type},
        )

        extractor = self.get(mime_type)

        try:
            return await extractor.extract(file_path, metadata)
        except ExtractionError:
            # [WHY] Re-raise ExtractionError as-is — it already has context.
            # Wrapping it again would lose the original message.
            raise
        except Exception as exc:
            raise ExtractionError(
                message=f"Unexpected error during extraction of {file_path.name}",
                detail=str(exc),
            ) from exc