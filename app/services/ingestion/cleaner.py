"""
Document cleaner — Sprint 2 implementation.
MVP: returns document untouched.
"""

import logging

from app.models.document import RawDocument

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Cleans raw extracted text.
    Sprint 2: whitespace normalisation, boilerplate removal, HTML stripping.
    MVP: pass-through.
    """

    async def clean(self, document: RawDocument) -> RawDocument:
        """
        Clean a raw document.

        Args:
            document: RawDocument from extractor.

        Returns:
            Cleaned RawDocument. Currently pass-through.
        """
        # [WHY] Stub — interface is locked, implementation comes Sprint 2.
        # Pipeline calls this normally — no changes needed when implemented.
        logger.debug(
            "Cleaner pass-through (Sprint 2)",
            extra={"document_id": document.metadata.document_id},
        )
        return document