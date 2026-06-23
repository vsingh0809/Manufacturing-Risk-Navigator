"""
Document deduplicator — Sprint 3 implementation.
MVP: never flags duplicates.
"""

import logging

from app.models.document import RawDocument

logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """
    Detects duplicate documents via content_hash lookup in Qdrant.
    Sprint 3: queries Qdrant payload by content_hash.
    MVP: always returns False.
    """

    async def is_duplicate(self, document: RawDocument) -> bool:
        """
        Check if document already exists in the vector store.

        Args:
            document: RawDocument to check.

        Returns:
            False always in MVP.
            Sprint 3: True if content_hash already exists in Qdrant.
        """
        # [WHY] content_hash already computed by extractors (Step 3).
        # Sprint 3 just queries Qdrant payload filter on that hash.
        # Zero schema changes needed.
        logger.debug(
            "Deduplicator pass-through (Sprint 3)",
            extra={"content_hash": document.metadata.content_hash},
        )
        return False