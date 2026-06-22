"""
BaseExtractor Protocol.

Defines the structural contract that every file extractor must satisfy.
Uses Protocol (not ABC) for structural subtyping — any class with the
correct method signatures qualifies without explicit inheritance.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.document import DocumentMetadata, RawDocument


# [WHY] @runtime_checkable allows isinstance(obj, BaseExtractor) checks
# at runtime — useful in the registry for validation without importing
# every concrete extractor class.
@runtime_checkable
class BaseExtractor(Protocol):
    """
    Structural contract for all file extractors.

    Every extractor must implement exactly two methods:
      - supported_mime_types: declares what it can handle
      - extract: performs the actual async extraction

    No __init__ signature is enforced — each extractor manages
    its own dependencies via constructor injection.
    """

    def supported_mime_types(self) -> list[str]:
        """
        Return the list of MIME types this extractor handles.

        Examples:
            ["application/pdf"]
            ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "text/csv"]

        Returns:
            List of MIME type strings this extractor can process.
        """
        ...

    async def extract(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> list[RawDocument]:
        """
        Extract content from a file and return one RawDocument per page/sheet.

        PDF       → one RawDocument per page
        XLSX/CSV  → one RawDocument per sheet or logical section
        Text      → one RawDocument for the whole file

        Args:
            file_path: Absolute path to the source file.
            metadata:  Pre-built DocumentMetadata to attach to each document.

        Returns:
            List of RawDocument instances. Never returns an empty list —
            raises ExtractionError instead.

        Raises:
            ExtractionError: If the file cannot be read or parsed.
        """
        ...