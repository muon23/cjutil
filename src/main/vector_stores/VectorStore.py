from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Match:
    """Standardized query match payload returned by vector stores."""
    record_id: str
    score: float
    metadata: dict[str, Any] | None = None
    document: str | None = None


class VectorStore(ABC):
    """Provider-agnostic vector store interface."""

    @abstractmethod
    def upsert(
            self,
            record_id: str,
            vector: list[float],
            metadata: dict[str, Any] | None = None,
            document: str | None = None,
    ) -> None:
        """
        Insert or replace one vector record.

        Args:
            record_id: Unique record identifier.
            vector: Embedding vector.
            metadata: Optional metadata payload.
            document: Optional original text/document.

        Raises:
            ValueError: If vector shape or payload is invalid for backend.
        """
        ...

    @abstractmethod
    def query(
            self,
            vector: list[float],
            top_k: int = 10,
            metadata_filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """
        Query nearest vectors for an input vector.

        Args:
            vector: Query vector.
            top_k: Maximum number of matches to return.
            metadata_filter: Optional backend-specific metadata filter.

        Returns:
            Ranked list of query matches.

        Raises:
            ValueError: If query arguments are invalid.
        """
        ...

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        Delete one vector record.

        Args:
            record_id: Record identifier to delete.

        Returns:
            True if deleted, otherwise False.
        """
        ...

    def upsert_many(
            self,
            items: list[tuple[str, list[float], dict[str, Any] | None, str | None]]
    ) -> None:
        """
        Default batch upsert implementation using repeated upsert().

        Args:
            items: List of tuples in the form
                (record_id, vector, metadata, document).

        Raises:
            ValueError: If any individual upsert payload is invalid.
        """
        for record_id, vector, metadata, document in items:
            self.upsert(record_id=record_id, vector=vector, metadata=metadata, document=document)
