from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Match:
    record_id: str
    score: float
    metadata: dict[str, Any] | None = None
    document: str | None = None


class VectorStore(ABC):
    @abstractmethod
    def upsert(
            self,
            record_id: str,
            vector: list[float],
            metadata: dict[str, Any] | None = None,
            document: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    def query(
            self,
            vector: list[float],
            top_k: int = 10,
            metadata_filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        ...

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        ...

    def upsert_many(
            self,
            items: list[tuple[str, list[float], dict[str, Any] | None, str | None]]
    ) -> None:
        for record_id, vector, metadata, document in items:
            self.upsert(record_id=record_id, vector=vector, metadata=metadata, document=document)
