from abc import ABC, abstractmethod
from typing import TypeAlias

EmbeddingVector: TypeAlias = list[float]
EmbeddingBatch: TypeAlias = list[EmbeddingVector]


class TextEmbedding(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> list[str]:
        ...
