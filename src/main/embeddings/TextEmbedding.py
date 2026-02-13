from abc import ABC, abstractmethod
from typing import TypeAlias

# Canonical embedding type aliases used across embedding providers.
EmbeddingVector: TypeAlias = list[float]
EmbeddingBatch: TypeAlias = list[EmbeddingVector]


class TextEmbedding(ABC):
    """Provider-agnostic text embedding interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        """
        Embed one or more documents/passages.

        Args:
            texts: Input document texts.

        Returns:
            A batch of embedding vectors in the same order as `texts`.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """
        Embed a user query text for retrieval/search.

        Args:
            text: Query text.

        Returns:
            The query embedding vector.
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the output embedding vector dimension for the active model.

        Returns:
            Number of dimensions in one embedding vector.

        Raises:
            RuntimeError: If the provider cannot resolve vector dimension.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the active concrete model identifier.

        Returns:
            Canonical active model name.
        """
        ...

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> list[str]:
        """
        Return canonical and alias names supported by this provider.

        Returns:
            Supported model names including aliases.
        """
        ...
