import logging
import os
from typing import Any

from embeddings.TextEmbedding import TextEmbedding, EmbeddingBatch


class LangChainEmbedding(TextEmbedding):
    """Embedding backend built on top of LangChain provider clients."""

    # Currently wired to OpenAI embeddings through langchain_openai.
    __OPENAI_MODELS = {
        "text-embedding-3-large": {"aliases": ["openai-embed-large", "embed-3-large"]},
        "text-embedding-3-small": {"aliases": ["openai-embed-small", "embed-3-small"]},
        "text-embedding-ada-002": {"aliases": ["ada-002"]},
    }

    SUPPORTED_MODELS = list(__OPENAI_MODELS.keys())

    MODEL_ALIASES = {
        alias: model
        for model, properties in __OPENAI_MODELS.items()
        for alias in properties.get("aliases", [])
    }

    def __init__(self, model_name: str = "text-embedding-3-large", model_key: str = None, **kwargs: Any):
        """
        Create a LangChain embedding client for a supported model.

        Args:
            model_name: Model name or alias from SUPPORTED_MODELS.
            model_key: API key override. Falls back to OPENAI_API_KEY.
            **kwargs: Additional keyword args passed to OpenAIEmbeddings().

        Raises:
            ValueError: If model_name is not supported.
            RuntimeError: If API key or langchain-openai is unavailable.
        """
        if model_name in self.MODEL_ALIASES:
            model_name = self.MODEL_ALIASES[model_name]

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Embedding model {model_name} not supported")

        api_key = model_key if model_key else os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            raise RuntimeError("OpenAI API key not provided")

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "langchain-openai is required for LangChainEmbedding. "
                "Install it with: pip install langchain-openai"
            ) from e

        self.model_name = model_name
        self.embedding = OpenAIEmbeddings(model=self.model_name, api_key=api_key, **kwargs)
        logging.info(f"Using LangChain embedding model {self.model_name}")

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        """
        Embed a batch of documents/passages.

        Args:
            texts: Document or chunk texts.

        Returns:
            Batch embeddings for the input texts.
        """
        if not texts:
            return []
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query text.

        Returns:
            Query embedding vector.
        """
        return self.embedding.embed_query(text)

    def get_model_name(self) -> str:
        """
        Get active model name.

        Returns:
            Canonical active model name.
        """
        return self.model_name

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        Return all supported model names.

        Returns:
            Canonical model names plus aliases.
        """
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS
