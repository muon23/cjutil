import logging
from typing import Any

from embeddings.TextEmbedding import TextEmbedding, EmbeddingBatch


class SentenceTransformerEmbedding(TextEmbedding):
    """Local embedding backend powered by sentence-transformers."""

    # Model catalog for this backend.
    # `instruct` is optional and only applied on embed_query().
    __MODELS = {
        "BAAI/bge-m3": {"aliases": ["bge-m3"]},
        "intfloat/multilingual-e5-large-instruct": {
            "aliases": ["multilingual-e5-large", "e5"],
            "instruct": "Instruct: Given a user search query, retrieve relevant passages that answer the query.\n"
                        "Query: {text}"
        },
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
            "aliases": ["multilingual-mpnet", "mpnet"],
        },
        "sentence-transformers/LaBSE": {"aliases": ["labse"]},
    }

    SUPPORTED_MODELS = list(__MODELS.keys())

    MODEL_ALIASES = {
        alias: model
        for model, properties in __MODELS.items()
        for alias in properties.get("aliases", [])
    }

    def __init__(
            self,
            model_name: str = "BAAI/bge-m3",
            normalize_embeddings: bool = True,
            **kwargs: Any
    ):
        """
        Create a sentence-transformers embedding model instance.

        Args:
            model_name: Model name or alias from SUPPORTED_MODELS.
            normalize_embeddings: Whether to L2-normalize output vectors.
            **kwargs: Additional keyword args passed to SentenceTransformer().

        Raises:
            ValueError: If model_name is not supported.
            RuntimeError: If sentence-transformers is not installed.
        """
        if model_name in self.MODEL_ALIASES:
            model_name = self.MODEL_ALIASES[model_name]

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Embedding model {model_name} not supported")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install sentence-transformers"
            ) from e

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, **kwargs)
        logging.info(f"Using sentence-transformers embedding model {self.model_name}")

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        """
        Embed raw document/passages as-is.

        Args:
            texts: Document or chunk texts.

        Returns:
            Batch embeddings for the input texts.
        """
        if not texts:
            return []
        return self.__embed_internal(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text, applying model-specific instruction template when configured.

        Args:
            text: Query text.

        Returns:
            Query embedding vector.
        """
        instruct: str = self.__MODELS[self.model_name].get("instruct")
        text = instruct.format(text=text) if instruct else text
        return self.__embed_internal([text])[0]

    def __embed_internal(self, texts: list[str]) -> EmbeddingBatch:
        # Keep one centralized encode path so document/query modes stay consistent.
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return vectors.tolist()

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
