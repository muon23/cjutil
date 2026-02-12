"""
Embedding Package Factory (__init__.py)

This module serves as the primary factory for creating TextEmbedding instances.
It transparently selects the correct concrete TextEmbedding subclass based on the
requested model name.
"""

from embeddings.LangChainEmbedding import LangChainEmbedding
from embeddings.SentenceTransformerEmbedding import SentenceTransformerEmbedding
from embeddings.TextEmbedding import TextEmbedding, EmbeddingBatch


def of(model_name: str, **kwargs) -> TextEmbedding:
    """
    Factory function to instantiate the correct TextEmbedding subclass based on model name.

    Args:
        model_name: Embedding model name or alias.
        **kwargs: Provider-specific constructor arguments.

    Returns:
        Instantiated TextEmbedding implementation.

    Raises:
        RuntimeError: If no provider supports model_name.
    """
    # Resolution order matters when model namespaces overlap across providers.
    embedders = [SentenceTransformerEmbedding, LangChainEmbedding]

    for embedder in embedders:
        if model_name in embedder.get_supported_models():
            return embedder(model_name, **kwargs)

    raise RuntimeError(f"Embedding model {model_name} not supported.")


active_embeddings: dict[str, TextEmbedding] = dict()


def embed_documents(model_name: str, texts: list[str], **kwargs) -> EmbeddingBatch:
    """
    Resolve/cache provider and embed document texts.

    Args:
        model_name: Embedding model name or alias.
        texts: Document or chunk texts.
        **kwargs: Provider-specific constructor arguments for first-time initialization.

    Returns:
        Batch embeddings for the input texts.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    if model_name not in active_embeddings:
        active_embeddings[model_name] = of(model_name, **kwargs)

    return active_embeddings[model_name].embed_documents(texts)


def embed_query(model_name: str, text: str, **kwargs) -> list[float]:
    """
    Resolve/cache provider and embed one query.

    Args:
        model_name: Embedding model name or alias.
        text: Query text.
        **kwargs: Provider-specific constructor arguments for first-time initialization.

    Returns:
        Query embedding vector.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    if model_name not in active_embeddings:
        active_embeddings[model_name] = of(model_name, **kwargs)

    return active_embeddings[model_name].embed_query(text)
