from vector_stores.VectorStore import VectorStore, Match
from vector_stores.PGVectorStore import PGVectorStore


def of(store_type: str, **kwargs) -> VectorStore:
    """
    Factory for VectorStore implementations.

    Args:
        store_type: Backend type name or alias.
        **kwargs: Backend-specific constructor arguments.

    Returns:
        A VectorStore implementation instance.

    Raises:
        RuntimeError: If store_type is unsupported.
    """
    store_type = store_type.lower()
    if store_type in {"pgvector", "postgres", "postgresql"}:
        return PGVectorStore(**kwargs)

    raise RuntimeError(f"Store type {store_type} not supported")
