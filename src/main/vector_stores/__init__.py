from vector_stores.VectorStore import VectorStore, Match
from vector_stores.PGVectorStore import PGVectorStore


def of(store_type: str, **kwargs) -> VectorStore:
    store_type = store_type.lower()
    if store_type in {"pgvector", "postgres", "postgresql"}:
        return PGVectorStore(**kwargs)

    raise RuntimeError(f"Store type {store_type} not supported")
