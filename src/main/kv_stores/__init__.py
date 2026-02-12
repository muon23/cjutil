from kv_stores.KeyValueStore import KeyValueStore
from kv_stores.DictKeyValueStore import DictKeyValueStore
from kv_stores.PostgresKeyValueStore import PostgresKeyValueStore, KeyValueSchemaSpec


def of(store_type: str, **kwargs) -> KeyValueStore:
    """
    Factory for KeyValueStore implementations.

    Args:
        store_type: Backend type name or alias.
        **kwargs: Backend-specific constructor arguments.

    Returns:
        A KeyValueStore implementation instance.

    Raises:
        RuntimeError: If store_type is unsupported.
    """
    store_type = store_type.lower()
    if store_type in {"dict", "memory", "in_memory"}:
        return DictKeyValueStore()
    if store_type in {"postgres", "postgresql"}:
        return PostgresKeyValueStore(**kwargs)

    raise RuntimeError(f"Store type {store_type} not supported")
