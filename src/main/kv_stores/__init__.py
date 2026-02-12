from kv_stores.KeyValueStore import KeyValueStore
from kv_stores.PostgresKeyValueStore import PostgresKeyValueStore, KeyValueSchemaSpec


def of(store_type: str, **kwargs) -> KeyValueStore:
    store_type = store_type.lower()
    if store_type in {"postgres", "postgresql"}:
        return PostgresKeyValueStore(**kwargs)

    raise RuntimeError(f"Store type {store_type} not supported")
