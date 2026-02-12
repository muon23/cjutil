from sql_stores.PostgresSqlStore import PostgresSqlStore
from sql_stores.SqlStore import SqlStore


def of(store_type: str, **kwargs) -> SqlStore:
    """
    Factory for SqlStore implementations.

    Args:
        store_type: Backend type name or alias.
        **kwargs: Backend-specific constructor arguments.

    Returns:
        A SqlStore implementation instance.

    Raises:
        RuntimeError: If store_type is unsupported.
    """
    store_type = store_type.lower()
    if store_type in {"postgres", "postgresql"}:
        return PostgresSqlStore(**kwargs)

    raise RuntimeError(f"Store type {store_type} not supported")
