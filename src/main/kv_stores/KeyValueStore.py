from abc import ABC, abstractmethod
from typing import Any

# Sentinel used to distinguish "no default provided" from "default=None".
MISSING = object()


class KeyNotFoundError(KeyError):
    """Raised when a key is missing and no default value is provided."""
    pass


class KeyValueStore(ABC):
    """
    Generic key-value store contract.

    - set(): full UPSERT (replace-style)
    - patch(): partial update
    - get(): returns stored record as dict; raises KeyNotFoundError when default is not provided
    """

    @abstractmethod
    def set(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        """
        Full UPSERT (replace-style) by key.

        Args:
            key: Store key (provider-specific type).
            value: Value payload to store.
            ttl_seconds: Optional time-to-live in seconds.
        """
        ...

    @abstractmethod
    def patch(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        """
        Partial update for an existing key.

        Args:
            key: Store key (provider-specific type).
            value: Partial payload to merge/apply.
            ttl_seconds: Optional TTL override in seconds.

        Raises:
            KeyNotFoundError: If key is missing and provider enforces strict patch.
        """
        ...

    @abstractmethod
    def get(self, key: Any, default: Any = MISSING) -> dict:
        """
        Fetch one value by key.

        Args:
            key: Store key (provider-specific type).
            default: Value returned when key is missing. If omitted, provider should raise.

        Returns:
            Stored record as a dict-like payload.

        Raises:
            KeyNotFoundError: If key is missing and default is not provided.
        """
        ...

    @abstractmethod
    def exists(self, key: Any) -> bool:
        """
        Check key existence.

        Args:
            key: Store key.

        Returns:
            True when key exists, otherwise False.
        """
        ...

    @abstractmethod
    def delete(self, key: Any) -> bool:
        """
        Delete one key.

        Args:
            key: Store key.

        Returns:
            True if deleted, False if key did not exist.
        """
        ...

    def set_many(self, items: dict[Any, Any], ttl_seconds: int | None = None) -> None:
        """
        Default batch set implementation using repeated set().

        Args:
            items: Mapping of key -> value payload.
            ttl_seconds: Optional TTL applied to all items.
        """
        for key, value in items.items():
            self.set(key=key, value=value, ttl_seconds=ttl_seconds)

    def get_many(self, keys: list[Any], default: Any = MISSING) -> dict[Any, Any]:
        """
        Default batch get implementation using repeated get().

        Args:
            keys: Keys to fetch.
            default: Missing-key fallback passed through to get().

        Returns:
            Mapping of key -> fetched value/default.

        Raises:
            KeyNotFoundError: If any key is missing and default is not provided.
        """
        return {key: self.get(key=key, default=default) for key in keys}
