from abc import ABC, abstractmethod
from typing import Any

MISSING = object()


class KeyNotFoundError(KeyError):
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
        ...

    @abstractmethod
    def patch(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        ...

    @abstractmethod
    def get(self, key: Any, default: Any = MISSING) -> dict:
        ...

    @abstractmethod
    def exists(self, key: Any) -> bool:
        ...

    @abstractmethod
    def delete(self, key: Any) -> bool:
        ...

    def set_many(self, items: dict[Any, Any], ttl_seconds: int | None = None) -> None:
        for key, value in items.items():
            self.set(key=key, value=value, ttl_seconds=ttl_seconds)

    def get_many(self, keys: list[Any], default: Any = MISSING) -> dict[Any, Any]:
        return {key: self.get(key=key, default=default) for key in keys}
