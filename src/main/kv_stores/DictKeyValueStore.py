import json
from datetime import datetime, timedelta, timezone
from typing import Any

from kv_stores.KeyValueStore import KeyValueStore, KeyNotFoundError, MISSING


class DictKeyValueStore(KeyValueStore):
    """
    In-memory key-value store backed by a Python dict.

    Behavior:
    - set(): full UPSERT replace.
    - patch(): partial update; key must exist.
    - get(): returns a shallow copy of stored dict.
    - TTL is enforced lazily on read/update operations.
    """

    def __init__(self):
        self._values: dict[Any, dict[str, Any]] = {}
        self._expires_at: dict[Any, datetime | None] = {}

    @staticmethod
    def _parse_value(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"value JSON string is invalid: {e}") from e
            if not isinstance(decoded, dict):
                raise ValueError("value must decode to a JSON object")
            return decoded
        raise ValueError("value must be a dict or JSON object string")

    def _is_expired(self, key: Any) -> bool:
        exp = self._expires_at.get(key)
        return exp is not None and exp <= datetime.now(timezone.utc)

    def _purge_if_expired(self, key: Any) -> None:
        if self._is_expired(key):
            self._values.pop(key, None)
            self._expires_at.pop(key, None)

    def set(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        parsed = self._parse_value(value)
        self._values[key] = parsed
        self._expires_at[key] = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None else None
        )

    def patch(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        self._purge_if_expired(key)
        if key not in self._values:
            raise KeyNotFoundError(f"Key not found: {key}")

        parsed = self._parse_value(value)
        self._values[key].update(parsed)

        if ttl_seconds is not None:
            self._expires_at[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    def get(self, key: Any, default: Any = MISSING) -> dict:
        self._purge_if_expired(key)
        if key not in self._values:
            if default is MISSING:
                raise KeyNotFoundError(f"Key not found: {key}")
            return default
        return dict(self._values[key])

    def exists(self, key: Any) -> bool:
        self._purge_if_expired(key)
        return key in self._values

    def delete(self, key: Any) -> bool:
        self._purge_if_expired(key)
        existed = key in self._values
        self._values.pop(key, None)
        self._expires_at.pop(key, None)
        return existed
