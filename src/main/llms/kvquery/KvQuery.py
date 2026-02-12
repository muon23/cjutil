from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.tools import tool

from kv_stores.KeyValueStore import KeyNotFoundError, KeyValueStore


class KvQuery:
    """
    LangChain-tool adapter around KeyValueStore.

    This class is intentionally a thin wrapper: it delegates data access to the
    provided KeyValueStore instance and only provides a tool-friendly interface.
    """

    @dataclass(frozen=True)
    class FieldDescription:
        """Human-readable field metadata for prompting/tool docs."""

        name: str
        description: str
        type: str

    @dataclass(frozen=True)
    class CollectionDescription:
        """Human-readable collection metadata for prompting/tool docs."""

        name: str
        description: str
        key: str
        key_description: str
        value: list["KvQuery.FieldDescription"]

    def __init__(
            self,
            store: KeyValueStore,
            name: str = "key_value_lookup",
            description: Optional[str] = None,
            collection_descriptions: Optional[list[CollectionDescription]] = None,
    ):
        """
        Initialize key-value query adapter.

        Args:
            store: Concrete KeyValueStore backend.
            name: LangChain tool name.
            description: LangChain tool description.
            collection_descriptions: Optional metadata for prompt/tool context.
        """
        self.store = store
        self.name = name
        self.description = description or "Look up one record from a key-value store by key."
        self._collection_descriptions = collection_descriptions or []

    def _collection_description_text(self) -> str:
        """
        Build compact collection text for tool description.

        Returns:
            Short collection summary text.
        """
        if not self._collection_descriptions:
            return "No collection metadata provided."
        parts: list[str] = []
        for collection in self._collection_descriptions:
            fields = ", ".join(f"{f.name}:{f.type}" for f in collection.value)
            parts.append(f"{collection.name}[key={collection.key}]({fields})")
        return "Known collections: " + "; ".join(parts)

    def get_collection_descriptions(self) -> list[CollectionDescription]:
        """
        Return collection metadata for prompt construction.

        Returns:
            Configured collection descriptions.
        """
        return list(self._collection_descriptions)

    def get(self, key: str) -> dict[str, Any]:
        """
        Get a single record by key.

        Args:
            key: Key string. If key is JSON object string, it is decoded and passed
                as composite-key mapping to KeyValueStore.

        Returns:
            Retrieved record as dictionary.

        Raises:
            KeyNotFoundError: If key does not exist.
            ValueError: If key JSON string is invalid.
        """
        parsed_key: Any = key
        if isinstance(key, str) and key.strip().startswith("{"):
            try:
                parsed_key = json.loads(key)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON key string: {e}") from e
        return self.store.get(parsed_key)

    def describe_collections(self) -> list[dict[str, Any]]:
        """
        Return serializable collection metadata.

        Returns:
            List of collection metadata dictionaries for LLM planning.
        """
        payload: list[dict[str, Any]] = []
        for c in self._collection_descriptions:
            payload.append({
                "name": c.name,
                "description": c.description,
                "key": c.key,
                "key_description": c.key_description,
                "value": [{"name": f.name, "description": f.description, "type": f.type} for f in c.value],
            })
        return payload

    def as_tool(self):
        """
        Expose get() as a LangChain tool.

        Returns:
            A LangChain tool function with schema `key: str -> dict`.
        """
        desc = f"{self.description} {self._collection_description_text()}".strip()

        @tool(self.name, description=desc)
        def _get_tool(key: str) -> dict[str, Any]:
            try:
                return self.get(key)
            except KeyNotFoundError:
                # Return a non-throwing structured payload for agent loops.
                return {"found": False, "key": key}

        return _get_tool

    def describe_collections_tool(self):
        """
        Expose collection metadata as a LangChain tool.

        Returns:
            A LangChain tool function with schema `() -> list[dict]`.
        """
        tool_name = f"{self.name}_schema"

        @tool(tool_name, description="Return collection/field metadata for key-value lookup planning.")
        def _describe_collections_tool() -> list[dict[str, Any]]:
            return self.describe_collections()

        return _describe_collections_tool

    def as_tools(self) -> list[Any]:
        """
        Return the full tool bundle for key-value usage.

        Returns:
            List containing lookup tool and collection description tool.
        """
        return [self.as_tool(), self.describe_collections_tool()]
