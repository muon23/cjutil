from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from langchain_core.tools import tool

from sql_stores.SqlStore import SqlStore, SqlRows


class SqlQuery:
    """
    LangChain-tool adapter around SqlStore.

    This class delegates actual SQL execution to a SqlStore instance and exposes
    a simple tool-friendly query API for LLM agents.
    """

    def __init__(
            self,
            store: SqlStore,
            name: str = "sql_query",
            description: Optional[str] = None,
            table_descriptions: Optional[list["SqlQuery.TableDescription"]] = None,
    ):
        """
        Initialize SQL query adapter.

        Args:
            store: Concrete SqlStore backend.
            name: LangChain tool name.
            description: LangChain tool description.
            table_descriptions: Optional metadata for prompt/tool context.
        """
        self.store = store
        self.name = name
        self.description = description or "Run a read-only SQL query against the database."
        self._table_descriptions = table_descriptions or []

    def _table_description_text(self) -> str:
        """
        Build compact schema text for tool description.

        Returns:
            Short schema summary text.
        """
        if not self._table_descriptions:
            return "No table schema metadata provided."
        lines: list[str] = []
        for table in self._table_descriptions:
            cols = ", ".join(f"{c.name}:{c.type}" for c in table.columns)
            lines.append(f"{table.name}({cols})")
        return "Known tables: " + "; ".join(lines)

    @dataclass
    class ColumnDescription:
        """Human-readable column metadata for prompting/tool docs."""

        name: str
        description: str
        type: str

    @dataclass
    class TableDescription:
        """Human-readable table metadata for prompting/tool docs."""

        name: str
        description: str
        columns: list["SqlQuery.ColumnDescription"]

    def get_table_descriptions(self) -> list[TableDescription]:
        """
        Return table metadata for prompt construction.

        Returns:
            Configured table descriptions.
        """
        return list(self._table_descriptions)

    def query(self, sql: str) -> SqlRows:
        """
        Execute SQL query and return all rows.

        Args:
            sql: SQL query string.

        Returns:
            Query result rows as list of dictionaries.

        Raises:
            Exception: Backend-specific SQL execution errors.
        """
        return self.store.query(sql=sql)

    def describe_tables(self) -> list[dict[str, Any]]:
        """
        Return serializable table schema metadata.

        Returns:
            List of table metadata dictionaries for LLM planning.
        """
        payload: list[dict[str, Any]] = []
        for t in self._table_descriptions:
            payload.append({
                "name": t.name,
                "description": t.description,
                "columns": [{"name": c.name, "description": c.description, "type": c.type} for c in t.columns],
            })
        return payload

    def as_tool(self):
        """
        Expose query() as a LangChain tool.

        Returns:
            A LangChain tool function with schema `sql: str -> list[dict]`.
        """
        desc = f"{self.description} {self._table_description_text()}".strip()

        @tool(self.name, description=desc)
        def _query_tool(sql: str) -> list[dict[str, Any]]:
            return self.query(sql)

        return _query_tool

    def describe_tables_tool(self):
        """
        Expose table schema metadata as a LangChain tool.

        Returns:
            A LangChain tool function with schema `() -> list[dict]`.
        """
        tool_name = f"{self.name}_schema"

        @tool(tool_name, description="Return table/column schema metadata for SQL planning.")
        def _describe_tables_tool() -> list[dict[str, Any]]:
            return self.describe_tables()

        return _describe_tables_tool

    def as_tools(self) -> list[Any]:
        """
        Return the full tool bundle for SQL usage.

        Returns:
            List containing SQL query tool and schema description tool.
        """
        return [self.as_tool(), self.describe_tables_tool()]
