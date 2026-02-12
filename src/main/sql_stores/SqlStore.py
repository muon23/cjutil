from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Iterable, TypeAlias

SqlRow: TypeAlias = dict[str, Any]
SqlRows: TypeAlias = list[SqlRow]
SqlParams: TypeAlias = dict[str, Any] | tuple[Any, ...] | list[Any] | None


class SqlStore(ABC):
    """
    Provider-agnostic synchronous SQL store interface.

    This abstraction intentionally stays minimal and backend-neutral:
    - execute(): run one write/DDL statement
    - execute_many(): run one write statement against many parameter sets
    - query(): run read statements and return rows as dictionaries
    - transaction(): explicit atomic unit-of-work boundary
    - close(): release database resources
    """

    @abstractmethod
    def execute(self, sql: str, params: SqlParams = None) -> int:
        """
        Run a single write/DDL SQL statement.

        Args:
            sql: SQL statement string.
            params: Positional or named bind parameters.

        Returns:
            Number of affected rows when available.

        Raises:
            Exception: Backend-specific SQL execution errors.
        """
        ...

    @abstractmethod
    def execute_many(self, sql: str, params_list: Iterable[SqlParams]) -> int:
        """
        Run one write SQL statement repeatedly with many parameter sets.

        Args:
            sql: SQL statement string.
            params_list: Iterable of parameter sets.

        Returns:
            Number of affected rows when available.

        Raises:
            Exception: Backend-specific SQL execution errors.
        """
        ...

    @abstractmethod
    def query(self, sql: str, params: SqlParams = None) -> SqlRows:
        """
        Run a read SQL statement and fetch all rows.

        Args:
            sql: SQL query string.
            params: Positional or named bind parameters.

        Returns:
            Query rows represented as a list of dictionaries.

        Raises:
            Exception: Backend-specific SQL execution errors.
        """
        ...

    @abstractmethod
    def transaction(self) -> AbstractContextManager["SqlStore"]:
        """
        Create a transaction context manager for atomic operations.

        Returns:
            A context manager that begins a transaction on enter and commits on success,
            or rolls back on exception.

        Raises:
            Exception: Backend-specific transaction errors.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Close underlying resources (connections/cursors/pools).

        Returns:
            None.
        """
        ...

    def __enter__(self) -> "SqlStore":
        """
        Enter context manager scope for the store itself.

        Returns:
            The same SqlStore instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit store context manager scope.

        Args:
            exc_type: Exception type if raised inside block.
            exc: Exception instance if raised inside block.
            tb: Traceback if raised inside block.

        Returns:
            None.
        """
        self.close()
