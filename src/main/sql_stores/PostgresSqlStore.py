from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import Iterable

import psycopg

from sql_stores.SqlStore import SqlStore, SqlParams, SqlRows


class PostgresSqlStore(SqlStore):
    """
    psycopg-backed synchronous SQL store for PostgreSQL.

    Behavior:
    - `execute()` / `execute_many()` auto-commit when not inside `transaction()`.
    - `query()` returns rows as list[dict[str, Any]].
    - `transaction()` provides explicit atomic grouping via `with db.transaction():`.
    """

    def __init__(self, dsn: str):
        """
        Initialize PostgreSQL SQL store.

        Args:
            dsn: PostgreSQL DSN string.

        Raises:
            psycopg.Error: If connection cannot be established.
        """
        self._conn = psycopg.connect(dsn)
        self._closed = False
        self._tx_depth = 0

    def execute(self, sql: str, params: SqlParams = None) -> int:
        """
        Run a single write/DDL SQL statement.

        Args:
            sql: SQL statement string.
            params: Positional or named bind parameters.

        Returns:
            Number of affected rows when available (driver `rowcount`).

        Raises:
            psycopg.Error: If SQL execution fails.
        """
        self._ensure_open()
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                affected = cur.rowcount if cur.rowcount is not None else 0
            if self._tx_depth == 0:
                self._conn.commit()
            return affected
        except Exception:
            if self._tx_depth == 0:
                self._conn.rollback()
            raise

    def execute_many(self, sql: str, params_list: Iterable[SqlParams]) -> int:
        """
        Run one write SQL statement repeatedly with many parameter sets.

        Args:
            sql: SQL statement string.
            params_list: Iterable of parameter sets.

        Returns:
            Total affected rows when available (sum of `rowcount`).

        Raises:
            psycopg.Error: If SQL execution fails.
        """
        self._ensure_open()
        total = 0
        try:
            with self._conn.cursor() as cur:
                for params in params_list:
                    cur.execute(sql, params)
                    total += cur.rowcount if cur.rowcount is not None else 0
            if self._tx_depth == 0:
                self._conn.commit()
            return total
        except Exception:
            if self._tx_depth == 0:
                self._conn.rollback()
            raise

    def query(self, sql: str, params: SqlParams = None) -> SqlRows:
        """
        Run a read SQL statement and fetch all rows as dictionaries.

        Args:
            sql: SQL query string.
            params: Positional or named bind parameters.

        Returns:
            Query rows represented as list of dict objects.

        Raises:
            psycopg.Error: If SQL execution fails.
        """
        self._ensure_open()
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(col_names, row)) for row in rows]

    @contextmanager
    def transaction(self) -> AbstractContextManager[SqlStore]:
        """
        Create a transaction scope for atomic operations.

        Args:
            None.

        Returns:
            Context manager yielding this store instance.

        Raises:
            psycopg.Error: If transaction begin/commit/rollback fails.
        """
        self._ensure_open()
        with self._conn.transaction():
            self._tx_depth += 1
            try:
                yield self
            finally:
                self._tx_depth -= 1

    def close(self) -> None:
        """
        Close underlying PostgreSQL connection.

        Returns:
            None.
        """
        if not self._closed:
            self._conn.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PostgresSqlStore is closed")
