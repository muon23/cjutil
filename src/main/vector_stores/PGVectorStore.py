import json
import re
from typing import Any, Literal

import psycopg

from vector_stores.VectorStore import VectorStore, Match


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PGVectorStore(VectorStore):
    """
    PostgreSQL + pgvector-backed vector store.

    Notes:
    - Requires pgvector extension in Postgres (created automatically when create_if_not_exist=True).
    - `score` is returned as a higher-is-better similarity-like value:
      - cosine: 1.0 - cosine_distance
      - l2 / inner_product: -distance
    """

    _METRIC_TO_OPERATOR = {
        "cosine": "<=>",
        "l2": "<->",
        "inner_product": "<#>",
    }

    def __init__(
            self,
            dsn: str,
            table_name: str,
            dimension: int,
            create_if_not_exist: bool = False,
            schema_name: str | None = None,
            distance: Literal["cosine", "l2", "inner_product"] = "cosine",
            id_column: str = "id",
            vector_column: str = "embedding",
            metadata_column: str = "metadata",
            document_column: str = "document",
    ):
        """
        Initialize pgvector store metadata and optionally bootstrap schema.

        Args:
            dsn: PostgreSQL DSN string.
            table_name: Target table name.
            dimension: Required vector dimension.
            create_if_not_exist: Whether to create extension/table if missing.
            schema_name: Optional schema name.
            distance: Distance metric (cosine, l2, inner_product).
            id_column: Record ID column name.
            vector_column: Vector column name.
            metadata_column: Metadata JSONB column name.
            document_column: Optional document text column name.

        Raises:
            ValueError: If identifiers, dimension, or distance metric are invalid.
        """
        self._dsn = dsn
        self._table_name = table_name
        self._dimension = dimension
        self._schema_name = schema_name
        self._distance = distance
        self._id_column = id_column
        self._vector_column = vector_column
        self._metadata_column = metadata_column
        self._document_column = document_column

        self._validate_identifier(self._table_name, "table_name")
        if self._schema_name:
            self._validate_identifier(self._schema_name, "schema_name")
        self._validate_identifier(self._id_column, "id_column")
        self._validate_identifier(self._vector_column, "vector_column")
        self._validate_identifier(self._metadata_column, "metadata_column")
        self._validate_identifier(self._document_column, "document_column")

        if self._distance not in self._METRIC_TO_OPERATOR:
            raise ValueError(f"Unsupported distance '{self._distance}'")
        if self._dimension <= 0:
            raise ValueError("dimension must be positive")

        if create_if_not_exist:
            self._create_if_not_exists()

    @classmethod
    def create_if_not_exist(
            cls,
            dsn: str,
            table_name: str,
            dimension: int,
            schema_name: str | None = None,
            distance: Literal["cosine", "l2", "inner_product"] = "cosine",
    ) -> "PGVectorStore":
        """
        Construct a store and create pgvector objects when missing.

        Args:
            dsn: PostgreSQL DSN string.
            table_name: Target table name.
            dimension: Required vector dimension.
            schema_name: Optional schema name.
            distance: Distance metric (cosine, l2, inner_product).

        Returns:
            Initialized PGVectorStore instance.
        """
        return cls(
            dsn=dsn,
            table_name=table_name,
            dimension=dimension,
            create_if_not_exist=True,
            schema_name=schema_name,
            distance=distance,
        )

    @staticmethod
    def _validate_identifier(identifier: str, name: str) -> None:
        if not _IDENTIFIER.match(identifier):
            raise ValueError(f"Invalid SQL identifier for {name}: {identifier}")

    def _qualified_table(self) -> str:
        return f"{self._schema_name}.{self._table_name}" if self._schema_name else self._table_name

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def _create_if_not_exists(self) -> None:
        """Create pgvector extension/table if not already present."""
        table = self._qualified_table()
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"{self._id_column} TEXT PRIMARY KEY, "
            f"{self._vector_column} VECTOR({self._dimension}) NOT NULL, "
            f"{self._metadata_column} JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
            f"{self._document_column} TEXT NULL"
            f")"
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(ddl)
            conn.commit()

    def upsert(
            self,
            record_id: str,
            vector: list[float],
            metadata: dict[str, Any] | None = None,
            document: str | None = None,
    ) -> None:
        """
        Insert or replace one vector record by record_id.

        Args:
            record_id: Unique record identifier.
            vector: Embedding vector with configured dimension.
            metadata: Optional metadata dict stored as JSONB.
            document: Optional document text.

        Raises:
            ValueError: If vector dimension does not match store dimension.
        """
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}"
            )
        metadata_json = json.dumps(metadata or {})
        vector_str = self._vector_literal(vector)
        table = self._qualified_table()

        sql = (
            f"INSERT INTO {table} "
            f"({self._id_column}, {self._vector_column}, {self._metadata_column}, {self._document_column}) "
            f"VALUES (%s, %s::vector, %s::jsonb, %s) "
            f"ON CONFLICT ({self._id_column}) DO UPDATE SET "
            f"{self._vector_column}=EXCLUDED.{self._vector_column}, "
            f"{self._metadata_column}=EXCLUDED.{self._metadata_column}, "
            f"{self._document_column}=EXCLUDED.{self._document_column}"
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (record_id, vector_str, metadata_json, document))
            conn.commit()

    def query(
            self,
            vector: list[float],
            top_k: int = 10,
            metadata_filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """
        Run nearest-neighbor lookup with optional metadata filter.

        Args:
            vector: Query vector with configured dimension.
            top_k: Maximum number of matches to return.
            metadata_filter: Optional JSONB containment filter.

        Returns:
            Ranked Match list (higher score is better).

        Raises:
            ValueError: If vector dimension is invalid.
        """
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}"
            )
        if top_k <= 0:
            return []

        op = self._METRIC_TO_OPERATOR[self._distance]
        vector_str = self._vector_literal(vector)
        table = self._qualified_table()

        where_clause = ""
        args: list[Any] = [vector_str, vector_str]
        if metadata_filter:
            where_clause = f"WHERE {self._metadata_column} @> %s::jsonb "
            args.append(json.dumps(metadata_filter))
        args.append(top_k)

        sql = (
            f"SELECT {self._id_column}, {self._metadata_column}, {self._document_column}, "
            f"({self._vector_column} {op} %s::vector) AS distance, "
            f"({self._vector_column} {op} %s::vector) AS distance_for_sort "
            f"FROM {table} "
            f"{where_clause}"
            f"ORDER BY distance_for_sort ASC "
            f"LIMIT %s"
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                rows = cur.fetchall()

        matches: list[Match] = []
        for row in rows:
            row_id, metadata, document, distance, _ = row
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            metadata = metadata if isinstance(metadata, dict) else {}

            # Normalize all distance modes to a higher-is-better score for callers.
            if self._distance == "cosine":
                score = 1.0 - float(distance)
            else:
                score = -float(distance)

            matches.append(Match(
                record_id=str(row_id),
                score=score,
                metadata=metadata,
                document=document,
            ))
        return matches

    def delete(self, record_id: str) -> bool:
        """
        Delete one vector record by ID.

        Args:
            record_id: Record identifier.

        Returns:
            True if a row was deleted, otherwise False.
        """
        table = self._qualified_table()
        sql = f"DELETE FROM {table} WHERE {self._id_column} = %s"
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (record_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
