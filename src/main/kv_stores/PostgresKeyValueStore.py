import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

from kv_stores.KeyValueStore import KeyValueStore, KeyNotFoundError, MISSING


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXTRA_DATA_COLUMN = "kv_store_extra_data"


@dataclass(frozen=True)
class KeyValueSchemaSpec:
    """
    Schema definition for typed PostgreSQL-backed key-value storage.

    - `columns` are user-managed columns and SQL type declarations.
    - `_EXTRA_DATA_COLUMN` is reserved and must not appear in `columns`.
    - `indexed_columns` are optional secondary index columns in addition to the key.
    """

    columns: dict[str, str]
    indexed_columns: list[str] = field(default_factory=list)

    @staticmethod
    def _validate_identifier(identifier: str, name: str) -> None:
        if not _IDENTIFIER.match(identifier):
            raise ValueError(f"Invalid SQL identifier for {name}: {identifier}")

    def validate(self, key_columns: list[str]) -> None:
        if _EXTRA_DATA_COLUMN in self.columns:
            raise ValueError(f"'{_EXTRA_DATA_COLUMN}' is reserved and cannot appear in schema columns")

        for col in self.columns:
            self._validate_identifier(col, f"schema column '{col}'")

        for key_col in key_columns:
            if key_col not in self.columns:
                raise ValueError(f"Key column '{key_col}' is missing from schema columns")

        for indexed in self.indexed_columns:
            self._validate_identifier(indexed, f"indexed column '{indexed}'")
            if indexed not in self.columns:
                raise ValueError(f"Indexed column '{indexed}' is missing from schema columns")


class PostgresKeyValueStore(KeyValueStore):
    """
    PostgreSQL typed key-value store with full UPSERT + partial patch support.

    Behavior:
    - set(): full UPSERT. Omitted optional columns are set to NULL.
    - patch(): partial update. Omitted columns are untouched.
    - get(): returns a flat dict of stored fields. Data from `kv_store_extra_data` is merged
      into top-level only for keys not already present as schema columns.
    - get(default=...): if default omitted and key is missing, raises KeyNotFoundError.

    Notes:
    - If `create_if_not_exist=True`, `schema_spec` is required and the table is created if needed.
    - If opening an existing table, schema is introspected from the database.
    - Composite keys require dict or JSON object key input.
    """

    def __init__(
            self,
            dsn: str,
            table_name: str,
            key_columns: str | list[str],
            schema_spec: KeyValueSchemaSpec | None = None,
            create_if_not_exist: bool = False,
            schema_name: str | None = None,
    ):
        self._dsn = dsn
        self._table_name = table_name
        self._schema_name = schema_name
        self._key_columns = [key_columns] if isinstance(key_columns, str) else list(key_columns)
        if not self._key_columns:
            raise ValueError("key_columns cannot be empty")
        for c in self._key_columns:
            KeyValueSchemaSpec._validate_identifier(c, f"key column '{c}'")
        KeyValueSchemaSpec._validate_identifier(self._table_name, "table_name")
        if self._schema_name:
            KeyValueSchemaSpec._validate_identifier(self._schema_name, "schema_name")

        self._schema_spec = schema_spec
        self._columns: dict[str, str] = {}
        self._nullable: dict[str, bool] = {}

        if create_if_not_exist and not self._schema_spec:
            raise ValueError("schema_spec is required when create_if_not_exist=True")

        if create_if_not_exist:
            self._create_table_if_not_exists()
        else:
            self._load_schema_from_existing_table()

        self._assert_key_is_unique_indexed()

    @classmethod
    def create_if_not_exist(
            cls,
            dsn: str,
            table_name: str,
            key_columns: str | list[str],
            schema_spec: KeyValueSchemaSpec,
            schema_name: str | None = None,
    ) -> "PostgresKeyValueStore":
        return cls(
            dsn=dsn,
            table_name=table_name,
            key_columns=key_columns,
            schema_spec=schema_spec,
            create_if_not_exist=True,
            schema_name=schema_name,
        )

    def _qualified_table(self) -> str:
        return f"{self._schema_name}.{self._table_name}" if self._schema_name else self._table_name

    def _create_table_if_not_exists(self) -> None:
        assert self._schema_spec is not None
        self._schema_spec.validate(self._key_columns)

        table = self._qualified_table()
        column_defs = [f"{name} {sql_type}" for name, sql_type in self._schema_spec.columns.items()]
        column_defs.append(f"{_EXTRA_DATA_COLUMN} JSONB NOT NULL DEFAULT '{{}}'::jsonb")

        key_expr = ", ".join(self._key_columns)
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"{', '.join(column_defs)}, "
            f"PRIMARY KEY ({key_expr})"
            f")"
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                for col in self._schema_spec.indexed_columns:
                    if col in self._key_columns:
                        continue
                    idx_name = f"{self._table_name}_{col}_idx"
                    cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col})")
            conn.commit()
        self._load_schema_from_existing_table()

    def _load_schema_from_existing_table(self) -> None:
        if self._schema_spec is not None and _EXTRA_DATA_COLUMN in self._schema_spec.columns:
            raise ValueError(f"'{_EXTRA_DATA_COLUMN}' is reserved and cannot appear in schema columns")

        sql = (
            "SELECT column_name, data_type, udt_name, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = %s "
        )
        args: list[Any] = [self._table_name]
        if self._schema_name:
            sql += "AND table_schema = %s "
            args.append(self._schema_name)
        sql += "ORDER BY ordinal_position"

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                rows = cur.fetchall()

        if not rows:
            raise RuntimeError(f"Table '{self._qualified_table()}' does not exist")

        self._columns = {}
        self._nullable = {}
        for name, data_type, udt_name, is_nullable in rows:
            self._columns[name] = data_type if data_type else udt_name
            self._nullable[name] = str(is_nullable).upper() == "YES"

    def _assert_key_is_unique_indexed(self) -> None:
        schema_sql = "AND n.nspname = %s " if self._schema_name else ""
        sql = (
            "SELECT i.indisunique, array_agg(a.attname ORDER BY ord.n) AS cols "
            "FROM pg_class t "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN pg_index i ON t.oid = i.indrelid "
            "JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS ord(attnum, n) ON TRUE "
            "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ord.attnum "
            "WHERE t.relname = %s "
            f"{schema_sql}"
            "GROUP BY i.indexrelid, i.indisunique"
        )
        args = (self._table_name, self._schema_name) if self._schema_name else (self._table_name,)
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()

        key_cols = self._key_columns
        for is_unique, cols in rows:
            if bool(is_unique) and list(cols) == key_cols:
                return
        raise RuntimeError(
            f"Table '{self._qualified_table()}' must have a UNIQUE/PRIMARY index on key columns {key_cols}"
        )

    def _parse_key(self, key: Any) -> dict[str, Any]:
        if len(self._key_columns) == 1:
            col = self._key_columns[0]
            if isinstance(key, dict):
                if col not in key:
                    raise ValueError(f"Single key column '{col}' missing in key dict")
                return {col: key[col]}
            return {col: key}

        key_data = self._parse_mapping_input(key, "key")
        missing = [c for c in self._key_columns if c not in key_data]
        if missing:
            raise ValueError(f"Missing key columns: {missing}")
        return {c: key_data[c] for c in self._key_columns}

    @staticmethod
    def _parse_mapping_input(payload: Any, label: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as e:
                raise ValueError(f"{label} JSON string is invalid: {e}") from e
            if not isinstance(decoded, dict):
                raise ValueError(f"{label} must decode to a JSON object")
            return decoded
        raise ValueError(f"{label} must be a dict or JSON object string")

    def _split_value_payload(self, value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._parse_mapping_input(value, "value")
        schema_cols = set(self._columns.keys()) - set(self._key_columns) - {_EXTRA_DATA_COLUMN}
        known = {k: payload[k] for k in payload if k in schema_cols}
        extra = {k: payload[k] for k in payload if k not in schema_cols}
        return known, extra

    def _typed_value(self, column: str, value: Any) -> Any:
        if value is None:
            return None
        type_name = self._columns[column].lower()
        try:
            if "int" in type_name:
                return int(value)
            if any(x in type_name for x in ("numeric", "decimal", "real", "double")):
                return float(value)
            if "bool" in type_name:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "1", "t", "yes", "y"}:
                        return True
                    if lowered in {"false", "0", "f", "no", "n"}:
                        return False
                raise ValueError(f"Cannot convert value '{value}' to boolean")
            if type_name in {"json", "jsonb"}:
                if isinstance(value, str):
                    parsed = json.loads(value)
                    return json.dumps(parsed)
                return json.dumps(value)
            if "timestamp" in type_name:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    return value
                raise ValueError(f"Cannot convert value '{value}' to timestamp")
            return value if isinstance(value, str) else str(value)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"Cannot convert column '{column}' with value '{value}': {e}") from e

    def _required_columns_missing(self, data: dict[str, Any]) -> list[str]:
        required = []
        for col in self._columns:
            if col in self._key_columns or col == _EXTRA_DATA_COLUMN:
                continue
            if not self._nullable.get(col, True) and data.get(col) is None:
                required.append(col)
        return required

    def set(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        key_map = self._parse_key(key)
        known, extra = self._split_value_payload(value)

        all_data_cols = [c for c in self._columns if c not in self._key_columns and c != _EXTRA_DATA_COLUMN]
        row: dict[str, Any] = {}
        for col in all_data_cols:
            row[col] = self._typed_value(col, known[col]) if col in known else None

        if ttl_seconds is not None:
            if "expires_at" not in all_data_cols:
                raise ValueError("ttl_seconds is provided but no 'expires_at' column exists in schema")
            row["expires_at"] = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        missing_required = self._required_columns_missing(row)
        if missing_required:
            raise ValueError(f"Missing required non-null columns for set(): {missing_required}")

        if _EXTRA_DATA_COLUMN in self._columns:
            row[_EXTRA_DATA_COLUMN] = json.dumps(extra)
        elif extra:
            raise ValueError(
                f"Unknown fields {list(extra.keys())} are present but table lacks '{_EXTRA_DATA_COLUMN}' column"
            )

        columns = list(key_map.keys()) + list(row.keys())
        values = [key_map[c] for c in key_map] + [row[c] for c in row]
        placeholders = ", ".join(["%s"] * len(columns))
        key_expr = ", ".join(self._key_columns)
        update_cols = [c for c in columns if c not in self._key_columns]
        update_expr = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols])
        sql = (
            f"INSERT INTO {self._qualified_table()} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({key_expr}) DO UPDATE SET {update_expr}"
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(values))
            conn.commit()

    def patch(self, key: Any, value: Any, ttl_seconds: int | None = None) -> None:
        key_map = self._parse_key(key)
        known, extra = self._split_value_payload(value)

        assignments: list[str] = []
        args: list[Any] = []
        for col, v in known.items():
            assignments.append(f"{col} = %s")
            args.append(self._typed_value(col, v))

        if ttl_seconds is not None:
            if "expires_at" not in self._columns:
                raise ValueError("ttl_seconds is provided but no 'expires_at' column exists in schema")
            assignments.append("expires_at = %s")
            args.append(datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds))

        if extra:
            if _EXTRA_DATA_COLUMN not in self._columns:
                raise ValueError(
                    f"Unknown fields {list(extra.keys())} are present but table lacks '{_EXTRA_DATA_COLUMN}' column"
                )
            assignments.append(
                f"{_EXTRA_DATA_COLUMN} = COALESCE({_EXTRA_DATA_COLUMN}, '{{}}'::jsonb) || %s::jsonb"
            )
            args.append(json.dumps(extra))

        if not assignments:
            return

        where = " AND ".join([f"{c} = %s" for c in self._key_columns])
        args.extend([key_map[c] for c in self._key_columns])
        sql = f"UPDATE {self._qualified_table()} SET {', '.join(assignments)} WHERE {where}"

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                updated = cur.rowcount
            conn.commit()
        if updated == 0:
            raise KeyNotFoundError(f"Key not found: {key}")

    def get(self, key: Any, default: Any = MISSING) -> dict:
        key_map = self._parse_key(key)
        where = " AND ".join([f"{c} = %s" for c in self._key_columns])
        sql = f"SELECT * FROM {self._qualified_table()} WHERE {where}"

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(key_map[c] for c in self._key_columns))
                row = cur.fetchone()
                if row is None:
                    if default is MISSING:
                        raise KeyNotFoundError(f"Key not found: {key}")
                    return default
                col_names = [d[0] for d in cur.description]

        data = dict(zip(col_names, row))
        extra = data.pop(_EXTRA_DATA_COLUMN, None)
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = None
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in data:
                    data[k] = v
        return data

    def exists(self, key: Any) -> bool:
        key_map = self._parse_key(key)
        where = " AND ".join([f"{c} = %s" for c in self._key_columns])
        sql = f"SELECT 1 FROM {self._qualified_table()} WHERE {where}"
        sql += " LIMIT 1"

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(key_map[c] for c in self._key_columns))
                return cur.fetchone() is not None

    def delete(self, key: Any) -> bool:
        key_map = self._parse_key(key)
        where = " AND ".join([f"{c} = %s" for c in self._key_columns])
        sql = f"DELETE FROM {self._qualified_table()} WHERE {where}"
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(key_map[c] for c in self._key_columns))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
