import json
import os
import unittest
import uuid

import psycopg

from kv_stores.KeyValueStore import KeyNotFoundError
from kv_stores.PostgresKeyValueStore import PostgresKeyValueStore, KeyValueSchemaSpec


class PostgresKeyValueStoreTest(unittest.TestCase):
    dsn: str | None = None
    table_single: str
    table_composite: str

    @classmethod
    def setUpClass(cls):
        cls.dsn = (
            os.environ.get("KV_STORE_TEST_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
            or os.environ.get("DATABASE_URL")
            or "postgresql://localhost:5432/postgres"
        )

        suffix = uuid.uuid4().hex[:10]
        cls.table_single = f"kv_store_single_{suffix}"
        cls.table_composite = f"kv_store_composite_{suffix}"

    @classmethod
    def tearDownClass(cls):
        if not cls.dsn:
            return
        with psycopg.connect(cls.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {cls.table_single}")
                cur.execute(f"DROP TABLE IF EXISTS {cls.table_composite}")
            conn.commit()

    def _single_store(self) -> PostgresKeyValueStore:
        schema = KeyValueSchemaSpec(
            columns={
                "user_id": "TEXT NOT NULL",
                "age": "INTEGER",
                "nickname": "TEXT",
                "expires_at": "TIMESTAMPTZ",
            },
            indexed_columns=["age"],
        )
        return PostgresKeyValueStore(
            dsn=self.dsn,
            table_name=self.table_single,
            key_columns="user_id",
            schema_spec=schema,
            create_if_not_exist=True,
        )

    def _composite_store(self) -> PostgresKeyValueStore:
        schema = KeyValueSchemaSpec(
            columns={
                "tenant_id": "TEXT NOT NULL",
                "doc_id": "TEXT NOT NULL",
                "content": "TEXT",
                "version": "INTEGER",
            }
        )
        return PostgresKeyValueStore(
            dsn=self.dsn,
            table_name=self.table_composite,
            key_columns=["tenant_id", "doc_id"],
            schema_spec=schema,
            create_if_not_exist=True,
        )

    def test_reject_reserved_extra_data_column(self):
        with self.assertRaises(ValueError):
            KeyValueSchemaSpec(
                columns={
                    "id": "TEXT NOT NULL",
                    "kv_store_extra_data": "JSONB",
                }
            ).validate(key_columns=["id"])

    def test_set_upsert_and_get_single_key(self):
        store = self._single_store()
        store.set(
            key="u1",
            value={"age": "42", "nickname": "cj", "favorite_color": "blue"},
        )
        data = store.get("u1")
        self.assertEqual("u1", data["user_id"])
        self.assertEqual(42, data["age"])
        self.assertEqual("cj", data["nickname"])
        self.assertEqual("blue", data["favorite_color"])

    def test_set_overwrite_omitted_optional_to_null(self):
        store = self._single_store()
        store.set(key="u2", value={"age": 18, "nickname": "first"})
        store.set(key="u2", value={"age": 19})
        data = store.get("u2")
        self.assertEqual(19, data["age"])
        self.assertIsNone(data["nickname"])

    def test_patch_partial_update(self):
        store = self._single_store()
        store.set(key="u3", value={"age": 20, "nickname": "old"})
        store.patch(key="u3", value={"nickname": "new", "new_tag": "vip"})
        data = store.get("u3")
        self.assertEqual(20, data["age"])
        self.assertEqual("new", data["nickname"])
        self.assertEqual("vip", data["new_tag"])

    def test_get_default_and_raise_behavior(self):
        store = self._single_store()
        self.assertIsNone(store.get("missing-user", default=None))
        self.assertEqual({"x": 1}, store.get("missing-user", default={"x": 1}))
        with self.assertRaises(KeyNotFoundError):
            store.get("missing-user")

    def test_composite_key_set_and_get_with_json_inputs(self):
        store = self._composite_store()
        key = json.dumps({"tenant_id": "t1", "doc_id": "d1"})
        value = json.dumps({"content": "hello", "version": "3", "source": "ingest"})
        store.set(key=key, value=value)
        data = store.get({"tenant_id": "t1", "doc_id": "d1"})
        self.assertEqual("t1", data["tenant_id"])
        self.assertEqual("d1", data["doc_id"])
        self.assertEqual("hello", data["content"])
        self.assertEqual(3, data["version"])
        self.assertEqual("ingest", data["source"])


if __name__ == "__main__":
    unittest.main()
