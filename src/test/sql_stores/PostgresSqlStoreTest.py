import os
import unittest
import uuid

import psycopg

import sql_stores
from sql_stores.PostgresSqlStore import PostgresSqlStore


class PostgresSqlStoreTest(unittest.TestCase):
    dsn: str | None = None
    table_name: str

    @classmethod
    def setUpClass(cls):
        cls.dsn = (
            os.environ.get("SQL_STORE_TEST_POSTGRES_DSN")
            or os.environ.get("KV_STORE_TEST_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
            or os.environ.get("DATABASE_URL")
            or "postgresql://localhost:5432/postgres"
        )
        cls.table_name = f"sql_store_{uuid.uuid4().hex[:10]}"

        try:
            with psycopg.connect(cls.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {cls.table_name} ("
                        f"id TEXT PRIMARY KEY, "
                        f"amount INTEGER NOT NULL, "
                        f"note TEXT NULL"
                        f")"
                    )
                conn.commit()
        except Exception as e:
            raise unittest.SkipTest(f"Skipping PostgresSqlStore tests: {e}")

    @classmethod
    def tearDownClass(cls):
        if not cls.dsn:
            return
        with psycopg.connect(cls.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {cls.table_name}")
            conn.commit()

    def setUp(self):
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
            conn.commit()

    def _store(self) -> PostgresSqlStore:
        return PostgresSqlStore(dsn=self.dsn)

    def test_factory_builds_postgres_store(self):
        store = sql_stores.of("postgres", dsn=self.dsn)
        self.assertIsInstance(store, PostgresSqlStore)
        store.close()

    def test_execute_and_query(self):
        with self._store() as store:
            affected = store.execute(
                f"INSERT INTO {self.table_name} (id, amount, note) VALUES (%s, %s, %s)",
                ("a1", 10, "first"),
            )
            self.assertEqual(1, affected)

            rows = store.query(
                f"SELECT id, amount, note FROM {self.table_name} WHERE id = %s",
                ("a1",),
            )
            self.assertEqual(1, len(rows))
            self.assertEqual("a1", rows[0]["id"])
            self.assertEqual(10, rows[0]["amount"])
            self.assertEqual("first", rows[0]["note"])

    def test_execute_many(self):
        with self._store() as store:
            total = store.execute_many(
                f"INSERT INTO {self.table_name} (id, amount, note) VALUES (%s, %s, %s)",
                [("m1", 1, "x"), ("m2", 2, "y"), ("m3", 3, "z")],
            )
            self.assertEqual(3, total)

            rows = store.query(f"SELECT id FROM {self.table_name} ORDER BY id")
            self.assertEqual(["m1", "m2", "m3"], [r["id"] for r in rows])

    def test_transaction_commit(self):
        with self._store() as store:
            with store.transaction():
                store.execute(
                    f"INSERT INTO {self.table_name} (id, amount, note) VALUES (%s, %s, %s)",
                    ("t1", 100, "tx"),
                )
                store.execute(
                    f"UPDATE {self.table_name} SET amount = amount + %s WHERE id = %s",
                    (50, "t1"),
                )

            rows = store.query(
                f"SELECT amount FROM {self.table_name} WHERE id = %s",
                ("t1",),
            )
            self.assertEqual(150, rows[0]["amount"])

    def test_transaction_rollback_on_error(self):
        with self._store() as store:
            with self.assertRaises(psycopg.Error):
                with store.transaction():
                    store.execute(
                        f"INSERT INTO {self.table_name} (id, amount, note) VALUES (%s, %s, %s)",
                        ("r1", 10, "ok"),
                    )
                    # Duplicate PK forces transaction failure and rollback.
                    store.execute(
                        f"INSERT INTO {self.table_name} (id, amount, note) VALUES (%s, %s, %s)",
                        ("r1", 20, "dup"),
                    )

            rows = store.query(f"SELECT id FROM {self.table_name} WHERE id = %s", ("r1",))
            self.assertEqual([], rows)

    def test_close_and_reuse_raises(self):
        store = self._store()
        store.close()
        with self.assertRaises(RuntimeError):
            store.query("SELECT 1")


if __name__ == "__main__":
    unittest.main()
