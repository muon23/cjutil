import os
import unittest
import uuid

import psycopg

import vector_stores
from vector_stores.PGVectorStore import PGVectorStore


class PGVectorStoreTest(unittest.TestCase):
    dsn: str | None = None
    table_name: str

    @classmethod
    def setUpClass(cls):
        cls.dsn = (
            os.environ.get("VECTOR_STORE_TEST_POSTGRES_DSN")
            or os.environ.get("KV_STORE_TEST_POSTGRES_DSN")
            or os.environ.get("POSTGRES_DSN")
            or os.environ.get("DATABASE_URL")
            or "postgresql://localhost:5432/postgres"
        )
        cls.table_name = f"vector_store_{uuid.uuid4().hex[:10]}"

        # Probe connectivity and pgvector extension capability early.
        try:
            _ = PGVectorStore(
                dsn=cls.dsn,
                table_name=cls.table_name,
                dimension=3,
                create_if_not_exist=True,
            )
        except Exception as e:
            raise unittest.SkipTest(f"Skipping PGVectorStore tests: {e}")

    @classmethod
    def tearDownClass(cls):
        if not cls.dsn:
            return
        with psycopg.connect(cls.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {cls.table_name}")
            conn.commit()

    def _store(self, distance: str = "cosine") -> PGVectorStore:
        return PGVectorStore(
            dsn=self.dsn,
            table_name=self.table_name,
            dimension=3,
            create_if_not_exist=True,
            distance=distance,  # type: ignore[arg-type]
        )

    def test_factory_builds_pgvector_store(self):
        store = vector_stores.of(
            "pgvector",
            dsn=self.dsn,
            table_name=self.table_name,
            dimension=3,
            create_if_not_exist=True,
        )
        self.assertIsInstance(store, PGVectorStore)

    def test_upsert_and_query_best_match(self):
        store = self._store(distance="cosine")
        store.upsert(record_id="a", vector=[1.0, 0.0, 0.0], metadata={"lang": "en"}, document="doc-a")
        store.upsert(record_id="b", vector=[0.0, 1.0, 0.0], metadata={"lang": "fr"}, document="doc-b")
        store.upsert(record_id="c", vector=[0.0, 0.0, 1.0], metadata={"lang": "en"}, document="doc-c")

        matches = store.query(vector=[0.95, 0.02, 0.01], top_k=2)
        self.assertEqual(2, len(matches))
        self.assertEqual("a", matches[0].record_id)
        self.assertGreaterEqual(matches[0].score, matches[1].score)

    def test_query_with_metadata_filter(self):
        store = self._store(distance="cosine")
        store.upsert(record_id="f1", vector=[0.9, 0.1, 0.0], metadata={"lang": "en"}, document="en-doc")
        store.upsert(record_id="f2", vector=[0.95, 0.0, 0.0], metadata={"lang": "es"}, document="es-doc")

        matches = store.query(
            vector=[1.0, 0.0, 0.0],
            top_k=5,
            metadata_filter={"lang": "es"},
        )
        self.assertEqual(1, len(matches))
        self.assertEqual("f2", matches[0].record_id)
        self.assertEqual("es-doc", matches[0].document)

    def test_upsert_overwrite_existing_record(self):
        store = self._store(distance="cosine")
        store.upsert(record_id="u1", vector=[1.0, 0.0, 0.0], metadata={"v": 1}, document="old")
        store.upsert(record_id="u1", vector=[0.0, 1.0, 0.0], metadata={"v": 2}, document="new")

        matches = store.query(vector=[0.0, 1.0, 0.0], top_k=1)
        self.assertEqual(1, len(matches))
        self.assertEqual("u1", matches[0].record_id)
        self.assertEqual("new", matches[0].document)
        self.assertEqual(2, matches[0].metadata.get("v"))

    def test_delete_record(self):
        store = self._store(distance="cosine")
        store.upsert(record_id="d1", vector=[1.0, 0.0, 0.0], metadata={"x": 1}, document="to-delete")
        self.assertTrue(store.delete("d1"))
        self.assertFalse(store.delete("d1"))

    def test_dimension_mismatch_raises(self):
        store = self._store(distance="cosine")
        with self.assertRaises(ValueError):
            store.upsert(record_id="bad", vector=[1.0, 0.0], metadata=None, document=None)
        with self.assertRaises(ValueError):
            store.query(vector=[1.0, 0.0], top_k=1)

    def test_query_non_positive_top_k_returns_empty(self):
        store = self._store(distance="cosine")
        store.upsert(record_id="z1", vector=[1.0, 0.0, 0.0], metadata={}, document="z")
        self.assertEqual([], store.query(vector=[1.0, 0.0, 0.0], top_k=0))


if __name__ == "__main__":
    unittest.main()
