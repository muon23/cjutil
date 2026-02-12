import json
import time
import unittest

import kv_stores
from kv_stores.DictKeyValueStore import DictKeyValueStore
from kv_stores.KeyValueStore import KeyNotFoundError


class DictKeyValueStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = DictKeyValueStore()

    def test_factory_builds_dict_store(self):
        self.assertIsInstance(kv_stores.of("dict"), DictKeyValueStore)
        self.assertIsInstance(kv_stores.of("memory"), DictKeyValueStore)
        self.assertIsInstance(kv_stores.of("in_memory"), DictKeyValueStore)

    def test_set_upsert_and_get(self):
        self.store.set("k1", {"a": 1, "b": "x"})
        self.assertEqual({"a": 1, "b": "x"}, self.store.get("k1"))

        self.store.set("k1", {"a": 2})
        self.assertEqual({"a": 2}, self.store.get("k1"))

    def test_patch_updates_partial_fields(self):
        self.store.set("k2", {"a": 1, "b": 2})
        self.store.patch("k2", {"b": 3, "c": 4})
        self.assertEqual({"a": 1, "b": 3, "c": 4}, self.store.get("k2"))

    def test_patch_missing_key_raises(self):
        with self.assertRaises(KeyNotFoundError):
            self.store.patch("missing", {"x": 1})

    def test_get_default_and_raise_behavior(self):
        self.assertIsNone(self.store.get("missing", default=None))
        self.assertEqual({"x": 1}, self.store.get("missing", default={"x": 1}))
        with self.assertRaises(KeyNotFoundError):
            self.store.get("missing")

    def test_exists_and_delete(self):
        self.assertFalse(self.store.exists("k3"))
        self.store.set("k3", {"a": 1})
        self.assertTrue(self.store.exists("k3"))
        self.assertTrue(self.store.delete("k3"))
        self.assertFalse(self.store.delete("k3"))
        self.assertFalse(self.store.exists("k3"))

    def test_json_string_inputs(self):
        self.store.set("k4", json.dumps({"a": 1, "b": "json"}))
        self.assertEqual({"a": 1, "b": "json"}, self.store.get("k4"))

        self.store.patch("k4", json.dumps({"b": "patched"}))
        self.assertEqual({"a": 1, "b": "patched"}, self.store.get("k4"))

    def test_ttl_expiry(self):
        self.store.set("k5", {"a": 1}, ttl_seconds=1)
        self.assertTrue(self.store.exists("k5"))
        time.sleep(1.2)
        self.assertFalse(self.store.exists("k5"))
        with self.assertRaises(KeyNotFoundError):
            self.store.get("k5")


if __name__ == "__main__":
    unittest.main()
