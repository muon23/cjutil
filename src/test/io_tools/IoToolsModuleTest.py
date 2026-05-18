import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from io_tools import (
    iter_json_array_stream,
    iter_json_objects,
    parse_json_text,
    write_json_array_stream,
)

SAMPLE = [
    {"id": 1, "name": "alpha"},
    {"id": 2, "name": "beta"},
]


class IoToolsModuleTest(unittest.TestCase):
    def test_iter_json_array_stream(self) -> None:
        payload = json.dumps(SAMPLE)
        items = list(iter_json_array_stream(StringIO(payload)))
        self.assertEqual(2, len(items))
        self.assertEqual(1, items[0]["id"])

    def test_iter_json_objects_from_list_and_file(self) -> None:
        self.assertEqual(2, len(list(iter_json_objects(SAMPLE))))

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as handle:
            json.dump(SAMPLE, handle)
            path = handle.name
        try:
            self.assertEqual(2, len(list(iter_json_objects(path))))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_json_text_strips_fence(self) -> None:
        payload = parse_json_text("```json\n" + json.dumps({"ok": True}) + "\n```")
        self.assertTrue(payload["ok"])

    def test_write_json_array_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            write_json_array_stream(path, SAMPLE)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(SAMPLE, loaded)


if __name__ == "__main__":
    unittest.main()
