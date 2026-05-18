import importlib.util
import tempfile
import unittest
from pathlib import Path

from nlp.BERTtopic import BERTtopic
from nlp.KeyBERT import KeyBERT
from nlp.TextInput import to_texts, wrap_list_of_lists

HAS_KEYBERT = importlib.util.find_spec("keybert") is not None
HAS_BERTOPIC = importlib.util.find_spec("bertopic") is not None

SAMPLE = (
    "Machine learning extracts patterns from data. "
    "Topic models discover themes across document collections."
)


class TextInputTest(unittest.TestCase):
    def test_single_string_is_not_batch(self):
        texts, is_batch = to_texts(SAMPLE)
        self.assertFalse(is_batch)
        self.assertEqual(1, len(texts))

    def test_list_is_batch(self):
        texts, is_batch = to_texts([SAMPLE, SAMPLE])
        self.assertTrue(is_batch)
        self.assertEqual(2, len(texts))

    def test_file_path_is_read(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write(SAMPLE)
            path = handle.name
        try:
            texts, is_batch = to_texts(path)
            self.assertFalse(is_batch)
            self.assertEqual(SAMPLE, texts[0])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_wrap_list_of_lists(self):
        self.assertEqual(["a"], wrap_list_of_lists([["a"]], False))
        self.assertEqual([["a"], ["b"]], wrap_list_of_lists([["a"], ["b"]], True))


@unittest.skipUnless(HAS_KEYBERT, "keybert is not installed")
class KeyBERTTest(unittest.TestCase):
    def test_string_returns_flat_list(self):
        extractor = KeyBERT()
        result = extractor.extract(SAMPLE, top_n=3)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, str) for item in result))
        self.assertGreater(len(result), 0)

    def test_list_returns_list_of_lists(self):
        extractor = KeyBERT()
        result = extractor.extract([SAMPLE, SAMPLE], top_n=3)
        self.assertEqual(2, len(result))
        self.assertTrue(all(isinstance(row, list) for row in result))


@unittest.skipUnless(HAS_BERTOPIC, "bertopic is not installed")
class BERTtopicTest(unittest.TestCase):
    CORPUS = [
        SAMPLE,
        "Neural networks learn hierarchical representations of text and images.",
        "Climate policy shapes renewable energy investment across many countries.",
        "Basketball teams adjust defensive schemes during playoff tournaments.",
        "Cell biology studies how organelles communicate inside living cells.",
    ]

    def test_small_list_returns_list_of_lists(self):
        extractor = BERTtopic()
        result = extractor.extract(self.CORPUS[:2], top_n=5)
        self.assertEqual(2, len(result))
        self.assertTrue(all(isinstance(row, list) for row in result))

    def test_larger_list_returns_list_of_lists(self):
        extractor = BERTtopic()
        result = extractor.extract(self.CORPUS, top_n=5)
        self.assertEqual(len(self.CORPUS), len(result))
        self.assertTrue(all(isinstance(row, list) for row in result))


if __name__ == "__main__":
    unittest.main()
