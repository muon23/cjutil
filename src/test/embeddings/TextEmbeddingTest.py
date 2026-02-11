import os
import time
import unittest

import embeddings
from embeddings.LangChainEmbedding import LangChainEmbedding
from embeddings.SentenceTransformerEmbedding import SentenceTransformerEmbedding


class TextEmbeddingTest(unittest.TestCase):
    SENTENCE_TEXTS = [
        "Hello world.",
        "Hola mundo.",
        "Bonjour le monde.",
        "Hallo Welt. Dies ist ein etwas laengerer deutscher Satz, um die Einbettungszeit zu beobachten.",
        "Ciao mondo. Questo e un esempio in italiano con piu dettagli per misurare meglio la latenza.",
        "Ola mundo. Este texto em portugues inclui mais contexto para um teste de desempenho mais realista.",
        "Hej varlden. Den har svenska meningen ar lite langre for att ge en battre tidsindikation.",
        "Merhaba dunya. Bu Turkce cumle, modelin farkli dil yapilarini nasil ele aldigini gostermek icin uzatildi.",
        "Xin chao the gioi. Day la mot cau tieng Viet dai hon de do thoi gian phan hoi.",
        "Selamat pagi dunia. Kalimat bahasa Indonesia ini sedikit lebih panjang untuk pengujian latensi.",
        "Konnichiwa sekai. Nihongo no bunsho wo sukoshi nagaku shite, embedding no jikkou jikan wo kakunin shimasu.",
        "Annyeong haseyo segye. I munjang-eun eungdab sigan cheugjeong-eul wihae jom deo gireo jyeosseumnida.",
        "Marhaban bialalam. Hatha nassun arabiyyun tawilun qaleelan liqiyas zamani alistijaba.",
        "Privet mir. Etot russkiy tekst nemnogo dlinnee, chtoby otsenit vremya polucheniya vektorov.",
        "Namaste duniya. Yah Hindi vakya thoda lamba hai taa ki response time aur achchhi tarah dikh sake.",
        "Sawasdee lok. Prayoek phasa Thai ni yao khuen leknoi phuea thotsop wela.",
        (
            "This is a longer English paragraph about embedding benchmarks. "
            "We include multiple clauses, punctuation, and mixed vocabulary so that timing reflects "
            "a more practical workload than tiny greeting phrases."
        ),
        (
            "Data engineering pipelines often embed titles, summaries, comments, and user profiles. "
            "This synthetic text imitates medium-length production content and can reveal model "
            "differences more clearly."
        ),
        (
            "In retrieval systems, embeddings are frequently generated in batches rather than one at a time. "
            "By sending a richer set of multilingual sentences, this test gives a more informative "
            "latency signal per model."
        ),
    ]

    # Best response times to encode the above.  (CPU: MacBook Pro m1.  Network: 580M D, 778M U.)
    #
    # bge-m3    422ms
    # e5        222ms   --> better relevancy local model
    # mpnet     110ms
    # labse     490ms
    #
    # embed-3l  271ms   --> better relevancy OpenAI
    # embed-3s  190ms
    # ada       155ms

    @staticmethod
    def _timed_embed(embedder, texts: list[str], provider: str, model_name: str):
        t0 = time.perf_counter()
        vectors = embedder.embed_documents(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[embed] provider={provider} model={model_name} elapsed_ms={elapsed_ms:.2f}")
        return vectors

    def test_sentence_transformer_supported_models(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self.skipTest("sentence-transformers is not installed")

        embeddings.active_embeddings.clear()
        for model_name in SentenceTransformerEmbedding.SUPPORTED_MODELS:
            with self.subTest(model_name=model_name):
                embedder = embeddings.of(model_name)
                self.assertIsInstance(embedder, SentenceTransformerEmbedding)
                vectors = self._timed_embed(
                    embedder=embedder,
                    texts=self.SENTENCE_TEXTS,
                    provider="sentence-transformers",
                    model_name=model_name
                )
                self.assertEqual(model_name, embedder.get_model_name())
                self.assertEqual(len(self.SENTENCE_TEXTS), len(vectors))
                self.assertGreater(len(vectors[0]), 0)
                self.assertIsInstance(vectors[0][0], float)

    def test_langchain_supported_models(self):
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            self.skipTest("langchain-openai is not installed")

        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            self.skipTest("OPENAI_API_KEY is not set")

        embeddings.active_embeddings.clear()
        for model_name in LangChainEmbedding.SUPPORTED_MODELS:
            with self.subTest(model_name=model_name):
                embedder = embeddings.of(model_name, model_key=api_key)
                self.assertIsInstance(embedder, LangChainEmbedding)
                vectors = self._timed_embed(
                    embedder=embedder,
                    texts=["hello", "hola"],
                    provider="openai",
                    model_name=model_name
                )
                self.assertEqual(model_name, embedder.get_model_name())
                self.assertEqual(2, len(vectors))
                self.assertGreater(len(vectors[0]), 0)
                self.assertIsInstance(vectors[0][0], float)

    def test_factory_of_supported_models(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self.skipTest("sentence-transformers is not installed")

        for model_name in SentenceTransformerEmbedding.SUPPORTED_MODELS:
            with self.subTest(provider="sentence-transformers", model_name=model_name):
                embedder = embeddings.of(model_name)
                self.assertIsInstance(embedder, SentenceTransformerEmbedding)
                vectors = embedder.embed_documents(["factory test"])
                self.assertEqual(1, len(vectors))
                self.assertGreater(len(vectors[0]), 0)

        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            self.skipTest("langchain-openai is not installed")

        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            self.skipTest("OPENAI_API_KEY is not set")

        for model_name in LangChainEmbedding.SUPPORTED_MODELS:
            with self.subTest(provider="langchain-openai", model_name=model_name):
                embedder = embeddings.of(model_name, model_key=api_key)
                self.assertIsInstance(embedder, LangChainEmbedding)
                vectors = embedder.embed_documents(["factory hello"])
                self.assertEqual(1, len(vectors))
                self.assertGreater(len(vectors[0]), 0)


if __name__ == "__main__":
    unittest.main()
