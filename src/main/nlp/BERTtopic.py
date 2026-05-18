import logging
from typing import Any

import numpy as np

from nlp.TextInput import NlpInput, to_texts, wrap_list_of_lists

_SMALL_CORPUS_THRESHOLD = 8


class _PassthroughUMAP:
    """Skip UMAP when there are too few documents for a stable projection."""

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_dims = min(5, X.shape[1], max(1, n_samples - 1))
        return X[:, :n_dims]


class BERTtopic:
    """
    Topic extraction backed by the BERTopic library.

    Accepts the same inputs as KeyBERT. A single input yields list[str] (topic terms);
    a list input yields list[list[str]].
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            **kwargs: Arguments forwarded to bertopic.BERTopic() (e.g. language, embedding_model).
        """
        self._kwargs = kwargs

    def _build_model(self, corpus_size: int):
        try:
            from bertopic import BERTopic
        except ImportError as e:
            raise RuntimeError(
                "bertopic is required for BERTtopic. Install it with: pip install bertopic"
            ) from e

        if corpus_size < _SMALL_CORPUS_THRESHOLD:
            from sklearn.cluster import KMeans

            n_clusters = max(1, min(corpus_size, 3))
            cluster_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            return BERTopic(
                umap_model=_PassthroughUMAP(),
                hdbscan_model=cluster_model,
                min_topic_size=1,
                calculate_probabilities=False,
                verbose=False,
                **self._kwargs,
            )

        min_topic_size = max(2, min(corpus_size, 10))
        return BERTopic(
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
            verbose=False,
            **self._kwargs,
        )

    def extract(
        self,
        data: NlpInput,
        top_n: int = 10,
        **kwargs: Any,
    ) -> list[str] | list[list[str]]:
        """
        Extract topic terms for each document in the input.

        Args:
            data: Document text(s) and/or path(s) to readable files.
            top_n: Maximum topic terms per document.
            **kwargs: Reserved for future use (fit/transform options).

        Returns:
            list[str] for one input, or list[list[str]] for a list input.
        """
        texts, is_batch = to_texts(data)
        if not texts or all(not text for text in texts):
            return wrap_list_of_lists([[] for _ in texts], is_batch)

        non_empty = [text for text in texts if text]
        model = self._build_model(len(non_empty))
        topics, _ = model.fit_transform(non_empty)

        topic_rows: dict[int, list[str]] = {}
        for topic_id in set(topics):
            if topic_id == -1:
                topic_rows[topic_id] = []
            else:
                terms = model.get_topic(topic_id) or []
                topic_rows[topic_id] = [word for word, _score in terms[:top_n]]

        non_empty_iter = iter(topics)
        rows: list[list[str]] = []
        for text in texts:
            if not text:
                rows.append([])
                continue
            topic_id = next(non_empty_iter)
            rows.append(topic_rows.get(topic_id, []))

        logging.info("BERTtopic extracted topics for %d document(s)", len(non_empty))
        return wrap_list_of_lists(rows, is_batch)
