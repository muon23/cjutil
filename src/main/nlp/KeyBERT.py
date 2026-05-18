import logging
from typing import Any

from nlp.TextInput import NlpInput, to_texts, wrap_list_of_lists


class KeyBERT:
    """
    Keyword extraction backed by the KeyBERT library.

    Accepts a single string, file path, or Path, or a list of those, and returns
    keywords for each document. A single input yields list[str]; a list input yields
    list[list[str]].
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", **kwargs: Any):
        """
        Args:
            model: sentence-transformers model name passed to KeyBERT.
            **kwargs: Additional arguments forwarded to keybert.KeyBERT().
        """
        try:
            from keybert import KeyBERT as _KeyBERTModel
        except ImportError as e:
            raise RuntimeError(
                "keybert is required for KeyBERT. Install it with: pip install keybert"
            ) from e

        self.model_name = model
        self._model = _KeyBERTModel(model=model, **kwargs)
        logging.info("KeyBERT using model %s", model)

    def extract(
        self,
        data: NlpInput,
        top_n: int = 10,
        **kwargs: Any,
    ) -> list[str] | list[list[str]]:
        """
        Extract keywords from text or file input.

        Args:
            data: Document text(s) and/or path(s) to readable files (.txt, .md, .pdf, ...).
            top_n: Maximum keywords per document.
            **kwargs: Forwarded to KeyBERT.extract_keywords() (e.g. keyphrase_ngram_range).

        Returns:
            list[str] for one input, or list[list[str]] for a list input.
        """
        texts, is_batch = to_texts(data)
        rows: list[list[str]] = []

        for text in texts:
            if not text:
                rows.append([])
                continue
            scored = self._model.extract_keywords(text, top_n=top_n, **kwargs)
            rows.append([keyword for keyword, _score in scored])

        return wrap_list_of_lists(rows, is_batch)
