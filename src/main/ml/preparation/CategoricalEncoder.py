from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder

ArrayLike = np.ndarray


class CategoricalEncoder:
    """
    Thin wrapper around sklearn OneHotEncoder.
    """

    def __init__(self, sparse_output: bool = False, handle_unknown: str = "ignore", **kwargs: Any):
        """
        Initialize one-hot encoder.

        Args:
            sparse_output: Whether to return sparse matrix output.
            handle_unknown: Unknown category handling strategy.
            **kwargs: Keyword args passed to OneHotEncoder.
        """
        self._encoder = OneHotEncoder(
            sparse_output=sparse_output,
            handle_unknown=handle_unknown,
            **kwargs,
        )

    def fit(self, X: ArrayLike) -> "CategoricalEncoder":
        """
        Fit category mappings.

        Args:
            X: Categorical feature matrix.

        Returns:
            The same CategoricalEncoder instance.
        """
        self._encoder.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        """
        One-hot encode input matrix.

        Args:
            X: Categorical feature matrix.

        Returns:
            Encoded feature matrix (dense or sparse based on configuration).
        """
        return self._encoder.transform(X)

    def fit_transform(self, X: ArrayLike) -> Any:
        """
        Fit encoder and transform in one step.

        Args:
            X: Categorical feature matrix.

        Returns:
            Encoded feature matrix (dense or sparse based on configuration).
        """
        return self._encoder.fit_transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """
        Return output feature names for transformed columns.

        Args:
            input_features: Optional input feature names.

        Returns:
            Array of encoded feature names.
        """
        return self._encoder.get_feature_names_out(input_features)
