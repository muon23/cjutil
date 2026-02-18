from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.impute import SimpleImputer

ArrayLike = np.ndarray


class Imputer:
    """
    Thin wrapper around sklearn SimpleImputer.
    """

    def __init__(self, strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean", **kwargs: Any):
        """
        Initialize imputer.

        Args:
            strategy: Imputation strategy.
            **kwargs: Keyword args passed to SimpleImputer.
        """
        self._imputer = SimpleImputer(strategy=strategy, **kwargs)

    def fit(self, X: ArrayLike) -> "Imputer":
        """
        Fit imputation statistics.

        Args:
            X: Feature matrix.

        Returns:
            The same Imputer instance.
        """
        self._imputer.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fill missing values with fitted imputation policy.

        Args:
            X: Feature matrix.

        Returns:
            Imputed feature matrix.
        """
        return self._imputer.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit imputer and transform in one step.

        Args:
            X: Feature matrix.

        Returns:
            Imputed feature matrix.
        """
        return self._imputer.fit_transform(X)
