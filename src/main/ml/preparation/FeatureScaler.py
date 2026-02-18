from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ArrayLike = np.ndarray


class FeatureScaler:
    """
    Thin wrapper for common feature scaling transformers.
    """

    def __init__(self, scaler: Literal["standard", "minmax"] = "standard", **kwargs: Any):
        """
        Initialize scaler.

        Args:
            scaler: Scaling strategy name.
            **kwargs: Keyword args passed to scaler constructor.

        Raises:
            ValueError: If scaler name is unsupported.
        """
        if scaler == "standard":
            self._scaler = StandardScaler(**kwargs)
        elif scaler == "minmax":
            self._scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler: {scaler}")

    def fit(self, X: ArrayLike) -> "FeatureScaler":
        """
        Fit scaler statistics.

        Args:
            X: Feature matrix.

        Returns:
            The same FeatureScaler instance.
        """
        self._scaler.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Transform features with fitted scaler.

        Args:
            X: Feature matrix.

        Returns:
            Scaled feature matrix.
        """
        return self._scaler.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit scaler then transform in one step.

        Args:
            X: Feature matrix.

        Returns:
            Scaled feature matrix.
        """
        return self._scaler.fit_transform(X)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Reverse scaling transformation.

        Args:
            X: Scaled feature matrix.

        Returns:
            Features in original scale.
        """
        return self._scaler.inverse_transform(X)
