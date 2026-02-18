from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

ArrayLike = np.ndarray


class PCATransformer:
    """
    Thin wrapper around sklearn PCA for dimensionality reduction.
    """

    def __init__(
            self,
            n_components: int | float | None = None,
            whiten: bool = False,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize PCA transformer.

        Args:
            n_components: Target dimensions (int), retained variance ratio (float in (0, 1]),
                or None to keep all components.
            whiten: Whether to whiten transformed components.
            random_state: Random seed used by randomized SVD solver.
            **kwargs: Additional keyword args passed to sklearn PCA.

        Raises:
            ValueError: If n_components value is invalid.
        """
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("n_components must be positive when provided as int")
        if isinstance(n_components, float) and not (0.0 < n_components <= 1.0):
            raise ValueError("n_components float must be in (0, 1]")
        self._pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike) -> "PCATransformer":
        """
        Fit PCA components on feature matrix.

        Args:
            X: Feature matrix.

        Returns:
            The same PCATransformer instance.
        """
        self._pca.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Project features into PCA space.

        Args:
            X: Feature matrix.

        Returns:
            PCA-transformed feature matrix.
        """
        return self._pca.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit PCA and transform features in one step.

        Args:
            X: Feature matrix.

        Returns:
            PCA-transformed feature matrix.
        """
        return self._pca.fit_transform(X)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Map PCA-space features back to original feature space.

        Args:
            X: PCA-space feature matrix.

        Returns:
            Approximate reconstruction in original feature space.
        """
        return self._pca.inverse_transform(X)

    def explained_variance_ratio(self) -> ArrayLike:
        """
        Return explained variance ratio per retained principal component.

        Returns:
            1D numpy array of explained variance ratios.
        """
        return self._pca.explained_variance_ratio_
