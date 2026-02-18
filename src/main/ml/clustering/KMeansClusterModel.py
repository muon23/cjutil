from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from ml.clustering.ClusterModel import ClusterModel

ArrayLike = np.ndarray


class KMeansClusterModel(ClusterModel):
    """
    K-means clustering wrapper.
    """

    def __init__(
            self,
            k: int,
            init: str = "k-means++",
            n_init: int | str = "auto",
            max_iter: int = 300,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize k-means clusterer.

        Args:
            k: Number of clusters.
            init: Initialization strategy.
            n_init: Number of initializations.
            max_iter: Maximum number of iterations for one run.
            random_state: Random seed.
            **kwargs: Additional parameters forwarded to sklearn KMeans.

        Raises:
            ValueError: If k is not positive.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.model = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "KMeansClusterModel":
        """
        Fit cluster model.

        Args:
            X: Feature matrix.
            y: Ignored for clustering models.

        Returns:
            The same KMeansClusterModel instance.
        """
        self.model.fit(X)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict nearest cluster for samples.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.
        """
        return self.model.predict(X)

    def fit_predict(self, X: ArrayLike) -> ArrayLike:
        """
        Fit model and return cluster assignments.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.
        """
        return self.model.fit_predict(X)

    def get_params(self) -> dict[str, Any]:
        """
        Return model hyperparameters.

        Returns:
            Hyperparameter dictionary.
        """
        return self.model.get_params(deep=True)

    def set_params(self, **params: Any) -> "KMeansClusterModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same KMeansClusterModel instance.
        """
        self.model.set_params(**params)
        return self
