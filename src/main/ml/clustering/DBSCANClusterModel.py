from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from ml.clustering.ClusterModel import ClusterModel

ArrayLike = np.ndarray


class DBSCANClusterModel(ClusterModel):
    """
    DBSCAN clustering wrapper.
    """

    def __init__(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            metric: str = "euclidean",
            **kwargs: Any,
    ):
        """
        Initialize DBSCAN clusterer.

        Args:
            eps: Maximum neighborhood radius.
            min_samples: Minimum points required to form a dense region.
            metric: Distance metric.
            **kwargs: Additional parameters forwarded to sklearn DBSCAN.

        Raises:
            ValueError: If eps or min_samples is invalid.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "DBSCANClusterModel":
        """
        Fit cluster model.

        Args:
            X: Feature matrix.
            y: Ignored for clustering models.

        Returns:
            The same DBSCANClusterModel instance.
        """
        self.model.fit(X)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict clusters for samples.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.

        Raises:
            NotImplementedError: DBSCAN has no native predict() in sklearn.
        """
        raise NotImplementedError("DBSCAN does not support predict(). Use fit_predict() on full dataset.")

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

    def set_params(self, **params: Any) -> "DBSCANClusterModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same DBSCANClusterModel instance.
        """
        self.model.set_params(**params)
        return self
