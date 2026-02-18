from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ml.MLModel import MlModel

ArrayLike = np.ndarray


class ClusterModel(MlModel, ABC):
    """
    Base interface for unsupervised clustering models.
    """

    @abstractmethod
    def fit_predict(self, X: ArrayLike) -> ArrayLike:
        """
        Fit model and return cluster assignments.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels for each sample.
        """
        ...

    def evaluate(self, X: ArrayLike, y: ArrayLike | None, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate clustering performance with requested metrics.

        Supported metrics:
        - silhouette
        - davies_bouldin
        - calinski_harabasz
        - ari (requires y labels)
        - nmi (requires y labels)

        Args:
            X: Feature matrix for evaluation.
            y: Optional reference labels for external clustering metrics.
            metrics: Metric names to calculate.

        Returns:
            Mapping of metric name to computed scalar value.

        Raises:
            ValueError: If required labels are missing or a metric is unsupported.
        """
        try:
            labels = self.predict(X)
        except NotImplementedError:
            labels = self.fit_predict(X)

        results: dict[str, float] = {}
        for metric_name in metrics:
            key = metric_name.lower().strip()
            if key == "silhouette":
                results[key] = float(silhouette_score(X, labels))
            elif key == "davies_bouldin":
                results[key] = float(davies_bouldin_score(X, labels))
            elif key == "calinski_harabasz":
                results[key] = float(calinski_harabasz_score(X, labels))
            elif key == "ari":
                if y is None:
                    raise ValueError("y must be provided for ari metric")
                results[key] = float(adjusted_rand_score(y, labels))
            elif key == "nmi":
                if y is None:
                    raise ValueError("y must be provided for nmi metric")
                results[key] = float(normalized_mutual_info_score(y, labels))
            else:
                raise ValueError(f"Unsupported clustering metric: {metric_name}")
        return results
