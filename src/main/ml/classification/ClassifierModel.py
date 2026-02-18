from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.MLModel import MlModel

ArrayLike = np.ndarray


class ClassifierModel(MlModel, ABC):
    """
    Base interface for supervised classification models.
    """

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix with one column per class.

        Raises:
            ValueError: If model is not fitted or input shape is invalid.
        """
        ...

    @abstractmethod
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute default classifier score.  (This is equivalent to evaluate(X, y, "accuracy".)

        Args:
            X: Feature matrix.
            y: Ground-truth labels.

        Returns:
            Accuracy score.
        """
        ...

    def evaluate(self, X: ArrayLike, y: ArrayLike | None, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate classification performance with requested metrics.

        Supported metrics:
        - accuracy
        - precision
        - recall
        - f1

        Args:
            X: Feature matrix for evaluation.
            y: Ground-truth labels.
            metrics: Metric names to calculate.

        Returns:
            Mapping of metric name to computed scalar value.

        Raises:
            ValueError: If y is missing or a metric is unsupported.
        """
        if y is None:
            raise ValueError("y must be provided for classification evaluate()")
        y_pred = self.predict(X)
        results: dict[str, float] = {}
        for metric_name in metrics:
            key = metric_name.lower().strip()
            if key == "accuracy":
                results[key] = float(accuracy_score(y, y_pred))
            elif key == "precision":
                results[key] = float(precision_score(y, y_pred, average="weighted", zero_division=0))
            elif key == "recall":
                results[key] = float(recall_score(y, y_pred, average="weighted", zero_division=0))
            elif key == "f1":
                results[key] = float(f1_score(y, y_pred, average="weighted", zero_division=0))
            else:
                raise ValueError(f"Unsupported classification metric: {metric_name}")
        return results
