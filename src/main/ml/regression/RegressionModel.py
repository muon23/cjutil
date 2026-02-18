from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.MLModel import MlModel

ArrayLike = np.ndarray


class RegressionModel(MlModel, ABC):
    """
    Base interface for supervised regression models.
    """

    @abstractmethod
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute default regressor score.

        Args:
            X: Feature matrix.
            y: Ground-truth targets.

        Returns:
            Regression score (R^2 for sklearn regressors).
        """
        ...

    def evaluate(self, X: ArrayLike, y: ArrayLike | None, metrics: list[str]) -> dict[str, float]:
        """
        Evaluate regression performance with requested metrics.

        Supported metrics:
        - r2
        - mse
        - rmse
        - mae

        Args:
            X: Feature matrix for evaluation.
            y: Ground-truth target values.
            metrics: Metric names to calculate.

        Returns:
            Mapping of metric name to computed scalar value.

        Raises:
            ValueError: If y is missing or a metric is unsupported.
        """
        if y is None:
            raise ValueError("y must be provided for regression evaluate()")
        y_pred = self.predict(X)
        results: dict[str, float] = {}
        for metric_name in metrics:
            key = metric_name.lower().strip()
            if key == "r2":
                results[key] = float(r2_score(y, y_pred))
            elif key == "mse":
                results[key] = float(mean_squared_error(y, y_pred))
            elif key == "rmse":
                results[key] = float(mean_squared_error(y, y_pred) ** 0.5)
            elif key == "mae":
                results[key] = float(mean_absolute_error(y, y_pred))
            else:
                raise ValueError(f"Unsupported regression metric: {metric_name}")
        return results
