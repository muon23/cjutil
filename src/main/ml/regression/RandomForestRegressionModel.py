from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml.regression.RegressionModel import RegressionModel

ArrayLike = np.ndarray


class RandomForestRegressionModel(RegressionModel):
    """
    Random forest regression wrapper.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int | None = None,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize random forest regressor.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth.
            random_state: Random seed.
            **kwargs: Additional parameters forwarded to sklearn RandomForestRegressor.

        Raises:
            ValueError: If n_estimators is not positive.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "RandomForestRegressionModel":
        """
        Train regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            The same RandomForestRegressionModel instance.

        Raises:
            ValueError: If y is not provided.
        """
        if y is None:
            raise ValueError("y must be provided for regression fit()")
        self.model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted target values.
        """
        return self.model.predict(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute default regressor score.

        Args:
            X: Feature matrix.
            y: Ground-truth targets.

        Returns:
            Regression score.
        """
        return float(self.model.score(X, y))

    def get_params(self) -> dict[str, Any]:
        """
        Return model hyperparameters.

        Returns:
            Hyperparameter dictionary.
        """
        return self.model.get_params(deep=True)

    def set_params(self, **params: Any) -> "RandomForestRegressionModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same RandomForestRegressionModel instance.
        """
        self.model.set_params(**params)
        return self
