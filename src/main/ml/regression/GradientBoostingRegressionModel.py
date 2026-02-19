from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ml.regression.RegressionModel import RegressionModel

ArrayLike = np.ndarray


class GradientBoostingRegressionModel(RegressionModel):
    """
    Gradient boosting regression wrapper.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 3,
            subsample: float = 1.0,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize gradient boosting regressor.

        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Shrinkage applied to each stage contribution.
            max_depth: Maximum depth of each regression tree learner.
            subsample: Fraction of samples used for fitting each stage.
            random_state: Random seed.
            **kwargs: Additional parameters forwarded to sklearn GradientBoostingRegressor.

        Raises:
            ValueError: If n_estimators, learning_rate, max_depth, or subsample are invalid.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if subsample <= 0 or subsample > 1:
            raise ValueError("subsample must be in (0, 1]")

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "GradientBoostingRegressionModel":
        """
        Train regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            The same GradientBoostingRegressionModel instance.

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

    def set_params(self, **params: Any) -> "GradientBoostingRegressionModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same GradientBoostingRegressionModel instance.
        """
        self.model.set_params(**params)
        return self
