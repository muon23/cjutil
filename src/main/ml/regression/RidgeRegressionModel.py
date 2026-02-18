from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from ml.regression.RegressionModel import RegressionModel

ArrayLike = np.ndarray


class RidgeRegressionModel(RegressionModel):
    """
    Ridge regression wrapper.
    """

    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize ridge regression model.

        Args:
            alpha: L2 regularization strength.
            fit_intercept: Whether to fit intercept term.
            random_state: Random seed for stochastic solvers.
            **kwargs: Additional parameters forwarded to sklearn Ridge.

        Raises:
            ValueError: If alpha is negative.
        """
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "RidgeRegressionModel":
        """
        Train regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            The same RidgeRegressionModel instance.

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

    def set_params(self, **params: Any) -> "RidgeRegressionModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same RidgeRegressionModel instance.
        """
        self.model.set_params(**params)
        return self
