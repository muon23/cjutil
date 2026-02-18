from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from ml.regression.RegressionModel import RegressionModel

ArrayLike = np.ndarray


class LinearRegressionModel(RegressionModel):
    """
    Linear regression wrapper.
    """

    def __init__(
            self,
            fit_intercept: bool = True,
            positive: bool = False,
            **kwargs: Any,
    ):
        """
        Initialize linear regression model.

        Args:
            fit_intercept: Whether to fit intercept term.
            positive: Whether to force positive coefficients.
            **kwargs: Additional parameters forwarded to sklearn LinearRegression.
        """
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            positive=positive,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "LinearRegressionModel":
        """
        Train regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            The same LinearRegressionModel instance.

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

    def set_params(self, **params: Any) -> "LinearRegressionModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same LinearRegressionModel instance.
        """
        self.model.set_params(**params)
        return self
