from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.classification.ClassifierModel import ClassifierModel

ArrayLike = np.ndarray


class LogisticRegressionClassifier(ClassifierModel):
    """
    Logistic regression classifier wrapper.
    """

    def __init__(
            self,
            max_iter: int = 100,
            C: float = 1.0,
            penalty: str = "l2",
            solver: str = "lbfgs",
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize logistic regression classifier.

        Args:
            max_iter: Maximum number of optimization iterations.
            C: Inverse regularization strength.
            penalty: Regularization penalty.
            solver: Optimization solver.
            random_state: Random seed used by selected solver when applicable.
            **kwargs: Additional parameters forwarded to sklearn LogisticRegression.
        """
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            solver=solver,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "LogisticRegressionClassifier":
        """
        Train classifier.

        Args:
            X: Feature matrix.
            y: Class labels.

        Returns:
            The same LogisticRegressionClassifier instance.

        Raises:
            ValueError: If y is not provided.
        """
        if y is None:
            raise ValueError("y must be provided for classification fit()")
        self.model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probability matrix.
        """
        return self.model.predict_proba(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute default classifier score.

        Args:
            X: Feature matrix.
            y: Ground-truth labels.

        Returns:
            Accuracy score.
        """
        return float(self.model.score(X, y))

    def get_params(self) -> dict[str, Any]:
        """
        Return model hyperparameters.

        Returns:
            Hyperparameter dictionary.
        """
        return self.model.get_params(deep=True)

    def set_params(self, **params: Any) -> "LogisticRegressionClassifier":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same LogisticRegressionClassifier instance.
        """
        self.model.set_params(**params)
        return self
