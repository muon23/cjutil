from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.classification.ClassifierModel import ClassifierModel

ArrayLike = np.ndarray


class RandomForestClassifierModel(ClassifierModel):
    """
    Random forest classifier wrapper.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int | None = None,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize random forest classifier.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth.
            random_state: Random seed.
            **kwargs: Additional parameters forwarded to sklearn RandomForestClassifier.

        Raises:
            ValueError: If n_estimators is not positive.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "RandomForestClassifierModel":
        """
        Train classifier.

        Args:
            X: Feature matrix.
            y: Class labels.

        Returns:
            The same RandomForestClassifierModel instance.

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

    def set_params(self, **params: Any) -> "RandomForestClassifierModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same RandomForestClassifierModel instance.
        """
        self.model.set_params(**params)
        return self
