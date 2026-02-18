from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

ArrayLike = np.ndarray


class MlModel(ABC):
    """
    Base interface for traditional machine learning models.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "MlModel":
        """
        Train model parameters on input data.

        Args:
            X: Feature matrix.
            y: Optional target labels/values.

        Returns:
            The same model instance.

        Raises:
            ValueError: If input shape/content is invalid for the concrete model.
        """
        ...

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Run model inference.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels/values.

        Raises:
            ValueError: If model is not fitted or input shape is invalid.
        """
        ...

    @abstractmethod
    def evaluate(
            self,
            X: ArrayLike,
            y: ArrayLike | None,
            metrics: list[str],
    ) -> dict[str, float]:
        """
        Evaluate model performance using requested metrics.

        Args:
            X: Feature matrix for evaluation.
            y: Ground-truth labels/targets when required by metric.
            metrics: Metric names to calculate.

        Returns:
            Mapping of metric name to computed scalar value.

        Raises:
            ValueError: If required labels are missing or metric name is unsupported.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """
        Return model hyperparameters.

        Returns:
            Hyperparameter dictionary.
        """
        ...

    @abstractmethod
    def set_params(self, **params: Any) -> "MlModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same model instance.
        """
        ...

    def save(self, path: str | Path) -> None:
        """
        Serialize this model instance to disk.

        Args:
            path: Destination file path.

        Returns:
            None.

        Raises:
            OSError: If file cannot be written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "MlModel":
        """
        Load a serialized model instance from disk.

        Args:
            path: Source file path.

        Returns:
            Deserialized model instance.

        Raises:
            FileNotFoundError: If path does not exist.
            pickle.PickleError: If payload is not a valid serialized model.
        """
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, MlModel):
            raise TypeError(f"Loaded object is not MlModel (was {type(model)})")
        return model
