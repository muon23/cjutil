from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import r2_score

from ml.regression.RegressionModel import RegressionModel

ArrayLike = np.ndarray

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as _torch_import_error:  # pragma: no cover - exercised via runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = _torch_import_error
else:
    _TORCH_IMPORT_ERROR = None


class NNRegressionModel(RegressionModel):
    """
    Feed-forward neural network regressor powered by PyTorch.
    """

    def __init__(
            self,
            hidden_dims: tuple[int, ...] = (64, 32),
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            dropout: float = 0.0,
            weight_decay: float = 0.0,
            random_state: int | None = None,
            device: str | None = None,
    ):
        """
        Initialize neural-network regressor.

        Args:
            hidden_dims: Hidden layer widths.
            learning_rate: Optimizer learning rate.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            dropout: Dropout probability for hidden layers.
            weight_decay: L2 regularization strength.
            random_state: Optional random seed.
            device: Optional device override ("cpu", "cuda", etc.).

        Raises:
            RuntimeError: If PyTorch is not installed.
            ValueError: If hyperparameters are invalid.
        """
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for NNRegressionModel") from _TORCH_IMPORT_ERROR
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("all hidden_dims values must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]

        self._model: nn.Module | None = None  # type: ignore[valid-type]

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "NNRegressionModel":
        """
        Train regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            The same NNRegressionModel instance.

        Raises:
            ValueError: If y is missing or if input shapes are invalid.
        """
        if y is None:
            raise ValueError("y must be provided for regression fit()")
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2 or y_arr.shape[1] != 1:
            raise ValueError("y must be shape (n_samples,) or (n_samples, 1)")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self._model = self._build_network(input_dim=X_arr.shape[1], output_dim=1)
        self._seed_if_needed()
        self._model.train()

        X_tensor = torch.from_numpy(X_arr).to(self.device)  # type: ignore[union-attr]
        y_tensor = torch.from_numpy(y_arr).to(self.device)  # type: ignore[union-attr]
        dataset = TensorDataset(X_tensor, y_tensor)  # type: ignore[misc]
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore[misc]

        criterion = nn.MSELoss()  # type: ignore[operator]
        optimizer = torch.optim.Adam(  # type: ignore[union-attr]
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted target values.

        Raises:
            ValueError: If model is not fitted.
        """
        model = self._require_model()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        model.eval()
        with torch.no_grad():  # type: ignore[union-attr]
            out = model(torch.from_numpy(X_arr).to(self.device))  # type: ignore[union-attr]
        return out.detach().cpu().numpy().reshape(-1)  # type: ignore[union-attr]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute default regressor score.

        Args:
            X: Feature matrix.
            y: Ground-truth targets.

        Returns:
            Regression score (R^2).
        """
        y_true = np.asarray(y, dtype=np.float32).reshape(-1)
        y_pred = self.predict(X).reshape(-1)
        return float(r2_score(y_true, y_pred))

    def get_params(self) -> dict[str, Any]:
        """
        Return model hyperparameters.

        Returns:
            Hyperparameter dictionary.
        """
        return {
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "NNRegressionModel":
        """
        Update model hyperparameters.

        Args:
            **params: Hyperparameter values.

        Returns:
            The same NNRegressionModel instance.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unsupported parameter: {key}")
            setattr(self, key, value)
        self._model = None
        return self

    def _build_network(self, input_dim: int, output_dim: int) -> nn.Module:  # type: ignore[valid-type]
        layers: list[nn.Module] = []  # type: ignore[valid-type]
        prev = input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev, dim))  # type: ignore[operator]
            layers.append(nn.ReLU())  # type: ignore[operator]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))  # type: ignore[operator]
            prev = dim
        layers.append(nn.Linear(prev, output_dim))  # type: ignore[operator]
        model = nn.Sequential(*layers)  # type: ignore[operator]
        return model.to(self.device)  # type: ignore[union-attr]

    def _seed_if_needed(self) -> None:
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)  # type: ignore[union-attr]
        if torch.cuda.is_available():  # type: ignore[union-attr]
            torch.cuda.manual_seed_all(self.random_state)  # type: ignore[union-attr]

    def _require_model(self) -> nn.Module:  # type: ignore[valid-type]
        if self._model is None:
            raise ValueError("Model must be fitted before inference")
        return self._model
