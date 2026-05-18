from __future__ import annotations

from typing import Any, Literal

import numpy as np

ArrayLike = np.ndarray

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as _torch_import_error:  # pragma: no cover - exercised via runtime guard
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = _torch_import_error
else:
    _TORCH_IMPORT_ERROR = None


def _build_tower(dims: tuple[int, ...], dropout: float) -> nn.Module:
    if len(dims) < 2:
        raise ValueError("tower dims must include input and output sizes, e.g. (input_dim, 64, 32)")
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != dims[-1]:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _TwoTowerNetwork(nn.Module):
    def __init__(
            self,
            user_dims: tuple[int, ...],
            item_dims: tuple[int, ...],
            dropout: float,
            similarity: Literal["dot", "cosine"],
    ):
        super().__init__()
        if user_dims[-1] != item_dims[-1]:
            raise ValueError("user_dims and item_dims must end with the same embedding size")
        self.similarity = similarity
        self.user_tower = _build_tower(user_dims, dropout)
        self.item_tower = _build_tower(item_dims, dropout)
        self.embedding_dim = user_dims[-1]

    def encode_user(self, user_x: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_x)

    def encode_item(self, item_x: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_x)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_x)
        item_emb = self.encode_item(item_x)
        if self.similarity == "cosine":
            user_emb = F.normalize(user_emb, dim=1)
            item_emb = F.normalize(item_emb, dim=1)
        return (user_emb * item_emb).sum(dim=1, keepdim=True)

    def train(self, mode: bool = True) -> "_TwoTowerNetwork":
        return super().train(mode)


class TwoTower:
    """
    Two-tower retrieval model with separate user and item encoders.

    Each tower is an MLP specified by a dimension tuple, e.g. ``user_dims=(32, 64, 16)``
    defines layers 32→64→16. Training follows the PyTorch mini-batch loop used by other
    ``ml`` neural models, with ``fit()`` / ``train()`` for the full epoch loop and
    ``module.train()`` / ``module.eval()`` for standard PyTorch mode switching.
    """

    def __init__(
            self,
            user_dims: tuple[int, ...],
            item_dims: tuple[int, ...],
            learning_rate: float = 1e-3,
            epochs: int = 10,
            batch_size: int = 256,
            dropout: float = 0.0,
            weight_decay: float = 0.0,
            loss: Literal["bce", "mse"] = "bce",
            similarity: Literal["dot", "cosine"] = "dot",
            random_state: int | None = None,
            device: str | None = None,
    ):
        """
        Args:
            user_dims: User tower layer sizes ``(input, hidden..., embedding)``.
            item_dims: Item tower layer sizes ``(input, hidden..., embedding)``.
            learning_rate: Adam learning rate.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            dropout: Dropout on hidden layers (not applied on the final embedding layer).
            weight_decay: Adam L2 penalty.
            loss: ``bce`` for binary labels (uses BCEWithLogitsLoss), ``mse`` for regression.
            similarity: ``dot`` or ``cosine`` similarity between tower embeddings.
            random_state: Optional RNG seed.
            device: ``cpu``, ``cuda``, etc. Defaults to CUDA when available.

        Raises:
            RuntimeError: If PyTorch is not installed.
            ValueError: If hyperparameters or tower shapes are invalid.
        """
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for TwoTower") from _TORCH_IMPORT_ERROR
        if len(user_dims) < 2 or len(item_dims) < 2:
            raise ValueError("user_dims and item_dims must include input and output sizes")
        if any(dim <= 0 for dim in user_dims):
            raise ValueError("all user_dims values must be positive")
        if any(dim <= 0 for dim in item_dims):
            raise ValueError("all item_dims values must be positive")
        if user_dims[-1] != item_dims[-1]:
            raise ValueError("user_dims and item_dims must end with the same embedding size")
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
        if loss not in {"bce", "mse"}:
            raise ValueError("loss must be 'bce' or 'mse'")
        if similarity not in {"dot", "cosine"}:
            raise ValueError("similarity must be 'dot' or 'cosine'")

        self.user_dims = user_dims
        self.item_dims = item_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.loss = loss
        self.similarity = similarity
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]

        self._module: _TwoTowerNetwork | None = None

    @property
    def module(self) -> _TwoTowerNetwork:
        """Underlying PyTorch module (available after ``fit`` / ``train``)."""
        return self._require_module()

    def fit(
            self,
            user_X: ArrayLike,
            item_X: ArrayLike,
            y: ArrayLike,
    ) -> "TwoTower":
        """
        Train the two towers on paired user/item examples.

        Args:
            user_X: User feature matrix ``(n_samples, user_input_dim)``.
            item_X: Item feature matrix ``(n_samples, item_input_dim)``.
            y: Target values; binary labels for ``loss='bce'``, floats for ``loss='mse'``.

        Returns:
            This TwoTower instance.
        """
        user_arr, item_arr, y_arr = self._validate_training_data(user_X, item_X, y)
        self._module = _TwoTowerNetwork(
            user_dims=self.user_dims,
            item_dims=self.item_dims,
            dropout=self.dropout,
            similarity=self.similarity,
        ).to(self.device)
        self._seed_if_needed()
        self._run_training_loop(user_arr, item_arr, y_arr)
        return self

    def train(
            self,
            user_X: ArrayLike,
            item_X: ArrayLike,
            y: ArrayLike,
    ) -> "TwoTower":
        """Alias for :meth:`fit` (full training loop, not ``nn.Module.train``)."""
        return self.fit(user_X, item_X, y)

    def predict(self, user_X: ArrayLike, item_X: ArrayLike) -> ArrayLike:
        """
        Predict similarity scores for user/item pairs.

        Returns:
            Scores shaped ``(n_samples,)``. Apply a sigmoid yourself when ``loss='bce'``.
        """
        module = self._require_module()
        user_arr, item_arr = self._validate_pair_features(user_X, item_X)
        module.eval()
        with torch.no_grad():  # type: ignore[union-attr]
            user_tensor = torch.from_numpy(user_arr).to(self.device)  # type: ignore[union-attr]
            item_tensor = torch.from_numpy(item_arr).to(self.device)  # type: ignore[union-attr]
            scores = module(user_tensor, item_tensor)
        return scores.detach().cpu().numpy().reshape(-1)  # type: ignore[union-attr]

    def encode_user(self, user_X: ArrayLike) -> ArrayLike:
        """Encode users into embedding vectors."""
        module = self._require_module()
        user_arr = np.asarray(user_X, dtype=np.float32)
        if user_arr.ndim != 2 or user_arr.shape[1] != self.user_dims[0]:
            raise ValueError(f"user_X must have shape (n_samples, {self.user_dims[0]})")
        module.eval()
        with torch.no_grad():  # type: ignore[union-attr]
            tensor = torch.from_numpy(user_arr).to(self.device)  # type: ignore[union-attr]
            emb = module.encode_user(tensor)
        return emb.detach().cpu().numpy()  # type: ignore[union-attr]

    def encode_item(self, item_X: ArrayLike) -> ArrayLike:
        """Encode items into embedding vectors."""
        module = self._require_module()
        item_arr = np.asarray(item_X, dtype=np.float32)
        if item_arr.ndim != 2 or item_arr.shape[1] != self.item_dims[0]:
            raise ValueError(f"item_X must have shape (n_samples, {self.item_dims[0]})")
        module.eval()
        with torch.no_grad():  # type: ignore[union-attr]
            tensor = torch.from_numpy(item_arr).to(self.device)  # type: ignore[union-attr]
            emb = module.encode_item(tensor)
        return emb.detach().cpu().numpy()  # type: ignore[union-attr]

    def evaluate(
            self,
            user_X: ArrayLike,
            item_X: ArrayLike,
            y: ArrayLike,
            metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the model on labeled pairs.

        Supported metrics: ``loss``, ``mse``, ``accuracy`` (binary labels, threshold 0.5).
        """
        metrics = metrics or ["loss"]
        scores = self.predict(user_X, item_X)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        result: dict[str, float] = {}

        if "loss" in metrics or "mse" in metrics:
            if self.loss == "bce":
                logits = torch.from_numpy(scores.astype(np.float32))  # type: ignore[union-attr]
                targets = torch.from_numpy(y_arr)  # type: ignore[union-attr]
                loss_value = float(F.binary_cross_entropy_with_logits(logits, targets).item())  # type: ignore[union-attr]
            else:
                loss_value = float(np.mean((scores.reshape(-1) - y_arr) ** 2))
            if "loss" in metrics:
                result["loss"] = loss_value
            if "mse" in metrics:
                result["mse"] = loss_value if self.loss == "mse" else float(np.mean((scores.reshape(-1) - y_arr) ** 2))

        if "accuracy" in metrics:
            probs = 1.0 / (1.0 + np.exp(-scores.reshape(-1)))
            preds = (probs >= 0.5).astype(np.float32)
            result["accuracy"] = float(np.mean(preds == y_arr))

        unknown = set(metrics) - set(result.keys())
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        return result

    def get_params(self) -> dict[str, Any]:
        return {
            "user_dims": self.user_dims,
            "item_dims": self.item_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "similarity": self.similarity,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "TwoTower":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unsupported parameter: {key}")
            setattr(self, key, value)
        self._module = None
        return self

    def _run_training_loop(self, user_arr: np.ndarray, item_arr: np.ndarray, y_arr: np.ndarray) -> None:
        module = self._require_module()
        module.train()

        user_tensor = torch.from_numpy(user_arr).to(self.device)  # type: ignore[union-attr]
        item_tensor = torch.from_numpy(item_arr).to(self.device)  # type: ignore[union-attr]
        y_tensor = torch.from_numpy(y_arr).to(self.device)  # type: ignore[union-attr]
        dataset = TensorDataset(user_tensor, item_tensor, y_tensor)  # type: ignore[misc]
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore[misc]

        if self.loss == "bce":
            criterion = nn.BCEWithLogitsLoss()  # type: ignore[operator]
        else:
            criterion = nn.MSELoss()  # type: ignore[operator]

        optimizer = torch.optim.Adam(  # type: ignore[union-attr]
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.epochs):
            for batch_user, batch_item, batch_y in loader:
                optimizer.zero_grad()
                logits = module(batch_user, batch_item).reshape_as(batch_y)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def _validate_training_data(
            self,
            user_X: ArrayLike,
            item_X: ArrayLike,
            y: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_arr, item_arr = self._validate_pair_features(user_X, item_X)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2 or y_arr.shape[1] != 1:
            raise ValueError("y must be shape (n_samples,) or (n_samples, 1)")
        if user_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("user_X, item_X, and y must have the same number of samples")
        return user_arr, item_arr, y_arr

    def _validate_pair_features(self, user_X: ArrayLike, item_X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        user_arr = np.asarray(user_X, dtype=np.float32)
        item_arr = np.asarray(item_X, dtype=np.float32)
        if user_arr.ndim != 2 or user_arr.shape[1] != self.user_dims[0]:
            raise ValueError(f"user_X must have shape (n_samples, {self.user_dims[0]})")
        if item_arr.ndim != 2 or item_arr.shape[1] != self.item_dims[0]:
            raise ValueError(f"item_X must have shape (n_samples, {self.item_dims[0]})")
        if user_arr.shape[0] != item_arr.shape[0]:
            raise ValueError("user_X and item_X must contain the same number of samples")
        return user_arr, item_arr

    def _seed_if_needed(self) -> None:
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)  # type: ignore[union-attr]
        if torch.cuda.is_available():  # type: ignore[union-attr]
            torch.cuda.manual_seed_all(self.random_state)  # type: ignore[union-attr]

    def _require_module(self) -> _TwoTowerNetwork:
        if self._module is None:
            raise ValueError("Model must be fitted before inference")
        return self._module
