from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ArrayLike = np.ndarray


@dataclass(frozen=True)
class SplitResult:
    """
    Container for train/test split output.
    """

    X_train: ArrayLike
    X_test: ArrayLike
    y_train: ArrayLike
    y_test: ArrayLike


@dataclass(frozen=True)
class Split3Result:
    """
    Container for train/validation/test split output.
    """

    X_train: ArrayLike
    X_val: ArrayLike
    X_test: ArrayLike
    y_train: ArrayLike
    y_val: ArrayLike
    y_test: ArrayLike


class DatasetSplitter:
    """
    Utility functions for dataset splitting.
    """

    @staticmethod
    def train_test(
            X: ArrayLike,
            y: ArrayLike,
            test_size: float = 0.2,
            random_state: int | None = 42,
            stratify: bool = False,
    ) -> SplitResult:
        """
        Split arrays into train and test subsets.

        Args:
            X: Feature matrix.
            y: Target labels/values.
            test_size: Proportion assigned to test split.
            random_state: Optional random seed.
            stratify: Whether to preserve class proportions using y labels.

        Returns:
            SplitResult containing train/test arrays.
        """
        stratify_labels = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )
        return SplitResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    @staticmethod
    def train_val_test(
            X: ArrayLike,
            y: ArrayLike,
            val_size: float = 0.1,
            test_size: float = 0.2,
            random_state: int | None = 42,
            stratify: bool = False,
    ) -> Split3Result:
        """
        Split arrays into train/validation/test subsets.

        Args:
            X: Feature matrix.
            y: Target labels/values.
            val_size: Validation proportion relative to full dataset.
            test_size: Test proportion relative to full dataset.
            random_state: Optional random seed.
            stratify: Whether to preserve class proportions using y labels.

        Returns:
            Split3Result containing train/validation/test arrays.

        Raises:
            ValueError: If split ratios are invalid.
        """
        if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1:
            raise ValueError("val_size and test_size must be > 0 and sum to < 1")

        stratify_labels = y if stratify else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )

        val_ratio_in_train_val = val_size / (1 - test_size)
        stratify_labels_2 = y_train_val if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio_in_train_val,
            random_state=random_state,
            stratify=stratify_labels_2,
        )
        return Split3Result(
            X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test
        )


class FeatureScaler:
    """
    Thin wrapper for common feature scaling transformers.
    """

    def __init__(self, scaler: Literal["standard", "minmax"] = "standard", **kwargs: Any):
        """
        Initialize scaler.

        Args:
            scaler: Scaling strategy name.
            **kwargs: Keyword args passed to scaler constructor.

        Raises:
            ValueError: If scaler name is unsupported.
        """
        if scaler == "standard":
            self._scaler = StandardScaler(**kwargs)
        elif scaler == "minmax":
            self._scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler: {scaler}")

    def fit(self, X: ArrayLike) -> "FeatureScaler":
        """
        Fit scaler statistics.

        Args:
            X: Feature matrix.

        Returns:
            The same FeatureScaler instance.
        """
        self._scaler.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Transform features with fitted scaler.

        Args:
            X: Feature matrix.

        Returns:
            Scaled feature matrix.
        """
        return self._scaler.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit scaler then transform in one step.

        Args:
            X: Feature matrix.

        Returns:
            Scaled feature matrix.
        """
        return self._scaler.fit_transform(X)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Reverse scaling transformation.

        Args:
            X: Scaled feature matrix.

        Returns:
            Features in original scale.
        """
        return self._scaler.inverse_transform(X)


class Imputer:
    """
    Thin wrapper around sklearn SimpleImputer.
    """

    def __init__(self, strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean", **kwargs: Any):
        """
        Initialize imputer.

        Args:
            strategy: Imputation strategy.
            **kwargs: Keyword args passed to SimpleImputer.
        """
        self._imputer = SimpleImputer(strategy=strategy, **kwargs)

    def fit(self, X: ArrayLike) -> "Imputer":
        """
        Fit imputation statistics.

        Args:
            X: Feature matrix.

        Returns:
            The same Imputer instance.
        """
        self._imputer.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fill missing values with fitted imputation policy.

        Args:
            X: Feature matrix.

        Returns:
            Imputed feature matrix.
        """
        return self._imputer.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit imputer and transform in one step.

        Args:
            X: Feature matrix.

        Returns:
            Imputed feature matrix.
        """
        return self._imputer.fit_transform(X)


class CategoricalEncoder:
    """
    Thin wrapper around sklearn OneHotEncoder.
    """

    def __init__(self, sparse_output: bool = False, handle_unknown: str = "ignore", **kwargs: Any):
        """
        Initialize one-hot encoder.

        Args:
            sparse_output: Whether to return sparse matrix output.
            handle_unknown: Unknown category handling strategy.
            **kwargs: Keyword args passed to OneHotEncoder.
        """
        self._encoder = OneHotEncoder(
            sparse_output=sparse_output,
            handle_unknown=handle_unknown,
            **kwargs,
        )

    def fit(self, X: ArrayLike) -> "CategoricalEncoder":
        """
        Fit category mappings.

        Args:
            X: Categorical feature matrix.

        Returns:
            The same CategoricalEncoder instance.
        """
        self._encoder.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        """
        One-hot encode input matrix.

        Args:
            X: Categorical feature matrix.

        Returns:
            Encoded feature matrix (dense or sparse based on configuration).
        """
        return self._encoder.transform(X)

    def fit_transform(self, X: ArrayLike) -> Any:
        """
        Fit encoder and transform in one step.

        Args:
            X: Categorical feature matrix.

        Returns:
            Encoded feature matrix (dense or sparse based on configuration).
        """
        return self._encoder.fit_transform(X)


class PCATransformer:
    """
    Thin wrapper around sklearn PCA for dimensionality reduction.
    """

    def __init__(
            self,
            n_components: int | float | None = None,
            whiten: bool = False,
            random_state: int | None = None,
            **kwargs: Any,
    ):
        """
        Initialize PCA transformer.

        Args:
            n_components: Target dimensions (int), retained variance ratio (float in (0, 1]),
                or None to keep all components.
            whiten: Whether to whiten transformed components.
            random_state: Random seed used by randomized SVD solver.
            **kwargs: Additional keyword args passed to sklearn PCA.

        Raises:
            ValueError: If n_components value is invalid.
        """
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("n_components must be positive when provided as int")
        if isinstance(n_components, float) and not (0.0 < n_components <= 1.0):
            raise ValueError("n_components float must be in (0, 1]")
        self._pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: ArrayLike) -> "PCATransformer":
        """
        Fit PCA components on feature matrix.

        Args:
            X: Feature matrix.

        Returns:
            The same PCATransformer instance.
        """
        self._pca.fit(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Project features into PCA space.

        Args:
            X: Feature matrix.

        Returns:
            PCA-transformed feature matrix.
        """
        return self._pca.transform(X)

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit PCA and transform features in one step.

        Args:
            X: Feature matrix.

        Returns:
            PCA-transformed feature matrix.
        """
        return self._pca.fit_transform(X)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Map PCA-space features back to original feature space.

        Args:
            X: PCA-space feature matrix.

        Returns:
            Approximate reconstruction in original feature space.
        """
        return self._pca.inverse_transform(X)

    def explained_variance_ratio(self) -> ArrayLike:
        """
        Return explained variance ratio per retained principal component.

        Returns:
            1D numpy array of explained variance ratios.
        """
        return self._pca.explained_variance_ratio_
