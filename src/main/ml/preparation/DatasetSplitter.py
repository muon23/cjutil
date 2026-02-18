from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

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
