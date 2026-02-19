from __future__ import annotations

import numpy as np

from ml.preparation.CategoricalEncoder import CategoricalEncoder
from ml.preparation.DataSet import DataSet
from ml.preparation.DatasetSplitter import DatasetSplitter, SplitResult, Split3Result
from ml.preparation.FeatureScaler import FeatureScaler
from ml.preparation.Imputer import Imputer
from ml.preparation.PCATransformer import PCATransformer

ArrayLike = np.ndarray


def train_test(
        X: ArrayLike,
        y: ArrayLike,
        test_size: float = 0.2,
        random_state: int | None = 42,
        stratify: bool = False,
) -> SplitResult:
    """
    Public convenience function for train/test splitting.

    Args:
        X: Feature matrix.
        y: Target labels/values.
        test_size: Proportion assigned to test split.
        random_state: Optional random seed.
        stratify: Whether to preserve class proportions using y labels.

    Returns:
        SplitResult containing train/test arrays.
    """
    return DatasetSplitter.train_test(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_val_test(
        X: ArrayLike,
        y: ArrayLike,
        val_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int | None = 42,
        stratify: bool = False,
) -> Split3Result:
    """
    Public convenience function for train/validation/test splitting.

    Args:
        X: Feature matrix.
        y: Target labels/values.
        val_size: Validation proportion relative to full dataset.
        test_size: Test proportion relative to full dataset.
        random_state: Optional random seed.
        stratify: Whether to preserve class proportions using y labels.

    Returns:
        Split3Result containing train/validation/test arrays.
    """
    return DatasetSplitter.train_val_test(
        X=X,
        y=y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


__all__ = [
    "SplitResult",
    "Split3Result",
    "DataSet",
    "DatasetSplitter",
    "FeatureScaler",
    "Imputer",
    "CategoricalEncoder",
    "PCATransformer",
    "train_test",
    "train_val_test",
]
