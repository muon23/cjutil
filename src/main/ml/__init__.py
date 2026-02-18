from typing import Any

from ml.classification import ClassifierModel, LogisticRegressionClassifier, RandomForestClassifierModel
from ml.clustering import ClusterModel, KMeansClusterModel, DBSCANClusterModel
from ml.data_prep import (
    SplitResult,
    Split3Result,
    DatasetSplitter,
    FeatureScaler,
    Imputer,
    CategoricalEncoder,
)
from ml.regression import RegressionModel, LinearRegressionModel, RidgeRegressionModel, RandomForestRegressionModel


def classifier_of(model_name: str, **kwargs: Any) -> ClassifierModel:
    """
    Factory for classification model implementations.

    Args:
        model_name: Classification algorithm name or alias.
        **kwargs: Constructor arguments passed to the concrete sklearn-backed wrapper.

    Returns:
        Instantiated classification model wrapper.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    model_name = model_name.lower().strip()
    mapping = {
        "logistic_regression": LogisticRegressionClassifier,
        "logreg": LogisticRegressionClassifier,
        "lr": LogisticRegressionClassifier,
        "random_forest": RandomForestClassifierModel,
        "rf": RandomForestClassifierModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Classifier model {model_name} not supported")
    return model_cls(**kwargs)


def regressor_of(model_name: str, **kwargs: Any) -> RegressionModel:
    """
    Factory for regression model implementations.

    Args:
        model_name: Regression algorithm name or alias.
        **kwargs: Constructor arguments passed to the concrete sklearn-backed wrapper.

    Returns:
        Instantiated regression model wrapper.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    model_name = model_name.lower().strip()
    mapping = {
        "linear_regression": LinearRegressionModel,
        "linear": LinearRegressionModel,
        "ridge": RidgeRegressionModel,
        "random_forest": RandomForestRegressionModel,
        "rf": RandomForestRegressionModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Regression model {model_name} not supported")
    return model_cls(**kwargs)


def cluster_of(model_name: str, **kwargs: Any) -> ClusterModel:
    """
    Factory for clustering model implementations.

    Args:
        model_name: Clustering algorithm name or alias.
        **kwargs: Constructor arguments passed to the concrete sklearn-backed wrapper.

    Returns:
        Instantiated clustering model wrapper.

    Raises:
        RuntimeError: If model_name is unsupported.
    """
    model_name = model_name.lower().strip()
    mapping = {
        "kmeans": KMeansClusterModel,
        "dbscan": DBSCANClusterModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Clustering model {model_name} not supported")
    # Backward-compatible alias for callers that used sklearn naming.
    if model_name == "kmeans" and "k" not in kwargs and "n_clusters" in kwargs:
        kwargs["k"] = kwargs.pop("n_clusters")
    return model_cls(**kwargs)


__all__ = [
    "ClassifierModel",
    "RegressionModel",
    "ClusterModel",
    "LogisticRegressionClassifier",
    "RandomForestClassifierModel",
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "RandomForestRegressionModel",
    "KMeansClusterModel",
    "DBSCANClusterModel",
    "DatasetSplitter",
    "SplitResult",
    "Split3Result",
    "FeatureScaler",
    "Imputer",
    "CategoricalEncoder",
    "classifier_of",
    "regressor_of",
    "cluster_of",
]
