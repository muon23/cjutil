from typing import Any

from ml.classification.ClassifierModel import ClassifierModel
from ml.classification.GradientBoostingClassifierModel import GradientBoostingClassifierModel
from ml.classification.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml.classification.NNClassificationModel import NNClassificationModel
from ml.classification.RandomForestClassifierModel import RandomForestClassifierModel


def of(model_name: str, **kwargs: Any) -> ClassifierModel:
    """
    Factory for classification model implementations.

    Args:
        model_name: Classification algorithm name or alias.
        **kwargs: Constructor arguments passed to concrete classifier model.

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
        "gradient_boosting": GradientBoostingClassifierModel,
        "gbdt": GradientBoostingClassifierModel,
        "gboost": GradientBoostingClassifierModel,
        "neural_network": NNClassificationModel,
        "nn": NNClassificationModel,
        "mlp": NNClassificationModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Classifier model {model_name} not supported")
    return model_cls(**kwargs)


__all__ = [
    "ClassifierModel",
    "LogisticRegressionClassifier",
    "RandomForestClassifierModel",
    "GradientBoostingClassifierModel",
    "NNClassificationModel",
    "of",
]
