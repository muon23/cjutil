from typing import Any

from ml.regression.GradientBoostingRegressionModel import GradientBoostingRegressionModel
from ml.regression.NNRegressionModel import NNRegressionModel
from ml.regression.RegressionModel import RegressionModel
from ml.regression.LinearRegressionModel import LinearRegressionModel
from ml.regression.RidgeRegressionModel import RidgeRegressionModel
from ml.regression.RandomForestRegressionModel import RandomForestRegressionModel


def of(model_name: str, **kwargs: Any) -> RegressionModel:
    """
    Factory for regression model implementations.

    Args:
        model_name: Regression algorithm name or alias.
        **kwargs: Constructor arguments passed to concrete regression model.

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
        "gradient_boosting": GradientBoostingRegressionModel,
        "gbdt": GradientBoostingRegressionModel,
        "gboost": GradientBoostingRegressionModel,
        "neural_network": NNRegressionModel,
        "nn": NNRegressionModel,
        "mlp": NNRegressionModel,
    }
    model_cls = mapping.get(model_name, None)
    if not model_cls:
        raise RuntimeError(f"Regression model {model_name} not supported")
    return model_cls(**kwargs)


__all__ = [
    "RegressionModel",
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "RandomForestRegressionModel",
    "GradientBoostingRegressionModel",
    "NNRegressionModel",
    "of",
]
