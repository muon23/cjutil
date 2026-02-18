from ml.classification import (
    ClassifierModel,
    LogisticRegressionClassifier,
    RandomForestClassifierModel,
    of as _classification_of,
)
from ml.clustering import (
    ClusterModel,
    KMeansClusterModel,
    DBSCANClusterModel,
    of as _clustering_of,
)
from ml.data_prep import (
    SplitResult,
    Split3Result,
    DatasetSplitter,
    FeatureScaler,
    Imputer,
    CategoricalEncoder,
    PCATransformer,
)
from ml.regression import (
    RegressionModel,
    LinearRegressionModel,
    RidgeRegressionModel,
    RandomForestRegressionModel,
    of as _regression_of,
)

# Alias top-level factories to task-specific module factories.
classifier_of = _classification_of
regressor_of = _regression_of
cluster_of = _clustering_of


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
    "PCATransformer",
    "classifier_of",
    "regressor_of",
    "cluster_of",
]
