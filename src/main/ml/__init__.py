from ml.classification import (
    ClassifierModel,
    GradientBoostingClassifierModel,
    LogisticRegressionClassifier,
    NNClassificationModel,
    RandomForestClassifierModel,
    of as _classification_of,
)
from ml.clustering import (
    ClusterModel,
    KMeansClusterModel,
    DBSCANClusterModel,
    of as _clustering_of,
)
from ml.preparation import (
    DataSet,
    SplitResult,
    Split3Result,
    DatasetSplitter,
    FeatureScaler,
    Imputer,
    CategoricalEncoder,
    PCATransformer,
    train_test,
    train_val_test,
)
from ml.regression import (
    GradientBoostingRegressionModel,
    RegressionModel,
    LinearRegressionModel,
    NNRegressionModel,
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
    "GradientBoostingClassifierModel",
    "NNClassificationModel",
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "RandomForestRegressionModel",
    "GradientBoostingRegressionModel",
    "NNRegressionModel",
    "KMeansClusterModel",
    "DBSCANClusterModel",
    "DataSet",
    "DatasetSplitter",
    "SplitResult",
    "Split3Result",
    "FeatureScaler",
    "Imputer",
    "CategoricalEncoder",
    "PCATransformer",
    "train_test",
    "train_val_test",
    "classifier_of",
    "regressor_of",
    "cluster_of",
]
