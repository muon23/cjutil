import tempfile
import unittest
from pathlib import Path

import numpy as np

import ml
from ml.MLModel import MlModel
from ml.classification import LogisticRegressionClassifier, RandomForestClassifierModel
from ml.clustering import DBSCANClusterModel, KMeansClusterModel
from ml.data_prep import CategoricalEncoder, DatasetSplitter, FeatureScaler, Imputer
from ml.regression import LinearRegressionModel, RandomForestRegressionModel, RidgeRegressionModel


class MlModuleTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
            [1.0, 1.1],
            [1.1, 1.0],
            [0.2, 0.2],
            [0.9, 0.8],
        ])
        self.y_cls = np.array([0, 0, 1, 1, 0, 1])
        self.y_reg = np.array([0.1, 0.0, 2.0, 2.1, 0.4, 1.8])

    def test_classification_models_fit_predict_evaluate(self):
        logreg = LogisticRegressionClassifier(max_iter=200).fit(self.X, self.y_cls)
        preds = logreg.predict(self.X)
        self.assertEqual(len(self.y_cls), len(preds))
        eval_result = logreg.evaluate(self.X, self.y_cls, ["accuracy", "precision", "recall", "f1"])
        self.assertEqual({"accuracy", "precision", "recall", "f1"}, set(eval_result.keys()))

        rf = RandomForestClassifierModel(n_estimators=10, random_state=42).fit(self.X, self.y_cls)
        eval_rf = rf.evaluate(self.X, self.y_cls, ["accuracy"])
        self.assertIn("accuracy", eval_rf)

    def test_classification_validation(self):
        with self.assertRaises(ValueError):
            RandomForestClassifierModel(n_estimators=0)

    def test_regression_models_fit_predict_evaluate(self):
        linear = LinearRegressionModel().fit(self.X, self.y_reg)
        preds = linear.predict(self.X)
        self.assertEqual(len(self.y_reg), len(preds))
        eval_result = linear.evaluate(self.X, self.y_reg, ["r2", "mse", "rmse", "mae"])
        self.assertEqual({"r2", "mse", "rmse", "mae"}, set(eval_result.keys()))

        ridge = RidgeRegressionModel(alpha=0.5).fit(self.X, self.y_reg)
        self.assertIn("r2", ridge.evaluate(self.X, self.y_reg, ["r2"]))

        rf_reg = RandomForestRegressionModel(n_estimators=8, random_state=42).fit(self.X, self.y_reg)
        self.assertIn("mae", rf_reg.evaluate(self.X, self.y_reg, ["mae"]))

    def test_regression_validation(self):
        with self.assertRaises(ValueError):
            RidgeRegressionModel(alpha=-0.1)
        with self.assertRaises(ValueError):
            RandomForestRegressionModel(n_estimators=0)

    def test_clustering_models_fit_predict_evaluate(self):
        kmeans = KMeansClusterModel(k=2, random_state=42).fit(self.X)
        labels = kmeans.predict(self.X)
        self.assertEqual(len(self.X), len(labels))
        eval_result = kmeans.evaluate(self.X, self.y_cls, ["silhouette", "ari"])
        self.assertEqual({"silhouette", "ari"}, set(eval_result.keys()))

        dbscan = DBSCANClusterModel(eps=0.6, min_samples=1).fit(self.X)
        with self.assertRaises(NotImplementedError):
            dbscan.predict(self.X)
        eval_dbscan = dbscan.evaluate(self.X, self.y_cls, ["ari"])
        self.assertIn("ari", eval_dbscan)

    def test_clustering_validation(self):
        with self.assertRaises(ValueError):
            KMeansClusterModel(k=0)
        with self.assertRaises(ValueError):
            DBSCANClusterModel(eps=0)
        with self.assertRaises(ValueError):
            DBSCANClusterModel(min_samples=0)

    def test_factory_builds_expected_models(self):
        self.assertIsInstance(ml.classifier_of("logreg"), LogisticRegressionClassifier)
        self.assertIsInstance(ml.classifier_of("rf", n_estimators=5), RandomForestClassifierModel)
        self.assertIsInstance(ml.regressor_of("linear"), LinearRegressionModel)
        self.assertIsInstance(ml.regressor_of("ridge"), RidgeRegressionModel)
        self.assertIsInstance(ml.regressor_of("rf", n_estimators=5), RandomForestRegressionModel)
        self.assertIsInstance(ml.cluster_of("kmeans", k=2), KMeansClusterModel)

        # Backward compatibility alias for sklearn naming.
        self.assertIsInstance(ml.cluster_of("kmeans", n_clusters=2), KMeansClusterModel)

        with self.assertRaises(RuntimeError):
            ml.classifier_of("unknown")
        with self.assertRaises(RuntimeError):
            ml.regressor_of("unknown")
        with self.assertRaises(RuntimeError):
            ml.cluster_of("unknown")

        # Backward compatibility alias for sklearn naming.
        self.assertIsInstance(ml.cluster_of("kmeans", n_clusters=2), KMeansClusterModel)

    def test_data_prep_utilities(self):
        split = DatasetSplitter.train_test(self.X, self.y_cls, test_size=0.33, random_state=7, stratify=True)
        self.assertEqual(len(split.X_train) + len(split.X_test), len(self.X))
        self.assertEqual(len(split.y_train) + len(split.y_test), len(self.y_cls))

        split3 = DatasetSplitter.train_val_test(
            self.X, self.y_cls, val_size=0.17, test_size=0.33, random_state=7, stratify=True
        )
        self.assertEqual(len(split3.X_train) + len(split3.X_val) + len(split3.X_test), len(self.X))
        self.assertEqual(len(split3.y_train) + len(split3.y_val) + len(split3.y_test), len(self.y_cls))

        with self.assertRaises(ValueError):
            DatasetSplitter.train_val_test(self.X, self.y_cls, val_size=0.6, test_size=0.5)

        scaler = FeatureScaler("standard").fit(split.X_train)
        transformed = scaler.transform(split.X_test)
        self.assertEqual(split.X_test.shape, transformed.shape)

        with_nan = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]])
        imputed = Imputer(strategy="mean").fit_transform(with_nan)
        self.assertFalse(np.isnan(imputed).any())

        categories = np.array([["red"], ["blue"], ["green"], ["red"]])
        encoded = CategoricalEncoder(sparse_output=False).fit_transform(categories)
        self.assertEqual(categories.shape[0], encoded.shape[0])
        self.assertGreater(encoded.shape[1], 0)

    def test_model_save_and_load(self):
        model = LogisticRegressionClassifier(max_iter=200).fit(self.X, self.y_cls)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classifier.pkl"
            model.save(path)
            loaded = MlModel.load(path)
            self.assertIsInstance(loaded, MlModel)
            preds = loaded.predict(self.X)
            self.assertEqual(len(self.X), len(preds))


if __name__ == "__main__":
    unittest.main()
