import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preparation import DataSet
from ml.preparation.CategoricalEncoder import CategoricalEncoder
from ml.preparation.FeatureScaler import FeatureScaler
from ml.preparation.Imputer import Imputer


class DataSetTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "age": [20, 21, np.nan, 40, 31, 28, 35, np.nan],
                "country": ["US", "US", "CA", None, "CA", "US", "FR", "FR"],
                "income": [1000.0, 1200.0, 1400.0, 3000.0, 2100.0, 1600.0, 2600.0, 2800.0],
                "clicked": [0, 0, 1, 1, 0, 1, 0, 1],
            }
        )

    def test_to_xy_with_inferred_defaults(self):
        ds = DataSet(self.df)
        X, y = ds.to_xy(target_column="clicked")
        self.assertEqual(8, X.shape[0])
        self.assertEqual(8, y.shape[0])
        self.assertGreater(X.shape[1], 0)

    def test_to_xy_with_user_defined_specs(self):
        ds = DataSet(self.df)
        ds.set_column_preparation("country", categorical=True, impute=True, impute_strategy="most_frequent")
        ds.set_column_preparation("age", categorical=False, scale=False, impute=True, impute_strategy="median")
        X, y = ds.to_xy(target_column="clicked")
        self.assertEqual(8, X.shape[0])
        self.assertEqual(8, y.shape[0])

    def test_to_xy_matches_helper_classes_behavior(self):
        ds = DataSet(self.df)
        ds.set_column_preparation("age", categorical=False, scale=True, impute=True, impute_strategy="mean")
        ds.set_column_preparation(
            "country",
            categorical=True,
            scale=False,
            impute=True,
            impute_strategy="most_frequent",
        )
        ds.set_column_preparation("income", categorical=False, scale=True, impute=False)

        X, y = ds.to_xy(target_column="clicked", feature_columns=["age", "country", "income"])
        self.assertEqual(8, y.shape[0])

        age = self.df["age"].to_numpy().reshape(-1, 1)
        age = Imputer(strategy="mean").fit_transform(age)
        age = FeatureScaler("standard").fit_transform(age)

        country = self.df["country"].astype("object").to_numpy().reshape(-1, 1)
        country = Imputer(strategy="most_frequent").fit_transform(country)
        enc = CategoricalEncoder(sparse_output=False, handle_unknown="ignore")
        country = enc.fit_transform(country)

        income = self.df["income"].to_numpy().reshape(-1, 1)
        income = FeatureScaler("standard").fit_transform(income)

        expected = np.hstack([age, country, income])
        self.assertTrue(np.allclose(X, expected))

    def test_save_load_csv_and_metadata(self):
        ds = DataSet(self.df)
        ds.set_column_preparation("country", categorical=True, impute=True, impute_strategy="most_frequent")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            md_path = Path(tmp_dir) / "sample.metadata.json"
            ds.save(csv_path, metadata_path=md_path)

            loaded = DataSet.load(csv_path, metadata_path=md_path)
            self.assertEqual(list(ds.df.columns), list(loaded.df.columns))
            self.assertIn("country", loaded.column_specs)
            self.assertTrue(loaded.column_specs["country"].categorical)

            loaded_2 = DataSet.load(str(csv_path), metadata_path=str(md_path))
            self.assertEqual(loaded.df.shape, loaded_2.df.shape)

    def test_save_load_csv_with_custom_delimiter(self):
        ds = DataSet(self.df)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample_semicolon.csv"
            ds.save_csv(csv_path, delimiter=";")
            loaded = DataSet.from_csv(csv_path, delimiter=";")
            self.assertEqual(ds.df.shape, loaded.df.shape)

            with self.assertRaises(ValueError):
                DataSet.load(csv_path, delimiter=";", sep=",")

    def test_save_load_tsv(self):
        ds = DataSet(self.df)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tsv_path = Path(tmp_dir) / "sample.tsv"
            ds.save(tsv_path)
            loaded = DataSet.load(tsv_path)
            self.assertEqual(ds.df.shape, loaded.df.shape)

    def test_split_dataset(self):
        ds = DataSet(self.df)
        train_ds, test_ds = ds.split(test_size=0.25, random_state=7, stratify_by="clicked")
        self.assertEqual(len(ds.df), len(train_ds.df) + len(test_ds.df))

        train_ds, val_ds, test_ds = ds.split_train_val_test(
            val_size=0.25, test_size=0.25, random_state=7, stratify_by="clicked"
        )
        self.assertEqual(len(ds.df), len(train_ds.df) + len(val_ds.df) + len(test_ds.df))

    def test_save_load_parquet_if_available(self):
        ds = DataSet(self.df)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pq_path = Path(tmp_dir) / "sample.parquet"
            try:
                ds.save(pq_path)
            except Exception as e:
                self.skipTest(f"Parquet engine unavailable: {e}")
            loaded = DataSet.load(pq_path)
            self.assertEqual(ds.df.shape, loaded.df.shape)


if __name__ == "__main__":
    unittest.main()
