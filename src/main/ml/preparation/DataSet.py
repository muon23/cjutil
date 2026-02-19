from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.preparation.CategoricalEncoder import CategoricalEncoder
from ml.preparation.DatasetSplitter import DatasetSplitter
from ml.preparation.FeatureScaler import FeatureScaler
from ml.preparation.Imputer import Imputer


@dataclass
class ColumnPreparationSpec:
    """
    Per-column preparation behavior.
    """

    categorical: bool
    scale: bool
    impute: bool
    impute_strategy: Literal["mean", "median", "most_frequent", "constant"] | None = None
    constant_value: Any = None


class DataSet:
    """
    Dataset orchestration utility for loading, saving, preprocessing, and splitting tabular data.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize DataSet with in-memory DataFrame and optional metadata.

        Args:
            dataframe: Source tabular data.
            metadata: Optional metadata dictionary containing column preparation specs.

        Returns:
            None.
        """
        self.df = dataframe.copy()
        self.metadata: dict[str, Any] = metadata.copy() if metadata else {}
        self.column_specs: dict[str, ColumnPreparationSpec] = {}
        self._load_specs_from_metadata()

    @classmethod
    def load(
            cls,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> "DataSet":
        """
        Load dataset from csv/tsv/parquet into DataFrame and optional metadata.

        Args:
            path: Data file path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional delimiter for delimited text formats (csv/tsv).
            **kwargs: Reader-specific arguments passed to pandas.

        Returns:
            DataSet instance.

        Raises:
            ValueError: If file extension is unsupported.
            ValueError: If both `delimiter` and `sep` are provided.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            read_kwargs = cls._resolve_sep_kwargs(kwargs, delimiter=delimiter, default_sep=",")
            df = pd.read_csv(p, **read_kwargs)
        elif suffix == ".tsv":
            read_kwargs = cls._resolve_sep_kwargs(kwargs, delimiter=delimiter, default_sep="\t")
            df = pd.read_csv(p, **read_kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(p, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

        md_path = Path(metadata_path) if metadata_path else cls._default_metadata_path(p)
        metadata = {}
        if md_path.exists():
            metadata = json.loads(md_path.read_text(encoding="utf-8"))
        return cls(dataframe=df, metadata=metadata)

    @classmethod
    def from_csv(
            cls,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> "DataSet":
        """
        Load dataset from CSV file.

        Args:
            path: CSV file path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional CSV delimiter.
            **kwargs: Arguments passed to pandas.read_csv.

        Returns:
            DataSet instance.
        """
        return cls.load(path=path, metadata_path=metadata_path, delimiter=delimiter, **kwargs)

    @classmethod
    def from_tsv(
            cls,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> "DataSet":
        """
        Load dataset from TSV file.

        Args:
            path: TSV file path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional delimiter; defaults to tab.
            **kwargs: Arguments passed to pandas.read_csv.

        Returns:
            DataSet instance.
        """
        return cls.load(path=path, metadata_path=metadata_path, delimiter=delimiter, **kwargs)

    @classmethod
    def from_parquet(cls, path: str | Path, metadata_path: str | Path | None = None, **kwargs: Any) -> "DataSet":
        """
        Load dataset from Parquet file.

        Args:
            path: Parquet file path.
            metadata_path: Optional metadata JSON path.
            **kwargs: Arguments passed to pandas.read_parquet.

        Returns:
            DataSet instance.
        """
        return cls.load(path=path, metadata_path=metadata_path, **kwargs)

    def save(
            self,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Save DataFrame to csv/tsv/parquet and write metadata JSON.

        Args:
            path: Output data file path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional delimiter for delimited text formats (csv/tsv).
            **kwargs: Writer-specific arguments passed to pandas.

        Returns:
            None.

        Raises:
            ValueError: If file extension is unsupported.
            ValueError: If both `delimiter` and `sep` are provided.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            write_kwargs = self._resolve_sep_kwargs(kwargs, delimiter=delimiter)
            self.df.to_csv(p, index=False, **write_kwargs)
        elif suffix == ".tsv":
            write_kwargs = self._resolve_sep_kwargs(kwargs, delimiter=delimiter, default_sep="\t")
            self.df.to_csv(p, index=False, **write_kwargs)
        elif suffix == ".parquet":
            self.df.to_parquet(p, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

        md_path = Path(metadata_path) if metadata_path else self._default_metadata_path(p)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(json.dumps(self._build_metadata_payload(), indent=2), encoding="utf-8")

    def save_csv(
            self,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Save dataset as CSV.

        Args:
            path: CSV output path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional CSV delimiter.
            **kwargs: Arguments passed to pandas.to_csv.

        Returns:
            None.
        """
        self.save(path=path, metadata_path=metadata_path, delimiter=delimiter, **kwargs)

    def save_tsv(
            self,
            path: str | Path,
            metadata_path: str | Path | None = None,
            delimiter: str | None = None,
            **kwargs: Any,
    ) -> None:
        """
        Save dataset as TSV.

        Args:
            path: TSV output path.
            metadata_path: Optional metadata JSON path.
            delimiter: Optional delimiter; defaults to tab.
            **kwargs: Arguments passed to pandas.to_csv.

        Returns:
            None.
        """
        self.save(path=path, metadata_path=metadata_path, delimiter=delimiter, **kwargs)

    def save_parquet(self, path: str | Path, metadata_path: str | Path | None = None, **kwargs: Any) -> None:
        """
        Save dataset as Parquet.

        Args:
            path: Parquet output path.
            metadata_path: Optional metadata JSON path.
            **kwargs: Arguments passed to pandas.to_parquet.

        Returns:
            None.
        """
        self.save(path=path, metadata_path=metadata_path, **kwargs)

    def set_column_preparation(
            self,
            column: str,
            categorical: bool | None = None,
            scale: bool | None = None,
            impute: bool | None = None,
            impute_strategy: Literal["mean", "median", "most_frequent", "constant"] | None = None,
            constant_value: Any = None,
    ) -> None:
        """
        Set/override preparation behavior for a column.

        Args:
            column: Column name.
            categorical: Whether to one-hot encode.
            scale: Whether to standard-scale (numeric only).
            impute: Whether to impute missing values.
            impute_strategy: Missing value strategy.
            constant_value: Fill value for `constant` imputation.

        Returns:
            None.

        Raises:
            ValueError: If column does not exist.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in dataset")
        inferred = self._infer_column_spec(column)
        self.column_specs[column] = ColumnPreparationSpec(
            categorical=inferred.categorical if categorical is None else categorical,
            scale=inferred.scale if scale is None else scale,
            impute=inferred.impute if impute is None else impute,
            impute_strategy=inferred.impute_strategy if impute_strategy is None else impute_strategy,
            constant_value=inferred.constant_value if constant_value is None else constant_value,
        )

    def to_xy(
            self,
            target_column: str,
            feature_columns: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract processed X and y arrays for ML training/inference.

        Args:
            target_column: Target label/value column.
            feature_columns: Optional explicit feature column list.

        Returns:
            Tuple of (X, y) numpy arrays.

        Raises:
            ValueError: If target/feature columns are invalid.
        """
        if target_column not in self.df.columns:
            raise ValueError(f"target_column '{target_column}' not found")

        if feature_columns is None:
            feature_columns = [c for c in self.df.columns if c != target_column]
        else:
            missing = [c for c in feature_columns if c not in self.df.columns]
            if missing:
                raise ValueError(f"feature_columns not found: {missing}")

        X_df = self.df[feature_columns].copy()
        X_processed = self._prepare_features(X_df)
        y = self.df[target_column].to_numpy()
        return X_processed.to_numpy(dtype=float), y

    def split(
            self,
            test_size: float = 0.2,
            random_state: int | None = 42,
            stratify_by: str | None = None,
    ) -> tuple["DataSet", "DataSet"]:
        """
        Split dataset into train and test DataSet objects.

        Args:
            test_size: Proportion assigned to test split.
            random_state: Optional random seed.
            stratify_by: Optional column name to stratify by.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        row_ids = np.arange(len(self.df))
        split_result = DatasetSplitter.train_test(
            X=row_ids,
            y=self.df[stratify_by].to_numpy() if stratify_by else row_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by is not None,
        )
        train_df = self.df.iloc[split_result.X_train]
        test_df = self.df.iloc[split_result.X_test]
        payload = self._build_metadata_payload()
        return (
            DataSet(train_df.reset_index(drop=True), metadata=payload),
            DataSet(test_df.reset_index(drop=True), metadata=payload),
        )

    def split_train_val_test(
            self,
            val_size: float = 0.1,
            test_size: float = 0.2,
            random_state: int | None = 42,
            stratify_by: str | None = None,
    ) -> tuple["DataSet", "DataSet", "DataSet"]:
        """
        Split dataset into train/validation/test DataSet objects.

        Args:
            val_size: Validation proportion relative to full dataset.
            test_size: Test proportion relative to full dataset.
            random_state: Optional random seed.
            stratify_by: Optional column name to stratify by.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If split ratios are invalid.
        """
        row_ids = np.arange(len(self.df))
        split_result = DatasetSplitter.train_val_test(
            X=row_ids,
            y=self.df[stratify_by].to_numpy() if stratify_by else row_ids,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by is not None,
        )
        train_df = self.df.iloc[split_result.X_train]
        val_df = self.df.iloc[split_result.X_val]
        test_df = self.df.iloc[split_result.X_test]

        payload = self._build_metadata_payload()
        return (
            DataSet(train_df.reset_index(drop=True), metadata=payload),
            DataSet(val_df.reset_index(drop=True), metadata=payload),
            DataSet(test_df.reset_index(drop=True), metadata=payload),
        )

    @staticmethod
    def _default_metadata_path(data_path: Path) -> Path:
        return data_path.with_suffix(data_path.suffix + ".metadata.json")

    @staticmethod
    def _resolve_sep_kwargs(
            kwargs: dict[str, Any],
            delimiter: str | None = None,
            default_sep: str | None = None,
    ) -> dict[str, Any]:
        merged = dict(kwargs)
        if delimiter is not None and "sep" in merged:
            raise ValueError("Provide either delimiter or sep, not both")
        if delimiter is not None:
            merged["sep"] = delimiter
        elif default_sep is not None and "sep" not in merged:
            merged["sep"] = default_sep
        return merged

    def _load_specs_from_metadata(self) -> None:
        specs_payload = self.metadata.get("column_specs", {})
        for column, spec in specs_payload.items():
            self.column_specs[column] = ColumnPreparationSpec(
                categorical=bool(spec.get("categorical", False)),
                scale=bool(spec.get("scale", False)),
                impute=bool(spec.get("impute", False)),
                impute_strategy=spec.get("impute_strategy"),
                constant_value=spec.get("constant_value"),
            )

    def _build_metadata_payload(self) -> dict[str, Any]:
        return {
            "column_specs": {
                col: {
                    "categorical": spec.categorical,
                    "scale": spec.scale,
                    "impute": spec.impute,
                    "impute_strategy": spec.impute_strategy,
                    "constant_value": spec.constant_value,
                }
                for col, spec in self.column_specs.items()
            }
        }

    def _infer_column_spec(self, column: str) -> ColumnPreparationSpec:
        series = self.df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        has_missing = bool(series.isna().any())
        if is_numeric:
            return ColumnPreparationSpec(
                categorical=False,
                scale=True,
                impute=has_missing,
                impute_strategy="mean",
            )
        return ColumnPreparationSpec(
            categorical=True,
            scale=False,
            impute=has_missing,
            impute_strategy="most_frequent",
        )

    def _resolve_column_spec(self, column: str) -> ColumnPreparationSpec:
        if column not in self.column_specs:
            self.column_specs[column] = self._infer_column_spec(column)
        return self.column_specs[column]

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        prepared: list[pd.DataFrame] = []
        for column in features_df.columns:
            spec = self._resolve_column_spec(column)
            series = features_df[column]
            if spec.categorical:
                values = series.astype("object").to_numpy().reshape(-1, 1)
                values = self._impute_array(values, spec)
                encoder = CategoricalEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(values)
                encoded_cols = encoder.get_feature_names_out([column]).tolist()
                prepared.append(pd.DataFrame(encoded, columns=encoded_cols, index=features_df.index))
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                arr = numeric.to_numpy().reshape(-1, 1)
                arr = self._impute_array(arr, spec)
                if spec.scale:
                    scaler = FeatureScaler("standard")
                    arr = scaler.fit_transform(arr)
                prepared.append(pd.DataFrame(arr, columns=[column], index=features_df.index))

        if not prepared:
            return pd.DataFrame(index=features_df.index)
        return pd.concat(prepared, axis=1)

    @staticmethod
    def _impute_array(values: np.ndarray, spec: ColumnPreparationSpec) -> np.ndarray:
        if not spec.impute:
            return values
        strategy = spec.impute_strategy or "most_frequent"
        kwargs: dict[str, Any] = {}
        if strategy == "constant":
            kwargs["fill_value"] = spec.constant_value
        imputer = Imputer(strategy=strategy, **kwargs)
        return imputer.fit_transform(values)
