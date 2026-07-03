"""Automatic per-column categorical encoding."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Picks label / one-hot / target encoding per categorical column based
    on cardinality.

    - cardinality == 2  -> label encoding (0/1)
    - 2 < cardinality <= max_onehot_cardinality -> one-hot encoding
    - cardinality > max_onehot_cardinality -> smoothed target encoding

    Parameters
    ----------
    max_onehot_cardinality : int, default=10
        Columns with more unique values than this use target encoding.
    smoothing : float, default=1.0
        Regularisation strength for target encoding (higher -> shrinks
        more toward the global mean).
    task : {"auto", "classification", "regression"}, default="auto"
    handle_unknown : {"mean", "zero"}, default="mean"
        How to fill categories seen at transform time but not at fit time
        (target-encoded columns only).
    """

    def __init__(self, max_onehot_cardinality=10, smoothing=1.0,
                 task="auto", handle_unknown="mean"):
        self.max_onehot_cardinality = max_onehot_cardinality
        self.smoothing = smoothing
        self.task = task
        self.handle_unknown = handle_unknown

    def _infer_task(self, y):
        if self.task != "auto":
            return self.task
        y = pd.Series(y)
        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category" or y.nunique() <= 20:
            return "classification"
        return "regression"

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cat_columns_ = [
            c for c in X.columns
            if pd.api.types.is_string_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)
        ]

        self.global_mean_ = float(pd.Series(y).astype(float).mean()) if y is not None else 0.0
        self.encoding_map_ = {}
        self.strategy_ = {}
        self.onehot_categories_ = {}
        self.label_map_ = {}

        y_series = pd.Series(y).reset_index(drop=True) if y is not None else None
        if y_series is not None and (pd.api.types.is_string_dtype(y_series) or str(y_series.dtype) == "category"):
            y_numeric = y_series.astype("category").cat.codes.astype(float)
        else:
            y_numeric = y_series.astype(float) if y_series is not None else None

        for col in self.cat_columns_:
            n_unique = X[col].nunique(dropna=True)

            if n_unique <= 2:
                self.strategy_[col] = "label"
                cats = sorted(X[col].dropna().unique().tolist(), key=str)
                self.label_map_[col] = {cat: i for i, cat in enumerate(cats)}

            elif n_unique <= self.max_onehot_cardinality:
                self.strategy_[col] = "onehot"
                self.onehot_categories_[col] = sorted(X[col].dropna().unique().tolist(), key=str)

            else:
                self.strategy_[col] = "target"
                if y_numeric is None:
                    raise ValueError(
                        f"SmartCategoricalEncoder needs y to target-encode column '{col}'"
                    )
                df = pd.DataFrame({col: X[col].reset_index(drop=True), "_y": y_numeric})
                stats = df.groupby(col)["_y"].agg(["mean", "count"])
                smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / \
                           (stats["count"] + self.smoothing)
                self.encoding_map_[col] = smoothed.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.cat_columns_:
            if col not in X.columns:
                continue
            strategy = self.strategy_[col]

            if strategy == "label":
                mapping = self.label_map_[col]
                X[col] = X[col].map(mapping).fillna(-1).astype(int)

            elif strategy == "onehot":
                categories = self.onehot_categories_[col]
                for cat in categories:
                    X[f"{col}_{cat}"] = (X[col] == cat).astype(int)
                X = X.drop(columns=[col])

            elif strategy == "target":
                mapping = self.encoding_map_[col]
                fill_value = self.global_mean_ if self.handle_unknown == "mean" else 0.0
                X[col] = X[col].map(mapping).fillna(fill_value).astype(float)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
