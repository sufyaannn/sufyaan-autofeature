"""Cyclical (sin/cos) encoding for periodic variables."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encodes periodic integer/float variables (hour, month, day-of-week, ...)
    as sin/cos pairs so that the "wrap-around" structure (e.g. hour 23 is
    close to hour 0) is preserved for downstream models.

    Parameters
    ----------
    columns : dict[str, int]
        Mapping of column name -> period (e.g. {"hour": 24, "month": 12}).
    drop_original : bool, default=True
        Whether to drop the original column after encoding.
    """

    def __init__(self, columns=None, drop_original=True):
        self.columns = columns or {}
        self.drop_original = drop_original

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"CyclicalEncoder: columns not found in X: {missing}")
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, period in self.columns.items():
            if col not in X.columns:
                continue
            values = X[col].astype(float)
            X[f"{col}_sin"] = np.sin(2 * np.pi * values / period)
            X[f"{col}_cos"] = np.cos(2 * np.pi * values / period)
            if self.drop_original:
                X = X.drop(columns=[col])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
