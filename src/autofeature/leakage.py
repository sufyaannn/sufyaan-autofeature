"""Leakage detection for tabular features."""
import numpy as np
import pandas as pd


class LeakageDetector:
    """Flags features that suspiciously correlate with the target, or whose
    names match common target-leakage patterns (e.g. 'label', 'target').

    Parameters
    ----------
    correlation_threshold : float, default=0.95
        Absolute Pearson correlation with the (numerically-encoded) target
        above which a feature is flagged as leaky.
    name_patterns : list of str, default=["label", "target", "outcome"]
        Case-insensitive substrings checked against column names.
    verbose : bool, default=True
        Whether to print warnings when leaky features are found.
    """

    def __init__(self, correlation_threshold=0.95,
                 name_patterns=None, verbose=True):
        self.correlation_threshold = correlation_threshold
        self.name_patterns = name_patterns or ["label", "target", "outcome"]
        self.verbose = verbose
        self._report = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category":
            y_numeric = y.astype("category").cat.codes.astype(float)
        else:
            y_numeric = y.astype(float)

        flagged = {}

        for col in X.columns:
            reasons = []

            # Name-pattern check
            lower = str(col).lower()
            if any(p.lower() in lower for p in self.name_patterns):
                reasons.append("name_pattern")

            # Correlation check (numeric columns only)
            if pd.api.types.is_numeric_dtype(X[col]):
                col_vals = X[col].astype(float)
                if col_vals.nunique() > 1 and y_numeric.nunique() > 1:
                    corr = np.corrcoef(col_vals, y_numeric)[0, 1]
                    if np.isfinite(corr) and abs(corr) >= self.correlation_threshold:
                        reasons.append(f"correlation={corr:.3f}")

            if reasons:
                flagged[col] = reasons

        self._report = flagged

        if self.verbose and flagged:
            print(f"[LeakageDetector] {len(flagged)} suspicious feature(s) found:")
            for col, reasons in flagged.items():
                print(f"  - {col}: {', '.join(reasons)}")

        return self

    def remove_leaky(self, X):
        X = pd.DataFrame(X).copy()
        cols_to_drop = [c for c in self._report if c in X.columns]
        return X.drop(columns=cols_to_drop)

    def get_report(self):
        return dict(self._report)
