"""Feature selection by mutual information with the target."""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class TargetAwareSelector:
    """Selects features by mutual information with the target rather than
    generic variance-based filtering.

    Parameters
    ----------
    k : int or "all", default=10
        Number of top features to keep.
    task : {"auto", "classification", "regression"}, default="auto"
    threshold : float or None, default=None
        If set, keep all features with MI score >= threshold instead of
        using k.
    random_state : int, default=42
    """

    def __init__(self, k=10, task="auto", threshold=None, random_state=42):
        self.k = k
        self.task = task
        self.threshold = threshold
        self.random_state = random_state

    def _infer_task(self, y):
        if self.task != "auto":
            return self.task
        y = pd.Series(y)
        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category" or y.nunique() <= 20:
            return "classification"
        return "regression"

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y).reset_index(drop=True)
        self.task_ = self._infer_task(y)

        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category":
            y_numeric = y.astype("category").cat.codes
        else:
            y_numeric = y

        X_numeric = X.select_dtypes(include="number").fillna(0)
        self._non_numeric_cols_ = [c for c in X.columns if c not in X_numeric.columns]

        if self.task_ == "classification":
            scores = mutual_info_classif(X_numeric, y_numeric, random_state=self.random_state)
        else:
            scores = mutual_info_regression(X_numeric, y_numeric, random_state=self.random_state)

        self.scores_ = pd.Series(scores, index=X_numeric.columns).sort_values(ascending=False)

        if self.threshold is not None:
            self.selected_columns_ = self.scores_[self.scores_ >= self.threshold].index.tolist()
        elif self.k == "all":
            self.selected_columns_ = self.scores_.index.tolist()
        else:
            self.selected_columns_ = self.scores_.head(self.k).index.tolist()

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = [c for c in self.selected_columns_ if c in X.columns]
        return X[cols]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_feature_scores(self):
        return self.scores_.copy()
