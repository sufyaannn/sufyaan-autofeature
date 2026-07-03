"""Importance-guided interaction feature generation."""
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class AutoFeatureEngineer:
    """Detects and generates useful interaction features (products, ratios,
    differences) using importance-guided search: candidate interactions are
    scored by the improvement they give a small internal Random Forest, and
    only the most useful ones are kept.

    Parameters
    ----------
    max_interaction_features : int, default=20
        Maximum number of interaction features to add.
    interaction_types : list of str, default=["product", "ratio", "difference"]
    interaction_threshold : float, default=0.01
        Minimum importance (relative to the internal RF's feature_importances_)
        a candidate interaction must reach to be kept.
    n_estimators : int, default=50
        Trees used by the internal evaluator.
    task : {"auto", "classification", "regression"}, default="auto"
    random_state : int, default=42
    verbose : bool, default=False
    """

    def __init__(self, max_interaction_features=20,
                 interaction_types=None, interaction_threshold=0.01,
                 n_estimators=50, task="auto", random_state=42, verbose=False):
        self.max_interaction_features = max_interaction_features
        self.interaction_types = interaction_types or ["product", "ratio", "difference"]
        self.interaction_threshold = interaction_threshold
        self.n_estimators = n_estimators
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

    def _infer_task(self, y):
        if self.task != "auto":
            return self.task
        y = pd.Series(y)
        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category" or y.nunique() <= 20:
            return "classification"
        return "regression"

    @staticmethod
    def _make_interaction(a, b, kind):
        if kind == "product":
            return a * b
        if kind == "ratio":
            return a / b.replace(0, np.nan)
        if kind == "difference":
            return a - b
        raise ValueError(f"Unknown interaction type: {kind}")

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y).reset_index(drop=True)
        self.task_ = self._infer_task(y)
        numeric_cols = X.select_dtypes(include="number").columns.tolist()

        if pd.api.types.is_string_dtype(y) or str(y.dtype) == "category":
            y_numeric = y.astype("category").cat.codes
        else:
            y_numeric = y

        RFModel = RandomForestClassifier if self.task_ == "classification" else RandomForestRegressor
        candidates = {}

        for col_a, col_b in itertools.combinations(numeric_cols, 2):
            a, b = X[col_a].astype(float), X[col_b].astype(float)
            for kind in self.interaction_types:
                name = f"{col_a}_{kind}_{col_b}"
                try:
                    values = self._make_interaction(a, b, kind)
                except Exception:
                    continue
                if values.replace([np.inf, -np.inf], np.nan).isna().all():
                    continue
                candidates[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if not candidates:
            self.selected_interactions_ = {}
            self.report_ = pd.Series(dtype=float)
            return self

        cand_df = pd.DataFrame(candidates)
        combined = pd.concat([X[numeric_cols].fillna(0), cand_df], axis=1)

        model = RFModel(n_estimators=self.n_estimators, random_state=self.random_state)
        model.fit(combined, y_numeric)
        importances = pd.Series(model.feature_importances_, index=combined.columns)

        cand_importances = importances[cand_df.columns].sort_values(ascending=False)
        cand_importances = cand_importances[cand_importances >= self.interaction_threshold]
        top_names = cand_importances.head(self.max_interaction_features).index.tolist()

        self.selected_interactions_ = {}
        for name in top_names:
            for kind in self.interaction_types:
                marker = f"_{kind}_"
                if marker in name:
                    col_a, col_b = name.split(marker, 1)
                    self.selected_interactions_[name] = (col_a, col_b, kind)
                    break

        self.report_ = cand_importances.loc[top_names]

        if self.verbose and top_names:
            print(f"[AutoFeatureEngineer] Selected {len(top_names)} interaction feature(s):")
            for name in top_names:
                print(f"  - {name} (importance={self.report_[name]:.4f})")

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for name, (col_a, col_b, kind) in self.selected_interactions_.items():
            a, b = X[col_a].astype(float), X[col_b].astype(float)
            values = self._make_interaction(a, b, kind)
            X[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_interaction_report(self):
        return self.report_.copy()
