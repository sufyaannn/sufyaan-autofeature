"""End-to-end automatic feature engineering pipeline."""
import pandas as pd

from .leakage import LeakageDetector
from .categorical import SmartCategoricalEncoder
from .cyclical import CyclicalEncoder
from .engineer import AutoFeatureEngineer
from .selector import TargetAwareSelector


class AutoFeaturePipeline:
    """Runs leakage detection, categorical encoding, cyclical encoding,
    interaction-feature generation, and target-aware selection end-to-end
    in one call.

    Parameters
    ----------
    cyclical_columns : dict[str, int] or None
        Column -> period mapping for CyclicalEncoder. If None, this step
        is skipped.
    max_interaction_features : int, default=20
    k : int, default=20
        Number of final features to keep.
    task : {"auto", "classification", "regression"}, default="auto"
    detect_leakage : bool, default=True
        Whether to run LeakageDetector and report findings.
    remove_leaky : bool, default=False
        Whether to actually drop flagged features.
    random_state : int, default=42
    verbose : bool, default=False
    """

    def __init__(self, cyclical_columns=None, max_interaction_features=20,
                 k=20, task="auto", detect_leakage=True, remove_leaky=False,
                 random_state=42, verbose=False):
        self.cyclical_columns = cyclical_columns
        self.max_interaction_features = max_interaction_features
        self.k = k
        self.task = task
        self.detect_leakage = detect_leakage
        self.remove_leaky = remove_leaky
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X, y):
        X = pd.DataFrame(X).copy()
        self._summary = {}

        if self.detect_leakage:
            self.leakage_detector_ = LeakageDetector(verbose=self.verbose)
            self.leakage_detector_.fit(X, y)
            self._summary["leakage_report"] = self.leakage_detector_.get_report()
            if self.remove_leaky:
                X = self.leakage_detector_.remove_leaky(X)

        self.cat_encoder_ = SmartCategoricalEncoder(task=self.task)
        X = self.cat_encoder_.fit_transform(X, y)

        if self.cyclical_columns:
            self.cyclical_encoder_ = CyclicalEncoder(columns=self.cyclical_columns)
            X = self.cyclical_encoder_.fit_transform(X)
        else:
            self.cyclical_encoder_ = None

        self.engineer_ = AutoFeatureEngineer(
            max_interaction_features=self.max_interaction_features,
            task=self.task,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        X = self.engineer_.fit_transform(X, y)
        self._summary["interactions"] = self.engineer_.get_interaction_report()

        self.selector_ = TargetAwareSelector(k=self.k, task=self.task,
                                              random_state=self.random_state)
        X = self.selector_.fit_transform(X, y)
        self._summary["selected_features"] = self.selector_.get_feature_scores()

        self._summary["n_input_features"] = None  # set by fit caller if desired
        self._summary["n_output_features"] = X.shape[1]

        return X

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.detect_leakage and self.remove_leaky:
            X = self.leakage_detector_.remove_leaky(X)
        X = self.cat_encoder_.transform(X)
        if self.cyclical_encoder_ is not None:
            X = self.cyclical_encoder_.transform(X)
        X = self.engineer_.transform(X)
        X = self.selector_.transform(X)
        return X

    def get_summary(self):
        return dict(self._summary)
