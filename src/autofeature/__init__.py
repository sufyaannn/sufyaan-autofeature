"""AutoFeature: intelligent automatic feature engineering for tabular ML."""

from .engineer import AutoFeatureEngineer
from .selector import TargetAwareSelector
from .cyclical import CyclicalEncoder
from .categorical import SmartCategoricalEncoder
from .leakage import LeakageDetector
from .pipeline import AutoFeaturePipeline

__version__ = "0.1.0"

__all__ = [
    "AutoFeatureEngineer",
    "TargetAwareSelector",
    "CyclicalEncoder",
    "SmartCategoricalEncoder",
    "LeakageDetector",
    "AutoFeaturePipeline",
]
