import numpy as np
import pandas as pd

from autofeature import (
    AutoFeaturePipeline,
    AutoFeatureEngineer,
    TargetAwareSelector,
    CyclicalEncoder,
    SmartCategoricalEncoder,
    LeakageDetector,
)


def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n) + 1,
        "hour": rng.integers(0, 24, size=n),
        "day_of_week": rng.integers(0, 7, size=n),
        "cat_low": rng.choice(["a", "b", "c"], size=n),
        "cat_high": rng.choice([f"v{i}" for i in range(20)], size=n),
    })
    y = (df["num1"] + df["num2"] + rng.normal(scale=0.1, size=n) > 1).astype(int)
    return df, y


def test_full_pipeline():
    X, y = make_data()
    split = 150
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipeline = AutoFeaturePipeline(
        cyclical_columns={"hour": 24, "day_of_week": 7},
        max_interaction_features=5,
        k=8,
        task="classification",
        verbose=False,
    )

    X_train_out = pipeline.fit_transform(X_train, y_train)
    X_test_out = pipeline.transform(X_test)

    assert X_train_out.shape[1] == X_test_out.shape[1]
    assert X_train_out.shape[0] == len(X_train)
    assert X_test_out.shape[0] == len(X_test)
    assert not X_train_out.isna().any().any()

    summary = pipeline.get_summary()
    assert "selected_features" in summary


def test_individual_components():
    X, y = make_data()

    ld = LeakageDetector(verbose=False)
    ld.fit(X, y)
    X1 = ld.remove_leaky(X)
    assert X1.shape[1] <= X.shape[1]

    enc = SmartCategoricalEncoder()
    X2 = enc.fit_transform(X1, y)
    remaining_str_cols = [c for c in X2.columns if pd.api.types.is_string_dtype(X2[c])]
    assert len(remaining_str_cols) == 0

    cyc = CyclicalEncoder(columns={"hour": 24, "day_of_week": 7})
    X3 = cyc.fit_transform(X2)
    assert "hour_sin" in X3.columns and "hour_cos" in X3.columns
    assert "hour" not in X3.columns

    afe = AutoFeatureEngineer(max_interaction_features=5, verbose=False)
    X4 = afe.fit_transform(X3, y)
    report = afe.get_interaction_report()
    assert len(report) <= 5

    sel = TargetAwareSelector(k=6)
    X5 = sel.fit_transform(X4, y)
    assert X5.shape[1] <= 6
    scores = sel.get_feature_scores()
    assert (scores >= 0).all()


def test_cyclical_wraparound():
    df = pd.DataFrame({"hour": [0, 23]})
    cyc = CyclicalEncoder(columns={"hour": 24})
    out = cyc.fit_transform(df)
    # hour 0 and hour 23 should be close in sin/cos space
    dist = np.sqrt((out.iloc[0] - out.iloc[1]).pow(2).sum())
    assert dist < 0.3
