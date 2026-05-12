"""Unit tests for feature engineering and model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer
from src.models import TargetTrainer


@pytest.fixture()
def raw_features() -> pd.DataFrame:
    """Synthetic raw feature dataframe with 10 rows."""
    rng = np.random.default_rng(42)
    rows = 10

    data: dict[str, list[float]] = {}
    fractions = rng.uniform(0.05, 0.4, size=(rows, 5))
    fractions = fractions / fractions.sum(axis=1, keepdims=True)

    for i in range(1, 6):
        data[f"Component{i}_fraction"] = fractions[:, i - 1].tolist()
        for j in range(1, 11):
            values = rng.uniform(0.5, 5.0, size=rows)
            data[f"Component{i}_Property{j}"] = values.tolist()

    return pd.DataFrame(data)


@pytest.fixture()
def feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture()
def engineered_features(raw_features: pd.DataFrame, feature_engineer: FeatureEngineer) -> pd.DataFrame:
    return feature_engineer.transform(raw_features)


@pytest.fixture()
def target_series(raw_features: pd.DataFrame) -> pd.Series:
    weights = np.array([0.2, 0.1, 0.3, 0.25, 0.15])
    base = np.zeros(len(raw_features))
    for comp_idx in range(1, 6):
        base += (
            raw_features[f"Component{comp_idx}_fraction"].values
            * raw_features[f"Component{comp_idx}_Property1"].values
            * weights[comp_idx - 1]
        )
    noise = np.linspace(0.0, 0.1, len(raw_features))
    return pd.Series(base + noise, name="BlendProperty1")


def test_feature_engineer_transform(
    raw_features: pd.DataFrame, engineered_features: pd.DataFrame
) -> None:
    assert not engineered_features.isna().any().any()
    assert np.isfinite(engineered_features.to_numpy()).all()
    assert engineered_features.shape[1] > raw_features.shape[1]
    assert (engineered_features["ShannonH"] >= 0).all()

    row = raw_features.iloc[0]
    manual = 0.0
    for comp_idx in range(1, 6):
        manual += (
            row[f"Component{comp_idx}_fraction"]
            * row[f"Component{comp_idx}_Property1"]
        )
    assert engineered_features.loc[0, "LinMix_P1"] == pytest.approx(manual)


def test_target_trainer_predict(
    engineered_features: pd.DataFrame, target_series: pd.Series
) -> None:
    trainer = TargetTrainer(n_folds=3, random_state=7)
    trainer.fit(engineered_features, target_series)
    preds = trainer.predict(engineered_features)

    assert preds.shape[0] == engineered_features.shape[0]
    assert np.isfinite(preds).all()
    assert np.min(preds) >= trainer.p01
    assert np.max(preds) <= trainer.p99
