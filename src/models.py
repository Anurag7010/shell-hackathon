"""Model definitions and training utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

logger = logging.getLogger("shell_hackathon.models")


def get_base_models(
    random_state: int = 42, lgb_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return the base learners used for stacking."""
    lgb_defaults = {
        "objective": "regression",
        "metric": "mape",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 800,
        "random_state": random_state,
        "verbosity": -1,
    }
    if lgb_params:
        lgb_defaults.update(lgb_params)
    return {
        "lightgbm": lgb.LGBMRegressor(**lgb_defaults),
        "xgboost": xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=800,
            random_state=random_state,
            verbosity=0,
        ),
        "catboost": cb.CatBoostRegressor(
            loss_function="MAPE",
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_state=random_state,
            verbose=False,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def tune_lightgbm(
    X: pd.DataFrame, y: pd.Series, random_state: int, n_trials: int = 40
) -> Dict[str, Any]:
    """Run Optuna tuning for LightGBM hyperparameters."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for --tune") from exc

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="mape",
            n_estimators=300,
            random_state=random_state,
            verbosity=-1,
            **params,
        )
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores.append(mean_absolute_percentage_error(y_val, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best MAPE: %.4f", study.best_value)
    return study.best_params


@dataclass
class TargetTrainingResult:
    """Container for per-target training artifacts."""

    base_models: Dict[str, Any]
    meta_model: Any
    scaler: RobustScaler
    oof_mape: float
    p01: float
    p99: float


class TargetTrainer:
    """Train stacked models for a single target."""

    def __init__(
        self,
        base_models: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = base_models or get_base_models(random_state)
        self.scaler = RobustScaler()
        self.meta_model = Ridge(alpha=1.0, random_state=random_state)
        self.fitted_models: Dict[str, Any] = {}
        self.oof_mape: float = float("nan")
        self.p01: float = float("nan")
        self.p99: float = float("nan")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetTrainer":
        """Fit base learners and meta-model with OOF stacking.

        Args:
            X: Training features.
            y: Training target.

        Returns:
            Self, fitted.
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((X.shape[0], len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                model_clone = clone(model)
                model_clone.fit(X_train_scaled, y_train)
                oof_preds[val_idx, model_idx] = model_clone.predict(X_val_scaled)

            logger.debug("Fold %d complete", fold_idx)

        self.meta_model.fit(oof_preds, y)
        self.oof_mape = mean_absolute_percentage_error(y, self.meta_model.predict(oof_preds))

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        for name, model in self.base_models.items():
            fitted = clone(model)
            fitted.fit(X_scaled, y)
            self.fitted_models[name] = fitted

        self.p01, self.p99 = np.percentile(y, [1, 99])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using stacked base learners and meta model.

        Args:
            X: Feature matrix for inference.

        Returns:
            Array of predictions with clipping applied.
        """
        X_scaled = self.scaler.transform(X)
        base_preds = np.zeros((X.shape[0], len(self.fitted_models)))

        for model_idx, model in enumerate(self.fitted_models.values()):
            base_preds[:, model_idx] = model.predict(X_scaled)

        preds = self.meta_model.predict(base_preds)
        return np.clip(preds, self.p01, self.p99)

    def as_result(self) -> TargetTrainingResult:
        """Expose the trained artifacts for serialization."""
        return TargetTrainingResult(
            base_models=self.fitted_models,
            meta_model=self.meta_model,
            scaler=self.scaler,
            oof_mape=self.oof_mape,
            p01=self.p01,
            p99=self.p99,
        )
