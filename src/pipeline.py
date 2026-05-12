"""End-to-end training and inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from .features import FeatureEngineer
from .models import TargetTrainer, get_base_models, tune_lightgbm
from .utils import configure_logging, print_report

logger = logging.getLogger("shell_hackathon.pipeline")


class ShellPipeline:
    """Pipeline orchestrating feature engineering, training, and inference."""

    def __init__(
        self,
        data_path: Path,
        output_path: Path,
        val_split: float = 0.15,
        n_folds: int = 5,
        random_state: int = 42,
        tune: bool = False,
    ) -> None:
        self.data_path = data_path
        self.output_path = output_path
        self.val_split = val_split
        self.n_folds = n_folds
        self.random_state = random_state
        self.tune = tune
        self.feature_engineer = FeatureEngineer()
        self.trainers: Dict[str, TargetTrainer] = {}
        self.feature_columns: Optional[list[str]] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from disk."""
        train_path = self.data_path / "train.csv"
        test_path = self.data_path / "test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("train.csv/test.csv not found in data path")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Train shape: %s", train_df.shape)
        logger.info("Test shape: %s", test_df.shape)
        return train_df, test_df

    def _prepare_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target_cols = [f"BlendProperty{i}" for i in range(1, 11)]
        feature_cols = [
            col for col in train_df.columns if col not in ["ID", *target_cols]
        ]

        X_train_raw = train_df[feature_cols]
        X_test_raw = test_df[feature_cols]
        y_train = train_df[target_cols]

        X_train = self.feature_engineer.transform(X_train_raw)
        X_test = self.feature_engineer.transform(X_test_raw)

        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())

        self.feature_columns = list(X_train.columns)
        return X_train, X_test, y_train

    def _train_target(
        self, X_train: pd.DataFrame, y_train: pd.Series, target: str
    ) -> TargetTrainer:
        lgb_params = None
        if self.tune:
            logger.info("Optuna tuning LightGBM for %s", target)
            lgb_params = tune_lightgbm(X_train, y_train, self.random_state)
        base_models = get_base_models(self.random_state, lgb_params=lgb_params)
        trainer = TargetTrainer(
            base_models=base_models, n_folds=self.n_folds, random_state=self.random_state
        )
        trainer.fit(X_train, y_train)
        logger.info("%s OOF MAPE: %.4f", target, trainer.oof_mape)
        return trainer

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, float]:
        """Train stacked models for each target and return OOF MAPEs."""
        oof_scores: Dict[str, float] = {}
        for target in y_train.columns:
            logger.info("Training %s", target)
            trainer = self._train_target(X_train, y_train[target], target)
            self.trainers[target] = trainer
            oof_scores[target] = trainer.oof_mape
        return oof_scores

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all targets for a feature matrix."""
        if not self.trainers:
            raise RuntimeError("Models are not trained or loaded")

        preds: Dict[str, np.ndarray] = {}
        for target, trainer in self.trainers.items():
            preds[target] = trainer.predict(X)
        return pd.DataFrame(preds)

    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """Evaluate predictions with MAPE per target."""
        scores: Dict[str, float] = {}
        for target in y_true.columns:
            scores[target] = mean_absolute_percentage_error(y_true[target], y_pred[target])
        return scores

    def save(self, filepath: Path) -> None:
        """Persist the pipeline artifacts to disk."""
        payload = {
            "feature_engineer": self.feature_engineer,
            "trainers": self.trainers,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: Path) -> None:
        """Load pipeline artifacts from disk."""
        payload = joblib.load(filepath)
        self.feature_engineer = payload["feature_engineer"]
        self.trainers = payload["trainers"]
        self.feature_columns = payload.get("feature_columns")

    def run(self, generate_submission: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run the full pipeline and return OOF/validation scores."""
        configure_logging()
        train_df, test_df = self.load_data()
        X_train, X_test, y_train = self._prepare_features(train_df, test_df)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.val_split,
            random_state=self.random_state,
        )

        logger.info("Training on %d rows, validating on %d rows", len(X_tr), len(X_val))
        oof_scores = self.train(X_tr, y_tr)
        val_predictions = self.predict(X_val)
        val_scores = self.evaluate(y_val, val_predictions)
        print_report(oof_scores, val_scores)

        if generate_submission:
            logger.info("Retraining on full data for submission")
            self.trainers = {}
            oof_scores = self.train(X_train, y_train)
            predictions = self.predict(X_test)
            self.output_path.mkdir(parents=True, exist_ok=True)
            submission = pd.DataFrame({"ID": test_df["ID"]})
            for col in predictions.columns:
                submission[col] = predictions[col]
            submission_path = self.output_path / "submission.csv"
            submission.to_csv(submission_path, index=False)
            self.save(self.output_path / "pipeline.pkl")

        return oof_scores, val_scores
