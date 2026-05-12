"""Feature engineering utilities for Shell Hackathon."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

logger = logging.getLogger("shell_hackathon.features")


class FeatureEngineer:
    """Create engineered features for fuel blend prediction."""

    def __init__(self, pca_components: int = 10, random_state: int = 42) -> None:
        self.pca_components = pca_components
        self.random_state = random_state
        self.feature_names: List[str] = []

    def _mixing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        for prop_idx in range(1, 11):
            lin_mix = 0.0
            geo_mix = 1.0
            harm_mix = 0.0
            log_mix = 0.0
            for comp_idx in range(1, 6):
                frac = features[f"Component{comp_idx}_fraction"]
                prop = features[f"Component{comp_idx}_Property{prop_idx}"]
                safe_prop = np.maximum(prop, 1e-10)

                lin_mix += frac * prop
                geo_mix *= np.power(safe_prop, frac)
                harm_mix += frac / safe_prop
                log_mix += frac * np.log(safe_prop)

            features[f"LinMix_P{prop_idx}"] = lin_mix
            features[f"GeoMix_P{prop_idx}"] = geo_mix
            features[f"HarmMix_P{prop_idx}"] = 1.0 / np.maximum(harm_mix, 1e-10)
            features[f"LogMix_P{prop_idx}"] = np.exp(log_mix)

        return features

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        for i in range(1, 5):
            for j in range(i + 1, 6):
                interaction = features[f"Component{i}_fraction"] * features[
                    f"Component{j}_fraction"
                ]
                features[f"Interaction_{i}_{j}"] = interaction

                for prop_idx in range(1, 11):
                    prop_diff = np.abs(
                        features[f"Component{i}_Property{prop_idx}"]
                        - features[f"Component{j}_Property{prop_idx}"]
                    )
                    features[f"PropDiff_{i}_{j}_P{prop_idx}"] = prop_diff
                    features[f"WeightedPropDiff_{i}_{j}_P{prop_idx}"] = (
                        prop_diff * interaction
                    )

        return features

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        for prop_idx in range(1, 11):
            prop_values = [
                features[f"Component{i}_Property{prop_idx}"] for i in range(1, 6)
            ]
            prop_matrix = np.column_stack(prop_values)

            features[f"Property{prop_idx}_Mean"] = np.mean(prop_matrix, axis=1)
            features[f"Property{prop_idx}_Std"] = np.std(prop_matrix, axis=1)
            features[f"Property{prop_idx}_Min"] = np.min(prop_matrix, axis=1)
            features[f"Property{prop_idx}_Max"] = np.max(prop_matrix, axis=1)
            features[f"Property{prop_idx}_Range"] = (
                features[f"Property{prop_idx}_Max"] - features[f"Property{prop_idx}_Min"]
            )
            features[f"Property{prop_idx}_Skew"] = stats.skew(prop_matrix, axis=1)
            features[f"Property{prop_idx}_Kurt"] = stats.kurtosis(prop_matrix, axis=1)

        for comp_idx in range(1, 6):
            comp_values = [
                features[f"Component{comp_idx}_Property{j}"] for j in range(1, 11)
            ]
            comp_matrix = np.column_stack(comp_values)

            features[f"Component{comp_idx}_Mean"] = np.mean(comp_matrix, axis=1)
            features[f"Component{comp_idx}_Std"] = np.std(comp_matrix, axis=1)
            features[f"Component{comp_idx}_Min"] = np.min(comp_matrix, axis=1)
            features[f"Component{comp_idx}_Max"] = np.max(comp_matrix, axis=1)

        return features

    def _entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        fractions = df[[f"Component{i}_fraction" for i in range(1, 6)]].values
        fractions = np.maximum(fractions, 1e-10)

        shannon_entropy = -np.sum(fractions * np.log(fractions), axis=1)
        features["ShannonH"] = shannon_entropy
        features["Effective_Components"] = np.exp(shannon_entropy)
        features["Dominant_Fraction"] = np.max(fractions, axis=1)
        features["Dominant_Component"] = np.argmax(fractions, axis=1) + 1

        return features

    def _pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        property_cols = [
            f"Component{i}_Property{j}" for i in range(1, 6) for j in range(1, 11)
        ]

        pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        pca_features = pca.fit_transform(features[property_cols])

        for i in range(self.pca_components):
            features[f"PCA_{i}"] = pca_features[:, i]

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate the full engineered feature set.

        Args:
            df: Raw feature dataframe containing component fractions and properties.

        Returns:
            Feature dataframe with engineered columns appended.
        """
        logger.info("Starting feature engineering")
        features = df.copy()
        features = self._mixing_features(features)
        features = self._interaction_features(features)
        features = self._statistical_features(features)
        features = self._entropy_features(features)
        features = self._pca_features(features)

        self.feature_names = list(features.columns)
        logger.info("Feature engineering complete: %s", features.shape)
        return features
