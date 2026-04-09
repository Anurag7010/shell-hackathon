"""
Shell.ai Hackathon - Complete ML Pipeline
Predict 10 continuous targets (BlendProperty1-10) based on 5 components and their properties
"""


import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy import stats
from scipy.signal import medfilt
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for chemical mixing predictions"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_mixing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mixing rule features based on chemical principles"""
        features = df.copy()
        
        # Component fractions
        fractions = [f'Component{i}_fraction' for i in range(1, 6)]
        
        # Properties for each component
        properties = []
        for i in range(1, 6):
            for j in range(1, 11):
                properties.append(f'Component{i}_Property{j}')
        
        # 1. Linear mixing (weighted average)
        for prop_idx in range(1, 11):
            linear_mix = 0
            for comp_idx in range(1, 6):
                linear_mix += (features[f'Component{comp_idx}_fraction'] * 
                              features[f'Component{comp_idx}_Property{prop_idx}'])
            features[f'Linear_Mix_P{prop_idx}'] = linear_mix
        
        # 2. Geometric mixing
        for prop_idx in range(1, 11):
            geometric_mix = 1
            for comp_idx in range(1, 6):
                prop_val = np.maximum(features[f'Component{comp_idx}_Property{prop_idx}'], 1e-10)
                geometric_mix *= np.power(prop_val, features[f'Component{comp_idx}_fraction'])
            features[f'Geometric_Mix_P{prop_idx}'] = geometric_mix
        
        # 3. Harmonic mixing
        for prop_idx in range(1, 11):
            harmonic_mix = 0
            for comp_idx in range(1, 6):
                prop_val = np.maximum(features[f'Component{comp_idx}_Property{prop_idx}'], 1e-10)
                harmonic_mix += features[f'Component{comp_idx}_fraction'] / prop_val
            features[f'Harmonic_Mix_P{prop_idx}'] = 1 / np.maximum(harmonic_mix, 1e-10)
        
        # 4. Logarithmic mixing (Antoine-like)
        for prop_idx in range(1, 11):
            log_mix = 0
            for comp_idx in range(1, 6):
                prop_val = np.maximum(features[f'Component{comp_idx}_Property{prop_idx}'], 1e-10)
                log_mix += features[f'Component{comp_idx}_fraction'] * np.log(prop_val)
            features[f'Log_Mix_P{prop_idx}'] = np.exp(log_mix)
        
        return features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create component interaction features"""
        features = df.copy()
        
        # Component-component interactions
        for i in range(1, 5):
            for j in range(i+1, 6):
                # Fraction interaction
                features[f'Interaction_{i}_{j}'] = (features[f'Component{i}_fraction'] * 
                                                   features[f'Component{j}_fraction'])
                
                # Property difference indicators
                for prop_idx in range(1, 11):
                    prop_diff = np.abs(features[f'Component{i}_Property{prop_idx}'] - 
                                      features[f'Component{j}_Property{prop_idx}'])
                    features[f'PropDiff_{i}_{j}_P{prop_idx}'] = prop_diff
                    
                    # Weighted property difference
                    features[f'WeightedPropDiff_{i}_{j}_P{prop_idx}'] = (
                        prop_diff * features[f'Interaction_{i}_{j}']
                    )
        
        return features
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features across components and properties"""
        features = df.copy()
        
        # For each property across all components
        for prop_idx in range(1, 11):
            prop_values = [features[f'Component{i}_Property{prop_idx}'] for i in range(1, 6)]
            prop_matrix = np.column_stack(prop_values)
            
            # Statistical measures
            features[f'Property{prop_idx}_Mean'] = np.mean(prop_matrix, axis=1)
            features[f'Property{prop_idx}_Std'] = np.std(prop_matrix, axis=1)
            features[f'Property{prop_idx}_Min'] = np.min(prop_matrix, axis=1)
            features[f'Property{prop_idx}_Max'] = np.max(prop_matrix, axis=1)
            features[f'Property{prop_idx}_Range'] = features[f'Property{prop_idx}_Max'] - features[f'Property{prop_idx}_Min']
            features[f'Property{prop_idx}_Skew'] = stats.skew(prop_matrix, axis=1)
            features[f'Property{prop_idx}_Kurt'] = stats.kurtosis(prop_matrix, axis=1)
        
        # For each component across all properties
        for comp_idx in range(1, 6):
            comp_values = [features[f'Component{comp_idx}_Property{j}'] for j in range(1, 11)]
            comp_matrix = np.column_stack(comp_values)
            
            features[f'Component{comp_idx}_Mean'] = np.mean(comp_matrix, axis=1)
            features[f'Component{comp_idx}_Std'] = np.std(comp_matrix, axis=1)
            features[f'Component{comp_idx}_Min'] = np.min(comp_matrix, axis=1)
            features[f'Component{comp_idx}_Max'] = np.max(comp_matrix, axis=1)
        
        return features
    
    def create_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create entropy-based features for mixture complexity"""
        features = df.copy()
        
        # Shannon entropy of component fractions
        fractions = df[[f'Component{i}_fraction' for i in range(1, 6)]].values
        fractions = np.maximum(fractions, 1e-10)  # Avoid log(0)
        
        shannon_entropy = -np.sum(fractions * np.log(fractions), axis=1)
        features['Shannon_Entropy'] = shannon_entropy
        
        # Effective number of components
        features['Effective_Components'] = np.exp(shannon_entropy)
        
        # Dominant component fraction
        features['Dominant_Fraction'] = np.max(fractions, axis=1)
        features['Dominant_Component'] = np.argmax(fractions, axis=1) + 1
        
        return features
    
    def create_pca_features(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Create PCA features from all properties"""
        features = df.copy()
        
        # Get all property columns
        property_cols = []
        for i in range(1, 6):
            for j in range(1, 11):
                property_cols.append(f'Component{i}_Property{j}')
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_features = pca.fit_transform(features[property_cols])
        
        # Add PCA features
        for i in range(n_components):
            features[f'PCA_{i}'] = pca_features[:, i]
        
        return features
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering...")
        
        features = df.copy()
        
        # Apply all feature engineering steps
        features = self.create_mixing_features(features)
        features = self.create_interaction_features(features)
        features = self.create_statistical_features(features)
        features = self.create_entropy_features(features)
        features = self.create_pca_features(features)
        
        logger.info(f"Feature engineering complete. Shape: {features.shape}")
        return features

class ModelTrainer:
    """Advanced model training with multiple algorithms and stacking"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.cv_scores = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get base models with optimized hyperparameters"""
        models = {
            'lightgbm': lgb.LGBMRegressor(
                objective='regression',
                metric='mape',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_estimators=1000,
                random_state=self.random_state,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_estimators=1000,
                random_state=self.random_state,
                verbosity=0
            ),
            'catboost': cb.CatBoostRegressor(
                loss_function='MAPE',
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_state=self.random_state,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        return models
    
    def get_scalers(self) -> Dict[str, Any]:
        """Get different scalers for preprocessing"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson', standardize=True)
        }
        return scalers
    
    def train_single_target(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """Train models for a single target with cross-validation"""
        logger.info(f"Training models for {target_name}...")
        
        target_models = {}
        target_scalers = {}
        target_scores = {}
        
        # Cross-validation setup
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Get base models and scalers
        base_models = self.get_base_models()
        scalers = self.get_scalers()
        
        # Train each model with each scaler
        for scaler_name, scaler in scalers.items():
            for model_name, model in base_models.items():
                combo_name = f"{model_name}_{scaler_name}"
                
                # Scale features
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=kf, 
                    scoring='neg_mean_absolute_percentage_error',
                    n_jobs=-1
                )
                
                # Train final model
                model.fit(X_scaled, y)
                
                # Store results
                target_models[combo_name] = model
                target_scalers[combo_name] = scaler
                target_scores[combo_name] = -cv_scores.mean()
                
                logger.info(f"{combo_name}: CV MAPE = {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'models': target_models,
            'scalers': target_scalers,
            'scores': target_scores
        }
    
    def create_stacking_features(self, X: pd.DataFrame, y: pd.Series, models: Dict, scalers: Dict) -> np.ndarray:
        """Create stacking features using cross-validation"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        stacking_features = np.zeros((X.shape[0], len(models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for model_idx, (model_name, model) in enumerate(models.items()):
                # Get corresponding scaler
                scaler = scalers[model_name]
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                # Train and predict
                model.fit(X_train_scaled, y_train_fold)
                predictions = model.predict(X_val_scaled)
                
                # Store predictions
                stacking_features[val_idx, model_idx] = predictions
        
        return stacking_features
    
    def train_meta_model(self, stacking_features: np.ndarray, y: pd.Series) -> Any:
        """Train meta-model for stacking"""
        # Try Ridge and MLP, choose best
        models = {
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True
            )
        }
        
        best_model = None
        best_score = float('inf')
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            scores = cross_val_score(
                model, stacking_features, y,
                cv=kf, scoring='neg_mean_absolute_percentage_error'
            )
            score = -scores.mean()
            
            if score < best_score:
                best_score = score
                best_model = model
        
        # Train final meta-model
        best_model.fit(stacking_features, y)
        return best_model

class ShellAIPipeline:
    """Complete pipeline for Shell.ai Hackathon"""
    
    def __init__(self, data_path: str = ".", n_folds: int = 5, random_state: int = 42):
        self.data_path = Path(data_path)
        self.n_folds = n_folds
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(n_folds, random_state)
        self.trained_models = {}
        self.results = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        logger.info("Loading data...")
        
        try:
            train_df = pd.read_csv(self.data_path / "train.csv")
            test_df = pd.read_csv(self.data_path / "test.csv")
            
            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape: {test_df.shape}")
            
            return train_df, test_df
        except FileNotFoundError:
            logger.error("Data files not found. Please ensure train.csv and test.csv are in the data directory.")
            raise
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess and engineer features"""
        logger.info("Preprocessing data...")
        
        # Separate features and targets
        feature_cols = [col for col in train_df.columns if col not in ['ID'] + [f'BlendProperty{i}' for i in range(1, 11)]]
        target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        
        # Extract features
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[target_cols]
        
        # Feature engineering
        X_train_engineered = self.feature_engineer.engineer_all_features(X_train)
        X_test_engineered = self.feature_engineer.engineer_all_features(X_test)
        
        # Remove ID column if present
        if 'ID' in X_train_engineered.columns:
            X_train_engineered = X_train_engineered.drop('ID', axis=1)
        if 'ID' in X_test_engineered.columns:
            X_test_engineered = X_test_engineered.drop('ID', axis=1)
        
        # Handle missing values
        X_train_engineered = X_train_engineered.fillna(X_train_engineered.median())
        X_test_engineered = X_test_engineered.fillna(X_test_engineered.median())
        
        # Handle infinite values
        X_train_engineered = X_train_engineered.replace([np.inf, -np.inf], np.nan)
        X_test_engineered = X_test_engineered.replace([np.inf, -np.inf], np.nan)
        X_train_engineered = X_train_engineered.fillna(X_train_engineered.median())
        X_test_engineered = X_test_engineered.fillna(X_test_engineered.median())
        
        return X_train_engineered, X_test_engineered, y_train
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train models for all targets"""
        logger.info("Training models for all targets...")
        
        target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        
        for target in target_cols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {target}")
            logger.info(f"{'='*50}")
            
            # Train base models
            target_results = self.model_trainer.train_single_target(
                X_train, y_train[target], target
            )
            
            # Create stacking features
            stacking_features = self.model_trainer.create_stacking_features(
                X_train, y_train[target], 
                target_results['models'], 
                target_results['scalers']
            )
            
            # Train meta-model
            meta_model = self.model_trainer.train_meta_model(stacking_features, y_train[target])
            
            # Store results
            self.trained_models[target] = {
                'base_models': target_results['models'],
                'scalers': target_results['scalers'],
                'meta_model': meta_model,
                'scores': target_results['scores']
            }
    
    def make_predictions(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for test data"""
        logger.info("Making predictions...")
        
        predictions = {}
        target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        
        for target in target_cols:
            logger.info(f"Predicting {target}...")
            
            # Get models for this target
            base_models = self.trained_models[target]['base_models']
            scalers = self.trained_models[target]['scalers']
            meta_model = self.trained_models[target]['meta_model']
            
            # Generate base predictions
            base_predictions = np.zeros((X_test.shape[0], len(base_models)))
            
            for model_idx, (model_name, model) in enumerate(base_models.items()):
                scaler = scalers[model_name]
                X_test_scaled = scaler.transform(X_test)
                base_predictions[:, model_idx] = model.predict(X_test_scaled)
            
            # Meta-model predictions
            final_predictions = meta_model.predict(base_predictions)
            
            # Post-processing: clip extreme values
            y_train_target = pd.concat([pd.Series(self.trained_models[target]['scores'].values()) 
                                      for _ in range(len(final_predictions))], ignore_index=True)
            q1, q99 = np.percentile(y_train_target, [1, 99])
            final_predictions = np.clip(final_predictions, q1, q99)
            
            predictions[target] = final_predictions
        
        return pd.DataFrame(predictions)
    
    def evaluate_performance(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating performance...")
        
        results = {}
        mape_scores = []
        
        target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        
        for target in target_cols:
            mape = mean_absolute_percentage_error(y_true[target], y_pred[target])
            results[target] = mape
            mape_scores.append(mape)
            logger.info(f"{target}: MAPE = {mape:.4f}")
        
        # Summary statistics
        results['mean_mape'] = np.mean(mape_scores)
        results['median_mape'] = np.median(mape_scores)
        results['best_mape'] = np.min(mape_scores)
        results['worst_mape'] = np.max(mape_scores)
        
        logger.info(f"\n{'='*50}")
        logger.info("PERFORMANCE SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Mean MAPE: {results['mean_mape']:.4f}")
        logger.info(f"Median MAPE: {results['median_mape']:.4f}")
        logger.info(f"Best MAPE: {results['best_mape']:.4f}")
        logger.info(f"Worst MAPE: {results['worst_mape']:.4f}")
        
        return results
    
    def create_submission(self, test_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
        """Create submission file"""
        logger.info("Creating submission file...")
        
        submission = pd.DataFrame()
        submission['ID'] = test_df['ID']
        
        target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        for target in target_cols:
            submission[target] = predictions[target]
        
        submission.to_csv('submission.csv', index=False)
        logger.info("Submission file created: submission.csv")
    
    def save_models(self, filepath: str = "models.pkl") -> None:
        """Save trained models"""
        logger.info(f"Saving models to {filepath}...")
        joblib.dump(self.trained_models, filepath)
    
    def load_models(self, filepath: str = "models.pkl") -> None:
        """Load trained models"""
        logger.info(f"Loading models from {filepath}...")
        self.trained_models = joblib.load(filepath)
    
    def run_complete_pipeline(self, train_on_full_data: bool = True) -> Dict[str, float]:
        """Run the complete pipeline"""
        logger.info("Starting Shell.ai Hackathon Pipeline...")
        
        # Load data
        train_df, test_df = self.load_data()
        
        # Preprocess data
        X_train, X_test, y_train = self.preprocess_data(train_df, test_df)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Make predictions
        predictions = self.make_predictions(X_test)
        
        # Create submission
        self.create_submission(test_df, predictions)
        
        # Save models
        self.save_models()
        
        # If we have validation data, evaluate performance
        if not train_on_full_data:
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state
            )
            
            # Train on split data
            self.train_models(X_train_split, y_train_split)
            
            # Evaluate on validation set
            val_predictions = self.make_predictions(X_val_split)
            results = self.evaluate_performance(y_val_split, val_predictions)
            
            return results
        
        logger.info("Pipeline completed successfully!")
        return {"status": "completed"}

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = ShellAIPipeline(
        data_path=".",  # Adjust path as needed
        n_folds=5,
        random_state=42
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(train_on_full_data=True)
    
    logger.info("Shell.ai Hackathon Pipeline completed!")
    return results

if __name__ == "__main__":
    main()