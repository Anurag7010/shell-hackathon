import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None

import warnings
warnings.filterwarnings('ignore')

class TargetWiseOptimizer:
    """
    Advanced target-wise optimization system for fuel blend prediction
    Focus on improving high-MAPE targets while preserving good ones
    """
    
    def __init__(self, error_threshold=0.6):
        self.error_threshold = error_threshold
        self.target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        self.volume_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        self.baseline_models = {}
        self.specialized_models = {}
        self.target_strategies = {}
        
        # Define target-specific strategies based on current performance
        self.target_specific_configs = {
            'BlendProperty1': {
                'strategy': 'nonlinear_heavy',
                'needs_improvement': True,
                'target_mape': 0.8,
                'focus': 'extreme_nonlinearity'
            },
            'BlendProperty3': {
                'strategy': 'interaction_heavy', 
                'needs_improvement': True,
                'target_mape': 0.8,
                'focus': 'complex_interactions'
            },
            'BlendProperty7': {
                'strategy': 'ensemble_boost',
                'needs_improvement': True,
                'target_mape': 0.6,
                'focus': 'ensemble_diversity'
            },
            'BlendProperty8': {
                'strategy': 'ensemble_boost',
                'needs_improvement': True,
                'target_mape': 0.6,
                'focus': 'ensemble_diversity'
            },
            'BlendProperty9': {
                'strategy': 'ensemble_boost',
                'needs_improvement': True,
                'target_mape': 0.6,
                'focus': 'ensemble_diversity'
            },
            # Keep good performing targets unchanged
            'BlendProperty2': {'strategy': 'preserve', 'needs_improvement': False},
            'BlendProperty4': {'strategy': 'preserve', 'needs_improvement': False},
            'BlendProperty5': {'strategy': 'preserve', 'needs_improvement': False},
            'BlendProperty6': {'strategy': 'preserve', 'needs_improvement': False},
            'BlendProperty10': {'strategy': 'preserve', 'needs_improvement': False},
        }
    
    def engineer_target_specific_features(self, df, target_name):
        """Engineer features specifically for each target"""
        df = df.copy()
        
        strategy = self.target_specific_configs[target_name]['strategy']
        
        if strategy == 'preserve':
            # Use standard features for well-performing targets
            return self.engineer_standard_features(df)
        
        elif strategy == 'nonlinear_heavy':
            # Extreme nonlinear features for BlendProperty1
            df = self.engineer_extreme_nonlinear_features(df)
            
        elif strategy == 'interaction_heavy':
            # Complex interaction features for BlendProperty3
            df = self.engineer_complex_interactions(df)
            
        elif strategy == 'ensemble_boost':
            # Enhanced ensemble features for moderately high MAPE targets
            df = self.engineer_ensemble_boost_features(df)
        
        return df
    
    def engineer_standard_features(self, df):
        """Standard feature engineering for well-performing targets"""
        df = df.copy()
        
        # 1. Normalize volume fractions
        df[self.volume_cols] = df[self.volume_cols].div(df[self.volume_cols].sum(axis=1), axis=0)
        
        # 2. Linear mixing rule
        for p in range(1, 11):
            df[f'Weighted_Property{p}'] = 0
            for i in range(1, 6):
                df[f'Weighted_Property{p}'] += df[f'Component{i}_Property{p}'] * df[f'Component{i}_fraction']
        
        # 3. Basic nonlinear mixing
        for p in range(1, 11):
            geometric_mean = 1
            for i in range(1, 6):
                prop_val = df[f'Component{i}_Property{p}']
                prop_val = np.maximum(prop_val, 1e-10)
                geometric_mean *= np.power(prop_val, df[f'Component{i}_fraction'])
            df[f'Geometric_Property{p}'] = geometric_mean
        
        return df
    
    def engineer_extreme_nonlinear_features(self, df):
        """Extreme nonlinear features for BlendProperty1"""
        df = self.engineer_standard_features(df)
        
        # 1. Higher-order polynomial interactions
        for p in range(1, 11):
            df[f'Weighted_Property{p}_cubed'] = df[f'Weighted_Property{p}'] ** 3
            df[f'Weighted_Property{p}_fourth'] = df[f'Weighted_Property{p}'] ** 4
            df[f'Weighted_Property{p}_log'] = np.log(np.abs(df[f'Weighted_Property{p}']) + 1)
            df[f'Weighted_Property{p}_exp'] = np.exp(np.clip(df[f'Weighted_Property{p}'], -5, 5))
        
        # 2. Trigonometric transformations
        for p in range(1, 6):
            df[f'Sin_Property{p}'] = np.sin(df[f'Weighted_Property{p}'])
            df[f'Cos_Property{p}'] = np.cos(df[f'Weighted_Property{p}'])
        
        # 3. Extreme mixing rules
        for p in range(1, 11):
            # Power law mixing
            power_sum = 0
            for i in range(1, 6):
                prop_val = df[f'Component{i}_Property{p}']
                power_sum += df[f'Component{i}_fraction'] * (prop_val ** 2)
            df[f'PowerLaw_Property{p}'] = np.sqrt(power_sum)
            
            # Exponential mixing
            exp_sum = 0
            for i in range(1, 6):
                prop_val = df[f'Component{i}_Property{p}']
                exp_sum += df[f'Component{i}_fraction'] * np.exp(np.clip(prop_val, -5, 5))
            df[f'ExpMix_Property{p}'] = np.log(exp_sum + 1e-10)
        
        # 4. Fraction-based nonlinear terms
        df['fraction_product'] = df[self.volume_cols].prod(axis=1)
        df['fraction_sum_squares'] = (df[self.volume_cols] ** 2).sum(axis=1)
        
        return df
    
    def engineer_complex_interactions(self, df):
        """Complex interaction features for BlendProperty3"""
        df = self.engineer_standard_features(df)
        
        # 1. All pairwise property interactions
        for p1 in range(1, 11):
            for p2 in range(p1+1, 11):
                df[f'PropPair_{p1}_{p2}'] = df[f'Weighted_Property{p1}'] * df[f'Weighted_Property{p2}']
                df[f'PropRatio_{p1}_{p2}'] = df[f'Weighted_Property{p1}'] / (df[f'Weighted_Property{p2}'] + 1e-10)
        
        # 2. Three-way interactions for key properties
        for p1 in range(1, 4):
            for p2 in range(p1+1, 5):
                for p3 in range(p2+1, 6):
                    df[f'Triple_{p1}_{p2}_{p3}'] = (df[f'Weighted_Property{p1}'] * 
                                                   df[f'Weighted_Property{p2}'] * 
                                                   df[f'Weighted_Property{p3}'])
        
        # 3. Component-property cross interactions
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j:
                    for p in range(1, 6):
                        df[f'CrossInt_{i}_{j}_P{p}'] = (df[f'Component{i}_fraction'] * 
                                                       df[f'Component{j}_Property{p}'])
        
        # 4. Clustering-based features
        prop_cols = [f'Weighted_Property{p}' for p in range(1, 11)]
        if len(prop_cols) > 0:
            # Create property clusters
            kmeans = KMeans(n_clusters=5, random_state=42)
            df['property_cluster'] = kmeans.fit_predict(df[prop_cols])
            
            # Cluster-based features
            for cluster in range(5):
                df[f'is_cluster_{cluster}'] = (df['property_cluster'] == cluster).astype(int)
        
        return df
    
    def engineer_ensemble_boost_features(self, df):
        """Enhanced features for ensemble boosting"""
        df = self.engineer_standard_features(df)
        
        # 1. Advanced statistical features
        prop_cols = [col for col in df.columns if 'Weighted_Property' in col]
        if len(prop_cols) > 0:
            df['prop_mean'] = df[prop_cols].mean(axis=1)
            df['prop_std'] = df[prop_cols].std(axis=1)
            df['prop_skew'] = df[prop_cols].skew(axis=1)
            df['prop_kurtosis'] = df[prop_cols].kurtosis(axis=1)
        
        # 2. Quantile-based features
        for q in [0.25, 0.5, 0.75]:
            df[f'prop_quantile_{q}'] = df[prop_cols].quantile(q, axis=1)
        
        # 3. Pairwise component interactions
        for i in range(1, 6):
            for j in range(i+1, 6):
                df[f'CompInt_{i}_{j}'] = df[f'Component{i}_fraction'] * df[f'Component{j}_fraction']
        
        return df
    
    def get_target_specific_models(self, target_name):
        """Get model configurations specific to each target"""
        
        strategy = self.target_specific_configs[target_name]['strategy']
        
        if strategy == 'preserve':
            # Keep successful models for well-performing targets
            return {
                'lgb': {
                    'model': lgb.LGBMRegressor,
                    'params': {
                        'n_estimators': 1000,
                        'learning_rate': 0.05,
                        'num_leaves': 31,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'random_state': 42,
                        'verbose': -1
                    }
                }
            }
        
        elif strategy == 'nonlinear_heavy':
            # Nonlinear models for BlendProperty1
            return {
                'mlp': {
                    'model': MLPRegressor,
                    'params': {
                        'hidden_layer_sizes': (200, 100, 50),
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.01,
                        'learning_rate': 'adaptive',
                        'max_iter': 1000,
                        'random_state': 42
                    }
                },
                'xgb_deep': {
                    'model': xgb.XGBRegressor,
                    'params': {
                        'n_estimators': 1500,
                        'learning_rate': 0.03,
                        'max_depth': 10,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'random_state': 42
                    }
                },
                'huber': {
                    'model': HuberRegressor,
                    'params': {
                        'epsilon': 1.35,
                        'alpha': 0.01,
                        'max_iter': 1000
                    }
                }
            }
        
        elif strategy == 'interaction_heavy':
            # Interaction-focused models for BlendProperty3
            return {
                'gbm': {
                    'model': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': 1000,
                        'learning_rate': 0.05,
                        'max_depth': 8,
                        'subsample': 0.8,
                        'random_state': 42
                    }
                },
                'rf_deep': {
                    'model': RandomForestRegressor,
                    'params': {
                        'n_estimators': 800,
                        'max_depth': 20,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'random_state': 42
                    }
                },
                'knn': {
                    'model': KNeighborsRegressor,
                    'params': {
                        'n_neighbors': 15,
                        'weights': 'distance',
                        'algorithm': 'auto'
                    }
                }
            }
        
        elif strategy == 'ensemble_boost':
            # Diverse ensemble for moderate MAPE targets
            return {
                'lgb_tuned': {
                    'model': lgb.LGBMRegressor,
                    'params': {
                        'n_estimators': 1500,
                        'learning_rate': 0.04,
                        'num_leaves': 63,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.9,
                        'reg_alpha': 0.05,
                        'reg_lambda': 0.05,
                        'random_state': 42,
                        'verbose': -1
                    }
                },
                'et_tuned': {
                    'model': ExtraTreesRegressor,
                    'params': {
                        'n_estimators': 600,
                        'max_depth': 18,
                        'min_samples_split': 3,
                        'min_samples_leaf': 1,
                        'max_features': 0.8,
                        'random_state': 42
                    }
                },
                'elastic': {
                    'model': ElasticNet,
                    'params': {
                        'alpha': 0.1,
                        'l1_ratio': 0.5,
                        'random_state': 42
                    }
                }
            }
    
    def asymmetric_loss_objective(self, y_true, y_pred):
        """Asymmetric loss function for better handling of outliers"""
        residual = y_true - y_pred
        return np.where(residual > 0, 2 * residual, 0.5 * np.abs(residual))
    
    def train_target_specific_model(self, X_train, y_train, X_test, target_name):
        """Train specialized model for specific target"""
        
        models = self.get_target_specific_models(target_name)
        kf = KFold(n_splits=7, shuffle=True, random_state=42)
        
        # Store predictions for ensemble
        oof_predictions = np.zeros((len(X_train), len(models)))
        test_predictions = np.zeros((len(X_test), len(models)))
        
        for model_idx, (model_name, config) in enumerate(models.items()):
            print(f"    Training {model_name} for {target_name}...")
            
            fold_test_preds = []
            fold_oof_preds = np.zeros(len(X_train))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_train_fold = X_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Train model
                model = config['model'](**config['params'])
                
                if 'mlp' in model_name:
                    # Scale features for neural network
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_fold)
                    X_val_scaled = scaler.transform(X_val_fold)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train_fold)
                    val_pred = model.predict(X_val_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train_fold, y_train_fold)
                    val_pred = model.predict(X_val_fold)
                    test_pred = model.predict(X_test)
                
                fold_oof_preds[val_idx] = val_pred
                fold_test_preds.append(test_pred)
            
            oof_predictions[:, model_idx] = fold_oof_preds
            test_predictions[:, model_idx] = np.mean(fold_test_preds, axis=0)
        
        # Meta-model ensemble
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(oof_predictions, y_train)
        
        final_test_pred = meta_model.predict(test_predictions)
        final_oof_pred = meta_model.predict(oof_predictions)
        
        return final_test_pred, final_oof_pred, meta_model
    
    def post_process_predictions(self, predictions, target_name):
        """Post-process predictions based on target-specific requirements"""
        
        strategy = self.target_specific_configs[target_name]['strategy']
        
        if strategy == 'nonlinear_heavy':
            # Smooth extreme predictions for BlendProperty1
            predictions = np.clip(predictions, 
                                np.percentile(predictions, 1), 
                                np.percentile(predictions, 99))
        
        elif strategy == 'interaction_heavy':
            # Apply median filtering for BlendProperty3
            from scipy.ndimage import median_filter
            predictions = median_filter(predictions, size=3)
        
        return predictions
    
    def fit_and_predict(self, train_data, test_data):
        """Main training and prediction pipeline"""
        
        print("🎯 Starting Target-Wise Optimization...")
        print("=" * 70)
        
        final_predictions = {}
        performance_metrics = {}
        
        for target in self.target_cols:
            print(f"\n🔧 Processing {target}...")
            
            config = self.target_specific_configs[target]
            
            if not config['needs_improvement']:
                print(f"  ✅ {target} performing well, using baseline model...")
                # Use your existing good model for this target
                # This is a placeholder - use your existing trained model
                X_train = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
                y_train = train_data[target]
                X_test = test_data.drop(columns=['ID'], errors='ignore')
                
                # Apply standard feature engineering
                X_train_eng = self.engineer_target_specific_features(X_train, target)
                X_test_eng = self.engineer_target_specific_features(X_test, target)
                
                # Use simple model for preserved targets
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)
                model.fit(X_train_eng, y_train)
                predictions = model.predict(X_test_eng)
                
                final_predictions[target] = predictions
                performance_metrics[target] = "PRESERVED"
                
            else:
                print(f"  🚀 {target} needs improvement, applying specialized optimization...")
                
                # Extract base features
                X_train_base = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
                y_train = train_data[target]
                X_test_base = test_data.drop(columns=['ID'], errors='ignore')
                
                # Apply target-specific feature engineering
                X_train_eng = self.engineer_target_specific_features(X_train_base, target)
                X_test_eng = self.engineer_target_specific_features(X_test_base, target)
                
                # Handle missing values
                X_train_eng = X_train_eng.fillna(X_train_eng.median())
                X_test_eng = X_test_eng.fillna(X_train_eng.median())
                
                # Feature selection for high-dimensional cases
                if X_train_eng.shape[1] > 200:
                    selector = SelectKBest(f_regression, k=200)
                    X_train_eng = pd.DataFrame(selector.fit_transform(X_train_eng, y_train))
                    X_test_eng = pd.DataFrame(selector.transform(X_test_eng))
                
                # Train specialized model
                predictions, oof_preds, meta_model = self.train_target_specific_model(
                    X_train_eng, y_train, X_test_eng, target
                )
                
                # Post-process predictions
                predictions = self.post_process_predictions(predictions, target)
                
                # Calculate improvement
                oof_mape = mean_absolute_percentage_error(y_train, oof_preds)
                performance_metrics[target] = f"MAPE: {oof_mape:.4f}"
                
                final_predictions[target] = predictions
                
                print(f"  📊 {target} optimized MAPE: {oof_mape:.4f}")
        
        print("\n" + "=" * 70)
        print("📈 TARGET-WISE OPTIMIZATION COMPLETE")
        print("=" * 70)
        
        for target, metric in performance_metrics.items():
            status = "✅" if metric == "PRESERVED" else "🔧"
            print(f"  {status} {target}: {metric}")
        
        return final_predictions, performance_metrics

# Usage example
def main():
    # Load your data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Initialize optimizer
    optimizer = TargetWiseOptimizer(error_threshold=0.6)
    
    # Fit and predict
    predictions, metrics = optimizer.fit_and_predict(train, test)
    
    # Create submission
    submission = pd.DataFrame()
    submission['ID'] = range(1, len(test) + 1)
    
    for target in optimizer.target_cols:
        submission[target] = predictions[target]
    
    submission.to_csv('submission_target_optimized.csv', index=False)
    print("\n✅ Optimized submission saved!")
    
    return submission, metrics

if __name__ == "__main__":
    submission, metrics = main()