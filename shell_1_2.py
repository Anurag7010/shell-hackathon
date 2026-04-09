import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MathematicalBlendOptimizer:
    """
    Advanced mathematical optimization for specific blend properties
    Focus on BlendProperty1 (MAPE: 2.7) and BlendProperty3 (MAPE: 1.0)
    """
    
    def __init__(self):
        self.target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        self.volume_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        self.critical_targets = ['BlendProperty1', 'BlendProperty3']
        
        # Mathematical mixing models
        self.mixing_models = {
            'linear': self._linear_mixing,
            'logarithmic': self._logarithmic_mixing,
            'exponential': self._exponential_mixing,
            'power_law': self._power_law_mixing,
            'quadratic': self._quadratic_mixing,
            'cubic': self._cubic_mixing,
            'exponential_quadratic': self._exponential_quadratic_mixing
        }
    
    def _linear_mixing(self, components, fractions):
        """Standard linear mixing rule"""
        return np.sum(components * fractions, axis=1)
    
    def _logarithmic_mixing(self, components, fractions):
        """Logarithmic mixing rule"""
        # Ensure positive values
        components_pos = np.maximum(components, 1e-10)
        log_sum = np.sum(fractions * np.log(components_pos), axis=1)
        return np.exp(log_sum)
    
    def _exponential_mixing(self, components, fractions):
        """Exponential mixing rule"""
        # Clip to prevent overflow
        components_clipped = np.clip(components, -5, 5)
        exp_sum = np.sum(fractions * np.exp(components_clipped), axis=1)
        return np.log(exp_sum + 1e-10)
    
    def _power_law_mixing(self, components, fractions):
        """Power law mixing rule"""
        # Ensure positive values
        components_pos = np.maximum(np.abs(components), 1e-10)
        power_sum = np.sum(fractions * (components_pos ** 2), axis=1)
        return np.sqrt(power_sum)
    
    def _quadratic_mixing(self, components, fractions):
        """Quadratic mixing rule"""
        linear_mix = self._linear_mixing(components, fractions)
        quadratic_term = np.sum(fractions * (components ** 2), axis=1)
        return linear_mix + 0.1 * quadratic_term
    
    def _cubic_mixing(self, components, fractions):
        """Cubic mixing rule"""
        linear_mix = self._linear_mixing(components, fractions)
        cubic_term = np.sum(fractions * (components ** 3), axis=1)
        return linear_mix + 0.01 * cubic_term
    
    def _exponential_quadratic_mixing(self, components, fractions):
        """Exponential-quadratic hybrid mixing"""
        exp_mix = self._exponential_mixing(components, fractions)
        quad_mix = self._quadratic_mixing(components, fractions)
        return 0.7 * exp_mix + 0.3 * quad_mix
    
    def engineer_mathematical_features(self, df, target_name):
        """Engineer mathematically-driven features"""
        df = df.copy()
        
        # Normalize fractions
        df[self.volume_cols] = df[self.volume_cols].div(df[self.volume_cols].sum(axis=1), axis=0)
        
        # Apply all mixing models for each property
        for property_idx in range(1, 11):
            component_props = np.array([df[f'Component{i}_Property{property_idx}'].values 
                                      for i in range(1, 6)]).T
            fractions = df[self.volume_cols].values
            
            for mixing_name, mixing_func in self.mixing_models.items():
                df[f'{mixing_name}_Property{property_idx}'] = mixing_func(component_props, fractions)
        
        # Target-specific advanced features
        if target_name == 'BlendProperty1':
            df = self._engineer_blendprop1_features(df)
        elif target_name == 'BlendProperty3':
            df = self._engineer_blendprop3_features(df)
        
        return df
    
    def _engineer_blendprop1_features(self, df):
        """Specific features for BlendProperty1 (highest MAPE)"""
        
        # Hypothesis: BlendProperty1 might follow complex nonlinear patterns
        
        # 1. Fraction-weighted property moments
        for moment in [2, 3, 4]:
            for prop in range(1, 11):
                weighted_moment = 0
                for i in range(1, 6):
                    component_val = df[f'Component{i}_Property{prop}']
                    fraction = df[f'Component{i}_fraction']
                    weighted_moment += fraction * (component_val ** moment)
                df[f'Moment{moment}_Property{prop}'] = weighted_moment
        
        # 2. Interaction complexity measures
        for i in range(1, 6):
            for j in range(i+1, 6):
                # Synergy/antagonism effects
                frac_i = df[f'Component{i}_fraction']
                frac_j = df[f'Component{j}_fraction']
                
                # Nonlinear interaction terms
                df[f'Synergy_{i}_{j}'] = frac_i * frac_j * np.log(frac_i + frac_j + 1e-10)
                df[f'Antagonism_{i}_{j}'] = (frac_i - frac_j) ** 2
                
                # Property-mediated interactions
                for prop in range(1, 6):  # First 5 properties
                    prop_i = df[f'Component{i}_Property{prop}']
                    prop_j = df[f'Component{j}_Property{prop}']
                    
                    # Cross-property effects
                    df[f'CrossProp_{i}_{j}_P{prop}'] = (prop_i * prop_j) * (frac_i * frac_j)
                    df[f'PropDiff_{i}_{j}_P{prop}'] = np.abs(prop_i - prop_j) * (frac_i + frac_j)
        
        # 3. Thermodynamic-inspired features
        # Entropy-like measures
        fractions = df[self.volume_cols].values
        df['mixing_entropy'] = -np.sum(fractions * np.log(fractions + 1e-10), axis=1)
        
        # Gibbs-like energy approximation
        for prop in range(1, 6):
            gibbs_approx = 0
            for i in range(1, 6):
                prop_val = df[f'Component{i}_Property{prop}']
                frac_val = df[f'Component{i}_fraction']
                gibbs_approx += frac_val * prop_val * np.log(frac_val + 1e-10)
            df[f'Gibbs_Property{prop}'] = gibbs_approx
        
        # 4. Fractal-like complexity measures
        df['complexity_index'] = 0
        for i in range(1, 6):
            frac = df[f'Component{i}_fraction']
            df['complexity_index'] += frac * np.log(frac + 1e-10) * np.log(np.log(frac + 1e-10) + 1e-10)
        
        return df
    
    def _engineer_blendprop3_features(self, df):
        """Specific features for BlendProperty3"""
        
        # Hypothesis: BlendProperty3 might depend on specific component interactions
        
        # 1. Selective property interactions
        # Focus on properties that might be most relevant for Property3
        key_properties = [1, 2, 3, 4, 5]
        
        for prop_a in key_properties:
            for prop_b in key_properties:
                if prop_a != prop_b:
                    # Weighted correlation-like features
                    corr_sum = 0
                    for i in range(1, 6):
                        val_a = df[f'Component{i}_Property{prop_a}']
                        val_b = df[f'Component{i}_Property{prop_b}']
                        frac = df[f'Component{i}_fraction']
                        corr_sum += frac * val_a * val_b
                    df[f'WeightedCorr_{prop_a}_{prop_b}'] = corr_sum
        
        # 2. Dominant component analysis
        # Find which component dominates and create features around it
        dominant_idx = df[self.volume_cols].idxmax(axis=1)
        df['dominant_component'] = dominant_idx.str.extract('(\d+)').astype(int)
        
        # Features based on dominant component
        for comp in range(1, 6):
            is_dominant = (df['dominant_component'] == comp).astype(int)
            df[f'is_dominant_{comp}'] = is_dominant
            
            # Dominant component properties
            for prop in range(1, 6):
                df[f'dominant_prop_{prop}'] = is_dominant * df[f'Component{comp}_Property{prop}']
        
        # 3. Minority component effects
        # Sometimes minority components have disproportionate effects
        df['minority_effect'] = 0
        for i in range(1, 6):
            frac = df[f'Component{i}_fraction']
            # Amplify effect of small fractions
            minority_weight = 1 / (frac + 0.01)  # Inverse weighting
            
            for prop in range(1, 4):  # First 3 properties
                prop_val = df[f'Component{i}_Property{prop}']
                df['minority_effect'] += minority_weight * frac * prop_val
        
        # 4. Threshold effects
        # Some properties might have threshold behaviors
        for prop in range(1, 6):
            prop_vals = df[f'linear_Property{prop}']  # Use linear mixing as baseline
            
            # Binary threshold features
            for threshold in [0.25, 0.5, 0.75]:
                df[f'above_threshold_{prop}_{threshold}'] = (prop_vals > np.quantile(prop_vals, threshold)).astype(int)
        
        return df
    
    def create_ensemble_with_optimization(self, X_train, y_train, X_test, target_name):
        """Create optimized ensemble for specific target"""
        
        kf = KFold(n_splits=7, shuffle=True, random_state=42)
        
        # Model configurations optimized for each target
        if target_name == 'BlendProperty1':
            models = {
                'huber': HuberRegressor(epsilon=1.1, alpha=0.001, max_iter=1000),
                'mlp_deep': MLPRegressor(
                    hidden_layer_sizes=(300, 150, 75),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                ),
                'xgb_specialized': xgb.XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.02,
                    max_depth=12,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.05,
                    reg_lambda=0.05,
                    random_state=42
                ),
                'lgb_specialized': lgb.LGBMRegressor(
                    n_estimators=2000,
                    learning_rate=0.02,
                    num_leaves=127,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    random_state=42,
                    verbose=-1
                )
            }
        
        elif target_name == 'BlendProperty3':
            models = {
                'gbm_tuned': GradientBoostingRegressor(
                    n_estimators=1500,
                    learning_rate=0.03,
                    max_depth=10,
                    subsample=0.8,
                    random_state=42
                ),
                'rf_deep': RandomForestRegressor(
                    n_estimators=1000,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=42
                ),
                'elastic_tuned': ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.3,
                    random_state=42
                ),
                'lgb_interaction': lgb.LGBMRegressor(
                    n_estimators=1500,
                    learning_rate=0.04,
                    num_leaves=63,
                    feature_fraction=0.9,
                    bagging_fraction=0.9,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1
                )
            }
        
        else:
            # Default ensemble for other targets
            models = {
                'lgb_default': lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
            }
        
        # Train ensemble
        oof_predictions = np.zeros((len(X_train), len(models)))
        test_predictions = np.zeros((len(X_test), len(models)))
        
        for model_idx, (model_name, model) in enumerate(models.items()):
            print(f"      Training {model_name}...")
            
            fold_test_preds = []
            fold_oof_preds = np.zeros(len(X_train))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_train_fold = X_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Handle neural network scaling
                if 'mlp' in model_name:
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
        
        # Optimized meta-model
        if target_name in ['BlendProperty1', 'BlendProperty3']:
            # Use more sophisticated meta-model for critical targets
            meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            meta_model = Ridge(alpha=1.0, random_state=42)
        
        meta_model.fit(oof_predictions, y_train)
        
        final_test_pred = meta_model.predict(test_predictions)
        final_oof_pred = meta_model.predict(oof_predictions)
        
        return final_test_pred, final_oof_pred
    
    def optimize_predictions(self, predictions, y_true, target_name):
        """Post-processing optimization"""
        
        def objective(params):
            # Apply transformation
            alpha, beta, gamma = params
            transformed = alpha * predictions + beta * (predictions ** 2) + gamma
            return mean_absolute_percentage_error(y_true, transformed)
        
        # Optimize transformation parameters
        result = minimize(
            objective,
            x0=[1.0, 0.0, 0.0],
            bounds=[(0.5, 1.5), (-0.1, 0.1), (-0.1, 0.1)],
            method='L-BFGS-B'
        )
        
        if result.success:
            alpha, beta, gamma = result.x
            optimized_preds = alpha * predictions + beta * (predictions ** 2) + gamma
            return optimized_preds
        else:
            return predictions
    
    def fit_and_predict(self, train_data, test_data, baseline_predictions=None):
        """Main optimization pipeline"""
        
        print("🧮 Starting Mathematical Optimization...")
        print("=" * 70)
        
        optimized_predictions = {}
        performance_metrics = {}
        
        for target in self.target_cols:
            print(f"\n🔬 Processing {target}...")
            
            # Extract features
            X_train = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
            y_train = train_data[target]
            X_test = test_data.drop(columns=['ID'], errors='ignore')
            
            # Apply mathematical feature engineering
            print(f"  🔧 Engineering mathematical features...")
            X_train_eng = self.engineer_mathematical_features(X_train, target)
            X_test_eng = self.engineer_mathematical_features(X_test, target)
            
            # Handle missing values
            X_train_eng = X_train_eng.fillna(X_train_eng.median())
            X_test_eng = X_test_eng.fillna(X_train_eng.median())
            
            # Feature selection for high-dimensional data
            if X_train_eng.shape[1] > 300:
                from sklearn.feature_selection import SelectKBest, f_regression
                selector = SelectKBest(f_regression, k=300)
                X_train_eng = pd.DataFrame(selector.fit_transform(X_train_eng, y_train))
                X_test_eng = pd.DataFrame(selector.transform(X_test_eng))
                print(f"  📊 Selected top 300 features")
            
            print(f"  🎯 Training optimized ensemble...")
            # Train specialized ensemble
            test_pred, oof_pred = self.create_ensemble_with_optimization(
                X_train_eng, y_train, X_test_eng, target
            )
            
            # Post-processing optimization for critical targets
            if target in self.critical_targets:
                print(f"  ⚡ Applying post-processing optimization...")
                test_pred = self.optimize_predictions(test_pred, y_train, target)
                oof_pred = self.optimize_predictions(oof_pred, y_train, target)
            
            # Calculate performance
            oof_mape = mean_absolute_percentage_error(y_train, oof_pred)
            performance_metrics[target] = oof_mape
            optimized_predictions[target] = test_pred
            
            improvement = ""
            if target == 'BlendProperty1':
                improvement = f" (Target: <0.8, Previous: 2.7)"
            elif target == 'BlendProperty3':
                improvement = f" (Target: <0.8, Previous: 1.0)"
            
            print(f"  📈 {target} MAPE: {oof_mape:.4f}{improvement}")
            
            # Decision logic: use optimized if significantly better
            if baseline_predictions is not None and target in baseline_predictions:
                # Use baseline for well-performing targets, optimized for poor ones
                if target in ['BlendProperty2', 'BlendProperty4', 'BlendProperty5', 'BlendProperty6', 'BlendProperty10']:
                    optimized_predictions[target] = baseline_predictions[target]
                    print(f"  ✅ Using baseline prediction for well-performing {target}")
        
        print("\n" + "=" * 70)
        print("🎯 MATHEMATICAL OPTIMIZATION COMPLETE")
        print("=" * 70)
        
        for target, mape in performance_metrics.items():
            status = "🔥" if mape < 0.5 else "✅" if mape < 0.8 else "📈" if mape < 1.2 else "⚠️"
            print(f"  {status} {target}: {mape:.4f}")
        
        avg_mape = np.mean(list(performance_metrics.values()))
        print(f"\n📊 Average MAPE: {avg_mape:.4f}")
        
        expected_score = 100 * (1 - avg_mape)
        print(f"🎲 Expected Score: {expected_score:.1f}%")
        
        return optimized_predictions, performance_metrics

# Integration script
def integrate_with_existing_pipeline(train_path, test_path, baseline_submission_path=None):
    """
    Integrate mathematical optimization with existing pipeline
    """
    
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Load baseline predictions if available
    baseline_predictions = None
    if baseline_submission_path:
        baseline_df = pd.read_csv(baseline_submission_path)
        baseline_predictions = {col: baseline_df[col].values for col in baseline_df.columns if col != 'ID'}
    
    # Run mathematical optimization
    optimizer = MathematicalBlendOptimizer()
    predictions, metrics = optimizer.fit_and_predict(train, test, baseline_predictions)
    
    # Create final submission
    submission = pd.DataFrame()
    submission['ID'] = range(1, len(test) + 1)
    
    for target in optimizer.target_cols:
        submission[target] = predictions[target]
    
    submission.to_csv('submission_mathematically_optimized.csv', index=False)
    print("\n✅ Mathematically optimized submission saved!")
    
    return submission, metrics

# Usage example
if __name__ == "__main__":
    # Run the optimization
    submission, metrics = integrate_with_existing_pipeline(
        'train.csv', 
        'test.csv',
        'submission_stacked_optimized.csv'  # Your baseline submission
    )
    
    print(f"\n📋 Final submission shape: {submission.shape}")
    print(f"📊 Performance metrics: {metrics}")