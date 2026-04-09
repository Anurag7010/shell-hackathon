import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class IntegratedOptimizationPipeline:
    """
    Integrated optimization pipeline that automatically selects the best approach
    for each target based on performance thresholds
    """
    
    def __init__(self, error_threshold=0.6, improvement_threshold=0.1):
        self.error_threshold = error_threshold
        self.improvement_threshold = improvement_threshold
        self.target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
        
        # Current known performance (from your results)
        self.current_performance = {
            'BlendProperty1': 2.7061,
            'BlendProperty2': 0.5233,
            'BlendProperty3': 1.0361,
            'BlendProperty4': 0.6371,
            'BlendProperty5': 0.2685,
            'BlendProperty6': 0.3255,
            'BlendProperty7': 0.8526,
            'BlendProperty8': 0.8993,
            'BlendProperty9': 0.8549,
            'BlendProperty10': 0.5081
        }
        
        # Strategy assignment
        self.strategy_map = self._assign_strategies()
    
    def _assign_strategies(self):
        """Assign optimization strategies based on current performance"""
        strategies = {}
        
        for target, current_mape in self.current_performance.items():
            if current_mape < self.error_threshold:
                strategies[target] = 'preserve'
            elif current_mape > 2.0:
                strategies[target] = 'aggressive_math'
            elif current_mape > 1.0:
                strategies[target] = 'moderate_math'
            else:
                strategies[target] = 'ensemble_boost'
        
        return strategies
    
    def run_baseline_model(self, train_data, test_data, target):
        """Run your existing baseline model (simplified version)"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        X_train = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
        y_train = train_data[target]
        X_test = test_data.drop(columns=['ID'], errors='ignore')
        
        # Simple preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train baseline model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        test_pred = model.predict(X_test_scaled)
        train_pred = model.predict(X_train_scaled)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_train, train_pred)
        
        return test_pred, mape
    
    def run_target_wise_optimization(self, train_data, test_data, target):
        """Run target-wise optimization (simplified version)"""
        
        from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        
        X_train = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
        y_train = train_data[target]
        X_test = test_data.drop(columns=['ID'], errors='ignore')
        
        # Enhanced feature engineering based on target
        X_train_eng = self._engineer_features_for_target(X_train, target)
        X_test_eng = self._engineer_features_for_target(X_test, target)
        
        # Handle missing values
        X_train_eng = X_train_eng.fillna(X_train_eng.median())
        X_test_eng = X_test_eng.fillna(X_train_eng.median())
        
        # Select model based on target
        if target == 'BlendProperty1':
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_eng)
            X_test_scaled = scaler.transform(X_test_eng)
        elif target == 'BlendProperty3':
            model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
            X_train_scaled = X_train_eng
            X_test_scaled = X_test_eng
        else:
            model = ExtraTreesRegressor(n_estimators=200, random_state=42)
            X_train_scaled = X_train_eng
            X_test_scaled = X_test_eng
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        test_pred = model.predict(X_test_scaled)
        train_pred = model.predict(X_train_scaled)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_train, train_pred)
        
        return test_pred, mape
    
    def run_mathematical_optimization(self, train_data, test_data, target):
        """Run mathematical optimization (simplified version)"""
        
        from sklearn.linear_model import HuberRegressor
        from sklearn.preprocessing import PolynomialFeatures
        
        X_train = train_data.drop(columns=self.target_cols + ['ID'], errors='ignore')
        y_train = train_data[target]
        X_test = test_data.drop(columns=['ID'], errors='ignore')
        
        # Mathematical feature engineering
        X_train_math = self._engineer_mathematical_features(X_train, target)
        X_test_math = self._engineer_mathematical_features(X_test, target)
        
        # Handle missing values
        X_train_math = X_train_math.fillna(X_train_math.median())
        X_test_math = X_test_math.fillna(X_train_math.median())
        
        # Use robust model for mathematical optimization
        model = HuberRegressor(epsilon=1.1, alpha=0.001, max_iter=1000)
        model.fit(X_train_math, y_train)
        
        test_pred = model.predict(X_test_math)
        train_pred = model.predict(X_train_math)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_train, train_pred)
        
        return test_pred, mape
    
    def _engineer_features_for_target(self, df, target):
        """Basic feature engineering for target-wise optimization"""
        df = df.copy()
        
        # Volume fraction normalization
        volume_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        df[volume_cols] = df[volume_cols].div(df[volume_cols].sum(axis=1), axis=0)
        
        # Linear mixing for all properties
        for p in range(1, 11):
            df[f'Linear_Mix_P{p}'] = 0
            for i in range(1, 6):
                df[f'Linear_Mix_P{p}'] += df[f'Component{i}_Property{p}'] * df[f'Component{i}_fraction']
        
        # Target-specific features
        if target == 'BlendProperty1':
            # Nonlinear terms
            for p in range(1, 6):
                df[f'Squared_P{p}'] = df[f'Linear_Mix_P{p}'] ** 2
                df[f'Cubed_P{p}'] = df[f'Linear_Mix_P{p}'] ** 3
        
        elif target == 'BlendProperty3':
            # Interaction terms
            for i in range(1, 6):
                for j in range(i+1, 6):
                    df[f'Interaction_{i}_{j}'] = df[f'Component{i}_fraction'] * df[f'Component{j}_fraction']
        
        return df
    
    def _engineer_mathematical_features(self, df, target):
        """Mathematical feature engineering"""
        df = df.copy()
        
        # Volume fraction normalization
        volume_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        df[volume_cols] = df[volume_cols].div(df[volume_cols].sum(axis=1), axis=0)
        
        # Mathematical mixing rules
        for p in range(1, 11):
            # Linear mixing
            linear_mix = 0
            geometric_mix = 1
            harmonic_mix = 0
            
            for i in range(1, 6):
                prop_val = df[f'Component{i}_Property{p}']
                frac_val = df[f'Component{i}_fraction']
                
                # Linear
                linear_mix += prop_val * frac_val
                
                # Geometric
                geometric_mix *= (prop_val + 1e-10) ** frac_val
                
                # Harmonic
                harmonic_mix += frac_val / (prop_val + 1e-10)
            
            df[f'Linear_P{p}'] = linear_mix
            df[f'Geometric_P{p}'] = geometric_mix
            df[f'Harmonic_P{p}'] = 1 / (harmonic_mix + 1e-10)
        
        # Advanced mathematical features for critical targets
        if target in ['BlendProperty1', 'BlendProperty3']:
            # Entropy-like features
            fractions = df[volume_cols].values
            df['entropy'] = -np.sum(fractions * np.log(fractions + 1e-10), axis=1)
            
            # Moment-based features
            for p in range(1, 6):
                prop_mean = df[f'Linear_P{p}']
                df[f'Moment2_P{p}'] = prop_mean ** 2
                df[f'Moment3_P{p}'] = prop_mean ** 3
        
        return df
    
    def evaluate_and_select_best_model(self, train_data, test_data, target):
        """Evaluate all applicable models and select the best one"""
        
        print(f"    🔍 Evaluating models for {target}...")
        
        results = {}
        strategy = self.strategy_map[target]
        
        # Always run baseline
        baseline_pred, baseline_mape = self.run_baseline_model(train_data, test_data, target)
        results['baseline'] = {'pred': baseline_pred, 'mape': baseline_mape}
        
        # Run optimization based on strategy
        if strategy == 'preserve':
            print(f"    ✅ {target} performing well, using baseline")
            return baseline_pred, baseline_mape, 'baseline'
        
        elif strategy == 'aggressive_math':
            print(f"    🧮 Running aggressive mathematical optimization...")
            math_pred, math_mape = self.run_mathematical_optimization(train_data, test_data, target)
            results['mathematical'] = {'pred': math_pred, 'mape': math_mape}
            
            target_pred, target_mape = self.run_target_wise_optimization(train_data, test_data, target)
            results['target_wise'] = {'pred': target_pred, 'mape': target_mape}
        
        elif strategy == 'moderate_math':
            print(f"    🔧 Running moderate optimization...")
            math_pred, math_mape = self.run_mathematical_optimization(train_data, test_data, target)
            results['mathematical'] = {'pred': math_pred, 'mape': math_mape}
        
        elif strategy == 'ensemble_boost':
            print(f"    🚀 Running ensemble boosting...")
            target_pred, target_mape = self.run_target_wise_optimization(train_data, test_data, target)
            results['target_wise'] = {'pred': target_pred, 'mape': target_mape}
        
        # Select best model
        best_model = min(results.keys(), key=lambda x: results[x]['mape'])
        best_pred = results[best_model]['pred']
        best_mape = results[best_model]['mape']
        
        # Check if improvement is significant
        improvement = (baseline_mape - best_mape) / baseline_mape
        if improvement < self.improvement_threshold:
            print(f"    ⚠️  No significant improvement, using baseline")
            return baseline_pred, baseline_mape, 'baseline'
        
        print(f"    ✅ Best model: {best_model} (MAPE: {best_mape:.4f}, Improvement: {improvement:.2%})")
        return best_pred, best_mape, best_model
    
    def run_complete_pipeline(self, train_data, test_data):
        """Run the complete integrated optimization pipeline"""
        
        print("🚀 Starting Integrated Optimization Pipeline")
        print("=" * 60)
        
        # Print strategy assignment
        print("\n📋 Strategy Assignment:")
        for target, strategy in self.strategy_map.items():
            current_mape = self.current_performance[target]
            print(f"  {target}: {strategy} (Current MAPE: {current_mape:.4f})")
        
        print("\n" + "=" * 60)
        print("🔄 Running Optimizations...")
        
        # Store results
        final_predictions = {}
        performance_report = {}
        
        # Process each target
        for target in self.target_cols:
            print(f"\n🎯 Processing {target}:")
            
            try:
                pred, mape, model_type = self.evaluate_and_select_best_model(
                    train_data, test_data, target
                )
                
                final_predictions[target] = pred
                performance_report[target] = {
                    'mape': mape,
                    'model_type': model_type,
                    'strategy': self.strategy_map[target],
                    'current_mape': self.current_performance[target],
                    'improvement': (self.current_performance[target] - mape) / self.current_performance[target]
                }
                
            except Exception as e:
                print(f"    ❌ Error processing {target}: {str(e)}")
                # Fallback to baseline
                pred, mape = self.run_baseline_model(train_data, test_data, target)
                final_predictions[target] = pred
                performance_report[target] = {
                    'mape': mape,
                    'model_type': 'baseline_fallback',
                    'strategy': 'fallback',
                    'current_mape': self.current_performance[target],
                    'improvement': (self.current_performance[target] - mape) / self.current_performance[target]
                }
        
        # Generate final results
        print("\n" + "=" * 60)
        print("📊 Final Performance Report:")
        print("=" * 60)
        
        total_improvement = 0
        improved_targets = 0
        
        for target, report in performance_report.items():
            improvement = report['improvement']
            if improvement > 0:
                improved_targets += 1
                total_improvement += improvement
                status = "✅ IMPROVED"
            else:
                status = "➡️  MAINTAINED"
            
            print(f"{target}:")
            print(f"  Model: {report['model_type']}")
            print(f"  Strategy: {report['strategy']}")
            print(f"  MAPE: {report['current_mape']:.4f} → {report['mape']:.4f}")
            print(f"  Improvement: {improvement:.2%} {status}")
            print()
        
        print(f"📈 Summary:")
        print(f"  Targets improved: {improved_targets}/{len(self.target_cols)}")
        print(f"  Average improvement: {total_improvement/len(self.target_cols):.2%}")
        
        # Create submission dataframe
        submission_df = test_data[['ID']].copy()
        for target in self.target_cols:
            submission_df[target] = final_predictions[target]
        
        return submission_df, performance_report
    
    def create_ensemble_predictions(self, train_data, test_data, target, models_to_ensemble):
        """Create ensemble predictions from multiple models"""
        
        predictions = []
        weights = []
        
        for model_name in models_to_ensemble:
            if model_name == 'baseline':
                pred, mape = self.run_baseline_model(train_data, test_data, target)
            elif model_name == 'target_wise':
                pred, mape = self.run_target_wise_optimization(train_data, test_data, target)
            elif model_name == 'mathematical':
                pred, mape = self.run_mathematical_optimization(train_data, test_data, target)
            
            predictions.append(pred)
            # Weight inversely proportional to MAPE
            weights.append(1 / (mape + 1e-10))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        
        return ensemble_pred

# Example usage function
def run_optimization_pipeline(train_path, test_path):
    """
    Main function to run the complete optimization pipeline
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
    
    Returns:
        submission_df: DataFrame with predictions
        performance_report: Dictionary with performance metrics
    """
    
    # Load data
    print("📂 Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Initialize pipeline
    pipeline = IntegratedOptimizationPipeline(
        error_threshold=0.6,
        improvement_threshold=0.05
    )
    
    # Run pipeline
    submission_df, performance_report = pipeline.run_complete_pipeline(
        train_data, test_data
    )
    
    # Save results
    submission_df.to_csv('integrated_optimization_submission.csv', index=False)
    print(f"\n💾 Submission saved to: integrated_optimization_submission.csv")
    
    return submission_df, performance_report

# Example usage:
# submission_df, report = run_optimization_pipeline('train.csv', 'test.csv')