import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model evaluation and analysis manager.
    
    Features:
    - Load and evaluate multiple models
    - Calculate comprehensive metrics
    - Residual analysis
    - Feature importance analysis
    - Overfitting detection
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model evaluator.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.evaluation_results = []
        self.predictions = {}
        self.best_model_name = None
        
    def load_models(self) -> Dict:
        """
        Load all trained models from directory.
        
        Returns:
            Dictionary of loaded models
        """
        import os
        
        model_files = {
            'Best Model': 'best_model.pkl',
            'XGBoost': 'xgboost_tuned.pkl',
            'LightGBM': 'lightgbm_tuned.pkl',
            'Random Forest': 'random_forest_tuned.pkl'
        }
        
        print("Loading models...\n")
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            try:
                self.models[name] = joblib.load(path)
                print(f"✓ {name} loaded")
            except FileNotFoundError:
                print(f"✗ {name} not found at {path}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded")
        except FileNotFoundError:
            print(f"✗ Scaler not found")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        return self.models
    
    def evaluate_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all loaded models.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80 + "\n")
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            metrics, y_pred = self._evaluate_model_detailed(
                model, X_train, X_test, y_train, y_test, name
            )
            self.evaluation_results.append(metrics)
            self.predictions[name] = y_pred
            
            print(f"  Test R²: {metrics['Test R²']:.4f}")
            print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
            print(f"  Overfitting: {metrics['Overfitting Status']}\n")
        
        results_df = pd.DataFrame(self.evaluation_results)
        results_df = results_df.set_index('Model')
        
        # Identify best model
        self.best_model_name = results_df['Test R²'].idxmax()
        
        print("\n" + "="*80)
        print("COMPLETE EVALUATION METRICS")
        print("="*80)
        print(results_df.round(4))
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test R²: {results_df.loc[self.best_model_name, 'Test R²']:.4f}")
        print(f"Test RMSE: {results_df.loc[self.best_model_name, 'Test RMSE']:.4f}")
        print(f"{'='*80}")
        
        return results_df
    
    def _evaluate_model_detailed(self, model, X_train: pd.DataFrame,
                                 X_test: pd.DataFrame, y_train: pd.Series,
                                 y_test: pd.Series, model_name: str) -> Tuple[Dict, np.ndarray]:
        """
        Detailed evaluation of a single model.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Tuple of (metrics dict, test predictions)
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            'Model': model_name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'Train MAPE (%)': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
            'Test MAPE (%)': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        }
        
        # Overfitting analysis
        r2_diff = train_r2 - test_r2
        metrics['Overfitting Gap'] = r2_diff
        
        if r2_diff < 0.05:
            metrics['Overfitting Status'] = 'Good'
        elif r2_diff < 0.10:
            metrics['Overfitting Status'] = 'Moderate'
        else:
            metrics['Overfitting Status'] = 'High'
        
        return metrics, y_test_pred
    
    def analyze_residuals(self, y_test: pd.Series, 
                         model_name: Optional[str] = None) -> Dict:
        """
        Analyze residuals for a model.
        
        Args:
            y_test: Test target values
            model_name: Name of model (uses best model if None)
            
        Returns:
            Dictionary of residual statistics
        """
        if model_name is None:
            model_name = self.best_model_name
        
        y_pred = self.predictions[model_name]
        residuals = y_test - y_pred
        residuals_percent = (residuals / y_test) * 100
        
        stats = {
            'model': model_name,
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'min_residual': residuals.min(),
            'max_residual': residuals.max(),
            'median_residual': np.median(residuals),
            'mape': np.mean(np.abs(residuals_percent)),
            'median_ape': np.median(np.abs(residuals_percent))
        }
        
        print("\n" + "="*80)
        print(f"RESIDUAL ANALYSIS - {model_name}")
        print("="*80)
        print(f"Mean Residual: {stats['mean_residual']:.4f}")
        print(f"Std Residual: {stats['std_residual']:.4f}")
        print(f"Min Residual: {stats['min_residual']:.4f}")
        print(f"Max Residual: {stats['max_residual']:.4f}")
        print(f"Median Residual: {stats['median_residual']:.4f}")
        print(f"\nMean Absolute Percentage Error: {stats['mape']:.2f}%")
        print(f"Median Absolute Percentage Error: {stats['median_ape']:.2f}%")
        
        return stats
    
    def get_feature_importance(self, feature_names: List[str],
                              model_name: Optional[str] = None,
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            feature_names: List of feature names
            model_name: Name of model (uses best model if None)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not provide feature importances.")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*80)
        print(f"TOP {top_n} FEATURE IMPORTANCES - {model_name}")
        print("="*80)
        print(importance_df.head(top_n).to_string(index=False))
        
        # Calculate cumulative importance
        cumulative = np.cumsum(importance_df['importance'].values)
        n_features_95 = np.argmax(cumulative >= 0.95) + 1
        print(f"\nFeatures needed for 95% importance: {n_features_95}/{len(importance_df)}")
        
        return importance_df
    
    def save_results(self, output_dir: str = 'models') -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        
        # Save detailed results
        results_df = pd.DataFrame(self.evaluation_results).set_index('Model')
        results_path = os.path.join(output_dir, 'model_evaluation_results.csv')
        results_df.to_csv(results_path)
        print(f"✓ Evaluation results saved to {results_path}")
        
        # Save summary
        if self.best_model_name:
            best_metrics = results_df.loc[self.best_model_name]
            summary = {
                'Best Model': self.best_model_name,
                'Test R²': best_metrics['Test R²'],
                'Test RMSE': best_metrics['Test RMSE'],
                'Test MAE': best_metrics['Test MAE'],
                'Test MAPE (%)': best_metrics['Test MAPE (%)'],
                'Overfitting Status': best_metrics['Overfitting Status'],
                'Total Models Evaluated': len(self.models)
            }
            summary_df = pd.DataFrame([summary])
            summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ Summary saved to {summary_path}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get evaluation results as DataFrame.
        
        Returns:
            DataFrame with all evaluation results
        """
        return pd.DataFrame(self.evaluation_results).set_index('Model')


def evaluate_models(data_path: str, models_dir: str = 'models',
                   test_size: float = 0.2, random_state: int = 42) -> ModelEvaluator:
    """
    Convenience function to evaluate models from file.
    
    Args:
        data_path: Path to featured data CSV
        models_dir: Directory containing models
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        ModelEvaluator instance with results
    """
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # Split data (same as training)
    X = df.drop('revenue', axis=1)
    y = df['revenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize evaluator and load models
    evaluator = ModelEvaluator(models_dir=models_dir)
    evaluator.load_models()
    
    # Apply scaling
    if evaluator.scaler:
        numeric_cols = ['budget', 'runtime']
        X_train[numeric_cols] = evaluator.scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = evaluator.scaler.transform(X_test[numeric_cols])
    
    # Evaluate all models
    evaluator.evaluate_all_models(X_train, X_test, y_train, y_test)
    
    # Analyze residuals
    evaluator.analyze_residuals(y_test)
    
    # Get feature importance
    evaluator.get_feature_importance(X.columns.tolist())
    
    # Save results
    evaluator.save_results(models_dir)
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    evaluator = evaluate_models(
        data_path='../../data/processed/movies_featured.csv',
        models_dir='../../models'
    )
