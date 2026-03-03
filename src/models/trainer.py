import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training and hyperparameter tuning manager.
    
    Features:
    - Train multiple baseline models
    - Hyperparameter tuning with regularization
    - Model comparison and selection
    - Automatic scaling and saving
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
            test_size: Proportion of data for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.numeric_cols = ['budget', 'runtime']
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'revenue') -> Tuple:
        """
        Prepare data for training (split and scale).
        
        Args:
            df: Input dataframe with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Apply scaling AFTER split to prevent data leakage
        self.scaler = StandardScaler()
        self.X_train[self.numeric_cols] = self.scaler.fit_transform(
            self.X_train[self.numeric_cols]
        )
        self.X_test[self.numeric_cols] = self.scaler.transform(
            self.X_test[self.numeric_cols]
        )
        
        print(f"✓ Data prepared:")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        print(f"  Features: {self.X_train.shape[1]}")
        print(f"  Scaling applied to: {self.numeric_cols}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_models(self) -> pd.DataFrame:
        """
        Train baseline models without tuning.
        
        Returns:
            DataFrame with model comparison results
        """
        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100, random_state=self.random_state, 
                n_jobs=-1, verbose=-1
            )
        }
        
        print("\nTraining baseline models...\n")
        for name, model in baseline_models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            metrics = self._evaluate_model(model, name)
            self.models[name] = model
            self.results[name] = metrics
            print(f"  Test R²: {metrics['Test R²']:.4f} | "
                  f"Test RMSE: {metrics['Test RMSE']:.4f}")
        
        results_df = pd.DataFrame(self.results).T
        print("\n" + "="*80)
        print("BASELINE MODEL COMPARISON")
        print("="*80)
        print(results_df.round(4))
        
        return results_df
    
    def tune_top_models(self, top_n: int = 3) -> pd.DataFrame:
        """
        Perform hyperparameter tuning on top performing models.
        
        Args:
            top_n: Number of top models to tune
            
        Returns:
            DataFrame with tuned model results
        """
        # Get top models by Test R²
        results_df = pd.DataFrame(self.results).T
        top_models = results_df.nlargest(top_n, 'Test R²').index.tolist()
        
        print(f"\nTuning top {top_n} models with regularization...\n")
        
        tuned_models = {}
        tuned_results = {}
        
        # XGBoost tuning
        if 'XGBoost' in top_models:
            print("Tuning XGBoost...")
            xgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 5, 10],
                'min_child_weight': [1, 3, 5]
            }
            xgb_grid = GridSearchCV(
                XGBRegressor(random_state=self.random_state, n_jobs=-1),
                xgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            xgb_grid.fit(self.X_train, self.y_train)
            tuned_models['XGBoost (Tuned)'] = xgb_grid.best_estimator_
            print(f"  Best params: {xgb_grid.best_params_}")
            print(f"  Best CV score: {xgb_grid.best_score_:.4f}\n")
        
        # LightGBM tuning
        if 'LightGBM' in top_models:
            print("Tuning LightGBM...")
            lgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 50],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 1, 5],
                'min_child_samples': [10, 20, 30]
            }
            lgb_grid = GridSearchCV(
                LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1),
                lgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            lgb_grid.fit(self.X_train, self.y_train)
            tuned_models['LightGBM (Tuned)'] = lgb_grid.best_estimator_
            print(f"  Best params: {lgb_grid.best_params_}")
            print(f"  Best CV score: {lgb_grid.best_score_:.4f}\n")
        
        # Random Forest tuning
        if 'Random Forest' in top_models:
            print("Tuning Random Forest...")
            rf_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8],
                'max_features': ['sqrt', 0.5, 0.7],
                'max_samples': [0.7, 0.8, 0.9]
            }
            rf_grid = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            rf_grid.fit(self.X_train, self.y_train)
            tuned_models['Random Forest (Tuned)'] = rf_grid.best_estimator_
            print(f"  Best params: {rf_grid.best_params_}")
            print(f"  Best CV score: {rf_grid.best_score_:.4f}")
        
        # Evaluate tuned models
        print("\n" + "="*80)
        print("TUNED MODEL PERFORMANCE")
        print("="*80)
        for name, model in tuned_models.items():
            metrics = self._evaluate_model(model, name)
            tuned_results[name] = metrics
            self.models[name] = model
            print(f"\n{name}:")
            print(f"  Test R²: {metrics['Test R²']:.4f}")
            print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
            print(f"  Test MAE: {metrics['Test MAE']:.4f}")
        
        tuned_results_df = pd.DataFrame(tuned_results).T
        
        # Update best model
        all_results = {**self.results, **tuned_results}
        all_results_df = pd.DataFrame(all_results).T
        self.best_model_name = all_results_df['Test R²'].idxmax()
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*80)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test R²: {all_results[self.best_model_name]['Test R²']:.4f}")
        print("="*80)
        
        return tuned_results_df
    
    def _evaluate_model(self, model, model_name: str) -> Dict:
        """
        Evaluate model with multiple metrics.
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        metrics = {
            'Train R²': r2_score(self.y_train, y_train_pred),
            'Test R²': r2_score(self.y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'Train MAE': mean_absolute_error(self.y_train, y_train_pred),
            'Test MAE': mean_absolute_error(self.y_test, y_test_pred),
            'Train MAPE': np.mean(np.abs((self.y_train - y_train_pred) / self.y_train)) * 100,
            'Test MAPE': np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
        }
        
        return metrics
    
    def save_models(self, output_dir: str = 'models') -> None:
        """
        Save trained models and scaler.
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Save best model
        if self.best_model:
            best_model_path = os.path.join(output_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"✓ Best model ({self.best_model_name}) saved to {best_model_path}")
        
        # Save tuned models
        tuned_model_files = {
            'XGBoost (Tuned)': 'xgboost_tuned.pkl',
            'LightGBM (Tuned)': 'lightgbm_tuned.pkl',
            'Random Forest (Tuned)': 'random_forest_tuned.pkl'
        }
        
        for name, filename in tuned_model_files.items():
            if name in self.models:
                path = os.path.join(output_dir, filename)
                joblib.dump(self.models[name], path)
                print(f"✓ {name} saved to {path}")
        
        print(f"\n✓ All models saved successfully to {output_dir}/")
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with all model results
        """
        return pd.DataFrame(self.results).T.round(4)


def train_models(data_path: str, output_dir: str = 'models', 
                tune_models: bool = True) -> ModelTrainer:
    """
    Convenience function to train models from file.
    
    Args:
        data_path: Path to featured data CSV
        output_dir: Directory to save models
        tune_models: Whether to perform hyperparameter tuning
        
    Returns:
        Trained ModelTrainer instance
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    trainer.prepare_data(df)
    
    # Train baseline models
    trainer.train_baseline_models()
    
    # Tune top models
    if tune_models:
        trainer.tune_top_models(top_n=3)
    
    # Save models
    trainer.save_models(output_dir)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    trainer = train_models(
        data_path='../../data/processed/movies_featured.csv',
        output_dir='../../models',
        tune_models=True
    )
