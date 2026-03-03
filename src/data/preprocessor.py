import pandas as pd
import numpy as np
from typing import Optional, Dict


class DataPreprocessor:
    """
    Data preprocessing pipeline for movie dataset.
    
    Features:
    - Log transformation of revenue and budget
    - Genre-based budget imputation for missing values
    - Missing value handling
    - Data validation
    """
    
    def __init__(self, transform_target: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            transform_target: Whether to log-transform the revenue target variable
        """
        self.transform_target = transform_target
        self.genre_budget_medians = None
        self.overall_budget_median = None
        self.preprocessing_stats = {}
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        Learns budget medians by genre for imputation.
        
        Args:
            df: Training dataframe
            
        Returns:
            self
        """
        # Calculate median budget by genre (excluding zeros)
        genre_budget_data = []
        for idx, row in df.iterrows():
            if pd.notna(row['genres']) and row['budget'] > 0:
                genres = str(row['genres']).split(', ')
                for genre in genres:
                    genre_budget_data.append({
                        'genre': genre,
                        'budget': row['budget']
                    })
        
        if genre_budget_data:
            genre_df = pd.DataFrame(genre_budget_data)
            self.genre_budget_medians = genre_df.groupby('genre')['budget'].median()
            self.overall_budget_median = genre_df['budget'].median()
        else:
            # Fallback if no genre data available
            self.overall_budget_median = df[df['budget'] > 0]['budget'].median()
            self.genre_budget_medians = pd.Series()
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe with preprocessing steps.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Record initial stats
        initial_shape = df.shape
        zero_budgets_initial = (df['budget'] == 0).sum()
        
        # 1. Log transform revenue (target variable)
        if self.transform_target and 'revenue' in df.columns:
            df['revenue'] = np.log1p(df['revenue'])
        
        # 2. Impute zero budgets with genre-based median
        df['budget'] = df.apply(self._impute_budget, axis=1)
        zero_budgets_after = (df['budget'] == 0).sum()
        
        # 3. Log transform budget after imputation
        df['budget'] = np.log1p(df['budget'])
        
        # 4. Handle missing values in production fields
        df['production_companies'] = df['production_companies'].fillna('Unknown')
        df['production_countries'] = df['production_countries'].fillna('Unknown')
        
        # 5. Drop records with missing critical features
        critical_cols = ['genres', 'cast', 'poster_brightness', 'poster_saturation',
                        'poster_dom_r', 'poster_dom_g', 'poster_dom_b']
        rows_before_drop = len(df)
        df = df.dropna(subset=critical_cols)
        rows_after_drop = len(df)
        
        # Save preprocessing statistics
        self.preprocessing_stats = {
            'initial_shape': initial_shape,
            'final_shape': df.shape,
            'zero_budgets_imputed': zero_budgets_initial - zero_budgets_after,
            'rows_dropped_missing': rows_before_drop - rows_after_drop,
            'revenue_log_transformed': self.transform_target,
            'budget_log_transformed': True
        }
        
        return df
    
    def _impute_budget(self, row: pd.Series) -> float:
        """
        Impute budget for a single row based on genres.
        
        Args:
            row: DataFrame row
            
        Returns:
            Imputed budget value
        """
        # Keep non-zero budgets
        if row['budget'] > 0:
            return row['budget']
        
        # Try genre-based imputation
        if pd.notna(row['genres']) and self.genre_budget_medians is not None:
            genres = str(row['genres']).split(', ')
            budgets_for_genres = [
                self.genre_budget_medians.get(g) 
                for g in genres 
                if g in self.genre_budget_medians.index
            ]
            budgets_for_genres = [b for b in budgets_for_genres if pd.notna(b)]
            
            if budgets_for_genres:
                return np.median(budgets_for_genres)
        
        # Fallback to overall median
        return self.overall_budget_median if self.overall_budget_median else 0
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        return self.fit(df).transform(df)
    
    def get_stats(self) -> Dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary of preprocessing stats
        """
        return self.preprocessing_stats.copy()
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse log transformation for revenue predictions.
        
        Args:
            y: Log-transformed revenue values
            
        Returns:
            Original scale revenue values
        """
        if self.transform_target:
            return np.expm1(y)
        return y


def preprocess_data(input_path: str, output_path: str, 
                   transform_target: bool = True) -> Dict:
    """
    Convenience function to preprocess data from file.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save preprocessed data
        transform_target: Whether to transform revenue target
        
    Returns:
        Preprocessing statistics
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Preprocess
    preprocessor = DataPreprocessor(transform_target=transform_target)
    df_processed = preprocessor.fit_transform(df)
    
    # Save
    df_processed.to_csv(output_path, index=False)
    
    # Get stats
    stats = preprocessor.get_stats()
    print(f"Preprocessing complete!")
    print(f"  Initial shape: {stats['initial_shape']}")
    print(f"  Final shape: {stats['final_shape']}")
    print(f"  Zero budgets imputed: {stats['zero_budgets_imputed']}")
    print(f"  Rows dropped (missing values): {stats['rows_dropped_missing']}")
    print(f"  Saved to: {output_path}")
    
    return stats


if __name__ == "__main__":
    # Example usage
    preprocess_data(
        input_path='../../data/raw/movies_dataset_revenue.csv',
        output_path='../../data/processed/movies_preprocessed.csv',
        transform_target=True
    )
