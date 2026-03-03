import pandas as pd
import numpy as np
from typing import Optional, List
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for movie dataset.
    
    Creates 31+ engineered features from raw movie data:
    - Production features (budget per minute, counts)
    - Genre features (flags for major genres)
    - Cast and crew features
    - Temporal features (release timing, seasonality)
    - Language and collection features
    """
    
    def __init__(self, use_post_release_features: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            use_post_release_features: Whether to include post-release features
                                       like popularity and rating (False for prediction)
        """
        self.use_post_release_features = use_post_release_features
        self.reference_year = None
        self.text_cols_to_drop = ['title', 'keywords', 'cast', 'director', 
                                   'genres', 'production_companies', 
                                   'production_countries', 'release_date']
        
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        """
        Fit the feature engineer (learns reference year).
        
        Args:
            X: Input dataframe
            y: Target variable (unused, for sklearn compatibility)
            
        Returns:
            self
        """
        # Extract release year to calculate reference year
        if 'release_date' in X.columns:
            temp_df = X.copy()
            temp_df['release_date'] = pd.to_datetime(temp_df['release_date'], errors='coerce')
            temp_df['release_year'] = temp_df['release_date'].dt.year
            self.reference_year = int(temp_df['release_year'].max()) if temp_df['release_year'].notna().any() else pd.Timestamp.now().year
        elif 'release_year' in X.columns:
            self.reference_year = int(X['release_year'].max()) if X['release_year'].notna().any() else pd.Timestamp.now().year
        else:
            self.reference_year = pd.Timestamp.now().year
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe with engineered features.
        
        Args:
            X: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df = X.copy()
        
        # 1. Production features
        df['num_production_companies'] = df['production_companies'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != 'Unknown' else 0
        )
        df['num_production_countries'] = df['production_countries'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != 'Unknown' else 0
        )
        
        # Budget features (safe division)
        df['budget_per_minute'] = np.where(df['runtime'] > 0, df['budget'] / df['runtime'], 0)
        
        if self.use_post_release_features and 'popularity' in df.columns:
            df['budget_popularity_ratio'] = (df['budget'] + 1) / (df['popularity'] + 1)
        
        # 2. Genre features
        df['num_genres'] = df['genres'].apply(
            lambda x: len(str(x).split(', ')) if pd.notna(x) else 0
        )
        df['is_action'] = df['genres'].apply(lambda x: 1 if 'Action' in str(x) else 0)
        df['is_animation'] = df['genres'].apply(lambda x: 1 if 'Animation' in str(x) else 0)
        df['is_comedy'] = df['genres'].apply(lambda x: 1 if 'Comedy' in str(x) else 0)
        df['is_drama'] = df['genres'].apply(lambda x: 1 if 'Drama' in str(x) else 0)
        df['is_scifi'] = df['genres'].apply(lambda x: 1 if 'Science Fiction' in str(x) else 0)
        
        # 3. Cast and crew features
        df['num_cast'] = df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        df['has_cast'] = (df['num_cast'] > 0).astype(int)
        
        df['num_directors'] = df['director'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        
        # 4. Temporal features
        if 'release_date' not in df.columns or df['release_date'].isna().all():
            # release_date might already be processed, check for release_year/month
            if 'release_year' not in df.columns:
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                df['release_year'] = df['release_date'].dt.year
                df['release_month'] = df['release_date'].dt.month
        else:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
        
        df['is_summer'] = df['release_month'].isin([5, 6, 7]).astype(int)
        df['is_holiday'] = df['release_month'].isin([11, 12]).astype(int)
        df['is_weekend_month'] = df['release_month'].isin([12, 7]).astype(int)
        
        # Use fitted reference year
        reference_year = self.reference_year if self.reference_year else df['release_year'].max()
        df['movie_age'] = reference_year - df['release_year']
        df['decade'] = (df['release_year'] // 10) * 10
        
        # 5. Keywords features
        df['num_keywords'] = df['keywords'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        df['has_keywords'] = (df['num_keywords'] > 0).astype(int)
        
        # 6. Collection features
        df['in_collection'] = (df['collection'].notna()).astype(int)
        
        # 7. Language features
        df['is_english'] = (df['original_language'] == 'en').astype(int)
        
        # Drop text columns that are no longer needed
        cols_to_drop = [col for col in self.text_cols_to_drop if col in df.columns]
        
        # Also drop metadata columns
        metadata_cols = ['id', 'popularity', 'rating', 'vote_count', 'collection', 'original_language']
        if not self.use_post_release_features:
            cols_to_drop.extend([col for col in metadata_cols if col in df.columns])
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input dataframe
            y: Target variable (unused)
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names.
        
        Returns:
            List of feature names created
        """
        engineered_features = [
            'num_production_companies', 'num_production_countries',
            'budget_per_minute', 'num_genres',
            'is_action', 'is_animation', 'is_comedy', 'is_drama', 'is_scifi',
            'num_cast', 'has_cast', 'num_directors',
            'release_year', 'release_month',
            'is_summer', 'is_holiday', 'is_weekend_month',
            'movie_age', 'decade',
            'num_keywords', 'has_keywords',
            'in_collection', 'is_english'
        ]
        
        if self.use_post_release_features:
            engineered_features.append('budget_popularity_ratio')
        
        return engineered_features


def engineer_features(input_path: str, output_path: str, 
                     use_post_release_features: bool = False) -> pd.DataFrame:
    """
    Convenience function to engineer features from file.
    
    Args:
        input_path: Path to preprocessed data CSV
        output_path: Path to save featured data
        use_post_release_features: Include post-release features
        
    Returns:
        Featured dataframe
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Data loaded: {df.shape}")
    
    # Engineer features
    engineer = FeatureEngineer(use_post_release_features=use_post_release_features)
    df_featured = engineer.fit_transform(df)
    
    # Save
    df_featured.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete!")
    print(f"  Final shape: {df_featured.shape}")
    print(f"  Features created: {len(engineer.get_feature_names())}")
    print(f"  Saved to: {output_path}")
    
    return df_featured