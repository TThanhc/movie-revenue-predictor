"""
Utility functions for movie revenue prediction
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import re


class MovieRevenuePredictor:
    """Handle model loading and prediction logic"""
    
    def __init__(self, model_dir="models"):
        """Initialize predictor with models and scaler"""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.best_model = None
        self._load_models()
        self._load_scaler()
    
    def _load_models(self):
        """Load all trained models"""
        model_files = {
            'Best Model': 'best_model.pkl',
            'XGBoost': 'xgboost_tuned.pkl',
            'LightGBM': 'lightgbm_tuned.pkl',
            'Random Forest': 'random_forest_tuned.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                path = self.model_dir / filename
                self.models[name] = joblib.load(path)
                if name == 'Best Model':
                    self.best_model = self.models[name]
            except FileNotFoundError:
                print(f"Warning: {name} not found at {path}")
    
    def _load_scaler(self):
        """Load the fitted scaler"""
        try:
            scaler_path = self.model_dir / 'scaler.pkl'
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print("Warning: Scaler not found")
    
    def parse_raw_input(self, raw_input):
        """
        Parse raw movie input into engineered features
        
        Args:
            raw_input (dict): Dictionary with raw movie data in original format
                Expected keys:
                - title (str): Movie title
                - release_date (str): 'YYYY-MM-DD' format
                - budget (int/float): Budget in USD
                - runtime (int): Runtime in minutes
                - genres (str): Comma-separated genre names
                - cast (str): Comma-separated cast names
                - director (str): Comma-separated director names
                - production_companies (str): Comma-separated company names
                - production_countries (str): Comma-separated country names
                - keywords (str): Comma-separated keywords
                - collection (str): Collection/franchise name or empty
                - poster_brightness (float, optional): From image analysis
                - poster_saturation (float, optional): From image analysis
                - poster_dom_r (float, optional): From image analysis
                - poster_dom_g (float, optional): From image analysis
                - poster_dom_b (float, optional): From image analysis
                - original_language (str, optional): Language code
                
        Returns:
            dict: Processed features ready for prediction
        """
        # Parse release date
        try:
            release_date = datetime.strptime(raw_input['release_date'], '%Y-%m-%d')
            release_year = release_date.year
            release_month = release_date.month
        except (ValueError, KeyError):
            # Default to current date if parsing fails
            release_year = 2024
            release_month = 6
        
        # Parse budget (keep as is, will be log-transformed later)
        budget = raw_input.get('budget', 100_000_000)
        
        # Runtime
        runtime = raw_input.get('runtime', 120)
        
        # Parse genres
        genres_str = raw_input.get('genres', '')
        genres_list = [g.strip() for g in genres_str.split(',') if g.strip()]
        num_genres = len(genres_list)
        
        # Genre flags
        is_action = 1 if any('action' in g.lower() for g in genres_list) else 0
        is_animation = 1 if any('animation' in g.lower() for g in genres_list) else 0
        is_comedy = 1 if any('comedy' in g.lower() for g in genres_list) else 0
        is_drama = 1 if any('drama' in g.lower() for g in genres_list) else 0
        is_scifi = 1 if any('science fiction' in g.lower() or 'sci-fi' in g.lower() for g in genres_list) else 0
        
        # Parse cast
        cast_str = raw_input.get('cast', '')
        cast_list = [c.strip() for c in cast_str.split(',') if c.strip()]
        num_cast = len(cast_list)
        has_cast = 1 if num_cast > 0 else 0
        
        # Parse directors
        director_str = raw_input.get('director', '')
        director_list = [d.strip() for d in director_str.split(',') if d.strip()]
        num_directors = len(director_list)
        
        # Parse production companies
        companies_str = raw_input.get('production_companies', '')
        companies_list = [c.strip() for c in companies_str.split(',') if c.strip()]
        # Remove duplicates
        companies_list = list(set(companies_list))
        num_production_companies = len(companies_list)
        
        # Parse production countries
        countries_str = raw_input.get('production_countries', '')
        countries_list = [c.strip() for c in countries_str.split(',') if c.strip()]
        countries_list = list(set(countries_list))
        num_production_countries = len(countries_list)
        
        # Parse keywords
        keywords_str = raw_input.get('keywords', '')
        keywords_list = [k.strip() for k in keywords_str.split(',') if k.strip()]
        num_keywords = len(keywords_list)
        has_keywords = 1 if num_keywords > 0 else 0
        
        # Collection status
        collection_str = raw_input.get('collection', '').strip()
        in_collection = 1 if collection_str else 0
        
        # Language detection
        language = raw_input.get('original_language', '').lower()
        # Check if English
        is_english = 1 if language in ['en', 'english', ''] else 0
        # Also check countries for English-speaking countries
        if not is_english:
            english_countries = ['united states', 'usa', 'united kingdom', 'uk', 'canada', 
                                'australia', 'new zealand', 'ireland']
            is_english = 1 if any(any(ec in country.lower() for ec in english_countries) 
                                 for country in countries_list) else 0
        
        # Default rating (we don't have this in raw input, use average)
        rating = raw_input.get('rating', 6.5)
        
        # Poster features (from image analysis or defaults)
        poster_brightness = raw_input.get('poster_brightness', 150.0)
        poster_saturation = raw_input.get('poster_saturation', 120.0)
        poster_dom_r = raw_input.get('poster_dom_r', 100.0)
        poster_dom_g = raw_input.get('poster_dom_g', 100.0)
        poster_dom_b = raw_input.get('poster_dom_b', 150.0)
        
        # Create processed input
        processed_input = {
            'budget': budget,
            'runtime': runtime,
            'rating': rating,
            'num_production_companies': num_production_companies,
            'num_production_countries': num_production_countries,
            'num_genres': num_genres,
            'num_cast': num_cast,
            'num_directors': num_directors,
            'num_keywords': num_keywords,
            'in_collection': in_collection,
            'is_english': is_english,
            'release_year': release_year,
            'release_month': release_month,
            'genres': genres_list,
            'poster_brightness': poster_brightness,
            'poster_saturation': poster_saturation,
            'poster_dom_r': poster_dom_r,
            'poster_dom_g': poster_dom_g,
            'poster_dom_b': poster_dom_b,
        }
        
        return processed_input
    
    def engineer_features(self, user_input):
        """
        Engineer features from user input
        
        Args:
            user_input (dict): Dictionary with raw user inputs
            
        Returns:
            pd.DataFrame: Engineered features ready for prediction
        """
        features = {}
        
        # Basic features (direct input)
        features['budget'] = np.log1p(user_input['budget'])
        features['runtime'] = user_input['runtime']
        features['rating'] = user_input['rating']
        features['num_production_companies'] = user_input['num_production_companies']
        features['num_production_countries'] = user_input['num_production_countries']
        features['num_genres'] = user_input['num_genres']
        features['num_cast'] = user_input['num_cast']
        features['has_cast'] = 1 if user_input['num_cast'] > 0 else 0
        features['num_directors'] = user_input['num_directors']
        features['num_keywords'] = user_input['num_keywords']
        features['has_keywords'] = 1 if user_input['num_keywords'] > 0 else 0
        features['in_collection'] = user_input['in_collection']
        features['is_english'] = user_input['is_english']
        
        # Calculated features
        features['budget_per_minute'] = (np.exp(features['budget']) - 1) / features['runtime'] if features['runtime'] > 0 else 0
        
        # Release date features
        release_year = user_input['release_year']
        release_month = user_input['release_month']
        features['release_year'] = release_year
        features['release_month'] = release_month
        features['movie_age'] = 2024 - release_year  # Current year
        features['decade'] = (release_year // 10) * 10
        
        # Seasonal features
        features['is_summer'] = 1 if release_month in [6, 7, 8] else 0
        features['is_holiday'] = 1 if release_month in [11, 12] else 0
        features['is_weekend_month'] = 1 if release_month in [6, 7, 11, 12] else 0
        
        # Genre features (binary)
        genres = user_input.get('genres', [])
        features['is_action'] = 1 if 'Action' in genres else 0
        features['is_animation'] = 1 if 'Animation' in genres else 0
        features['is_comedy'] = 1 if 'Comedy' in genres else 0
        features['is_drama'] = 1 if 'Drama' in genres else 0
        features['is_scifi'] = 1 if 'Science Fiction' in genres else 0
        
        # Poster color features
        features['poster_brightness'] = user_input['poster_brightness']
        features['poster_saturation'] = user_input['poster_saturation']
        features['poster_dom_r'] = user_input['poster_dom_r']
        features['poster_dom_g'] = user_input['poster_dom_g']
        features['poster_dom_b'] = user_input['poster_dom_b']
        
        return pd.DataFrame([features])
    
    def predict(self, user_input, model_name='Best Model'):
        """
        Make revenue prediction
        
        Args:
            user_input (dict): User input data
            model_name (str): Name of model to use for prediction
            
        Returns:
            tuple: (predicted_revenue_actual, predicted_revenue_log, numeric_prediction)
        """
        # Engineer features
        X = self.engineer_features(user_input)
        
        # Reorder columns to match training data
        expected_columns = [
            'budget', 'runtime', 'rating', 'poster_brightness', 'poster_saturation',
            'poster_dom_r', 'poster_dom_g', 'poster_dom_b', 'num_production_companies',
            'num_production_countries', 'budget_per_minute', 'num_genres', 'is_action',
            'is_animation', 'is_comedy', 'is_drama', 'is_scifi', 'num_cast', 'has_cast',
            'num_directors', 'release_year', 'release_month', 'is_summer', 'is_holiday',
            'is_weekend_month', 'movie_age', 'decade', 'num_keywords', 'has_keywords',
            'in_collection', 'is_english'
        ]
        X = X[expected_columns]
        
        # Scale numeric features
        numeric_cols = ['budget', 'runtime']
        X_scaled = X.copy()
        if self.scaler:
            X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
        else:
            # Fallback if scaler is not loaded
            from sklearn.preprocessing import StandardScaler
            fallback_scaler = StandardScaler()
            X_scaled[numeric_cols] = fallback_scaler.fit_transform(X[numeric_cols])
        
        # Get model
        model = self.models.get(model_name, self.best_model)
        
        # Make prediction
        prediction_log = model.predict(X_scaled)[0]
        prediction_actual = np.exp(prediction_log)
        
        return {
            'revenue_actual': prediction_actual,
            'revenue_log': prediction_log,
            'model_used': model_name if model_name in self.models else 'Best Model'
        }
    
    def predict_all_models(self, user_input):
        """
        Get predictions from all models
        
        Args:
            user_input (dict): User input data
            
        Returns:
            dict: Predictions from each model
        """
        results = {}
        for model_name in self.models.keys():
            pred = self.predict(user_input, model_name)
            results[model_name] = pred['revenue_actual']
        
        return results
    
    def predict_from_raw(self, raw_input, model_name='Best Model'):
        """
        Make prediction from raw movie data
        
        Args:
            raw_input (dict): Raw movie data in original format
            model_name (str): Name of model to use
            
        Returns:
            dict: Prediction results
        """
        # Parse raw input to processed features
        processed_input = self.parse_raw_input(raw_input)
        
        # Make prediction
        return self.predict(processed_input, model_name)
    
    def predict_all_models_from_raw(self, raw_input):
        """
        Get predictions from all models using raw input
        
        Args:
            raw_input (dict): Raw movie data in original format
            
        Returns:
            dict: Predictions from each model
        """
        # Parse raw input to processed features
        processed_input = self.parse_raw_input(raw_input)
        
        # Get predictions from all models
        return self.predict_all_models(processed_input)
    
    def get_feature_names(self):
        """Get list of all feature names in correct order"""
        return [
            'budget', 'runtime', 'rating', 'num_production_companies',
            'num_production_countries', 'num_genres', 'num_cast', 'has_cast',
            'num_directors', 'num_keywords', 'has_keywords', 'in_collection',
            'is_english', 'budget_per_minute', 'release_year', 'release_month',
            'movie_age', 'decade', 'is_summer', 'is_holiday', 'is_weekend_month',
            'is_action', 'is_animation', 'is_comedy', 'is_drama', 'is_scifi',
            'poster_brightness', 'poster_saturation', 'poster_dom_r',
            'poster_dom_g', 'poster_dom_b'
        ]
