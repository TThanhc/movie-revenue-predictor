# Movie Revenue Predictor - Notebooks

This directory contains Jupyter notebooks for the movie revenue prediction project, organized in a logical workflow from data exploration to business insights.

## Notebook Overview

### 01_data_exploration.ipynb
**Exploratory Data Analysis (EDA)**
- Load and examine the raw dataset
- Identify data types and structure
- Check for missing values and anomalies
- Analyze distributions of key variables
- Detect outliers and patterns
- Initial visualization of relationships

### 02_data_preprocessing.ipynb
**Data Cleaning & Validation**
- Handle missing values (imputation or dropping)
- Remove duplicates
- Detect and handle outliers
- Standardize data types and formats
- Data validation and consistency checks
- Export cleaned dataset for next steps

### 03_feature_engineering.ipynb
**Feature Creation & Transformation**
- Create derived features from existing variables
- Feature scaling and normalization
- Encode categorical variables (one-hot, label encoding, etc.)
- Feature selection techniques
- Dimensionality reduction (if applicable)
- Export engineered features dataset

### 04_model_training.ipynb
**Model Development & Hyperparameter Tuning**
- Train/test split
- Train baseline models (Linear Regression, Random Forest, Gradient Boosting, etc.)
- Perform cross-validation
- Hyperparameter optimization using GridSearchCV
- Model comparison and selection
- Save best performing model

### 05_model_evaluation.ipynb
**Performance Analysis & Diagnostics**
- Calculate evaluation metrics (MSE, RMSE, MAE, R² Score, etc.)
- Generate prediction vs actual value plots
- Residual analysis
- Feature importance visualization
- Model diagnostics and error analysis
- Identify potential improvements

### 06_meaningful_insights.ipynb
**Business Analysis & Recommendations**
- Extract key findings from model results
- Revenue analysis by genre, budget, release timing
- Identify success factors for movie revenue
- Analyze cast and crew influence
- Provide actionable business recommendations
- Summary of key metrics and insights

## Workflow

Follow the notebooks in order for a complete analysis pipeline:

```
01 → 02 → 03 → 04 → 05 → 06
```

Each notebook builds on the outputs of the previous one and produces artifacts (datasets, models) for the next stage.

## Data Files

- **Raw Data**: `../data/raw/movies.csv`
- **Preprocessed Data**: `../data/processed/movies_preprocessed.csv`
- **Engineered Features**: `../data/processed/movies_engineered.csv`
- **Trained Model**: `../models/best_model.pkl`

## Requirements

See `requirements.txt` in the project root for all dependencies. Key packages include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Getting Started

1. Ensure all dependencies are installed: `pip install -r ../requirements.txt`
2. Launch Jupyter: `jupyter notebook`
3. Open and run notebooks in order from 01 to 06
4. Modify code as needed for your specific dataset and use case

## Notes

- All paths in notebooks are relative to the `notebooks/` directory
- Commented code blocks provide templates that should be customized for your data
- Adjust train/test split, hyperparameters, and model selections based on results
- Document your findings at each stage before moving to the next notebook

## Author
Movie Revenue Predictor Project

## Date Created
February 2026
