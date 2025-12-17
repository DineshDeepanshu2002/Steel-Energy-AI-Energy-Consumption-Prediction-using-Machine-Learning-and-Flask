"""
models.py
---------
This module contains machine learning models for:
1. Energy Consumption Prediction (Regression)
2. Load Type Classification (Classification)

Author: [Your Name]
Date: December 2025
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib


class EnergyPredictionModels:
    """
    A class containing regression models for energy consumption prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all regression models with default parameters."""
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        }
    
    def train_model(self, name: str, X_train, y_train):
        """
        Train a specific model.
        
        Parameters:
        -----------
        name : str
            Name of the model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        
        Returns:
        --------
        Trained model object
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[name]
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        return model
    
    def train_all(self, X_train, y_train) -> dict:
        """
        Train all available models.
        
        Returns:
        --------
        dict : Dictionary of trained models
        """
        for name in self.models:
            print(f"Training {name}...")
            self.train_model(name, X_train, y_train)
        return self.trained_models
    
    def predict(self, name: str, X):
        """Make predictions using a trained model."""
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        return self.trained_models[name].predict(X)
    
    def cross_validate(self, name: str, X, y, cv: int = 5) -> dict:
        """
        Perform cross-validation for a model.
        
        Parameters:
        -----------
        name : str
            Model name
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Cross-validation scores
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")
        
        model = self.models[name]
        
        # R2 score
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        # Negative MSE (sklearn uses negative for maximization)
        mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': np.sqrt(mse_scores.mean()),
            'rmse_std': np.sqrt(mse_scores.std())
        }
    
    def get_feature_importance(self, name: str, feature_names: list) -> dict:
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        name : str
            Model name
        feature_names : list
            List of feature names
        
        Returns:
        --------
        dict : Feature importance dictionary
        """
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        
        model = self.trained_models[name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_)))
        else:
            return {}
    
    def save_model(self, name: str, filepath: str):
        """Save a trained model to disk."""
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        joblib.dump(self.trained_models[name], filepath)
    
    def load_model(self, name: str, filepath: str):
        """Load a model from disk."""
        self.trained_models[name] = joblib.load(filepath)


class LoadTypeClassificationModels:
    """
    A class containing classification models for load type prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classification models with default parameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        }
    
    def train_model(self, name: str, X_train, y_train):
        """Train a specific model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")
        
        model = self.models[name]
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        return model
    
    def train_all(self, X_train, y_train) -> dict:
        """Train all available models."""
        for name in self.models:
            print(f"Training {name}...")
            self.train_model(name, X_train, y_train)
        return self.trained_models
    
    def predict(self, name: str, X):
        """Make predictions using a trained model."""
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        return self.trained_models[name].predict(X)
    
    def predict_proba(self, name: str, X):
        """Get prediction probabilities."""
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        return self.trained_models[name].predict_proba(X)
    
    def cross_validate(self, name: str, X, y, cv: int = 5) -> dict:
        """Perform cross-validation for a model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")
        
        model = self.models[name]
        
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        return {
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std()
        }
    
    def get_feature_importance(self, name: str, feature_names: list) -> dict:
        """Get feature importance for tree-based models."""
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained yet.")
        
        model = self.trained_models[name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            # For multi-class, take mean absolute value across classes
            importance = np.abs(model.coef_).mean(axis=0)
            return dict(zip(feature_names, importance))
        else:
            return {}


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_pipeline
    
    # Test regression models
    print("Testing Regression Models...")
    df = load_data()
    X_train, X_test, y_train, y_test, prep = preprocess_pipeline(df, task='regression')
    
    reg_models = EnergyPredictionModels()
    reg_models.train_all(X_train, y_train)
    
    for name in reg_models.trained_models:
        preds = reg_models.predict(name, X_test)
        print(f"{name}: First 5 predictions = {preds[:5]}")
    
    # Test classification models
    print("\nTesting Classification Models...")
    X_train, X_test, y_train, y_test, prep = preprocess_pipeline(df, target='Load_Type', task='classification')
    
    clf_models = LoadTypeClassificationModels()
    clf_models.train_all(X_train, y_train)
    
    for name in clf_models.trained_models:
        preds = clf_models.predict(name, X_test)
        print(f"{name}: First 5 predictions = {preds[:5]}")
