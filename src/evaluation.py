"""
evaluation.py
-------------
This module contains evaluation metrics and functions for both
regression and classification tasks.

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


class RegressionEvaluator:
    """
    Evaluator for regression models (Energy Consumption Prediction).
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> dict:
        """
        Calculate comprehensive regression metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Coefficient of Variation of RMSE
        cv_rmse = (rmse / np.mean(y_true)) * 100
        
        return {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2_Score': round(r2, 4),
            'MAPE': round(mape, 2),
            'CV_RMSE': round(cv_rmse, 2)
        }
    
    @staticmethod
    def evaluate_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate multiple regression models and return comparison.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of {model_name: trained_model}
        X_test : array-like
            Test features
        y_test : array-like
            True test values
        
        Returns:
        --------
        pd.DataFrame : Comparison of all models
        """
        results = []
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            metrics = RegressionEvaluator.calculate_metrics(y_test, y_pred)
            metrics['Model'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df[['Model', 'R2_Score', 'RMSE', 'MAE', 'MSE', 'MAPE', 'CV_RMSE']]
        return df.sort_values('R2_Score', ascending=False)
    
    @staticmethod
    def get_residual_analysis(y_true, y_pred) -> dict:
        """
        Perform residual analysis.
        
        Returns:
        --------
        dict : Residual statistics
        """
        residuals = y_true - y_pred
        
        return {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'residuals': residuals
        }


class ClassificationEvaluator:
    """
    Evaluator for classification models (Load Type Classification).
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, average: str = 'weighted') -> dict:
        """
        Calculate comprehensive classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        average : str
            Averaging method for multiclass ('weighted', 'macro', 'micro')
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        return {
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1, 4)
        }
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred, labels: list = None) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
        --------
        np.ndarray : Confusion matrix
        """
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def get_classification_report(y_true, y_pred, target_names: list = None) -> str:
        """
        Get detailed classification report.
        
        Returns:
        --------
        str : Classification report
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def evaluate_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate multiple classification models and return comparison.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of {model_name: trained_model}
        X_test : array-like
            Test features
        y_test : array-like
            True test labels
        
        Returns:
        --------
        pd.DataFrame : Comparison of all models
        """
        results = []
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            metrics = ClassificationEvaluator.calculate_metrics(y_test, y_pred)
            metrics['Model'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df[['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall']]
        return df.sort_values('Accuracy', ascending=False)


def print_model_comparison(df: pd.DataFrame, task: str = 'regression'):
    """
    Print formatted model comparison table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Comparison dataframe from evaluate_models
    task : str
        'regression' or 'classification'
    """
    print("\n" + "="*60)
    if task == 'regression':
        print("REGRESSION MODEL COMPARISON")
        print("="*60)
        print(f"{'Model':<25} {'R2 Score':<12} {'RMSE':<10} {'MAE':<10}")
        print("-"*60)
        for _, row in df.iterrows():
            print(f"{row['Model']:<25} {row['R2_Score']:<12.4f} {row['RMSE']:<10.4f} {row['MAE']:<10.4f}")
    else:
        print("CLASSIFICATION MODEL COMPARISON")
        print("="*60)
        print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<10} {'Precision':<10}")
        print("-"*60)
        for _, row in df.iterrows():
            print(f"{row['Model']:<25} {row['Accuracy']:<12.4f} {row['F1_Score']:<10.4f} {row['Precision']:<10.4f}")
    print("="*60)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_pipeline
    from models import EnergyPredictionModels, LoadTypeClassificationModels
    
    # Load and preprocess data
    df = load_data()
    
    # Test Regression Evaluation
    print("Testing Regression Evaluation...")
    X_train, X_test, y_train, y_test, prep = preprocess_pipeline(df, task='regression')
    
    reg_models = EnergyPredictionModels()
    reg_models.train_all(X_train, y_train)
    
    reg_results = RegressionEvaluator.evaluate_models(reg_models.trained_models, X_test, y_test)
    print_model_comparison(reg_results, task='regression')
    
    # Test Classification Evaluation
    print("\nTesting Classification Evaluation...")
    X_train, X_test, y_train, y_test, prep = preprocess_pipeline(df, target='Load_Type', task='classification')
    
    clf_models = LoadTypeClassificationModels()
    clf_models.train_all(X_train, y_train)
    
    clf_results = ClassificationEvaluator.evaluate_models(clf_models.trained_models, X_test, y_test)
    print_model_comparison(clf_results, task='classification')
