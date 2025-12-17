"""
preprocessing.py
-----------------
This module handles data preprocessing and feature engineering
for the Steel Industry Energy Consumption dataset.

Author: [Your Name]
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    A class to preprocess the Steel Industry Energy Consumption dataset.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional time-based features from the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with 'date' and 'NSM' columns
        
        Returns:
        --------
        pd.DataFrame : DataFrame with additional time features
        """
        df = df.copy()
        
        # Extract hour from NSM (Number of Seconds from Midnight)
        df['Hour'] = (df['NSM'] / 3600).astype(int)
        
        # Create time period categories
        def get_time_period(hour):
            if 0 <= hour < 6:
                return 'Night'
            elif 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            else:
                return 'Evening'
        
        df['Time_Period'] = df['Hour'].apply(get_time_period)
        
        # Is it a peak hour? (8 AM - 6 PM on weekdays)
        df['Is_Peak_Hour'] = ((df['Hour'] >= 8) & (df['Hour'] <= 18) & 
                              (df['WeekStatus'] == 'Weekday')).astype(int)
        
        # Extract month if date column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['Month'] = df['date'].dt.month
            df['Quarter'] = df['date'].dt.quarter
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to encode. If None, encodes all object columns.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with encoded columns
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, columns: list, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            List of columns to scale
        method : str
            Scaling method: 'standard' or 'minmax'
        
        Returns:
        --------
        pd.DataFrame : DataFrame with scaled features
        """
        df = df.copy()
        
        if self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Usage_kWh', 
                        task: str = 'regression') -> tuple:
        """
        Prepare features and target for model training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        task : str
            'regression' for energy prediction, 'classification' for load type
        
        Returns:
        --------
        tuple : (X, y) - Features and target
        """
        df = df.copy()
        
        # Feature engineering
        df = self.extract_time_features(df)
        
        # Encode categorical variables
        categorical_cols = ['WeekStatus', 'Day_of_week', 'Time_Period']
        if task == 'regression':
            categorical_cols.append('Load_Type')
        df = self.encode_categorical(df, categorical_cols)
        
        # Select feature columns
        feature_cols = [
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor',
            'NSM',
            'Hour',
            'Is_Peak_Hour',
            'WeekStatus_encoded',
            'Day_of_week_encoded',
            'Time_Period_encoded'
        ]
        
        if task == 'regression':
            feature_cols.append('Load_Type_encoded')
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        
        if task == 'regression':
            y = df[target_col]
        else:  # classification
            if target_col + '_encoded' not in df.columns:
                le = LabelEncoder()
                y = le.fit_transform(df[target_col])
                self.label_encoders[target_col] = le
            else:
                y = df[target_col + '_encoded']
        
        return X, y
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        test_size : float
            Proportion for test set
        random_state : int
            Random seed
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_pipeline(df: pd.DataFrame, target: str = 'Usage_kWh', 
                       task: str = 'regression') -> tuple:
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    target : str
        Target column name
    task : str
        'regression' or 'classification'
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, preprocessor)
    """
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(df, target_col=target, task=task)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    from data_loader import load_data
    
    # Test preprocessing
    df = load_data()
    X_train, X_test, y_train, y_test, prep = preprocess_pipeline(df, task='regression')
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {prep.feature_columns}")
