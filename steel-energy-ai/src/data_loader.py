"""
data_loader.py
--------------
This module handles loading the Steel Industry Energy Consumption dataset
and provides basic data inspection functions.

Author: [Your Name]
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the Steel Industry Energy Consumption dataset.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the CSV file. If None, uses default path.
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    if filepath is None:
        # Default path relative to project root
        filepath = Path(__file__).parent.parent / "data" / "Steel_industry_data.csv"
    
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    
    Returns:
    --------
    dict : Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return info


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistical summary for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    
    Returns:
    --------
    pd.DataFrame : Statistical summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    summary['skewness'] = numeric_df.skew()
    summary['kurtosis'] = numeric_df.kurtosis()
    return summary


def get_categorical_summary(df: pd.DataFrame) -> dict:
    """
    Get summary for categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    
    Returns:
    --------
    dict : Summary for each categorical column
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    summary = {}
    for col in cat_cols:
        summary[col] = {
            'unique_values': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict(),
            'mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
        }
    return summary


if __name__ == "__main__":
    # Test the functions
    df = load_data()
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    info = get_data_info(df)
    print(f"\nColumns: {info['columns']}")
    print(f"Missing values: {info['missing_values']}")
    
    stats = get_statistical_summary(df)
    print(f"\nStatistical Summary:\n{stats}")
