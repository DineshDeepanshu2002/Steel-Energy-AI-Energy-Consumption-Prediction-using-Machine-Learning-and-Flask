"""
visualization.py
-----------------
This module contains visualization functions for data exploration,
model evaluation, and results presentation.

Author: [Your Name]
Date: December 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_distribution(df: pd.DataFrame, column: str, figsize: tuple = (10, 6), 
                     save_path: str = None):
    """
    Plot distribution of a numerical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(df[column], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {column}')
    
    # Box plot
    axes[1].boxplot(df[column].dropna())
    axes[1].set_ylabel(column)
    axes[1].set_title(f'Box Plot of {column}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 10), 
                           save_path: str = None):
    """
    Plot correlation heatmap for numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5)
    
    ax.set_title('Correlation Matrix - Steel Industry Energy Data', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, 
                                  figsize: tuple = (10, 6), save_path: str = None):
    """
    Plot distribution of a categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    value_counts = df[column].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    
    bars = ax.bar(value_counts.index, value_counts.values, color=colors, edgecolor='black')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column}')
    
    # Add value labels on bars
    for bar, val in zip(bars, value_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_energy_by_load_type(df: pd.DataFrame, figsize: tuple = (12, 5), 
                             save_path: str = None):
    """
    Plot energy consumption by load type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    load_order = ['Light_Load', 'Medium_Load', 'Maximum_Load']
    available_loads = [l for l in load_order if l in df['Load_Type'].unique()]
    
    sns.boxplot(x='Load_Type', y='Usage_kWh', data=df, order=available_loads, ax=axes[0])
    axes[0].set_title('Energy Consumption by Load Type')
    axes[0].set_xlabel('Load Type')
    axes[0].set_ylabel('Energy Consumption (kWh)')
    
    # Violin plot
    sns.violinplot(x='Load_Type', y='Usage_kWh', data=df, order=available_loads, ax=axes[1])
    axes[1].set_title('Energy Consumption Distribution by Load Type')
    axes[1].set_xlabel('Load Type')
    axes[1].set_ylabel('Energy Consumption (kWh)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_time_series_analysis(df: pd.DataFrame, figsize: tuple = (14, 10), 
                              save_path: str = None):
    """
    Plot time series analysis of energy consumption.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'date' column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['Hour'] = (df['NSM'] / 3600).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Hourly average consumption
    hourly_avg = df.groupby('Hour')['Usage_kWh'].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0, 0].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Energy (kWh)')
    axes[0, 0].set_title('Average Energy Consumption by Hour')
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Daily consumption by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    available_days = [d for d in day_order if d in df['Day_of_week'].unique()]
    daily_avg = df.groupby('Day_of_week')['Usage_kWh'].mean().reindex(available_days)
    axes[0, 1].bar(daily_avg.index, daily_avg.values, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Energy (kWh)')
    axes[0, 1].set_title('Average Energy Consumption by Day')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Weekend vs Weekday
    week_status_avg = df.groupby('WeekStatus')['Usage_kWh'].mean()
    axes[1, 0].bar(week_status_avg.index, week_status_avg.values, 
                   color=['coral', 'lightgreen'], edgecolor='black')
    axes[1, 0].set_xlabel('Week Status')
    axes[1, 0].set_ylabel('Average Energy (kWh)')
    axes[1, 0].set_title('Energy Consumption: Weekday vs Weekend')
    
    # CO2 vs Energy consumption
    sample = df.sample(min(1000, len(df)), random_state=42)
    axes[1, 1].scatter(sample['Usage_kWh'], sample['CO2(tCO2)'], alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Energy Consumption (kWh)')
    axes[1, 1].set_ylabel('CO2 Emissions (tCO2)')
    axes[1, 1].set_title('Energy Consumption vs CO2 Emissions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str, 
                          task: str = 'regression', figsize: tuple = (10, 6),
                          save_path: str = None):
    """
    Plot model comparison bar chart.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from evaluate_models
    metric : str
        Metric to compare (e.g., 'R2_Score', 'Accuracy')
    task : str
        'regression' or 'classification'
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    
    bars = ax.barh(results_df['Model'], results_df[metric], color=colors, edgecolor='black')
    
    ax.set_xlabel(metric.replace('_', ' '))
    ax.set_title(f'Model Comparison - {metric.replace("_", " ")}')
    
    # Add value labels
    for bar, val in zip(bars, results_df[metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names: list = None, 
                                   figsize: tuple = (8, 6), save_path: str = None):
    """
    Plot confusion matrix as heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(importance_dict: dict, figsize: tuple = (10, 8), 
                            save_path: str = None):
    """
    Plot feature importance bar chart.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary of {feature_name: importance_value}
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    features = list(sorted_importance.keys())
    importances = list(sorted_importance.values())
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]
    
    ax.barh(features, importances, color=colors, edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()  # Highest importance at top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_actual_vs_predicted(y_true, y_pred, figsize: tuple = (10, 5), 
                             save_path: str = None):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_dashboard_plots(df: pd.DataFrame, save_dir: str = None):
    """
    Create all dashboard plots and optionally save them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_dir : str, optional
        Directory to save plots
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    plots = {}
    
    # Distribution plot
    save_path = os.path.join(save_dir, 'energy_distribution.png') if save_dir else None
    plots['energy_distribution'] = plot_distribution(df, 'Usage_kWh', save_path=save_path)
    
    # Correlation matrix
    save_path = os.path.join(save_dir, 'correlation_matrix.png') if save_dir else None
    plots['correlation'] = plot_correlation_matrix(df, save_path=save_path)
    
    # Load type distribution
    save_path = os.path.join(save_dir, 'load_type_distribution.png') if save_dir else None
    plots['load_type'] = plot_categorical_distribution(df, 'Load_Type', save_path=save_path)
    
    # Energy by load type
    save_path = os.path.join(save_dir, 'energy_by_load.png') if save_dir else None
    plots['energy_by_load'] = plot_energy_by_load_type(df, save_path=save_path)
    
    # Time series analysis
    save_path = os.path.join(save_dir, 'time_series_analysis.png') if save_dir else None
    plots['time_series'] = plot_time_series_analysis(df, save_path=save_path)
    
    plt.close('all')
    
    return plots


if __name__ == "__main__":
    from data_loader import load_data
    
    # Test visualizations
    df = load_data()
    
    print("Creating visualizations...")
    create_dashboard_plots(df, save_dir='/home/claude/steel-energy-ai/results/figures')
    print("Visualizations saved to results/figures/")
