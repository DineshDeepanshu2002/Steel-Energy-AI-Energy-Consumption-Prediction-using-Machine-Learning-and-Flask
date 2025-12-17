"""
main.py
-------
Main entry point for Steel Industry Energy Consumption Prediction project.
Run this script to execute the complete analysis pipeline.

Author: [Your Name]
Date: December 2025

Usage:
    python main.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, get_data_info, get_statistical_summary
from preprocessing import preprocess_pipeline
from models import EnergyPredictionModels, LoadTypeClassificationModels
from evaluation import RegressionEvaluator, ClassificationEvaluator, print_model_comparison
from visualization import create_dashboard_plots

import warnings
warnings.filterwarnings('ignore')


def main():
    """Execute the complete analysis pipeline."""
    
    print("="*70)
    print("STEEL INDUSTRY ENERGY CONSUMPTION PREDICTION")
    print("AI-Powered Energy Analytics for Sustainable Manufacturing")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[Step 1/6] Loading Dataset...")
    df = load_data()
    info = get_data_info(df)
    print(f"  ✓ Dataset loaded: {info['shape'][0]:,} records, {info['shape'][1]} features")
    print(f"  ✓ Missing values: {sum(info['missing_values'].values())}")
    
    # Step 2: Create Visualizations
    print("\n[Step 2/6] Generating Visualizations...")
    os.makedirs('results/figures', exist_ok=True)
    create_dashboard_plots(df, save_dir='results/figures')
    print("  ✓ Visualizations saved to results/figures/")
    
    # Step 3: Preprocess Data for Regression
    print("\n[Step 3/6] Preprocessing Data for Regression...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, prep_reg = preprocess_pipeline(
        df, target='Usage_kWh', task='regression'
    )
    print(f"  ✓ Training samples: {len(X_train_reg):,}")
    print(f"  ✓ Test samples: {len(X_test_reg):,}")
    print(f"  ✓ Features: {len(prep_reg.feature_columns)}")
    
    # Step 4: Train Regression Models
    print("\n[Step 4/6] Training Regression Models...")
    reg_models = EnergyPredictionModels()
    reg_models.train_all(X_train_reg, y_train_reg)
    
    # Evaluate Regression
    reg_results = RegressionEvaluator.evaluate_models(
        reg_models.trained_models, X_test_reg, y_test_reg
    )
    print_model_comparison(reg_results, task='regression')
    reg_results.to_csv('results/regression_results.csv', index=False)
    
    # Step 5: Train Classification Models
    print("\n[Step 5/6] Training Classification Models...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, prep_clf = preprocess_pipeline(
        df, target='Load_Type', task='classification'
    )
    
    clf_models = LoadTypeClassificationModels()
    clf_models.train_all(X_train_clf, y_train_clf)
    
    # Evaluate Classification
    clf_results = ClassificationEvaluator.evaluate_models(
        clf_models.trained_models, X_test_clf, y_test_clf
    )
    print_model_comparison(clf_results, task='classification')
    clf_results.to_csv('results/classification_results.csv', index=False)
    
    # Step 6: Summary
    print("\n[Step 6/6] Generating Summary...")
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    
    best_reg = reg_results.iloc[0]
    best_clf = clf_results.iloc[0]
    
    print(f"\n📊 Dataset: {info['shape'][0]:,} records from DAEWOO Steel Co. Ltd")
    print(f"\n📈 Best Regression Model: {best_reg['Model']}")
    print(f"   - R² Score: {best_reg['R2_Score']:.4f}")
    print(f"   - RMSE: {best_reg['RMSE']:.4f} kWh")
    
    print(f"\n🎯 Best Classification Model: {best_clf['Model']}")
    print(f"   - Accuracy: {best_clf['Accuracy']:.4f} ({best_clf['Accuracy']*100:.2f}%)")
    
    co2_corr = df['CO2(tCO2)'].corr(df['Usage_kWh'])
    print(f"\n🌱 Sustainability Insight:")
    print(f"   - CO2-Energy Correlation: {co2_corr:.4f}")
    
    print("\n" + "="*70)
    print("Analysis complete! Results saved to results/ directory.")
    print("="*70)
    
    return reg_results, clf_results


if __name__ == "__main__":
    main()
