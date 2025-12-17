"""
Flask Application for Steel Industry Energy Consumption Prediction
===================================================================
A web-based interface for predicting energy consumption and load type
in steel manufacturing using machine learning models.

Author: Dinesh
Date: December 2025
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
import io
import base64
import os
import warnings
import threading
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models and encoders
models = {}
encoders = {}
df_global = None

def load_and_train_models():
    """Load dataset and train models on startup."""
    global models, encoders, df_global
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Steel_industry_data.csv')
    df = pd.read_csv(data_path)
    df_global = df.copy()
    
    # Feature engineering
    df['Hour'] = (df['NSM'] / 3600).astype(int)
    
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
    df['Is_Peak_Hour'] = ((df['Hour'] >= 8) & (df['Hour'] <= 18) & 
                          (df['WeekStatus'] == 'Weekday')).astype(int)
    
    # Encode categorical variables
    encoders['week'] = LabelEncoder()
    encoders['day'] = LabelEncoder()
    encoders['time'] = LabelEncoder()
    encoders['load'] = LabelEncoder()
    
    df['WeekStatus_encoded'] = encoders['week'].fit_transform(df['WeekStatus'])
    df['Day_of_week_encoded'] = encoders['day'].fit_transform(df['Day_of_week'])
    df['Time_Period_encoded'] = encoders['time'].fit_transform(df['Time_Period'])
    df['Load_Type_encoded'] = encoders['load'].fit_transform(df['Load_Type'])
    
    # Regression features and target
    reg_features = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM', 'Hour', 'Is_Peak_Hour',
        'WeekStatus_encoded', 'Day_of_week_encoded',
        'Time_Period_encoded', 'Load_Type_encoded'
    ]
    
    X_reg = df[reg_features]
    y_reg = df['Usage_kWh']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Train regression model
    models['regression'] = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
    models['regression'].fit(X_train_r, y_train_r)
    
    # Classification features and target
    clf_features = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM', 'Hour', 'Is_Peak_Hour',
        'WeekStatus_encoded', 'Day_of_week_encoded',
        'Time_Period_encoded'
    ]
    
    X_clf = df[clf_features]
    y_clf = df['Load_Type_encoded']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Train classification model
    models['classification'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    models['classification'].fit(X_train_c, y_train_c)
    
    print("Models trained successfully!")
    return df


def create_prediction_graph(input_data, energy_pred, load_pred_label):
    """Create visualization graphs for predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Steel Industry Energy Prediction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Energy Prediction Gauge-like visualization
    ax1 = axes[0, 0]
    categories = ['Light\n(0-30 kWh)', 'Medium\n(30-60 kWh)', 'High\n(60-100 kWh)', 'Very High\n(>100 kWh)']
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    
    # Determine category
    if energy_pred < 30:
        highlight_idx = 0
    elif energy_pred < 60:
        highlight_idx = 1
    elif energy_pred < 100:
        highlight_idx = 2
    else:
        highlight_idx = 3
    
    bar_colors = ['lightgray'] * 4
    bar_colors[highlight_idx] = colors[highlight_idx]
    
    bars = ax1.bar(categories, [30, 30, 40, 50], color=bar_colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=energy_pred, color='red', linestyle='--', linewidth=3, label=f'Predicted: {energy_pred:.2f} kWh')
    ax1.set_ylabel('Energy (kWh)', fontsize=11)
    ax1.set_title('Energy Consumption Category', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(150, energy_pred + 20))
    
    # 2. Load Type Prediction
    ax2 = axes[0, 1]
    load_types = ['Light_Load', 'Medium_Load', 'Maximum_Load']
    load_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Highlight predicted load
    bar_alphas = [0.3, 0.3, 0.3]
    load_idx = load_types.index(load_pred_label)
    bar_alphas[load_idx] = 1.0
    
    for i, (lt, color, alpha) in enumerate(zip(load_types, load_colors, bar_alphas)):
        ax2.bar(lt, 1, color=color, alpha=alpha, edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Prediction', fontsize=11)
    ax2.set_title(f'Predicted Load Type: {load_pred_label}', fontsize=12, fontweight='bold', color=load_colors[load_idx])
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])
    
    # 3. Input Features Bar Chart
    ax3 = axes[1, 0]
    feature_names = ['Lagging\nReactive Power', 'Leading\nReactive Power', 'CO2\n(×100)', 
                     'Lagging PF', 'Leading PF', 'Hour']
    feature_values = [
        input_data['lagging_power'],
        input_data['leading_power'],
        input_data['co2'] * 100,  # Scale for visibility
        input_data['lagging_pf'],
        input_data['leading_pf'],
        input_data['hour']
    ]
    
    colors_feat = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
    ax3.barh(feature_names, feature_values, color=colors_feat, edgecolor='black')
    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_title('Input Feature Values', fontsize=12, fontweight='bold')
    
    # 4. CO2 vs Energy Relationship with prediction point
    ax4 = axes[1, 1]
    
    # Sample historical data
    sample_df = df_global.sample(min(500, len(df_global)), random_state=42)
    ax4.scatter(sample_df['Usage_kWh'], sample_df['CO2(tCO2)'] * 1000, 
                alpha=0.3, s=20, c='gray', label='Historical Data')
    
    # Plot prediction point
    ax4.scatter([energy_pred], [input_data['co2'] * 1000], 
                s=200, c='red', marker='*', edgecolors='black', linewidth=2,
                label=f'Your Prediction', zorder=5)
    
    ax4.set_xlabel('Energy Consumption (kWh)', fontsize=11)
    ax4.set_ylabel('CO2 Emissions (kg)', fontsize=11)
    ax4.set_title('Energy vs CO2: Your Prediction in Context', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    
    # Convert to base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_historical_comparison_graph(energy_pred, hour, day_of_week):
    """Create comparison with historical data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Hourly comparison
    ax1 = axes[0]
    hourly_avg = df_global.groupby((df_global['NSM'] / 3600).astype(int))['Usage_kWh'].mean()
    
    ax1.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='steelblue')
    ax1.plot(hourly_avg.index, hourly_avg.values, 'o-', color='steelblue', linewidth=2, label='Historical Avg')
    ax1.axvline(x=hour, color='red', linestyle='--', linewidth=2, label=f'Your Hour ({hour}:00)')
    ax1.scatter([hour], [energy_pred], s=150, c='red', marker='*', zorder=5, label=f'Your Prediction: {energy_pred:.2f}')
    
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax1.set_title('Your Prediction vs Hourly Average', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(range(0, 24, 2))
    
    # 2. Daily comparison
    ax2 = axes[1]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df_global.groupby('Day_of_week')['Usage_kWh'].mean().reindex(day_order)
    
    colors = ['lightblue'] * 7
    day_idx = day_order.index(day_of_week)
    colors[day_idx] = 'steelblue'
    
    bars = ax2.bar(day_order, daily_avg.values, color=colors, edgecolor='black')
    ax2.axhline(y=energy_pred, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {energy_pred:.2f}')
    
    ax2.set_xlabel('Day of Week', fontsize=11)
    ax2.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax2.set_title(f'Your Prediction vs Daily Average ({day_of_week})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


@app.route('/')
def index():
    """Render the main input form page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and return results."""
    try:
        # Get form data
        lagging_power = float(request.form['lagging_power'])
        leading_power = float(request.form['leading_power'])
        co2 = float(request.form['co2'])
        lagging_pf = float(request.form['lagging_pf'])
        leading_pf = float(request.form['leading_pf'])
        hour = int(request.form['hour'])
        day_of_week = request.form['day_of_week']
        week_status = request.form['week_status']
        load_type = request.form['load_type']
        
        # Calculate derived features
        nsm = hour * 3600
        
        if hour < 6:
            time_period = 'Night'
        elif hour < 12:
            time_period = 'Morning'
        elif hour < 18:
            time_period = 'Afternoon'
        else:
            time_period = 'Evening'
        
        is_peak = 1 if (8 <= hour <= 18 and week_status == 'Weekday') else 0
        
        # Encode categorical variables
        week_encoded = encoders['week'].transform([week_status])[0]
        day_encoded = encoders['day'].transform([day_of_week])[0]
        time_encoded = encoders['time'].transform([time_period])[0]
        load_encoded = encoders['load'].transform([load_type])[0]
        
        # Prepare input for regression
        reg_input = np.array([[
            lagging_power, leading_power, co2, lagging_pf, leading_pf,
            nsm, hour, is_peak, week_encoded, day_encoded, time_encoded, load_encoded
        ]])
        
        # Predict energy consumption
        energy_pred = models['regression'].predict(reg_input)[0]
        
        # Prepare input for classification (using predicted energy)
        clf_input = np.array([[
            energy_pred, lagging_power, leading_power, co2, lagging_pf, leading_pf,
            nsm, hour, is_peak, week_encoded, day_encoded, time_encoded
        ]])
        
        # Predict load type
        load_pred_encoded = models['classification'].predict(clf_input)[0]
        load_pred_label = encoders['load'].inverse_transform([load_pred_encoded])[0]
        
        # Calculate estimated CO2 from energy
        estimated_co2 = energy_pred * 0.0005
        
        # Determine energy category
        if energy_pred < 30:
            energy_category = 'Low'
            category_color = '#2ecc71'
        elif energy_pred < 60:
            energy_category = 'Medium'
            category_color = '#f39c12'
        elif energy_pred < 100:
            energy_category = 'High'
            category_color = '#e67e22'
        else:
            energy_category = 'Very High'
            category_color = '#e74c3c'
        
        # Create input data dict for graphs
        input_data = {
            'lagging_power': lagging_power,
            'leading_power': leading_power,
            'co2': co2,
            'lagging_pf': lagging_pf,
            'leading_pf': leading_pf,
            'hour': hour
        }
        
        # Generate graphs
        prediction_graph = create_prediction_graph(input_data, energy_pred, load_pred_label)
        comparison_graph = create_historical_comparison_graph(energy_pred, hour, day_of_week)
        
        # Prepare result data
        result = {
            'energy_prediction': round(energy_pred, 2),
            'energy_category': energy_category,
            'category_color': category_color,
            'load_prediction': load_pred_label,
            'estimated_co2': round(estimated_co2 * 1000, 2),  # Convert to kg
            'time_period': time_period,
            'is_peak': 'Yes' if is_peak else 'No',
            'prediction_graph': prediction_graph,
            'comparison_graph': comparison_graph,
            'input_summary': {
                'hour': hour,
                'day': day_of_week,
                'week_status': week_status,
                'lagging_power': lagging_power,
                'leading_power': leading_power,
                'co2': co2,
                'lagging_pf': lagging_pf,
                'leading_pf': leading_pf,
                'load_type': load_type
            }
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)."""
    try:
        data = request.get_json()
        
        # Get data
        lagging_power = float(data['lagging_power'])
        leading_power = float(data['leading_power'])
        co2 = float(data['co2'])
        lagging_pf = float(data['lagging_pf'])
        leading_pf = float(data['leading_pf'])
        hour = int(data['hour'])
        day_of_week = data['day_of_week']
        week_status = data['week_status']
        load_type = data['load_type']
        
        # Calculate derived features
        nsm = hour * 3600
        
        if hour < 6:
            time_period = 'Night'
        elif hour < 12:
            time_period = 'Morning'
        elif hour < 18:
            time_period = 'Afternoon'
        else:
            time_period = 'Evening'
        
        is_peak = 1 if (8 <= hour <= 18 and week_status == 'Weekday') else 0
        
        # Encode
        week_encoded = encoders['week'].transform([week_status])[0]
        day_encoded = encoders['day'].transform([day_of_week])[0]
        time_encoded = encoders['time'].transform([time_period])[0]
        load_encoded = encoders['load'].transform([load_type])[0]
        
        # Predict
        reg_input = np.array([[
            lagging_power, leading_power, co2, lagging_pf, leading_pf,
            nsm, hour, is_peak, week_encoded, day_encoded, time_encoded, load_encoded
        ]])
        
        energy_pred = models['regression'].predict(reg_input)[0]
        
        return jsonify({
            'success': True,
            'energy_prediction': round(energy_pred, 2),
            'unit': 'kWh'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def train_models_async():
    print("Loading data and training models...")
    load_and_train_models()
    print("Models trained successfully!")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    # Start training in background so port opens immediately
    training_thread = threading.Thread(target=train_models_async)
    training_thread.start()

    print("Starting Flask server...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )

