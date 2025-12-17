"""
Steel Industry Energy Consumption Prediction - Web Application
================================================================
A Streamlit-based interactive dashboard for energy consumption 
prediction and sustainability analytics.

Author: [Your Name]
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Steel Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset."""
    df = pd.read_csv('data/Steel_industry_data.csv')
    return df


@st.cache_data
def preprocess_data(df):
    """Preprocess data for modeling."""
    df = df.copy()
    
    # Extract features
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
    le_week = LabelEncoder()
    le_day = LabelEncoder()
    le_time = LabelEncoder()
    le_load = LabelEncoder()
    
    df['WeekStatus_encoded'] = le_week.fit_transform(df['WeekStatus'])
    df['Day_of_week_encoded'] = le_day.fit_transform(df['Day_of_week'])
    df['Time_Period_encoded'] = le_time.fit_transform(df['Time_Period'])
    df['Load_Type_encoded'] = le_load.fit_transform(df['Load_Type'])
    
    return df, le_load


@st.cache_resource
def train_models(df):
    """Train ML models."""
    # Feature columns
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
    
    # Regression
    X_reg = df[reg_features]
    y_reg = df['Usage_kWh']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
    reg_model.fit(X_train_r, y_train_r)
    reg_pred = reg_model.predict(X_test_r)
    reg_r2 = r2_score(y_test_r, reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_test_r, reg_pred))
    
    # Classification
    X_clf = df[clf_features]
    y_clf = df['Load_Type_encoded']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf_model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0, eval_metric='mlogloss')
    clf_model.fit(X_train_c, y_train_c)
    clf_pred = clf_model.predict(X_test_c)
    clf_acc = accuracy_score(y_test_c, clf_pred)
    
    return {
        'reg_model': reg_model,
        'clf_model': clf_model,
        'reg_r2': reg_r2,
        'reg_rmse': reg_rmse,
        'clf_acc': clf_acc,
        'reg_features': reg_features,
        'clf_features': clf_features,
        'y_test_r': y_test_r,
        'reg_pred': reg_pred,
        'y_test_c': y_test_c,
        'clf_pred': clf_pred
    }


def main():
    # Header
    st.markdown('<div class="main-header">⚡ Steel Industry Energy AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Energy Consumption Prediction & Sustainability Analytics</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        df_processed, load_encoder = preprocess_data(df)
        models = train_models(df_processed)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data file is in the correct location: data/Steel_industry_data.csv")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.radio("Select Page:", 
                            ["📊 Dashboard", "🔮 Predictions", "📈 Model Performance", "🌱 Sustainability"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.write(f"Total Records: **{len(df):,}**")
    st.sidebar.write(f"Features: **{len(df.columns)}**")
    st.sidebar.write(f"Time Range: **2018**")
    
    # Pages
    if page == "📊 Dashboard":
        show_dashboard(df, df_processed)
    elif page == "🔮 Predictions":
        show_predictions(models, load_encoder)
    elif page == "📈 Model Performance":
        show_model_performance(models, load_encoder)
    else:
        show_sustainability(df, df_processed)


def show_dashboard(df, df_processed):
    """Dashboard page with data overview."""
    st.header("📊 Data Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Energy (kWh)", f"{df['Usage_kWh'].mean():.2f}")
    with col2:
        st.metric("Total Records", f"{len(df):,}")
    with col3:
        st.metric("Avg CO2 (kg)", f"{df['CO2(tCO2)'].mean()*1000:.2f}")
    with col4:
        max_load_pct = (df['Load_Type'] == 'Maximum_Load').mean() * 100
        st.metric("Max Load %", f"{max_load_pct:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Usage_kWh'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Energy Consumption (kWh)')
        ax.set_ylabel('Frequency')
        ax.axvline(df['Usage_kWh'].mean(), color='red', linestyle='--', label=f'Mean: {df["Usage_kWh"].mean():.2f}')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Load Type Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        load_counts = df['Load_Type'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(load_counts.index, load_counts.values, color=colors, edgecolor='black')
        ax.set_xlabel('Load Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()
    
    # Time Analysis
    st.subheader("Energy Consumption Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_avg = df_processed.groupby('Hour')['Usage_kWh'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='steelblue')
        ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Energy (kWh)')
        ax.set_title('Hourly Energy Consumption')
        ax.set_xticks(range(0, 24, 2))
        st.pyplot(fig)
        plt.close()
    
    with col2:
        week_avg = df.groupby('WeekStatus')['Usage_kWh'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(week_avg.index, week_avg.values, color=['coral', 'lightgreen'], edgecolor='black')
        ax.set_xlabel('Week Status')
        ax.set_ylabel('Avg Energy (kWh)')
        ax.set_title('Weekday vs Weekend')
        st.pyplot(fig)
        plt.close()


def show_predictions(models, load_encoder):
    """Prediction page for making new predictions."""
    st.header("🔮 Make Predictions")
    
    st.markdown("Enter the parameters below to predict energy consumption and load type.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        lagging_power = st.slider("Lagging Reactive Power (kVarh)", 0.0, 50.0, 10.0)
        leading_power = st.slider("Leading Reactive Power (kVarh)", 0.0, 10.0, 2.0)
        co2 = st.slider("CO2 Emissions (tCO2)", 0.0, 0.1, 0.02)
        lagging_pf = st.slider("Lagging Power Factor (%)", 50.0, 100.0, 85.0)
        leading_pf = st.slider("Leading Power Factor (%)", 50.0, 100.0, 85.0)
        
    with col2:
        st.subheader("Time Parameters")
        
        hour = st.slider("Hour of Day", 0, 23, 12)
        day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        week_status = 'Weekend' if day in ['Saturday', 'Sunday'] else 'Weekday'
        st.write(f"Week Status: **{week_status}**")
        
        load_type = st.selectbox("Load Type", ['Light_Load', 'Medium_Load', 'Maximum_Load'])
    
    # Make prediction
    if st.button("🔮 Predict Energy Consumption", type="primary"):
        # Prepare input
        nsm = hour * 3600
        is_peak = 1 if (8 <= hour <= 18 and week_status == 'Weekday') else 0
        
        if hour < 6:
            time_period = 0  # Night
        elif hour < 12:
            time_period = 1  # Morning
        elif hour < 18:
            time_period = 2  # Afternoon
        else:
            time_period = 3  # Evening
        
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        load_map = {'Light_Load': 0, 'Maximum_Load': 1, 'Medium_Load': 2}
        week_map = {'Weekday': 0, 'Weekend': 1}
        
        input_data = np.array([[
            lagging_power, leading_power, co2, lagging_pf, leading_pf,
            nsm, hour, is_peak, week_map[week_status], day_map[day],
            time_period, load_map[load_type]
        ]])
        
        # Predict
        energy_pred = models['reg_model'].predict(input_data)[0]
        
        st.success(f"### Predicted Energy Consumption: **{energy_pred:.2f} kWh**")
        
        # Show context
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted", f"{energy_pred:.2f} kWh")
        with col2:
            if energy_pred < 30:
                st.metric("Category", "Low 🟢")
            elif energy_pred < 60:
                st.metric("Category", "Medium 🟡")
            else:
                st.metric("Category", "High 🔴")
        with col3:
            est_co2 = energy_pred * 0.0005
            st.metric("Est. CO2", f"{est_co2*1000:.2f} kg")


def show_model_performance(models, load_encoder):
    """Model performance page."""
    st.header("📈 Model Performance")
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Regression Model (Energy Prediction)")
        st.metric("R² Score", f"{models['reg_r2']:.4f}")
        st.metric("RMSE", f"{models['reg_rmse']:.4f} kWh")
        
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(models['y_test_r'], models['reg_pred'], alpha=0.5, s=20)
        ax.plot([models['y_test_r'].min(), models['y_test_r'].max()], 
                [models['y_test_r'].min(), models['y_test_r'].max()], 
                'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Energy (kWh)')
        ax.set_ylabel('Predicted Energy (kWh)')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🎯 Classification Model (Load Type)")
        st.metric("Accuracy", f"{models['clf_acc']:.4f}")
        st.metric("Accuracy %", f"{models['clf_acc']*100:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(models['y_test_c'], models['clf_pred'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=load_encoder.classes_, yticklabels=load_encoder.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        plt.close()
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    importance = models['reg_model'].feature_importances_
    features = models['reg_features']
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance - XGBoost Regressor')
    st.pyplot(fig)
    plt.close()


def show_sustainability(df, df_processed):
    """Sustainability insights page."""
    st.header("🌱 Sustainability Insights")
    
    # CO2 Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_co2 = df['CO2(tCO2)'].sum()
        st.metric("Total CO2 (tonnes)", f"{total_co2:.2f}")
    with col2:
        avg_co2 = df['CO2(tCO2)'].mean() * 1000
        st.metric("Avg CO2 per Reading (kg)", f"{avg_co2:.2f}")
    with col3:
        corr = df['CO2(tCO2)'].corr(df['Usage_kWh'])
        st.metric("CO2-Energy Correlation", f"{corr:.4f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CO2 by Load Type")
        load_order = ['Light_Load', 'Medium_Load', 'Maximum_Load']
        co2_by_load = df.groupby('Load_Type')['CO2(tCO2)'].mean().reindex(load_order) * 1000
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(co2_by_load.index, co2_by_load.values, color=colors, edgecolor='black')
        ax.set_xlabel('Load Type')
        ax.set_ylabel('Avg CO2 (kg)')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("CO2 vs Energy Consumption")
        sample = df.sample(min(2000, len(df)), random_state=42)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample['Usage_kWh'], sample['CO2(tCO2)'] * 1000, alpha=0.5, s=20, c='green')
        ax.set_xlabel('Energy (kWh)')
        ax.set_ylabel('CO2 (kg)')
        st.pyplot(fig)
        plt.close()
    
    # Recommendations
    st.subheader("💡 Sustainability Recommendations")
    
    st.markdown("""
    Based on the analysis, here are key recommendations for reducing energy consumption and CO2 emissions:
    
    1. **Shift Heavy Operations**: Move Maximum Load operations to off-peak hours (before 8 AM or after 6 PM) to reduce peak demand.
    
    2. **Weekend Optimization**: Consider scheduling maintenance and light operations during weekends when overall consumption is lower.
    
    3. **Power Factor Improvement**: Monitor and improve power factor to reduce reactive power losses.
    
    4. **Real-time Monitoring**: Implement the predictive models for real-time energy monitoring and anomaly detection.
    
    5. **CO2 Tracking**: Use the strong correlation between energy and CO2 to set emission reduction targets.
    """)


if __name__ == "__main__":
    main()
