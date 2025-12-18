# Steel Industry Energy Consumption Prediction

## AI-Powered Energy Analytics for Sustainable Manufacturing

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Author** | Dinesh |
| **Student ID** | GH1040084 |
| **Module** | M516 Business Project in Big Data & AI |
| **Institution** | GISMA University of Applied Sciences |
| **Date** | December 2025 |

### 🔗 Project Links

- **Live project link **: https://steel-energy-ai-energy-consumption.onrender.com
- **Video Demonstration**: https://youtu.be/_rfLcE8P0n0

---

## 📖 Project Overview

This project develops an **AI-based system for predicting energy consumption** in steel manufacturing to support sustainability goals. The system addresses the following objectives:

1. **Energy Consumption Prediction** - Regression models to forecast kWh usage
2. **Load Type Classification** - Classify operational load (Light/Medium/Maximum)
3. **Sustainability Analytics** - CO2 emission analysis and optimization recommendations

### Problem Domain

Steel manufacturing is one of the most energy-intensive industries globally. Efficient energy management is crucial for:
- Reducing operational costs
- Minimizing carbon footprint
- Meeting sustainability regulations
- Optimizing production schedules

### Solution Approach

This project applies machine learning techniques to predict energy consumption patterns and classify load types, enabling:
- Proactive energy management
- Data-driven operational decisions
- Sustainability monitoring

---

## 📊 Dataset

**Source**: [Kaggle - Steel Industry Energy Consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption)

**Original Source**: DAEWOO Steel Co. Ltd, Gwangyang, South Korea

### Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `Usage_kWh` | Continuous | Energy consumption in kilowatt-hours |
| `Lagging_Current_Reactive.Power_kVarh` | Continuous | Lagging reactive power |
| `Leading_Current_Reactive_Power_kVarh` | Continuous | Leading reactive power |
| `CO2(tCO2)` | Continuous | Carbon dioxide emissions |
| `Lagging_Current_Power_Factor` | Continuous | Lagging power factor (%) |
| `Leading_Current_Power_Factor` | Continuous | Leading power factor (%) |
| `NSM` | Continuous | Number of seconds from midnight |
| `WeekStatus` | Categorical | Weekend or Weekday |
| `Day_of_week` | Categorical | Day name (Monday-Sunday) |
| `Load_Type` | Categorical | Light_Load, Medium_Load, Maximum_Load |

### Dataset Statistics

- **Total Records**: 35,040 (one year at 15-minute intervals)
- **Time Period**: 2018
- **Missing Values**: None

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Steel Energy AI System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Raw Data   │───>│ Preprocessing │───>│   Feature    │       │
│  │  (CSV/API)   │    │   Pipeline    │    │ Engineering  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ML Models Layer                        │   │
│  │  ┌────────────────────┐  ┌────────────────────┐          │   │
│  │  │  Regression Models │  │Classification Models│          │   │
│  │  │  - Linear Reg.     │  │  - Logistic Reg.   │          │   │
│  │  │  - Random Forest   │  │  - Random Forest   │          │   │
│  │  │  - Gradient Boost  │  │  - Gradient Boost  │          │   │
│  │  │  - XGBoost         │  │  - XGBoost         │          │   │
│  │  └────────────────────┘  └────────────────────┘          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Output & Visualization Layer                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ Predictions │ │  Analytics  │ │  Streamlit Dashboard │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
steel-energy-ai/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── run_flask.sh                 # Script to run Flask app
│
├── data/
│   └── Steel_industry_data.csv  # Dataset
│
├── notebooks/
│   └── complete_analysis.ipynb  # Full analysis notebook
│
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing
│   ├── models.py                # ML model definitions
│   ├── evaluation.py            # Evaluation metrics
│   └── visualization.py         # Visualization functions
│
├── flask_app/                   # Flask Web Application
│   ├── app.py                   # Main Flask application
│   └── templates/
│       ├── index.html           # Input form page
│       ├── result.html          # Results & graphs page
│       └── error.html           # Error page
│
├── app/
│   └── streamlit_app.py         # Streamlit dashboard (alternative)
│
├── results/
│   ├── figures/                 # Generated plots
│   ├── regression_results.csv   # Model results
│   └── classification_results.csv
│
└── docs/
    └── project_report.pdf       # Final report
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/[your-username]/steel-energy-ai.git
cd steel-energy-ai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Analysis

```bash
# Run Jupyter Notebook
jupyter notebook notebooks/complete_analysis.ipynb
```

### Step 5: Launch Web Application

#### Option A: Flask Application 
```bash
cd flask_app
python app.py
# Open http://localhost:5000 in your browser
```

#### Option B: Streamlit Application
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🌐 Flask Web Application

The Flask application provides an interactive web interface for energy prediction:

### Features
- **User-friendly Input Form**: Enter all required parameters
- **Real-time Predictions**: Get instant energy consumption predictions
- **Visual Analytics**: Interactive graphs comparing predictions with historical data
- **AI Insights**: Automated recommendations based on prediction results

### Input Parameters
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Hour | Select | 0-23 | Hour of the day |
| Day of Week | Select | Mon-Sun | Day of the week |
| Week Status | Select | Weekday/Weekend | Working day status |
| Load Type | Select | Light/Medium/Maximum | Expected operational load |
| Lagging Reactive Power | Number | 0-100 kVarh | Lagging reactive power |
| Leading Reactive Power | Number | 0-20 kVarh | Leading reactive power |
| Lagging Power Factor | Number | 50-100% | Lagging power factor |
| Leading Power Factor | Number | 50-100% | Leading power factor |
| CO2 Emissions | Number | 0-0.1 tCO2 | Carbon dioxide emissions |

### Output
- **Energy Prediction**: Predicted kWh consumption
- **Load Classification**: Predicted load type
- **CO2 Estimation**: Estimated carbon emissions
- **Visualization Graphs**: 
  - Prediction analysis dashboard
  - Historical comparison charts
- **AI Recommendations**: Insights for optimization

---

## 📈 Results

### Regression Model Performance (Energy Prediction)

| Model | R² Score | RMSE (kWh) | MAE (kWh) |
|-------|----------|------------|-----------|
| **XGBoost** | **0.9847** | **3.42** | **2.15** |
| Random Forest | 0.9756 | 4.31 | 2.78 |
| Gradient Boosting | 0.9689 | 4.87 | 3.12 |
| Linear Regression | 0.9234 | 7.65 | 5.43 |

### Classification Model Performance (Load Type)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **XGBoost** | **0.9912** | **0.9908** |
| Random Forest | 0.9867 | 0.9854 |
| Gradient Boosting | 0.9823 | 0.9812 |
| Logistic Regression | 0.9456 | 0.9423 |

### Key Findings

1. **XGBoost** achieves the best performance for both tasks
2. **CO2 emissions** are strongly correlated with energy consumption (r = 0.97)
3. **Maximum Load** operations consume 5x more energy than Light Load
4. **Peak hours** (8 AM - 6 PM) show 35% higher energy consumption

---

## 🌱 Sustainability Insights

1. **CO2-Energy Correlation**: Strong positive correlation enables CO2 prediction from energy usage
2. **Load Optimization**: Shifting Maximum Load operations to off-peak hours can reduce peak demand by 25%
3. **Weekend Efficiency**: Weekend operations show 15% lower energy consumption
4. **Predictive Monitoring**: Real-time predictions enable proactive energy management

---

## 💡 Future Work

1. **Real-time Integration**: Connect with IoT sensors for live predictions
2. **Deep Learning**: Implement LSTM models for time-series forecasting
3. **Anomaly Detection**: Add real-time anomaly detection for equipment failures
4. **Mobile App**: Develop mobile dashboard for on-the-go monitoring
5. **Multi-site Deployment**: Scale to multiple manufacturing facilities

---

## 📚 References

1. Sathishkumar V E, Shin C., Cho Y., "Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city", Building Research & Information, 2021.

2. Dataset: [Kaggle - Steel Industry Energy Consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption)

3. Scikit-learn Documentation: https://scikit-learn.org/stable/

4. XGBoost Documentation: https://xgboost.readthedocs.io/

5. Streamlit Documentation: https://docs.streamlit.io/

---

## 📝 License

This project is submitted as part of academic coursework for M516 Business Project in Big Data & AI at GISMA University of Applied Sciences.

---

## 👤 Author

   Dinesh 
- Student ID: Gh1040084
- Program: MSc Computer Science
- Institution: GISMA University of Applied Sciences

---

*Last Updated: December 2025*
