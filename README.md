# Health Risk Predictor

A comprehensive machine learning system for health risk assessment and analysis. This project implements an advanced predictive model that evaluates individual health metrics, generates detailed risk assessments, and provides personalized health recommendations through interactive visualizations and detailed reports.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technical Requirements](#technical-requirements)
- [Installation](#installation)
- [Detailed Usage Guide](#detailed-usage-guide)
- [Implementation Details](#implementation-details)
- [Customization](#customization)
- [Model Performance](#model-performance)
- [Output Examples](#output-examples)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

The Health Risk Predictor is built using scikit-learn's Random Forest Classifier and incorporates multiple health metrics to assess individual health risks. The system generates synthetic training data, creates interactive visualizations, and produces comprehensive health reports with actionable insights.

### Key Components
- HealthRiskPredictor: Core prediction and visualization engine
- HealthRiskReportGenerator: Report generation and analysis system
- Synthetic Data Generator: Creates realistic health data for model training

## System Architecture

```
health_risk_predictor/
├── __init__.py
├── predictor/
│   ├── __init__.py
│   ├── model.py           # Core prediction engine
│   ├── visualizer.py      # Visualization components
│   └── data_generator.py  # Synthetic data generation
├── report/
│   ├── __init__.py
│   ├── generator.py       # Report generation logic
│   └── templates/         # Report templates
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
└── health_risk_reports/   # Generated reports directory
```

## Features

### 1. Health Metrics Analysis
- **Core Health Indicators**:
  - Age (18-90 years)
  - Heart Rate (60-100 bpm)
  - Blood Pressure (Systolic: 90-120 mmHg, Diastolic: 60-80 mmHg)
  - Glucose Level (70-99 mg/dL)
  - BMI (18.5-24.9)
  - Cholesterol (125-200 mg/dL)
  - Exercise Hours (2.5-5 hours/week)
  - Smoking Status (binary)

### 2. Machine Learning Model
- Random Forest Classifier with optimized hyperparameters
- Standardized feature scaling
- Cross-validation support
- Feature importance analysis

### 3. Visualization Capabilities
- ROC curves with AUC scores
- Feature importance plots
- Patient value comparisons
- Health range visualizations
- Interactive matplotlib/seaborn plots

### 4. Report Generation
- Comprehensive risk assessment
- Detailed metric analysis
- Personalized recommendations
- Model performance metrics
- PDF and text report formats

## Technical Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
joblib>=1.0.0
```

## Installation

### 1. Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/health-risk-predictor.git
cd health-risk-predictor

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Development Installation

```bash
# Install additional development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Detailed Usage Guide

### 1. Basic Usage

```python
from health_risk_predictor import HealthRiskReportGenerator

# Initialize the generator
generator = HealthRiskReportGenerator()

# Create patient data dictionary
patient_data = {
    'Age': 45,
    'Heart_Rate': 75,
    'Systolic_BP': 130,
    'Diastolic_BP': 85,
    'Glucose_Level': 110,
    'BMI': 27.5,
    'Cholesterol': 210,
    'Exercise_Hours_Week': 2,
    'Smoking_Status': 0
}

# Generate comprehensive report
report_file = generator.generate_report(patient_data)
```

### 2. Advanced Usage

```python
from health_risk_predictor import HealthRiskPredictor
import pandas as pd

# Initialize predictor
predictor = HealthRiskPredictor()

# Generate custom synthetic dataset
custom_data = predictor.generate_synthetic_data(n_samples=2000)

# Custom data preparation
X_train, X_test, y_train, y_test = predictor.prepare_data(custom_data)

# Train model with custom parameters
predictor.pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ))
])
predictor.train_model(X_train, y_train)

# Generate visualizations with custom patient data
patient_df = pd.DataFrame([patient_data])
predictor.create_visualizations(X_test, y_test, patient_df)
```

## Implementation Details

### Model Architecture
- **Preprocessing**: StandardScaler for feature normalization
- **Classifier**: RandomForestClassifier with following default parameters:
  ```python
  {
      'n_estimators': 100,
      'max_depth': None,
      'min_samples_split': 2,
      'min_samples_leaf': 1,
      'random_state': 42
  }
  ```

### Synthetic Data Generation
The system generates realistic health data using normal distributions with clinically relevant parameters:
```python
data = {
    'Age': np.random.normal(50, 15, n_samples).clip(18, 90),
    'Heart_Rate': np.random.normal(75, 12, n_samples).clip(45, 120),
    # ... other metrics
}
```

### Risk Assessment Logic
Risk levels are determined by counting risk factors:
- High Risk: ≥ 3 risk factors
- Moderate Risk: 1-2 risk factors
- Low Risk: 0 risk factors

## Customization

### 1. Modifying Health Ranges
```python
predictor.healthy_ranges = {
    'Age': {'min': 18, 'max': 90, 'optimal': 'N/A'},
    'Heart_Rate': {'min': 60, 'max': 100, 'optimal': '60-100 bpm'},
    # Add or modify ranges
}
```

### 2. Custom Visualization Settings
```python
def create_visualizations(self, X_test, y_test, current_patient=None):
    plt.style.use('your_preferred_style')
    # Modify visualization parameters
    plt.figure(figsize=(12, 8))
    # ... rest of the visualization code
```

## Model Performance

Typical performance metrics on synthetic data:
- Accuracy: 0.85-0.90
- ROC-AUC: 0.88-0.92
- F1 Score: 0.84-0.89

## Output Examples

### 1. Sample Report Structure
```
HEALTH RISK ANALYSIS REPORT
==========================
1. MODEL PERFORMANCE METRICS
2. PATIENT RISK ANALYSIS
   - Overall Risk Assessment
   - Critical Factors
   - Recommendations
3. DETAILED MEASUREMENTS
```

### 2. Visualization Outputs
- ROC curves showing model performance
- Feature importance plots with patient comparisons
- Health metric comparisons against normal ranges

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add AmazingFeature'`)
5. Push to branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## Troubleshooting

### Common Issues and Solutions

1. **Installation Issues**
   ```bash
   # If facing numpy/pandas installation issues
   pip install --upgrade pip
   pip install numpy pandas --force-reinstall
   ```

2. **Memory Errors**
   - Reduce synthetic data sample size
   - Adjust random forest parameters
   - Use batch processing for large datasets

3. **Visualization Errors**
   - Check matplotlib backend
   - Ensure all required data is available
   - Verify data types and ranges


