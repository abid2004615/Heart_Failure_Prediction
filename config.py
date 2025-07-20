"""
Configuration settings for the Heart Disease Prediction Application.

This module contains all configuration parameters, constants, and settings
used throughout the application.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Application Configuration
APP_CONFIG = {
    'name': 'Heart Disease Prediction App',
    'version': '1.0.0',
    'description': 'A Streamlit web application for predicting heart disease risk',
    'author': '[Your Name]',
    'contact': '[Your Contact Info]',
    'icon': '❤️',
    'layout': 'wide',
    'page_title': 'Heart Disease Prediction'
}

# Model Configuration
MODEL_CONFIG = {
    'path': 'heart_model.pkl',
    'input_features': [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca'
    ],
    'output_classes': ['Low Risk', 'High Risk']
}

# Input Validation Ranges
VALIDATION_RANGES = {
    'age': {'min': 1, 'max': 120, 'unit': 'years'},
    'trestbps': {'min': 50, 'max': 200, 'unit': 'mm Hg'},
    'chol': {'min': 100, 'max': 600, 'unit': 'mg/dL'},
    'thalach': {'min': 60, 'max': 220, 'unit': 'bpm'},
    'oldpeak': {'min': 0.0, 'max': 6.0, 'unit': ''}
}

# Feature Descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Patient age in years',
    'sex': 'Gender (0: Female, 1: Male)',
    'cp': 'Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
    'trestbps': 'Resting blood pressure in mm Hg',
    'chol': 'Serum cholesterol in mg/dL',
    'fbs': 'Fasting blood sugar > 120 mg/dL (1: True, 0: False)',
    'restecg': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved during exercise',
    'exang': 'Exercise induced angina (1: Yes, 0: No)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
    'ca': 'Number of major vessels (0-4) colored by fluoroscopy'
}

# Feature Options for Select Boxes
FEATURE_OPTIONS = {
    'sex': {
        'options': [0, 1],
        'labels': ['Female', 'Male']
    },
    'cp': {
        'options': [0, 1, 2, 3],
        'labels': ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic']
    },
    'fbs': {
        'options': [0, 1],
        'labels': ['False', 'True']
    },
    'restecg': {
        'options': [0, 1, 2],
        'labels': ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']
    },
    'exang': {
        'options': [0, 1],
        'labels': ['No', 'Yes']
    },
    'slope': {
        'options': [0, 1, 2],
        'labels': ['Upsloping', 'Flat', 'Downsloping']
    },
    'ca': {
        'options': [0, 1, 2, 3, 4],
        'labels': ['0', '1', '2', '3', '4']
    }
}

# Default Values
DEFAULT_VALUES = {
    'age': 50,
    'sex': 0,
    'cp': 0,
    'trestbps': 120,
    'chol': 200,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 0.0,
    'slope': 0,
    'ca': 0
}

# Risk Assessment Messages
RISK_MESSAGES = {
    'high_risk': {
        'title': '⚠️ **High Risk of Heart Disease**',
        'message': 'Please consult with a healthcare professional for further evaluation.',
        'color': 'error',
        'recommendations': [
            'Schedule an appointment with a cardiologist',
            'Monitor blood pressure regularly',
            'Follow a heart-healthy diet',
            'Engage in regular physical activity as recommended by your doctor',
            'Avoid smoking and limit alcohol consumption'
        ]
    },
    'low_risk': {
        'title': '✅ **Low Risk of Heart Disease**',
        'message': 'Continue maintaining a healthy lifestyle with regular check-ups.',
        'color': 'success',
        'recommendations': [
            'Maintain regular exercise routine',
            'Follow a balanced diet',
            'Get regular health check-ups',
            'Monitor cholesterol and blood pressure',
            'Avoid smoking and excessive alcohol consumption'
        ]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'app.log'
}

# Export Configuration
EXPORT_CONFIG = {
    'format': 'json',
    'filename_template': 'heart_disease_prediction_{age}_{gender}.json',
    'include_features': True,
    'include_recommendations': True
}

def get_model_path() -> Path:
    """Get the path to the model file."""
    return Path(MODEL_CONFIG['path'])

def get_feature_label(feature: str, value: int) -> str:
    """Get the human-readable label for a feature value."""
    if feature in FEATURE_OPTIONS:
        try:
            index = FEATURE_OPTIONS[feature]['options'].index(value)
            return FEATURE_OPTIONS[feature]['labels'][index]
        except ValueError:
            return str(value)
    return str(value)

def validate_feature_value(feature: str, value: Any) -> bool:
    """Validate if a feature value is within acceptable range."""
    if feature in VALIDATION_RANGES:
        range_config = VALIDATION_RANGES[feature]
        return range_config['min'] <= value <= range_config['max']
    return True

def get_export_filename(age: int, sex: int) -> str:
    """Generate export filename based on patient demographics."""
    gender = 'M' if sex == 1 else 'F'
    return EXPORT_CONFIG['filename_template'].format(age=age, gender=gender) 