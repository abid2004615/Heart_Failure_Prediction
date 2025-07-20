"""
Utility functions for the Heart Disease Prediction Application.

This module contains helper functions for data processing, validation,
and common operations used throughout the application.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from config import (
    VALIDATION_RANGES,
    FEATURE_OPTIONS,
    DEFAULT_VALUES,
    get_feature_label,
    validate_feature_value
)

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = 'app.log', level: str = 'INFO') -> None:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def validate_inputs(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input features for reasonable medical ranges.

    Args:
        features: Dictionary of input features

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        # Validate each feature
        for feature, value in features.items():
            if not validate_feature_value(feature, value):
                range_config = VALIDATION_RANGES.get(feature, {})
                unit = range_config.get('unit', '')
                min_val = range_config.get('min', '')
                max_val = range_config.get('max', '')

                error_msg = f"{feature.title()} must be between {min_val} and {max_val} {unit}".strip()
                errors.append(error_msg)

        # Additional medical logic validations
        if features.get('age', 0) < 18 and features.get('sex', 0) == 1:
            errors.append("Age validation: Consider pediatric cardiology for patients under 18")

        if features.get('trestbps', 0) > 180:
            errors.append("Blood pressure is very high - consider immediate medical attention")

        if features.get('chol', 0) > 500:
            errors.append("Cholesterol level is very high - consider medical consultation")

        return len(errors) == 0, errors

    except Exception as e:
        logger.error(f"Input validation error: {e}")
        errors.append(f"Validation error: {str(e)}")
        return False, errors


def prepare_features_for_model(features: Dict[str, Any]) -> List[float]:
    """
    Prepare features dictionary for model prediction.

    Args:
        features: Dictionary of input features

    Returns:
        List of feature values in the correct order for the model
    """
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca'
    ]

    return [features[feature] for feature in feature_order]


def format_feature_display(features: Dict[str, Any]) -> Dict[str, str]:
    """
    Format features for display with human-readable labels.

    Args:
        features: Dictionary of input features

    Returns:
        Dictionary of formatted feature displays
    """
    formatted = {}

    for feature, value in features.items():
        if feature in FEATURE_OPTIONS:
            formatted[feature] = get_feature_label(feature, value)
        elif feature in VALIDATION_RANGES:
            unit = VALIDATION_RANGES[feature].get('unit', '')
            formatted[feature] = f"{value} {unit}".strip()
        else:
            formatted[feature] = str(value)

    return formatted


def calculate_risk_factors(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional risk factors based on input features.

    Args:
        features: Dictionary of input features

    Returns:
        Dictionary of calculated risk factors
    """
    risk_factors = {
        'high_blood_pressure': features.get('trestbps', 0) > 140,
        'high_cholesterol': features.get('chol', 0) > 240,
        'diabetes_risk': features.get('fbs', 0) == 1,
        'exercise_induced_angina': features.get('exang', 0) == 1,
        'abnormal_ecg': features.get('restecg', 0) > 0,
        'age_risk': features.get('age', 0) > 65,
        'male_gender': features.get('sex', 0) == 1
    }

    # Calculate total risk score
    risk_score = sum(risk_factors.values())
    risk_factors['total_risk_score'] = risk_score
    risk_factors['risk_level'] = 'High' if risk_score >= 4 else 'Medium' if risk_score >= 2 else 'Low'

    return risk_factors


def export_results(
        prediction: int,
        features: Dict[str, Any],
        recommendations: List[str],
        filename: str
) -> str:
    """
    Export prediction results to JSON file.

    Args:
        prediction: Model prediction (0 or 1)
        features: Input features
        recommendations: List of recommendations
        filename: Output filename

    Returns:
        JSON string of results
    """
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'risk_level': 'High' if prediction == 1 else 'Low',
        'features': features,
        'formatted_features': format_feature_display(features),
        'risk_factors': calculate_risk_factors(features),
        'recommendations': recommendations,
        'model_info': {
            'version': '1.0.0',
            'features_used': list(features.keys())
        }
    }

    return json.dumps(results_data, indent=2)


def load_model_safely(model_path: Path) -> Optional[Any]:
    """
    Safely load the machine learning model with error handling.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model or None if loading fails
    """
    try:
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        import pickle
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def get_default_features() -> Dict[str, Any]:
    """
    Get default feature values for the application.

    Returns:
        Dictionary of default feature values
    """
    return DEFAULT_VALUES.copy()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def create_backup(features: Dict[str, Any], prediction: int) -> None:
    """
    Create a backup of the prediction results.

    Args:
        features: Input features
        prediction: Model prediction
    """
    try:
        backup_dir = Path('backups')
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_dir / f"prediction_backup_{timestamp}.json"

        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'risk_level': 'High' if prediction == 1 else 'Low'
        }

        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

        logger.info(f"Backup created: {backup_file}")

    except Exception as e:
        logger.error(f"Error creating backup: {e}")


def get_statistics() -> Dict[str, Any]:
    """
    Get application usage statistics.

    Returns:
        Dictionary of statistics
    """
    try:
        stats_file = Path('app_statistics.json')
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_predictions': 0,
                'high_risk_predictions': 0,
                'low_risk_predictions': 0,
                'last_updated': datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        return {}


def update_statistics(prediction: int) -> None:
    """
    Update application usage statistics.

    Args:
        prediction: Model prediction (0 or 1)
    """
    try:
        stats = get_statistics()
        stats['total_predictions'] = stats.get('total_predictions', 0) + 1

        if prediction == 1:
            stats['high_risk_predictions'] = stats.get('high_risk_predictions', 0) + 1
        else:
            stats['low_risk_predictions'] = stats.get('low_risk_predictions', 0) + 1

        stats['last_updated'] = datetime.now().isoformat()

        with open('app_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        logger.error(f"Error updating statistics: {e}")