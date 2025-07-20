"""
Unit tests for the Heart Disease Prediction Application.

This module contains comprehensive tests for all major functions
and components of the application.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from config import (
    VALIDATION_RANGES,
    FEATURE_OPTIONS,
    DEFAULT_VALUES,
    get_feature_label,
    validate_feature_value,
    get_model_path
)
from utils import (
    validate_inputs,
    prepare_features_for_model,
    format_feature_display,
    calculate_risk_factors,
    export_results,
    get_default_features,
    sanitize_filename
)


class TestConfig(unittest.TestCase):
    """Test configuration functions."""

    def test_get_feature_label(self):
        """Test feature label retrieval."""
        # Test valid feature
        self.assertEqual(get_feature_label('sex', 0), 'Female')
        self.assertEqual(get_feature_label('sex', 1), 'Male')

        # Test invalid feature
        self.assertEqual(get_feature_label('invalid', 0), '0')

        # Test invalid value
        self.assertEqual(get_feature_label('sex', 99), '99')

    def test_validate_feature_value(self):
        """Test feature value validation."""
        # Test valid values
        self.assertTrue(validate_feature_value('age', 50))
        self.assertTrue(validate_feature_value('trestbps', 120))

        # Test invalid values
        self.assertFalse(validate_feature_value('age', 150))  # Too high
        self.assertFalse(validate_feature_value('age', 0))  # Too low

        # Test non-validated feature
        self.assertTrue(validate_feature_value('sex', 0))

    def test_get_model_path(self):
        """Test model path retrieval."""
        path = get_model_path()
        self.assertIsInstance(path, Path)
        self.assertEqual(path.name, 'heart_model.pkl')


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_validate_inputs(self):
        """Test input validation."""
        # Test valid inputs
        valid_features = {
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
        is_valid, errors = validate_inputs(valid_features)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Test invalid inputs
        invalid_features = {
            'age': 150,  # Too high
            'trestbps': 300,  # Too high
            'chol': 700,  # Too high
            'thalach': 50,  # Too low
            'oldpeak': 10.0  # Too high
        }
        is_valid, errors = validate_inputs(invalid_features)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_prepare_features_for_model(self):
        """Test feature preparation for model."""
        features = {
            'age': 50,
            'sex': 0,
            'cp': 1,
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

        feature_list = prepare_features_for_model(features)
        self.assertEqual(len(feature_list), 12)
        self.assertEqual(feature_list[0], 50)  # age
        self.assertEqual(feature_list[1], 0)  # sex
        self.assertEqual(feature_list[2], 1)  # cp

    def test_format_feature_display(self):
        """Test feature display formatting."""
        features = {
            'age': 50,
            'sex': 0,
            'cp': 1,
            'trestbps': 120,
            'chol': 200
        }

        formatted = format_feature_display(features)
        self.assertEqual(formatted['age'], '50 years')
        self.assertEqual(formatted['sex'], 'Female')
        self.assertEqual(formatted['cp'], 'Atypical angina')
        self.assertEqual(formatted['trestbps'], '120 mm Hg')
        self.assertEqual(formatted['chol'], '200 mg/dL')

    def test_calculate_risk_factors(self):
        """Test risk factor calculation."""
        # Test low risk
        low_risk_features = {
            'age': 30,
            'sex': 0,
            'trestbps': 110,
            'chol': 180,
            'fbs': 0,
            'exang': 0,
            'restecg': 0
        }

        risk_factors = calculate_risk_factors(low_risk_features)
        self.assertEqual(risk_factors['risk_level'], 'Low')
        self.assertLess(risk_factors['total_risk_score'], 4)

        # Test high risk
        high_risk_features = {
            'age': 70,
            'sex': 1,
            'trestbps': 160,
            'chol': 280,
            'fbs': 1,
            'exang': 1,
            'restecg': 1
        }

        risk_factors = calculate_risk_factors(high_risk_features)
        self.assertEqual(risk_factors['risk_level'], 'High')
        self.assertGreaterEqual(risk_factors['total_risk_score'], 4)

    def test_export_results(self):
        """Test results export."""
        features = {
            'age': 50,
            'sex': 0,
            'cp': 1,
            'trestbps': 120,
            'chol': 200
        }
        recommendations = ['Test recommendation 1', 'Test recommendation 2']

        json_str = export_results(1, features, recommendations, 'test.json')
        results = json.loads(json_str)

        self.assertEqual(results['prediction'], 1)
        self.assertEqual(results['risk_level'], 'High')
        self.assertEqual(results['features'], features)
        self.assertEqual(results['recommendations'], recommendations)
        self.assertIn('timestamp', results)
        self.assertIn('model_info', results)

    def test_get_default_features(self):
        """Test default features retrieval."""
        defaults = get_default_features()
        self.assertEqual(defaults['age'], 50)
        self.assertEqual(defaults['sex'], 0)
        self.assertEqual(defaults['trestbps'], 120)
        # Ensure it's a copy, not the original
        self.assertIsNot(defaults, DEFAULT_VALUES)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test normal filename
        self.assertEqual(sanitize_filename('normal_file.json'), 'normal_file.json')

        # Test filename with invalid characters
        self.assertEqual(sanitize_filename('file<>:"/\\|?*.json'), 'file_______.json')

        # Test very long filename
        long_filename = 'a' * 150 + '.json'
        sanitized = sanitize_filename(long_filename)
        self.assertLessEqual(len(sanitized), 100)


class TestModelIntegration(unittest.TestCase):
    """Test model integration functions."""

    @patch('utils.load_model_safely')
    def test_model_loading(self, mock_load):
        """Test model loading with mock."""
        # Mock successful loading
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        from utils import load_model_safely
        model = load_model_safely(Path('test_model.pkl'))

        self.assertIsNotNone(model)
        mock_load.assert_called_once()

    @patch('utils.load_model_safely')
    def test_model_loading_failure(self, mock_load):
        """Test model loading failure."""
        # Mock loading failure
        mock_load.return_value = None

        from utils import load_model_safely
        model = load_model_safely(Path('nonexistent_model.pkl'))

        self.assertIsNone(model)


class TestDataValidation(unittest.TestCase):
    """Test data validation scenarios."""

    def test_edge_cases(self):
        """Test edge case validations."""
        # Test boundary values
        boundary_features = {
            'age': 1,  # Minimum age
            'trestbps': 50,  # Minimum blood pressure
            'chol': 100,  # Minimum cholesterol
            'thalach': 60,  # Minimum heart rate
            'oldpeak': 0.0  # Minimum ST depression
        }
        is_valid, errors = validate_inputs(boundary_features)
        self.assertTrue(is_valid)

        # Test maximum boundary values
        max_boundary_features = {
            'age': 120,  # Maximum age
            'trestbps': 200,  # Maximum blood pressure
            'chol': 600,  # Maximum cholesterol
            'thalach': 220,  # Maximum heart rate
            'oldpeak': 6.0  # Maximum ST depression
        }
        is_valid, errors = validate_inputs(max_boundary_features)
        self.assertTrue(is_valid)

    def test_medical_logic_validations(self):
        """Test medical logic validations."""
        # Test pediatric case
        pediatric_features = {
            'age': 15,
            'sex': 1,
            'trestbps': 120,
            'chol': 200
        }
        is_valid, errors = validate_inputs(pediatric_features)
        self.assertTrue(is_valid)  # Should pass validation but with warning

        # Test very high blood pressure
        high_bp_features = {
            'age': 50,
            'trestbps': 190,
            'chol': 200
        }
        is_valid, errors = validate_inputs(high_bp_features)
        self.assertFalse(is_valid)  # Should fail due to very high BP


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestModelIntegration))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)