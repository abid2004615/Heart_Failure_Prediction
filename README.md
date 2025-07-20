# Heart Disease Prediction App

A professional Streamlit web application for predicting heart disease risk based on patient medical parameters. Built with modern Python practices, comprehensive testing, and robust error handling.

## ✨ Features

- **Interactive Web Interface**: User-friendly form with clear labels and helpful tooltips
- **Real-time Risk Prediction**: Instant heart disease risk assessment
- **Comprehensive Validation**: Medical range validation and input sanitization
- **Detailed Results**: Risk assessment with personalized recommendations
- **Export Functionality**: Download results as JSON files
- **Responsive Design**: Two-column layout with sidebar information
- **Professional Logging**: Comprehensive logging for debugging and monitoring
- **Statistics Tracking**: Usage statistics and prediction analytics
- **Backup System**: Automatic backup of prediction results

## 🚀 Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

1. Ensure you have the `heart_model.pkl` file in the same directory as `app.py`
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`)

## 🔧 Input Parameters

The application analyzes 12 key medical parameters:

- **Age**: Patient's age (1-120 years)
- **Sex**: Gender (Female/Male)
- **Chest Pain Type**: Type of chest pain experienced
  - Typical angina
  - Atypical angina
  - Non-anginal pain
  - Asymptomatic
- **Resting Blood Pressure**: Blood pressure in mm Hg (50-200)
- **Cholesterol**: Serum cholesterol in mg/dL (100-600)
- **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dL
- **Resting ECG**: Results of resting electrocardiogram
  - Normal
  - ST-T wave abnormality
  - Left ventricular hypertrophy
- **Max Heart Rate**: Maximum heart rate achieved during exercise (60-220 bpm)
- **Exercise Induced Angina**: Whether angina is induced by exercise
- **ST Depression**: ST depression induced by exercise relative to rest (0.0-6.0)
- **ST Slope**: Slope of the peak exercise ST segment
  - Upsloping
  - Flat
  - Downsloping
- **Major Vessels**: Number of major vessels colored by fluoroscopy (0-4)

## 📊 Output

The application provides comprehensive results:

- **Risk Assessment**: High or Low risk of heart disease
- **Personalized Recommendations**: Actionable medical advice
- **Risk Factor Analysis**: Detailed breakdown of contributing factors
- **Input Summary**: Complete review of all entered parameters
- **Export Options**: Download results for medical records

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_app.py

# Run with pytest (if installed)
pytest test_app.py -v

# Run with coverage
pytest test_app.py --cov=. --cov-report=html
```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings and constants
├── utils.py               # Utility functions and helpers
├── test_app.py            # Comprehensive unit tests
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This documentation
├── app.log               # Application logs (generated)
├── backups/              # Prediction backups (generated)
└── heart_model.pkl       # Trained machine learning model (not included)
```

## 🔒 Security & Privacy

- **Input Validation**: Comprehensive validation of all medical parameters
- **Error Handling**: Robust error handling with user-friendly messages
- **Data Sanitization**: Safe filename handling and data processing
- **Logging**: Secure logging without sensitive data exposure

## 📈 Monitoring & Analytics

- **Usage Statistics**: Track prediction counts and risk distributions
- **Performance Monitoring**: Log application performance and errors
- **Backup System**: Automatic backup of prediction results
- **Export Tracking**: Monitor data export activities

## 🛠️ Development

### Code Quality Tools

```bash
# Code formatting
black app.py config.py utils.py

# Linting
flake8 app.py config.py utils.py

# Type checking
mypy app.py config.py utils.py
```

### Adding New Features

1. Update `config.py` for new configuration parameters
2. Add utility functions to `utils.py`
3. Update the main application in `app.py`
4. Add comprehensive tests to `test_app.py`
5. Update documentation in `README.md`

## ⚠️ Disclaimer

This application is for **educational purposes only**. Always consult healthcare professionals for medical advice and diagnosis. The predictions are based on machine learning models and should not replace professional medical evaluation.

## 📋 Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt`
- **Model**: A trained machine learning model (`heart_model.pkl`)
- **Memory**: Minimum 512MB RAM
- **Storage**: 100MB free space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please contact: [Your Contact Info] 