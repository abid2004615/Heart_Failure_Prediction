"""
Heart Disease Prediction Application

A Streamlit web application for predicting heart disease risk based on patient medical parameters.
This application uses machine learning to assess cardiovascular health risk.

Author: [Your Name]
Version: 1.0.0
"""

import streamlit as st
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model_path': 'heart_model.pkl',
    'app_title': 'Heart Disease Prediction App',
    'app_icon': '❤️',
    'layout': 'wide',
    'page_title': 'Heart Disease Prediction'
}

# Feature descriptions for better user understanding
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

def validate_inputs(features: Dict[str, Any]) -> bool:
    """
    Validate input features for reasonable medical ranges.

    Args:
        features: Dictionary of input features

    Returns:
        bool: True if all inputs are valid, False otherwise
    """
    try:
        # Age validation
        if not (1 <= features['age'] <= 120):
            st.error("Age must be between 1 and 120 years")
            return False

        # Blood pressure validation
        if not (50 <= features['trestbps'] <= 200):
            st.error("Resting blood pressure must be between 50 and 200 mm Hg")
            return False

        # Cholesterol validation
        if not (100 <= features['chol'] <= 600):
            st.error("Cholesterol must be between 100 and 600 mg/dL")
            return False

        # Heart rate validation
        if not (60 <= features['thalach'] <= 220):
            st.error("Maximum heart rate must be between 60 and 220 bpm")
            return False

        # ST depression validation
        if not (0.0 <= features['oldpeak'] <= 6.0):
            st.error("ST depression must be between 0.0 and 6.0")
            return False

        return True

    except KeyError as e:
        st.error(f"Missing required input: {e}")
        return False
    except Exception as e:
        st.error(f"Input validation error: {e}")
        return False

def load_model() -> Optional[Any]:
    """
    Load the trained heart disease prediction model.

    Returns:
        The loaded model or None if loading fails
    """
    try:
        model_path = Path(CONFIG['model_path'])
        if not model_path.exists():
            st.error(f"Model file '{CONFIG['model_path']}' not found. Please ensure the model file is in the same directory.")
            logger.error(f"Model file not found: {CONFIG['model_path']}")
            return None

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return None

def predict_heart_disease(model: Any, features: List[float]) -> Optional[int]:
    """
    Make prediction using the loaded model.

    Args:
        model: The trained machine learning model
        features: List of input features

    Returns:
        Prediction result (0 or 1) or None if prediction fails
    """
    try:
        input_data = np.array([features])
        result = model.predict(input_data)
        logger.info(f"Prediction made successfully: {result[0]}")
        return result[0]

    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return None

def get_risk_description(risk_level: int) -> Dict[str, str]:
    """
    Get description and recommendations based on risk level.

    Args:
        risk_level: 0 for low risk, 1 for high risk

    Returns:
        Dictionary containing title, message, and recommendations
    """
    if risk_level == 1:
        return {
            'title': '⚠️ **High Risk of Heart Disease**',
            'message': 'Please consult with a healthcare professional for further evaluation.',
            'recommendations': [
                'Schedule an appointment with a cardiologist',
                'Monitor blood pressure regularly',
                'Follow a heart-healthy diet',
                'Engage in regular physical activity as recommended by your doctor',
                'Avoid smoking and limit alcohol consumption'
            ]
        }
    else:
        return {
            'title': '✅ **Low Risk of Heart Disease**',
            'message': 'Continue maintaining a healthy lifestyle with regular check-ups.',
            'recommendations': [
                'Maintain regular exercise routine',
                'Follow a balanced diet',
                'Get regular health check-ups',
                'Monitor cholesterol and blood pressure',
                'Avoid smoking and excessive alcohol consumption'
        ]
    }

def generate_pdf_report(prediction: int, features: Dict[str, Any], recommendations: List[str], age: int, sex: int) -> bytes:
    """
    Generate a PDF report for the heart disease prediction.

    Args:
        prediction: Model prediction (0 or 1)
        features: Input features
        recommendations: List of recommendations
        age: Patient age
        sex: Patient sex

    Returns:
        PDF file as bytes
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO

        # Create buffer for PDF
        buffer = BytesIO()

        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Get styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )

        normal_style = styles['Normal']

        # Header
        story.append(Paragraph("Heart Disease Prediction Report", title_style))
        story.append(Spacer(1, 20))

        # Report information
        story.append(Paragraph("Report Information", heading_style))
        report_info = [
            ["Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Report Time:", datetime.now().strftime("%I:%M %p")],
            ["Risk Assessment:", "High Risk" if prediction == 1 else "Low Risk"],
            ["Model Version:", "1.0.0"]
        ]

        info_table = Table(report_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))

        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))

        # Format features for display
        feature_labels = {
            'age': 'Age',
            'sex': 'Sex',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar > 120 mg/dL',
            'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels'
        }

        sex_labels = {0: 'Female', 1: 'Male'}
        cp_labels = {0: 'Typical angina', 1: 'Atypical angina', 2: 'Non-anginal pain', 3: 'Asymptomatic'}
        fbs_labels = {0: 'False', 1: 'True'}
        restecg_labels = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}
        exang_labels = {0: 'No', 1: 'Yes'}
        slope_labels = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}

        # Create patient info table
        patient_data = []
        for feature, value in features.items():
            feature_name = feature_labels.get(feature, feature.title())

            # Format value based on feature type
            if feature == 'sex':
                formatted_value = sex_labels.get(value, str(value))
            elif feature == 'cp':
                formatted_value = cp_labels.get(value, str(value))
            elif feature == 'fbs':
                formatted_value = fbs_labels.get(value, str(value))
            elif feature == 'restecg':
                formatted_value = restecg_labels.get(value, str(value))
            elif feature == 'exang':
                formatted_value = exang_labels.get(value, str(value))
            elif feature == 'slope':
                formatted_value = slope_labels.get(value, str(value))
            elif feature == 'trestbps':
                formatted_value = f"{value} mm Hg"
            elif feature == 'chol':
                formatted_value = f"{value} mg/dL"
            elif feature == 'thalach':
                formatted_value = f"{value} bpm"
            else:
                formatted_value = str(value)

            patient_data.append([feature_name, formatted_value])

        patient_table = Table(patient_data, colWidths=[2.5*inch, 3.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))

        # Risk Assessment
        story.append(Paragraph("Risk Assessment", heading_style))
        risk_text = f"Based on the analysis of {len(features)} medical parameters, "
        risk_text += "the patient has been assessed as having a "
        risk_text += f"<b>{'HIGH' if prediction == 1 else 'LOW'}</b> risk of heart disease."

        story.append(Paragraph(risk_text, normal_style))
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Medical Recommendations", heading_style))
        story.append(Paragraph("Based on the risk assessment, the following recommendations are provided:", normal_style))
        story.append(Spacer(1, 12))

        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", normal_style))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 20))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )

        disclaimer_text = """
        <b>DISCLAIMER:</b> This report is generated by an automated system for educational purposes only. 
        The predictions are based on machine learning models and should not replace professional medical evaluation. 
        Always consult with qualified healthcare professionals for medical advice and diagnosis.
        """
        story.append(Paragraph(disclaimer_text, disclaimer_style))

        # Build PDF
        doc.build(story)

        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()

        return pdf_content

    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

def display_input_summary(features: Dict[str, Any]) -> None:
    """
    Display a summary of all input parameters.

    Args:
        features: Dictionary of input features
    """
    with st.expander("📋 Input Summary"):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Age:** {features['age']} years")
            st.write(f"**Sex:** {'Female' if features['sex'] == 0 else 'Male'}")
            st.write(f"**Chest Pain Type:** {['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'][features['cp']]}")
            st.write(f"**Resting Blood Pressure:** {features['trestbps']} mm Hg")
            st.write(f"**Cholesterol:** {features['chol']} mg/dL")
            st.write(f"**Fasting Blood Sugar > 120 mg/dL:** {'True' if features['fbs'] == 1 else 'False'}")

        with col2:
            st.write(f"**Resting ECG:** {['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'][features['restecg']]}")
            st.write(f"**Max Heart Rate:** {features['thalach']} bpm")
            st.write(f"**Exercise Induced Angina:** {'Yes' if features['exang'] == 1 else 'No'}")
            st.write(f"**ST Depression:** {features['oldpeak']}")
            st.write(f"**ST Slope:** {['Upsloping', 'Flat', 'Downsloping'][features['slope']]}")
            st.write(f"**Major Vessels:** {features['ca']}")

def main() -> None:
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title=CONFIG['page_title'],
        page_icon=CONFIG['app_icon'],
        layout=CONFIG['layout']
    )

    # Load the model
    model = load_model()
    if model is None:
        return

    # Header
    st.title(f"{CONFIG['app_icon']} {CONFIG['app_title']}")
    st.markdown("---")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This application uses machine learning to predict heart disease risk based on medical parameters.
        
        **Note:** This is for educational purposes only. Always consult healthcare professionals for medical advice.
        """)

        st.header("📊 Model Information")
        st.write("The model analyzes 12 key medical parameters to assess cardiovascular health risk.")

        # Feature descriptions
        with st.expander("📖 Feature Descriptions"):
            for feature, description in FEATURE_DESCRIPTIONS.items():
                st.write(f"**{feature.title()}:** {description}")

    # Input form
    with st.form("prediction_form"):
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=50, help="Patient age in years")
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Patient gender")
            cp = st.selectbox('Chest Pain Type',
                            options=[0, 1, 2, 3],
                            format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x],
                            help="Type of chest pain experienced")
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=120, help="Blood pressure in mm Hg")
            chol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=600, value=200, help="Serum cholesterol level")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL',
                             options=[0, 1],
                             format_func=lambda x: "False" if x == 0 else "True",
                             help="Whether fasting blood sugar exceeds 120 mg/dL")

        with col2:
            restecg = st.selectbox('Resting ECG',
                                 options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x],
                                 help="Results of resting electrocardiogram")
            thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150, help="Maximum heart rate during exercise")
            exang = st.selectbox('Exercise Induced Angina',
                               options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               help="Whether angina is induced by exercise")
            oldpeak = st.number_input('ST Depression Induced by Exercise',
                                    min_value=0.0, max_value=6.0,
                                    value=0.0, step=0.1, format="%.1f",
                                    help="ST depression relative to rest")
            slope = st.selectbox('Slope of ST Segment',
                               options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                               help="Slope of peak exercise ST segment")
            ca = st.selectbox('Number of Major Vessels', options=[0, 1, 2, 3, 4], help="Number of major vessels colored by fluoroscopy")

        # Predict button
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

        if submitted:
            # Prepare features
            features = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca
            }

            # Validate inputs
            if not validate_inputs(features):
                return

            # Make prediction
            feature_list = list(features.values())
            result = predict_heart_disease(model, feature_list)

            if result is not None:
                st.markdown("---")
                st.subheader("📊 Prediction Result")

                # Get risk description
                risk_info = get_risk_description(result)

                if result == 1:
                    st.error(risk_info['title'])
                    st.warning(risk_info['message'])
                else:
                    st.success(risk_info['title'])
                    st.info(risk_info['message'])

                # Display recommendations
                st.subheader("💡 Recommendations")
                for i, recommendation in enumerate(risk_info['recommendations'], 1):
                    st.write(f"{i}. {recommendation}")

                # Display input summary
                display_input_summary(features)

                # Store results in session state for export
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy_types(obj):
                    """Convert numpy types to native Python types."""
                    if hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj

                features_serializable = convert_numpy_types(features)
                result_serializable = convert_numpy_types(result)

                st.session_state.prediction_results = {
                    'prediction': result_serializable,
                    'risk_level': 'High' if result_serializable == 1 else 'Low',
                    'features': features_serializable,
                    'recommendations': risk_info['recommendations'],
                    'age': convert_numpy_types(age),
                    'sex': convert_numpy_types(sex)
                }

    # Export functionality (outside the form)
    if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
        st.markdown("---")
        st.subheader("📤 Export Results")

        results = st.session_state.prediction_results
        age = results['age']
        sex = results['sex']
        prediction = results['prediction']
        features = results['features']
        recommendations = results['recommendations']

        # Create two columns for export buttons
        col1, col2 = st.columns(2)

        with col1:
            # Create JSON file for download
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_str,
                file_name=f"heart_disease_prediction_{age}_{'M' if sex == 1 else 'F'}.json",
                mime="application/json",
                help="Download the prediction results as a JSON file"
            )

        with col2:
            # Generate and download PDF report
            pdf_content = generate_pdf_report(prediction, features, recommendations, age, sex)
            if pdf_content:
                st.download_button(
                    label="📋 Download PDF Report",
                    data=pdf_content,
                    file_name=f"heart_disease_prediction_{age}_{'M' if sex == 1 else 'F'}.pdf",
                    mime="application/pdf",
                    help="Download a professional PDF report with all details"
                )
            else:
                st.error("Failed to generate PDF report")

        # Clear results after export (optional)
        if st.button("🗑️ Clear Results", help="Clear the current prediction results"):
            del st.session_state.prediction_results
            st.rerun()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Disclaimer:** Educational purposes only")
    with col2:
        st.markdown("**Version:** 1.0.0")
    with col3:
        st.markdown("**Contact:** [Your Contact Info]")

if __name__ == "__main__":
    main() 