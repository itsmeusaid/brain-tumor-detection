import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import base64
from io import BytesIO

# TensorFlow imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Scikit-learn imports for NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorAnalyzer:
    def __init__(self):
        self.img_width, self.img_height = 224, 224
        self.class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.model_path = r"D:\Intern base\Project\my_brain_tumor_mobilenetv2.h5"
        self.confidence_threshold = 0.7
        
        #  symptom mapping 
        self.symptoms_texts = [
            "headache dizziness blurred vision", "severe headache memory loss confusion",
            "nausea seizures vision problem", "seizures vomiting and nausea",
            "hormone issues weight gain fatigue", "growth problems infertility hormonal",
            "no headache no tumor normal", "healthy normal no symptoms",
        ]
        self.symptoms_labels = [
            "glioma", "glioma",
            "meningioma", "meningioma",
            "pituitary", "pituitary",
            "notumor", "notumor"
        ]
    
    @st.cache_resource
    def load_cnn_model(_self):
        """Load the trained CNN model"""
        try:
            if os.path.exists(_self.model_path):
                model = load_model(_self.model_path)
                logger.info("CNN Model loaded successfully")
                return model
            else:
                st.error(f"Model file not found at: {_self.model_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            st.error(f"Failed to load CNN model: {str(e)}")
            return None
    
    @st.cache_resource
    def load_nlp_model(_self):
        """Load/train the NLP model for symptom analysis"""
        try:
            tfidf = TfidfVectorizer()
            X_train = tfidf.fit_transform(_self.symptoms_texts)
            text_clf = LogisticRegression(max_iter=200)
            text_clf.fit(X_train, _self.symptoms_labels)
            logger.info("NLP Model trained successfully")
            return tfidf, text_clf
        except Exception as e:
            logger.error(f"Error training NLP model: {e}")
            st.error(f"Failed to train NLP model: {str(e)}")
            return None, None
    
    def preprocess_image(self, uploaded_file):
        """Enhanced image preprocessing with validation"""
        try:
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                raise ValueError("File size too large. Please upload images smaller than 10MB.")
            
            # Load and validate image
            img = Image.open(uploaded_file).resize((self.img_width, self.img_height)).convert('RGB')
            img_array = np.array(img) / 255.0
            
            return np.expand_dims(img_array, axis=0)
        
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise e
    
    def analyze_image(self, processed_image, model):
        """Analyze image using the trained CNN model"""
        try:
            if model is None:
                raise ValueError("Model not loaded")
            
            # Get predictions from your trained model
            predictions = model.predict(processed_image)
            
            # Get the predicted class index and confidence
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx]) * 100
            predicted_class = self.class_labels[predicted_idx]
            
            # Create detailed predictions dictionary
            all_predictions = {}
            for i, label in enumerate(self.class_labels):
                all_predictions[label] = float(predictions[0][i]) * 100
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'predicted_idx': predicted_idx,
                'all_predictions': all_predictions,
                'raw_predictions': predictions[0].tolist()
            }
        
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            raise e
    
    def analyze_symptoms(self, symptoms_text, tfidf, text_clf):
        """Enhanced symptom analysis using your trained NLP model"""
        try:
            if not symptoms_text.strip():
                return {'error': 'Please enter symptoms'}
            
            if tfidf is None or text_clf is None:
                return {'error': 'NLP model not loaded'}
            
            # Use your trained model for prediction
            X_test = tfidf.transform([symptoms_text.lower()])
            prediction = text_clf.predict(X_test)[0]
            
            # Get prediction probabilities if available
            try:
                probabilities = text_clf.predict_proba(X_test)[0]
                confidence = max(probabilities) * 100
                
                # Create detailed scores
                all_scores = {}
                for i, class_name in enumerate(text_clf.classes_):
                    all_scores[class_name] = probabilities[i] * 100
            except:
                confidence = 75.0  # Default confidence if probabilities not available
                all_scores = {prediction: 75.0}
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'all_scores': all_scores,
                'input_text': symptoms_text
            }
        
        except Exception as e:
            logger.error(f"Error in symptom analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

class PatientManager:
    def __init__(self):
        self.history_file = "patient_history.json"
    
    def save_patient_data(self, patient_id, patient_name, analysis_results, entry_number=None):
        """Save patient data with improved structure and entry numbering"""
        history = self.load_history()
        
        # Get next entry number for this patient
        if patient_id not in history:
            history[patient_id] = []
        
        if entry_number is None:
            entry_number = len(history[patient_id]) + 1
        
        patient_data = {
            'patient_id': patient_id,
            'patient_name': patient_name,
            'entry_number': entry_number,
            'timestamp': datetime.now().isoformat(),
            'analysis_results': analysis_results
        }
        
        history[patient_id].append(patient_data)
        
        # Save to session state
        st.session_state.history = history
        return entry_number
    
    def get_current_session_results(self):
        """Get all analysis results from current session"""
        results = {}
        
        # Check if there are any analysis results in session state
        if hasattr(st.session_state, 'current_mri_results'):
            results['mri_analysis'] = st.session_state.current_mri_results
            
        if hasattr(st.session_state, 'current_symptom_results'):
            results['symptom_analysis'] = st.session_state.current_symptom_results
            
        return results if results else None
    
    def clear_current_session(self):
        """Clear current session results after saving"""
        if hasattr(st.session_state, 'current_mri_results'):
            del st.session_state.current_mri_results
        if hasattr(st.session_state, 'current_symptom_results'):
            del st.session_state.current_symptom_results
    
    def load_history(self):
        """Load patient history from session state"""
        return st.session_state.get('history', {})
    
    def get_patient_summary(self, patient_id):
        """Generate patient summary statistics"""
        history = self.load_history()
        if patient_id not in history:
            return None
        
        records = history[patient_id]
        return {
            'total_entries': len(records),
            'last_visit': records[-1]['timestamp'] if records else None,
            'analysis_types': [list(r['analysis_results'].keys()) for r in records]
        }

def create_enhanced_ui():
    """Create enhanced Streamlit UI"""
    st.set_page_config(
        page_title="🩺 Brain Tumor Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS with better accessibility
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 20px;
            color: white;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin: 10px 0;
            color: white;
        }
        
        .prediction-high {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .prediction-medium {
            background: linear-gradient(45deg, #feca57, #ff9ff3);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .prediction-low {
            background: linear-gradient(45deg, #48dbfb, #0abde3);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .confidence-bar {
            height: 25px;
            border-radius: 12px;
            background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
            position: relative;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .sidebar .stSelectbox label, .sidebar .stTextInput label {
            color: white !important;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .main { padding: 10px; }
            .metric-card { margin: 5px 0; }
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Initialize components
    analyzer = BrainTumorAnalyzer()
    patient_manager = PatientManager()
    
    # Create UI
    create_enhanced_ui()
    
    # Header
    st.title("🩺 Advanced Brain Tumor Analyzer")
    st.markdown("### AI-Powered Medical Image Analysis & Symptom Assessment")
    
    # Load models
    cnn_model = analyzer.load_cnn_model()
    tfidf, text_clf = analyzer.load_nlp_model()
    
    if cnn_model is None:
        st.error("❌ Cannot proceed without CNN model. Please check your model path.")
        st.stop()
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("👤 Patient Information")
        
        # Patient details
        patient_id = st.text_input("🆔 Patient ID", key="patient_id")
        patient_name = st.text_input("📝 Patient Name", key="patient_name")
        patient_age = st.number_input("🎂 Age", min_value=1, max_value=120, value=30)
        patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
        
        # Analysis settings
        st.subheader("⚙ Analysis Settings")
        show_confidence_details = st.checkbox("Show detailed confidence scores", True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Image analysis section
    with col1:
        st.subheader(" MRI Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload MRI scan", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file:
            # Always show the image
            st.image(uploaded_file, caption="Uploaded MRI Scan", width=300)
            
            # Always run analysis when image is uploaded
            try:
                with st.spinner(" Analyzing MRI scan..."):
                    # Process image using your preprocessing
                    processed_img = analyzer.preprocess_image(uploaded_file)
                    results = analyzer.analyze_image(processed_img, cnn_model)
                    
                    # Display results
                    prediction = results['prediction'].capitalize()
                    confidence = results['confidence']
                    
                    # Choose styling based on confidence
                    if confidence >= 80:
                        style_class = "prediction-high"
                    elif confidence >= 60:
                        style_class = "prediction-medium"
                    else:
                        style_class = "prediction-low"
                    
                    st.markdown(f"""
                        <div class="{style_class}">
                            🎯 Prediction: {prediction}<br>
                            📊 Confidence: {confidence:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed confidence scores
                    if show_confidence_details:
                        st.subheader(" Detailed Analysis")
                        for label, score in results['all_predictions'].items():
                            # Calculate delta from average (25%)
                            delta_val = score - 25.0
                            delta_str = f"{delta_val:+.1f}%" if delta_val != 0 else None
                            
                            st.metric(
                                label.capitalize(),
                                f"{score:.1f}%",
                                delta=delta_str
                            )
                    
                    # Store results in session state for later saving
                    st.session_state.current_mri_results = results
                    
                    st.success("✅ MRI analysis complete! Use 'Save Results' button below to save.")
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    # Symptom analysis section
    with col2:
        st.subheader("Symptom Analysis")
        
        symptoms_input = st.text_area(
            "Describe symptoms",
            height=100,
            placeholder="e.g., persistent headaches, vision problems, memory issues...",
            help="Enter all relevant symptoms"
        )
        
        if st.button("🔬 Analyze Symptoms"):
            if not symptoms_input.strip():
                st.warning("⚠ Please enter symptoms to analyze!")
            else:
                with st.spinner(" Analyzing symptoms..."):
                    symptom_results = analyzer.analyze_symptoms(symptoms_input, tfidf, text_clf)
                    
                    if 'error' in symptom_results:
                        st.error(f"❌ {symptom_results['error']}")
                    else:
                        prediction = symptom_results['prediction'].capitalize()
                        
                        # Display main result without confidence percentage
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>🎯 Symptom-Based Analysis</h3>
                                <p><strong>Result:</strong> There is a chance of {prediction}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Store results in session state for later saving
                        st.session_state.current_symptom_results = symptom_results
                        
                        st.success("✅ Symptom analysis complete! Use 'Save Results' button below to save.")
    
    # Dedicated Save Section
    st.markdown("---")
    st.subheader("💾 Save Results")
    
    # Check if there are any results to save
    current_results = patient_manager.get_current_session_results()
    
    if current_results:
        # Show what will be saved in a nice format
        result_types = []
        if 'mri_analysis' in current_results:
            result_types.append(" MRI Analysis")
        if 'symptom_analysis' in current_results:
            result_types.append(" Symptom Analysis")
        
        st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Ready to Save:</h4>
                <p>{' • '.join(result_types)}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Save button
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
        with col_save2:
            if st.button("💾 Save Results", key="save_results_btn", type="primary", use_container_width=True):
                if patient_id.strip() and patient_name.strip():
                    # Save all current results
                    entry_number = patient_manager.save_patient_data(
                        patient_id, 
                        patient_name, 
                        current_results
                    )
                    
                    # Clear current session results
                    patient_manager.clear_current_session()
                    
                    st.success(f"✅ Results saved as Entry #{entry_number} for {patient_name}!")
                    st.balloons()  # Fun celebration effect
                    
                    # Force refresh to update the UI
                    st.rerun()
                else:
                    st.error("❌ Please enter both Patient ID and Patient Name to save results!")
    else:
        st.info("📝 No analysis results to save. Perform MRI or symptom analysis first.")
    
    st.markdown("---")
    
    # Combined analysis section
    st.subheader(" Comprehensive Analysis")
    
    if st.button("📊 Generate Full Report"):
        if patient_id.strip() and patient_name.strip():
            # Generate comprehensive report
            st.success(f"📋 Comprehensive medical report generated for {patient_name}!")
            
            # Patient summary
            summary = patient_manager.get_patient_summary(patient_id)
            
            # Display patient info
            col3, col4, col5 = st.columns(3)
            with col3:
                entries_count = summary['total_entries'] if summary else 0
                st.metric("Total Entries", entries_count)
            with col4:
                st.metric("Age", patient_age)
            with col5:
                st.metric("Gender", patient_gender)
            
            # Additional report info
            st.markdown("### 📋 Patient Summary")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Age:** {patient_age}")
            st.write(f"**Gender:** {patient_gender}")
            
            if summary and summary['last_visit']:
                st.write(f"**Last Entry:** {summary['last_visit'][:19]}")
            else:
                st.write("**Last Entry:** No previous entries")
                
        else:
            st.warning("⚠ Please enter both Patient ID and Patient Name to generate a full report!")
    
    # Patient history section
    st.subheader("📜 Patient History")
    
    if patient_manager.load_history():
        # Add search functionality
        col_search, col_select = st.columns([1, 1])
        
        with col_search:
            search_id = st.text_input(" Search by Patient ID", placeholder="Enter Patient ID to search")
        
        # Filter patients based on search
        all_patient_ids = list(patient_manager.load_history().keys())
        if search_id:
            filtered_ids = [pid for pid in all_patient_ids if search_id.lower() in pid.lower()]
        else:
            filtered_ids = all_patient_ids
        
        with col_select:
            selected_patient = st.selectbox("Select Patient", [""] + filtered_ids)
        
        if selected_patient:
            history = patient_manager.load_history()[selected_patient]
            
            # Display patient info
            if history:
                st.write(f"**Patient:** {history[0].get('patient_name', 'Unknown')}")
                st.write(f"**Total Entries:** {len(history)}")
            
            # Display history in expandable sections
            for i, record in enumerate(history):
                entry_num = record.get('entry_number', i+1)
                with st.expander(f"Entry #{entry_num} - {record['timestamp'][:19]}"):
                    st.json(record['analysis_results'])
                    
                    # Fixed delete button
                    if st.button(f"🗑 Delete Entry #{entry_num}", key=f"delete_{selected_patient}_{i}_{record['timestamp']}"):
                        # Properly delete the record
                        if selected_patient in st.session_state.history:
                            if len(st.session_state.history[selected_patient]) > i:
                                st.session_state.history[selected_patient].pop(i)
                                # If no more records for this patient, remove patient entirely
                                if not st.session_state.history[selected_patient]:
                                    del st.session_state.history[selected_patient]
                                st.success("✅ Entry deleted successfully!")
                                st.rerun()
    else:
        st.info("📝 No patient history available yet.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.7);'>
            🏥 Advanced Brain Tumor Analyzer v2.0<br>
            ⚠ For research and educational purposes only. Always consult healthcare professionals.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()