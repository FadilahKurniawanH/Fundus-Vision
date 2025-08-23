import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# Page configuration
st.set_page_config(
    page_title="Fundus Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Functions
@st.cache_resource
MODEL_PATH = "best_fundus_model.h5"
if not os.path.exists(MODEL_PATH):
    url = "https://github.com/FadilahKurniawanH/Fundus-Vision/blob/main/best_fundus_model.h5"
    gdown.download(url, MODEL_PATH, quiet=False)


def preprocess_image(image, img_size=224):
    """Preprocess image for prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize pixel values
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_disease(model, image, class_names):
    """Make prediction on the image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return predicted_class, confidence, class_probabilities
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

def create_probability_chart(probabilities):
    """Create a bar chart of class probabilities"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = px.bar(
        x=probs, 
        y=classes, 
        orientation='h',
        title="Class Probabilities",
        labels={'x': 'Probability', 'y': 'Disease Class'},
        color=probs,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def get_disease_info(disease_name):
    """Get information about the disease"""
    disease_info = {
        'Cataract': {
            'description': 'A clouding of the lens in the eye that affects vision. Most cataracts are related to aging.',
            'symptoms': ['Cloudy or blurry vision', 'Colors seem faded', 'Glare', 'Poor night vision'],
            'treatment': 'Surgery is the most effective treatment for cataracts.',
            'prevention': 'Protect eyes from UV light, quit smoking, maintain healthy diet'
        },
        'Diabetic Retinopathy': {
            'description': 'A diabetes complication that affects eyes caused by damage to blood vessels of the retina.',
            'symptoms': ['Spots or dark strings floating in vision', 'Blurred vision', 'Fluctuating vision', 'Dark areas in vision'],
            'treatment': 'Blood sugar control, laser treatment, vitrectomy, medication injections',
            'prevention': 'Manage diabetes, regular eye exams, maintain healthy blood pressure'
        },
        'Glaucoma': {
            'description': 'A group of eye conditions that damage the optic nerve, often caused by abnormally high eye pressure.',
            'symptoms': ['Gradual loss of peripheral vision', 'Tunnel vision', 'Eye pain', 'Nausea and vomiting'],
            'treatment': 'Eye drops, laser treatment, surgery to lower eye pressure',
            'prevention': 'Regular eye exams, exercise regularly, limit caffeine intake'
        },
        'Normal': {
            'description': 'Healthy eye with no detected abnormalities in the fundus image.',
            'symptoms': ['Clear vision', 'No visual disturbances', 'Healthy retinal appearance'],
            'treatment': 'Continue regular eye check-ups to maintain eye health',
            'prevention': 'Maintain healthy lifestyle, protect eyes from UV, regular eye exams'
        }
    }
    return disease_info.get(disease_name, {})

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Fundus Eye Disease Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Disease Detection", "Model Information", "Dataset Statistics", "About"])
    
    if page == "Home":
        home_page()
    elif page == "Disease Detection":
        detection_page()
    elif page == "Model Information":
        model_info_page()
    elif page == "Dataset Statistics":
        dataset_stats_page()
    elif page == "About":
        about_page()

def home_page():
    """Home page content"""
    st.markdown('<h2 class="subheader">Welcome to Fundus Eye Disease Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        This AI-powered system uses deep learning to analyze fundus images and detect various eye diseases including:
        - **Cataract**: Clouding of the eye's lens
        - **Diabetic Retinopathy**: Damage to retinal blood vessels due to diabetes
        - **Glaucoma**: Damage to the optic nerve
        - **Normal**: Healthy eye condition
        
        ### 🔬 How it Works
        1. **Upload** a fundus image
        2. **AI Analysis** using trained CNN models
        3. **Results** with confidence scores and recommendations
        
        ### 📊 Model Performance
        Our system achieves high accuracy in detecting eye diseases with multiple CNN architectures including custom models and transfer learning approaches.
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x400/1f77b4/ffffff?text=Fundus+Image+Analysis", 
                caption="AI-Powered Eye Disease Detection")
        
        st.markdown("""
        <div class="info-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This system is for educational and research purposes only. 
        Always consult with qualified healthcare professionals for medical diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

def detection_page():
    """Disease detection page"""
    st.markdown('<h2 class="subheader">🔍 Eye Disease Detection</h2>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Cannot load the model. Please check if the model file exists.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload Fundus Image")
        uploaded_file = st.file_uploader(
            "Choose a fundus image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear fundus (retinal) image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.markdown("**Image Details:**")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Format: {uploaded_file.type}")
            
            # Predict button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, probabilities = predict_disease(
                        st.session_state.model, image, st.session_state.class_names
                    )
                    
                    if predicted_class is not None:
                        # Store in history
                        st.session_state.prediction_history.append({
                            'image_name': uploaded_file.name,
                            'prediction': predicted_class,
                            'confidence': confidence,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        # Display results in col2
                        with col2:
                            display_prediction_results(predicted_class, confidence, probabilities)
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload Image**: Select a clear fundus image
        2. **Wait for Analysis**: AI processes the image
        3. **View Results**: Get prediction with confidence
        4. **Read Information**: Learn about detected condition
        
        **Image Requirements:**
        - Clear fundus/retinal image
        - Good lighting and focus
        - Standard fundus photography
        - Supported formats: PNG, JPG, JPEG
        """)
        
        # Show prediction history
        if st.session_state.prediction_history:
            st.markdown("### 📈 Recent Predictions")
            history_df = pd.DataFrame(st.session_state.prediction_history[-5:])  # Last 5 predictions
            st.dataframe(history_df[['image_name', 'prediction', 'confidence']], use_container_width=True)

def display_prediction_results(predicted_class, confidence, probabilities):
    """Display prediction results"""
    st.markdown("### 🎯 Analysis Results")
    
    # Main prediction
    st.markdown(f"""
    <div class="prediction-box">
    <h3>Detected Condition: <strong>{predicted_class}</strong></h3>
    <h4>Confidence: <strong>{confidence:.2%}</strong></h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence indicator
    if confidence > 0.8:
        st.success("High confidence prediction")
    elif confidence > 0.6:
        st.warning("Moderate confidence prediction")
    else:
        st.error("Low confidence prediction - consider consulting a specialist")
    
    # Probability chart
    st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
    
    # Disease information
    disease_info = get_disease_info(predicted_class)
    if disease_info:
        st.markdown("### 📚 Condition Information")
        
        with st.expander(f"Learn more about {predicted_class}", expanded=True):
            st.markdown(f"**Description:** {disease_info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Common Symptoms:**")
                for symptom in disease_info['symptoms']:
                    st.write(f"• {symptom}")
            
            with col2:
                st.markdown("**Treatment Options:**")
                st.write(disease_info['treatment'])
                
                st.markdown("**Prevention:**")
                st.write(disease_info['prevention'])

def model_info_page():
    """Model information page"""
    st.markdown('<h2 class="subheader">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Architecture Details")
        st.markdown("""
        **Model Types Implemented:**
        1. **Custom CNN**: Built from scratch with enhanced architecture with modern techniques
        3. **VGG16 Transfer Learning**: Pre-trained on ImageNet
        
        **Key Features:**
        - Batch Normalization for stable training
        - Dropout layers for regularization
        - Early Stopping to prevent overfitting
        - Data augmentation for better generalization
        """)
    
    with col2:
        st.markdown("### 📊 Training Configuration")
        st.markdown("""
        **Dataset Split:**
        - Training: 70%
        - Validation: 15% 
        - Testing: 15%
        
        **Hyperparameters:**
        - Image Size: 224x224 pixels
        - Batch Size: 64
        - Optimizer: Adam
        - Loss Function: Sparse Categorical Crossentropy
        - Epochs: Up to 50 (with early stopping)
        """)
    
    st.markdown("### 🎯 Model Performance Comparison")
    
    # Sample performance data (replace with actual results)
    performance_data = {
        'Model': ['Custom CNN', 'VGG16 Transfer'],
        'Test Accuracy': [0.7607, 0.8938],
        'Training Time (min)': [45, 30],
        'Parameters (M)': [26.3, 15.1]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Performance chart
    fig = px.bar(performance_df, x='Model', y='Test Accuracy', 
                title="Model Performance Comparison",
                color='Test Accuracy',
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

def dataset_stats_page():
    """Dataset statistics page"""
    st.markdown('<h2 class="subheader">📊 Dataset Statistics</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    dataset_info = {
        'Class': ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Total'],
        'Images': [1038, 1098, 1007, 1074, 4217],
        'Percentage': [24.6, 26.0, 23.9, 25.5, 100.0]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Class Distribution")
        df = pd.DataFrame(dataset_info)
        st.dataframe(df, use_container_width=True)
        
        # Pie chart
        fig_pie = px.pie(df[:-1], values='Images', names='Class', 
                        title="Dataset Class Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Dataset Details")
        st.markdown("""
        **Image Specifications:**
        - Resolution: 512x512 pixels
        - Format: Various (PNG, JPG, JPEG)
        - Color: RGB (3 channels)
        - Total Size: 4,217 images
        
        **Data Quality:**
        - High-resolution fundus images
        - Professional medical photography
        - Balanced class distribution
        - Verified medical annotations
        
        **Augmentation Applied:**
        - Random rotation (±15°)
        - Horizontal flipping
        - Brightness variation
        - Normalization (0-1 scale)
        """)
        
        # Bar chart
        fig_bar = px.bar(df[:-1], x='Class', y='Images',
                        title="Images per Class",
                        color='Images',
                        color_continuous_scale='blues')
        st.plotly_chart(fig_bar, use_container_width=True)

def about_page():
    """About page"""
    st.markdown('<h2 class="subheader">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This Fundus Eye Disease Detection System is an AI-powered tool designed to assist in the early detection 
        of common eye diseases using fundus (retinal) photography. The system employs advanced deep learning 
        techniques to analyze retinal images and provide accurate classifications.
        
        ### 🔬 Technology Stack
        - **Deep Learning Framework**: TensorFlow/Keras
        - **CNN Architectures**: Custom CNN, VGG16
        - **Web Framework**: Streamlit
        - **Image Processing**: OpenCV, PIL
        - **Data Visualization**: Plotly, Matplotlib
        - **Development Environment**: Google Colab, VS Code
        
        ### 🎓 Educational Purpose
        This system is developed for educational and research purposes to demonstrate:
        - Implementation of CNN for medical image classification
        - Transfer learning techniques in healthcare AI
        - Model comparison and evaluation methods
        - Deployment of AI models using Streamlit
        
        ### 📚 Dataset Information
        The model is trained on a comprehensive dataset containing 4,217 high-quality fundus images 
        across four categories: Cataract, Diabetic Retinopathy, Glaucoma, and Normal eyes.
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍💻 Development Info
        **Created by**: Fadilah Kurniawan Hadi  
        **Version**: 1.0   
        **Last Updated**: 2025             
        **License**: Educational Use  
        
        ### 🔗 Key Features
        - Multiple CNN architectures
        - Real-time image analysis
        - Confidence scoring
        - Disease information
        - Performance metrics
        - User-friendly interface
        
        ### ⚠️ Important Notice
        This system is intended for educational 
        and research purposes only. It should not 
        be used as a substitute for professional 
        medical diagnosis or treatment.
        
        Always consult qualified healthcare 
        professionals for medical advice.
        """)
        
        st.markdown("""
        <div class="info-box">
        <strong>📧 Contact</strong><br>
        For questions or feedback about this system, 
        please consult your instructor or the 
        development team.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
