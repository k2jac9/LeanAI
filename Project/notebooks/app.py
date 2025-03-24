import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Body Fat Prediction App",
    page_icon="💪",
    layout="wide"
)

# Function to load model with error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to load dataset with error handling
@st.cache_data
def load_data(data_url):
    try:
        return pd.read_csv(data_url)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main title
st.title("Body Fat Prediction App 💪")

# Load the trained model
model_path = "bodyfat_model.pkl"
model = load_model(model_path)

# Sidebar for user inputs
with st.sidebar:
    st.header("Enter Your Measurements:")
    
    # Input fields
    abdomen = st.number_input("Abdomen (cm)", min_value=50.0, max_value=150.0, value=85.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
    chest = st.number_input("Chest (cm)", min_value=50.0, max_value=150.0, value=95.0)
    hip = st.number_input("Hip (cm)", min_value=50.0, max_value=150.0, value=100.0)
    thigh = st.number_input("Thigh (cm)", min_value=30.0, max_value=100.0, value=60.0)
    knee = st.number_input("Knee (cm)", min_value=20.0, max_value=50.0, value=38.0)
    biceps = st.number_input("Biceps (cm)", min_value=20.0, max_value=50.0, value=32.0)
    density = st.number_input("Density", min_value=0.9, max_value=1.2, value=1.07)
    
    # Predict button
    predict_btn = st.button("Predict")

# Load the dataset
data_url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/bodyfat.csv"
data = load_data(data_url)

# Make prediction
if predict_btn and model is not None:
    input_data = pd.DataFrame(
        [[density, weight, chest, abdomen, hip, thigh, knee, biceps]],
        columns=['Density', 'Weight', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Biceps']
    )
    
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Body Fat Percentage: {prediction[0]:.2f}%")
        st.info("Note: The prediction is based on the Extended Body Fat Dataset")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Main content using tabs
if data is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Visualizations", "📝 Model Information"])
    
    with tab1:
        st.header("Body Fat Dataset")
        st.write("""
        This dataset is from the UCI Machine Learning Repository.
        It contains measurements of body fat percentage and various body circumference measurements.
        """)
        
        # Dataset overview
        st.subheader("Preview")
        st.dataframe(data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
        with col2:
            st.subheader("Missing Values")
            st.write(data.isnull().sum())
        
        with st.expander("Dataset Description"):
            st.dataframe(data.describe())
        
        with st.expander("Dataset Columns"):
            st.write(data.columns.tolist())

    with tab2:
        st.header("Data Visualizations")
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        # Scatter plot with regression line
        st.subheader("Body Fat vs Weight")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Weight', y='BodyFat', data=data, scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        ax.set_xlabel('Weight (kg)')
        ax.set_ylabel('Body Fat Percentage (%)')
        ax.set_title('Relationship Between Weight and Body Fat Percentage')
        st.pyplot(fig)
        
        st.write("""
        The scatter plot shows the relationship between weight and body fat percentage.
        - Each point represents an individual in the dataset
        - The red line is the linear regression line showing the trend
        - The positive slope indicates that as weight increases, body fat percentage tends to increase
        - The correlation coefficient between weight and body fat percentage is approximately 0.61,
          indicating a moderate positive correlation
        """)
        
        # Histograms
        st.subheader("Distribution of Key Features")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.histplot(data['BodyFat'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Body Fat Distribution')
        
        sns.histplot(data['Weight'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Weight Distribution')
        
        sns.histplot(data['Abdomen'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Abdomen Circumference Distribution')
        
        sns.histplot(data['Hip'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Hip Circumference Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.header("Model Information")
        st.write("""
        ### About the Prediction Model
        
        This application uses a machine learning model trained on the UCI Body Fat dataset 
        to predict body fat percentage based on various body measurements.
        
        #### Features Used:
        - Weight
        - Density
        - Chest circumference
        - Abdomen circumference
        - Hip circumference
        - Thigh circumference
        - Knee circumference
        - Biceps circumference
        
        #### How to Use:
        1. Enter your measurements in the sidebar
        2. Click the "Predict" button
        3. View your predicted body fat percentage
        
        #### Limitations:
        - The model accuracy depends on the quality of the training data
        - The predictions are estimates and should not replace medical advice
        - The model was trained on a specific population and may not generalize perfectly to all individuals
        """)
else:
    st.error("Failed to load dataset. Please check your internet connection and try again.")

# Add footer
st.markdown("---")
st.markdown("© 2025 Body Fat Prediction App | Created with Streamlit")