import streamlit as st
import pandas as pd
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Placement Readiness & Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# Header Section
st.title("🎓 Placement Readiness & Risk Prediction System")
st.markdown("Enter student details below to check placement readiness and risk assessment.")

# Load Model Artifacts directly from current working directory
@st.cache_resource
def load_artifacts():
    # Direct paths without models/ folder dependency
    model_path = "placement_model.pkl"
    features_path = "feature_names.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("Model files not found in the repository! Please ensure placement_model.pkl and feature_names.pkl exist.")
        st.stop()
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)
        
    return model, feature_names

try:
    model, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# User Inputs Form
st.sidebar.header("Student Parameters")

def user_input_features():
    # Dynamic form fields matching model feature names
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.sidebar.number_input(
            f"Enter {feature}", 
            value=0.0
        )
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Display Input Data
st.subheader("Selected Student Profile")
st.dataframe(input_df)

# Predict Button
if st.button("Predict Placement Status"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        
        st.markdown("---")
        if prediction == 1:
            st.success("🎉 **Status:** Student is **Ready for Placement**!")
        else:
            st.warning("⚠️ **Status:** Student is **At Risk / Needs Improvement**.")
            
        if prediction_proba is not None:
            st.write(f"**Confidence Score:** {max(prediction_proba)*100:.2f}%")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

