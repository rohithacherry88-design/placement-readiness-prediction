import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Readiness & Risk Prediction", page_icon="🎓", layout="wide")

# Paths check
model_path = 'models/placement_model.pkl'
features_path = 'models/feature_names.pkl'

if not os.path.exists(model_path):
    model_path = 'data/models/placement_model.pkl'
    features_path = 'data/models/feature_names.pkl'

@st.cache_resource
def load_ml_model():
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

try:
    model, feature_names = load_ml_model()
except Exception as e:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

st.title("🎓 Placement Readiness & Risk Prediction System")
st.write("Student academic and skill parameters enter chesi prediction metrics chudandi:")
st.markdown("---")

input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    current_col = col1 if i % 2 == 0 else col2
    f_lower = feature.lower()
    
    if 'cgpa' in f_lower:
        input_data[feature] = current_col.number_input(f"Enter {feature}", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    elif 'percentage' in f_lower or 'pct' in f_lower or 'score' in f_lower:
        input_data[feature] = current_col.slider(f"Select {feature}", min_value=0.0, max_value=100.0, value=70.0)
    elif 'backlog' in f_lower:
        input_data[feature] = current_col.number_input(f"Active {feature}", min_value=0, max_value=10, value=0)
    elif 'internship' in f_lower or 'project' in f_lower or 'certific' in f_lower:
        input_data[feature] = current_col.number_input(f"Number of {feature}", min_value=0, max_value=10, value=1)
    else:
        input_data[feature] = current_col.number_input(f"Enter {feature}", min_value=0.0, value=50.0)

st.markdown("---")

if st.button("🚀 Predict Placement Readiness", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = probabilities[1] * 100 if len(probabilities) > 1 else probabilities[0] * 100
    else:
        confidence = 100.0 if prediction == 1 else 0.0

    st.subheader("📊 Evaluation Output:")
    
    if prediction == 1:
        st.success(f"✅ **PLACEMENT READY!** (Placement Chance: **{confidence:.1f}%**)")
        st.info("💡 **Recommendation:** Ready for placements.")
    else:
        st.error(f"⚠️ **HIGH RISK CATEGORY!** (Placement Chance: **{confidence:.1f}%**)")
        st.warning("💡 **Action Required:** Focus on skill improvement & clearing backlogs.")