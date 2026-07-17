import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Rain Prediction in Australia", layout="wide", page_icon="🌦️")

@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('rain_prediction_ann.h5')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        shap_background = joblib.load('shap_background.pkl')
        # Create shap explainer
        explainer = shap.DeepExplainer(model, shap_background.values)
        return model, scaler, encoders, feature_columns, explainer
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please make sure you have run `python train.py` first to generate the necessary files.")
        return None, None, None, None, None

model, scaler, encoders, feature_columns, explainer = load_artifacts()

if model is None:
    st.stop()

st.title("🌦️ Rain Prediction in Australia")
st.markdown("""
This application predicts whether it will rain tomorrow based on today's weather observations using an Artificial Neural Network (ANN).
It also uses **Explainable AI (SHAP)** to show exactly *why* the model made its prediction.
""")

st.header("Input Weather Data")

# Create two columns for inputs
col1, col2 = st.columns(2)

input_data = {}

# Split the features between the two columns
half = len(feature_columns) // 2

for i, feature in enumerate(feature_columns):
    col = col1 if i < half else col2
    
    if feature in encoders:
        # Categorical feature
        classes = encoders[feature].classes_
        # Filter out 'Unknown' if we want, but let's just show all
        val = col.selectbox(f"{feature}", classes)
        input_data[feature] = val
    else:
        # Numerical feature
        # Provide some reasonable defaults based on the feature name
        if "Humidity" in feature:
            val = col.slider(f"{feature} (%)", 0.0, 100.0, 50.0)
        elif "Pressure" in feature:
            val = col.number_input(f"{feature} (hPa)", min_value=900.0, max_value=1100.0, value=1015.0)
        elif "Temp" in feature:
            val = col.number_input(f"{feature} (°C)", min_value=-15.0, max_value=50.0, value=20.0)
        elif "Speed" in feature:
            val = col.number_input(f"{feature} (km/h)", min_value=0.0, max_value=150.0, value=15.0)
        elif "Rainfall" in feature:
            val = col.number_input(f"{feature} (mm)", min_value=0.0, max_value=400.0, value=0.0)
        else:
            val = col.number_input(f"{feature}", value=0.0)
            
        input_data[feature] = val

st.markdown("---")

if st.button("Predict Rain Tomorrow", type="primary"):
    with st.spinner("Analyzing weather data and generating explanation..."):
        # 1. Preprocess the input
        df_input = pd.DataFrame([input_data])
        
        # Encode categorical
        for col_name, le in encoders.items():
            if df_input[col_name].values[0] not in le.classes_:
                df_input[col_name] = le.transform(['Unknown'])
            else:
                df_input[col_name] = le.transform(df_input[col_name])
                
        # Ensure correct column order
        df_input = df_input[feature_columns]
        
        # Scale numerical
        scaled_input = scaler.transform(df_input)
        
        # 2. Predict
        prob = model.predict(scaled_input, verbose=0)[0][0]
        is_rain = prob > 0.5
        
        # Display Prediction
        st.header("Prediction Results")
        if is_rain:
            st.error(f"🌧️ **It WILL RAIN tomorrow.** (Confidence: {prob:.1%})")
        else:
            st.success(f"☀️ **It will NOT RAIN tomorrow.** (Confidence: {1-prob:.1%})")
            
        # 3. Explainable AI (SHAP)
        st.subheader("Explainable AI (XAI) - Why did the model make this prediction?")
        
        st.markdown("""
        The plot below shows how each weather feature contributed to pushing the model's prediction 
        from the base value (average prediction) to the final output.
        - **Red bars** push the prediction higher (towards Rain).
        - **Blue bars** push the prediction lower (towards No Rain).
        """)
        
        try:
            # SHAP DeepExplainer expects a tensor/array
            shap_values = explainer.shap_values(scaled_input)
            
            # For a single output model, shap_values might be a list or array
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][0]
            else:
                shap_vals = shap_values[0]
                
            # Create Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Since SHAP deep explainer provides values in log-odds or probability depending on model
            # We'll use the generic decision plot or waterfall if expected_value is available
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[0]
                
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=expected_value,
                data=df_input.iloc[0].values, # Original unscaled data for display
                feature_names=feature_columns
            )
            
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
