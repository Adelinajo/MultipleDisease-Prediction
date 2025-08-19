import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load pickle files
with open('parkinsons_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pfeature_names.pkl','rb') as f:
    feature_names = pickle.load(f)  # all feature names


st.title("🧠 Parkinson's Disease Detection")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file with patient data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure the uploaded file has all required features
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in CSV: {missing_cols}")
        else:
            # Keep only required features in correct order
            input_df = df[feature_names]

            # Scale
            input_scaled = scaler.transform(input_df)

            # Predictions
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)[:, 1]  # prob of disease

            # Combine results
            df_results = df.copy()
            df_results["Prediction"] = predictions
            df_results["Probability"] = probabilities

            # Map prediction to readable text
            df_results["Result"] = df_results["Prediction"].map({1: "Parkinson's Detected", 0: " Healthy"})

            # Show table
            st.subheader("📊 Prediction Results")
            st.dataframe(df_results)

            # Allow download
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="parkinsons_predictions.csv",
                mime="text/csv" )

    except Exception as e:
        st.error(f"Error processing file: {e}")

