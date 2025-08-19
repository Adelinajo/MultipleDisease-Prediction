import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load all objects

with open('lok_reg.pkl', 'rb') as f:
    lok_reg = pickle.load(f)

with open('kidneyscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ordinal_encoder.pkl', 'rb') as f:
     enc = pickle.load(f)

with open('kidney_features.pkl', 'rb') as f:
    kidney_features= pickle.load(f)


st.title("🩺 Kidney Disease Prediction App")

uploaded_file = st.file_uploader("📂 Upload CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # st.write("### 🔍 Data Preview")
    # st.dataframe(df.head())

    # Keep original copy for displaying results later
    df_original = df.copy()

    # Reindex to match training features
    df = df.reindex(columns=kidney_features, fill_value=0)

    # Identify categorical & numeric columns
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(exclude="object").columns

     # Encode categorical columns using the loaded ordinal encoder
    if len(cat_cols) > 0:
        df[cat_cols] = enc.transform(df[cat_cols].astype(str))

    # Handle missing values for numeric columns (fill with mean)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Handle missing values for categorical columns (fill with mode)
    for col in cat_cols:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna(df[col].mode()[0])    

   
    # Scale numeric features
    df[num_cols] = scaler.transform(df[num_cols])

    # --- Predictions ---
    preds = lok_reg.predict(df)
    probs = lok_reg.predict_proba(df)[:, 1]  # CKD probability

    # Build results by combining original data + predictions
    results = df_original.copy()
    results["Prediction"] = preds
    results["Probability (%)"] = (probs * 100).round(2).astype(str) + "%"
    results["Result"] = ["⚠️ CKD Detected" if p == 1 else "✅ Healthy" for p in preds]

    # Show final table
    st.write("### 📊 Prediction Results")
    st.dataframe(results)

    # Download as CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Predictions as CSV",
        data=csv,
        file_name="kidney_predictions.csv",
        mime="text/csv",
    )

    
