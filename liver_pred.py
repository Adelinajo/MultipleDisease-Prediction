import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load pickle files
with open('rf_ilmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('lscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)    

with open('liver_feature.pkl','rb') as f:
    feature_names = pickle.load(f)  # all feature names

# --- Function to load defaults from uploaded file ---
# ---- Load defaults from CSV ----
def load_defaults(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # If Feature,Value format
    if set(["Feature", "Value"]).issubset(df.columns):
        defaults = {str(row["Feature"]).strip(): row["Value"] for _, row in df.iterrows()}
    else:
        # Wide format (columns = features, one row of values)
        defaults = df.iloc[0].to_dict()

    return defaults

# --- Streamlit App ---
st.title("🫀Indian liver Disease Prediction")
uploaded_file = st.file_uploader("Upload defaults CSV file", type=["csv"])

if uploaded_file:
    defaults = load_defaults(uploaded_file)

    # ✅ Only 3 inputs from user
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total_Bilirubin", min_value=0.0, value=1.0)

    # Merge user inputs with defaults
    features = pd.DataFrame({
        'Age':[age],
        'Gender':[gender.capitalize()],
        'Total_Bilirubin':[total_bilirubin],
        'Direct_Bilirubin':[defaults["Direct_Bilirubin"]], 
        'Alkaline_Phosphotase':[defaults["Alkaline_Phosphotase"]],
        'Alamine_Aminotransferase':[defaults["Alamine_Aminotransferase"]],
        'Aspartate_Aminotransferase':[defaults["Aspartate_Aminotransferase"]],
        'Total_Protiens':[defaults["Total_Protiens"]],
        'Albumin':[defaults["Albumin"]],
        'Albumin_and_Globulin_Ratio':[defaults["Albumin_and_Globulin_Ratio"]]
})

    # Ensure feature names match the model
    features['Gender'] = label_encoder.transform(features['Gender'])

    input_df = features[feature_names]

# Ensure all feature names are present
# Scale features
    features_scaled = scaler.transform(input_df)
    features_scaled = pd.DataFrame(features_scaled, columns=feature_names)


#Predict
    if st.button("Predict Liver Disease"):
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        st.subheader("Results")
        if prediction == 1:
            st.error("⚠️ The patient has liver disease.")
        else:
            st.success("✅ The patient is healthy.")

        st.write("### 🔢 Prediction Probabilities:")
        st.write(f"🧬 Probability of Liver Disease (Class 1): `{prediction_proba[0]:.2%}`")
        st.write(f"💚 Probability of Being Healthy (Class 2): `{prediction_proba[1]:.2%}`")
    

