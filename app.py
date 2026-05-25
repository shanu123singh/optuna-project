import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction App")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.joblib")

st.write("📂 Model path:", MODEL_PATH)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Run train.py first.")
    st.stop()

# Load model safely
try:
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    scaler = data["scaler"]
    ct = data["column_transformer"]
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Enter House Details")

Condition = st.selectbox("Condition", ["Poor", "Average", "Good"])
Location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
Garage = st.selectbox("Garage", ["No", "Yes"])

Area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
Bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
Bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
YearBuilt = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
Floors = st.number_input("Floors", min_value=1, max_value=5, value=2)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Price"):

    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            "Condition": Condition,
            "Location": Location,
            "Garage": Garage,
            "Area": Area,
            "Bedrooms": Bedrooms,
            "Bathrooms": Bathrooms,
            "YearBuilt": YearBuilt,
            "Floors": Floors
        }])

        # Match training columns
        expected_cols = ct.feature_names_in_

        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_cols]

        # Transform and scale
        transformed = ct.transform(input_df)
        scaled = scaler.transform(transformed)

        # Predict
        prediction = model.predict(scaled)[0]

        # Output
        st.success(f"💰 Predicted House Price: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("⚡ Built using Streamlit | Model: XGBoost + Optuna")