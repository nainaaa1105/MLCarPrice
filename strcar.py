import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction")

# ---------- LOAD ARTIFACTS ----------
model = joblib.load("xgb.pkl")
scaler = joblib.load("scaler.pkl")   # make sure filename matches
columns = joblib.load("columns.pkl")

numeric_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']



# ---------- USER INPUT ----------
year = st.number_input("Year", 2000, 2025, 2018)
mileage = st.number_input("Mileage", 0.0, 300000.0, 30000.0)
tax = st.number_input("Tax", 0.0, 1000.0, 150.0)
mpg = st.number_input("MPG", 0.0, 200.0, 45.0)
engineSize = st.number_input("Engine Size", 0.5, 6.0, 1.5)

model_name = st.selectbox("Model", sorted(
    [c.replace("model_", "") for c in columns if c.startswith("model_")]
))
transmission = st.selectbox(
    "Transmission",
    sorted([c.replace("transmission_", "") for c in columns if c.startswith("transmission_")] + ["Automatic"])
)
fuelType = st.selectbox("Fuel Type", sorted(
    [c.replace("fuelType_", "") for c in columns if c.startswith("fuelType_")]
))

# ---------- PREDICTION ----------
if st.button("Predict Price"):

    try:
        # Initialize input with ALL columns = 0
        input_data = {col: 0 for col in columns}

        # Fill numeric values
        input_data['year'] = year
        input_data['mileage'] = mileage
        input_data['tax'] = tax
        input_data['mpg'] = mpg
        input_data['engineSize'] = engineSize

        # One-hot encoding (ONLY if column exists)
        model_col = f"model_{model_name}"
        trans_col = f"transmission_{transmission}"
        fuel_col = f"fuelType_{fuelType}"

        if model_col in input_data:
            input_data[model_col] = 1
        if trans_col in input_data:
            input_data[trans_col] = 1
        if fuel_col in input_data:
            input_data[fuel_col] = 1

        # Create DataFrame in EXACT order
        input_df = pd.DataFrame([input_data])[columns]

        # Scale numeric columns ONLY
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
