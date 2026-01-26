import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ARTIFACTS ---
# Loading the model, scaler, and the list of columns used during training
model = joblib.load('xgb.pkl')
scaler = joblib.load('Scaler.pkl')
model_columns = joblib.load('columns.pkl')

# --- 2. HELPER FUNCTIONS ---
def get_categories(prefix):
    """Extracts unique category names from the encoded column list."""
    return [col.split(prefix)[1] for col in model_columns if col.startswith(prefix)]

# Identify unique values for dropdowns
car_models = get_categories('model_')
transmissions = get_categories('transmission_')
fuel_types = get_categories('fuelType_')

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.write("Fill in the details below to get an estimated market price.")

# Input fields organized into columns
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2020)
    mileage = st.number_input("Mileage", min_value=0, value=15000, step=500)
    tax = st.number_input("Road Tax (£)", min_value=0, value=145)

with col2:
    mpg = st.number_input("MPG (Efficiency)", min_value=0.0, value=50.0, step=0.1)
    engineSize = st.number_input("Engine Size (Liters)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

st.subheader("Vehicle Specifications")
sel_model = st.selectbox("Select Model", sorted(car_models))
sel_trans = st.selectbox("Select Transmission", sorted(transmissions))
sel_fuel = st.selectbox("Select Fuel Type", sorted(fuel_types))

# --- 4. PREDICTION LOGIC ---
if st.button("Calculate Estimated Price", type="primary"):
    # Create a DataFrame with all zeros based on training columns
    # This ensures that any unselected category is automatically 0 (False)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Map numerical inputs
    input_df['year'] = year
    input_df['mileage'] = mileage
    input_df['tax'] = tax
    input_df['mpg'] = mpg
    input_df['engineSize'] = engineSize
    
    # Map one-hot encoded selections to 1 (True)
    if f'model_{sel_model}' in input_df.columns:
        input_df[f'model_{sel_model}'] = 1
        
    if f'transmission_{sel_trans}' in input_df.columns:
        input_df[f'transmission_{sel_trans}'] = 1
        
    if f'fuelType_{sel_fuel}' in input_df.columns:
        input_df[f'fuelType_{sel_fuel}'] = 1

    # CRITICAL STEP: Re-index to ensure column order matches Scaler.fit() exactly
    # This prevents the "Feature names seen at fit time, yet now missing" error
    input_df = input_df[model_columns]

    try:
        # Scale numerical features
        # Note: Scaler expects the exact same shape it was trained on
        input_scaled = scaler.transform(input_df)
        
        # Predict using XGBoost
        prediction = model.predict(input_scaled)
        
        st.divider()
        st.success(f"### Estimated Market Price: £{prediction[0]:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Check if your Scaler was fit on the same columns stored in columns.pkl.")