import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ---- Load pre-trained model and scaler ----
with open("gb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---- UI Header ----
st.title("Used Car Price Prediction - Gradient Boosting")

# ---- Input fields ----
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Car Name (for display)", value="Maruti Swift")
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, value=50000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
with col2:
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# ---- Feature Engineering ----
car_age = 2025 - year

# Prepare features as per training
input_dict = {
    "kmdriven": km_driven,
    "carage": car_age,
    "fuel_" + fuel: 1,
    "sellertype_" + seller_type: 1,
    "transmission_" + transmission: 1,
    "owner_" + owner: 1
}

# List all encoded columns from training
base_features = ["kmdriven", "carage"]
cat_features = [
    "fuel_Petrol", "fuel_Diesel", "fuel_CNG", "fuel_LPG", "fuel_Electric",
    "sellertype_Individual", "sellertype_Dealer", "sellertype_Trustmark Dealer",
    "transmission_Manual", "transmission_Automatic",
    "owner_First Owner", "owner_Second Owner", "owner_Third Owner", "owner_Fourth & Above Owner", "owner_Test Drive Car"
]
# Fill missing one-hot columns with 0
full_input = {}
for col in base_features + cat_features:
    full_input[col] = input_dict.get(col, 0)

X_input = pd.DataFrame([full_input])

# ---- Scale numeric features ----
X_input[base_features] = scaler.transform(X_input[base_features])

# ---- Prediction ----
if st.button("Predict Selling Price"):
    y_pred = model.predict(X_input)[0]
    st.success(f"Predicted Selling Price: ₹ {int(y_pred)}")

# ---- Footer ----
st.markdown("---")
st.markdown("Powered by Gradient Boosting Regression. Model trained on CAR-DEKHO dataset.")
