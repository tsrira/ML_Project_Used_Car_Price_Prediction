import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained Gradient Boosting model and scaler
with open("gb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚗 Used Car Price Prediction")

# Input fields
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Car Name (for display)", "Maruti Swift")
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, value=50000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
with col2:
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Normalize categorical inputs to match training data format
fuel = fuel.strip().title()
seller_type = seller_type.strip().title()
transmission = transmission.strip().title()
owner = owner.strip().title()

# Feature Engineering
car_age = 2025 - year

# Prepare input features to match training columns exactly
input_dict = {
    "km_driven": km_driven,
    "car_age": car_age,
    "fuel_" + fuel: 1,
    "seller_type_" + seller_type: 1,
    "transmission_" + transmission: 1,
    "owner_" + owner: 1
}

base_features = ["km_driven", "car_age"]
cat_features = [
    "fuel_Petrol", "fuel_Diesel", "fuel_Cng", "fuel_Lpg", "fuel_Electric",
    "seller_type_Individual", "seller_type_Dealer", "seller_type_Trustmark Dealer",
    "transmission_Manual", "transmission_Automatic",
    "owner_First Owner", "owner_Second Owner", "owner_Third Owner",
    "owner_Fourth & Above Owner", "owner_Test Drive Car"
]

# Initialize all features with zeros and update with input
full_input = {}
for col in base_features + cat_features:
    full_input[col] = input_dict.get(col, 0)

X_input = pd.DataFrame([full_input])

# Remove selling_price if it exists in input dataframe (to avoid mismatch)
if "selling_price" in X_input.columns:
    X_input = X_input.drop(columns=["selling_price"])

# Scale numerical features
X_input[base_features] = scaler.transform(X_input[base_features])

# Predict on button click
if st.button("Predict Selling Price"):
    y_pred = model.predict(X_input)[0]
    st.success(f"Predicted Selling Price: ₹ {int(y_pred)}")

st.markdown("---")
st.markdown("Model: Gradient Boosting Regressor | Dataset: CAR-DEKHO")
