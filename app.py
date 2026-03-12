
import streamlit as st
import numpy as np
import joblib

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter house details")

squareMeters = st.number_input("Square Meters")
numPrevOwners = st.number_input("Previous Owners")
numberOfRooms = st.number_input("Number of Rooms")
cityPartRange = st.number_input("City Part Range")
hasStormProtector = st.number_input("Storm Protector (0 or 1)")
floors = st.number_input("Floors")

if st.button("Predict Price"):

    features = np.array([[squareMeters,
                          numPrevOwners,
                          numberOfRooms,
                          cityPartRange,
                          hasStormProtector,
                          floors]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Predicted Price: {prediction[0]:,.2f}")
