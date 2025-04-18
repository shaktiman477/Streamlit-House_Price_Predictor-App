import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model

with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("California House Price Predictor")
st.write("Enter details to predect the house price (in 100, 000 USD): \n")

# Inputs fields

MedInc = st.slider("Median Income", 0.0, 15.0, 3.0)
HouseAge = st.slider("House Age", 1, 60, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bed Rooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 0, 4000, 1000)
AveAccups = st.slider("Average Accupants", 0.5, 10.0, 3.0)
Latitude = st.slider("Latitude", 32, 42, 36)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# display input summary

st.subheader("Your Input Summary")
st.write({
    "Median Income": MedInc,
    "House Age": HouseAge,
    "Average Rooms": AveRooms, 
    "Average Bed Rooms": AveBedrms,
    "Population": Population, 
    "Average Accupants": AveAccups, 
    "Latitude": Latitude,
    "Longitude": Longitude
})

if st.button("Predict House Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveAccups, Latitude, Longitude]])


    prediction = model.predict(input_data)

    st.success(f"Predicted house Price: ${prediction[0] * 100000:.2f}")

st.subheader("Feature Importance (XGBoost)")

importances = model.feature_importances_
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveAccups', 'Latitude', 'Longitude']
plt.figure(figsize=(8, 4))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.ylabel('XGBoost Feature Importance')
st.pyplot(plt)

