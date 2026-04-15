import streamlit as st
import pandas as pd
import json

# Setup page properties
st.set_page_config(page_title="Real Estate Dashboard", layout="centered")

st.title("🏡 Real Estate Price Estimator")
st.write("Enter the property details below to get an estimated price.")

# 1. Load your saved model and Label Encoders 
# (Label Encoders here were saved as json mappings per train.ipynb)
with open('models/best_model.pkl', 'rb') as f:
    import joblib
    model = joblib.load(f)

with open('frontend_mappings.json', 'r') as f:
    mappings = json.load(f)

city_options = list(mappings['city'].keys())
location_options = list(mappings['location'].keys())
status_options = list(mappings['Status'].keys())
property_type_options = list(mappings['property_type'].keys())

# Create inputs
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", city_options)
    location = st.selectbox("Location", location_options)
    latitude = st.number_input("Latitude", value=19.21)
    longitude = st.number_input("Longitude", value=72.98)
    numBathrooms = st.slider("Number of Bathrooms", min_value=0, max_value=20, value=2)
    numBalconies = st.slider("Number of Balconies", min_value=0, max_value=10, value=1)
    isNegotiable = st.selectbox("Is Negotiable", [0, 1])
    securityDeposit = st.number_input("Security Deposit", min_value=0, value=0)

with col2:
    status = st.selectbox("Status", status_options)
    size_ft2 = st.number_input("Size (ft²)", min_value=0, value=500)
    price_per_sqft = st.number_input("Price per sqft", min_value=0.0, value=50.0)
    bhk = st.slider("BHK", min_value=1, max_value=10, value=1)
    rooms_num = st.slider("Number of Rooms", min_value=1, max_value=20, value=2)
    property_type = st.selectbox("Property Type", property_type_options)
    verification_days = st.number_input("Verification Days", min_value=0.0, value=10.0)

# Predict button
if st.button("Predict"):
    # Convert categorical variables using mappings
    loc_val = mappings['location'][location]
    city_val = mappings['city'][city]
    status_val = mappings['Status'][status]
    prop_val = mappings['property_type'][property_type]
    
    # Store in dataframe with exact column names as expected by model
    input_data = pd.DataFrame([[
        loc_val, city_val, latitude, longitude, 
        numBathrooms, numBalconies, isNegotiable, 
        securityDeposit, status_val, size_ft2, 
        price_per_sqft, bhk, rooms_num, 
        prop_val, verification_days
    ]], columns=[
        'location', 'city', 'latitude', 'longitude', 
        'numBathrooms', 'numBalconies', 'isNegotiable', 
        'SecurityDeposit', 'Status', 'Size_ft²', 
        'Price_per_sqft', 'BHK', 'rooms_num', 
        'property_type', 'verification_days'
    ])
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")