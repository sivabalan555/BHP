import streamlit as st
import pickle
import json
import numpy as np

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load column information
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Streamlit UI
st.title("Bengaluru House Price Prediction")

# Input features
st.sidebar.header("Enter House Details")

area_type = st.sidebar.selectbox(
    "Area Type",
    ['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area']
)

location = st.sidebar.selectbox(
    "Location",
    data_columns[:-3]  # Assuming last 3 columns are not locations
)

total_sqft = st.sidebar.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)

bath = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, step=1)

balcony = st.sidebar.slider("Number of Balconies", min_value=0, max_value=3, step=1)

size = st.sidebar.selectbox("Size (BHK)", [f"{i} BHK" for i in range(1, 6)])

# Preprocess user input
def get_prediction(area_type, location, total_sqft, bath, balcony, size):
    # Convert categorical features to the expected format
    loc_index = data_columns.index(location.lower())
    size_value = int(size.split()[0])
    area_type_value = area_type.lower()

    # Create input array
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = balcony
    x[3] = size_value
    x[loc_index] = 1

    return model.predict([x])[0]

# Predict and Display the result
if st.button("Predict Price"):
    predicted_price = get_prediction(area_type, location, total_sqft, bath, balcony, size)
    st.success(f"The predicted price of the house is ₹{predicted_price:,.2f} Lakhs")

