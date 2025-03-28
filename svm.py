import streamlit as st
import numpy as np
import pickle

# Load the trained SVM model
with open("svm_model.pkl", "rb") as file:    # Load the Scaler Object in a read-binary mode.
    model = pickle.load(file)

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Title for the Streamlit app.
st.title("Diabetic Retinopathy Prediction")
st.write("Please enter patient details:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, step=1)
systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=200, step=1)
diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=120, step=1)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=300, step=1)

input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])


# Predict button
if st.button("Predict"):

    # Scale input data
    input_data_scaled = scaler.transform(input_data)       

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Display result
    if prediction == 1:
        st.error("The patient may have Diabetic Retinopathy.")
    else:
        st.success("The patient may not have Diabetic Retinopathy.")