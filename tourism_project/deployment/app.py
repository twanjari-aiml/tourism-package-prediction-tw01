import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_predictor_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package
based on their details and interaction data.
Enter the customer information below to get a prediction.
""")

# Input widgets for the tourism dataset features
Age = st.number_input("Age", 18, 90, 30)
TypeofContact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
CityTier = st.selectbox("City Tier (1=Highest, 3=Lowest)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0.0, 60.0, 10.0, 0.5)
Occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government', 'Unemployed'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 1)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
NumberOfTrips = st.number_input("NumberOfTrips (annually)", 0, 50, 5)
Passport = st.checkbox("Has Passport")
OwnCar = st.checkbox("Owns Car")
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (<5 years old)", 0, 5, 0)
Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
MonthlyIncome = st.number_input("Monthly Income", 0.0, 200000.0, 50000.0, 1000.0)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (1=Low, 5=High)", [1, 2, 3, 4, 5])
ProductPitched = st.selectbox("Product Pitched", ['Wellness', 'Domestic', 'International', 'Glamping', 'Adventure', 'Resort'])
NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)

# Prepare input data for prediction
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport else 0,  # Convert boolean to 0/1
    "OwnCar": 1 if OwnCar else 0,      # Convert boolean to 0/1
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups
}])

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer Will Purchase Package" if prediction == 1 else "Customer Will Not Purchase Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
