import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('random_forest_fraud_model_top15.pkl')

# Define the raw features
st.title("Fraud Detection App")
st.write("Enter values for the raw features to predict if a fraud will occur.")

# Define the raw input features
incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
insured_hobbies = st.selectbox("Insured Hobbies", ["Other", "chess", "cross-fit", "reading", "paintball", "none"])
property_claim = st.number_input("Property Claim", min_value=0.0, format="%.2f")
vehicle_claim = st.number_input("Vehicle Claim", min_value=0.0, format="%.2f")
total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, format="%.2f")
insured_zip = st.number_input("Insured Zip", min_value=0, format="%d")
injury_claim = st.number_input("Injury Claim", min_value=0.0, format="%.2f")
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0.0, format="%.2f")
months_as_customer = st.number_input("Months as Customer", min_value=0, format="%d")
age = st.number_input("Age", min_value=18, format="%d")

# Collect all the inputs in a dictionary for one-hot encoding
input_data = {
    'incident_severity': incident_severity,
    'insured_hobbies': insured_hobbies,
    'property_claim': property_claim,
    'vehicle_claim': vehicle_claim,
    'total_claim_amount': total_claim_amount,
    'insured_zip': insured_zip,
    'injury_claim': injury_claim,
    'policy_annual_premium': policy_annual_premium,
    'months_as_customer': months_as_customer,
    'age': age
}

# Convert input data into DataFrame
input_df = pd.DataFrame([input_data])

# Perform one-hot encoding on the input data
input_encoded = pd.get_dummies(input_df)

# Align input features with the model’s training features (in case of missing columns)
model_features = ['incident_severity_Major Damage', 'incident_severity_Minor Damage', 'incident_severity_Total Loss',
                  'insured_hobbies_Other', 'insured_hobbies_chess', 'insured_hobbies_cross-fit', 
                  'property_claim', 'vehicle_claim', 'total_claim_amount', 'insured_zip', 
                  'injury_claim', 'policy_annual_premium', 'months_as_customer', 'age']

# Reindex the encoded input to match the model’s feature columns
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Perform prediction when button is clicked
if st.button("Predict Fraud"):
    # Predict and get probability
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    # Display results
    if prediction[0] == 1:
        st.error("Warning: This transaction is predicted to be fraudulent.")
    else:
        st.success("This transaction is predicted to be legitimate.")
    
    # Display probability scores
    st.write(f"Fraud Probability: {prediction_proba[0][1]:.2f}")
    st.write(f"Legitimate Probability: {prediction_proba[0][0]:.2f}")
