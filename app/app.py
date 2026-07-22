import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../models/churn_model.pkl')
feature_columns = joblib.load('../models/feature_columns.pkl')

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📊")
st.title("📊 Telecom Customer Churn Prediction")
st.write("Enter customer details below to predict if they are likely to churn.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

if st.button("Predict Churn"):

    input_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([input_dict])

    
    from sklearn.preprocessing import LabelEncoder
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is LIKELY TO CHURN (Probability: {probability:.2%})")
    else:
        st.success(f"✅ This customer is LIKELY TO STAY (Probability of churn: {probability:.2%})")