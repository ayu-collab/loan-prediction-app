import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved artifacts
model = joblib.load("model_rf.pkl")
feature_list = joblib.load("feature_list.pkl")
preproc_meta = joblib.load("preproc_meta.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoders.pkl")


st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title(" Loan Approval Prediction App")
st.write("Fill in applicant details to check if the loan is likely to be approved.")

# Sidebar Info 
st.sidebar.header("About")
st.sidebar.write("""
This app uses a **Machine Learning model ** trained on historical loan data.  
It predicts whether a loan application will be **Approved  or Rejected **.
""")

#  Input Sections 
st.subheader(" Applicant Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with col2:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

st.subheader(" Financial Information")
applicantincome = st.number_input("Applicant Income", min_value=0, step=100)
coapplicantincome = st.number_input("Coapplicant Income", min_value=0, step=100)
loanamount = st.number_input("Loan Amount", min_value=0, step=1)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12, step=12)
credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1.0, 0.0])

#  Preprocessing function
def preprocess_input():
    data = {
        "gender": 1 if gender == "Male" else 0,
        "married": 1 if married == "Yes" else 0,
        "education": 0 if education == "Graduate" else 1,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "applicantincome": applicantincome,
        "coapplicantincome": coapplicantincome,
        "loanamount": loanamount,
        "loan_amount_term": loan_amount_term,
        "credit_history": credit_history,
        "applicantincome_log": np.log1p(applicantincome),
        "coapplicantincome_log": np.log1p(coapplicantincome),
        "loanamount_log": np.log1p(loanamount),
        "dependents_1": int(dependents == "1"),
        "dependents_2": int(dependents == "2"),
        "dependents_3+": int(dependents == "3+"),
        "property_area_Semiurban": int(property_area == "Semiurban"),
        "property_area_Urban": int(property_area == "Urban"),


    }
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_list, fill_value=0)
    return df

#  Prediction 
if st.button(" Predict Loan Approval"):
    X_new = preprocess_input()
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f" Loan Approved (Confidence: {prob:.2%})")
    else:
        st.error(f" Loan Not Approved (Confidence: {prob:.2%})")
