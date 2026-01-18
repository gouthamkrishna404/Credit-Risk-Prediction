import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Credit Risk Prediction", 
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("models/credit_risk_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        train_cols = joblib.load("models/training_columns.pkl")
        return model, scaler, train_cols
    except FileNotFoundError:
        return None, None, None

model, scaler, training_columns = load_artifacts()

if model is None:
    st.error("Model files not found in 'models/'. Please run the training script first.")
    st.stop()

def run_policy_guardrails(inputs):
    if inputs['Age'] < 18:
        return "REJECT", "Eligibility Decline: Applicant must be at least 18 years old."
    if inputs['Age'] > 75:
        return "REJECT", "Policy Decline: Applicant age exceeds maximum threshold (75)."
    if not (300 <= inputs['CreditScore'] <= 850):
        return "REJECT", "Data Error: Credit Score must be between 300 and 850."
    if inputs['Income'] < 0 or inputs['LoanAmount'] < 0:
        return "REJECT", "Data Error: Income and Loan Amount must be positive."
    
    if inputs['EmploymentType'] == "Unemployed":
        return "WARN", "High Risk Flag: Applicant is currently unemployed."
    if inputs['Income'] < 10000:
        return "WARN", "High Risk Flag: Income is below standard thresholds."

    return "PASS", "Application meets standard policy criteria."

def manual_label_encoder(value, options_list):
    options_list = sorted([str(x) for x in options_list])
    try:
        return options_list.index(str(value))
    except ValueError:
        return 0 

st.title("🏦 Credit Risk Prediction System")
st.markdown("Enter applicant details below to assess loan default risk.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
    loan_term = st.number_input("Loan Term (months)", min_value=6, value=36)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)

with col2:
    credit = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    dti = st.number_input("DTI Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.0, step=0.1)
    
st.divider()

col3, col4 = st.columns(2)
with col3: 
    edu_opts = ["Bachelor's", "High School", "Master's", "PhD"]
    education = st.selectbox("Education", edu_opts)
    
    emp_opts = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    employment = st.selectbox("Employment Type", emp_opts)
    
    mar_opts = ["Divorced", "Married", "Single"]
    marital = st.selectbox("Marital Status", mar_opts)
    
    mort_opts = ["No", "Yes"]
    mortgage = st.selectbox("Has Mortgage", mort_opts)

with col4:
    dep_opts = ["No", "Yes"]
    dependents = st.selectbox("Has Dependents", dep_opts)
    
    purp_opts = ["Auto", "Business", "Education", "Home", "Other"]
    loan_purpose = st.selectbox("Loan Purpose", purp_opts)
    
    co_opts = ["No", "Yes"]
    co_signer = st.selectbox("Has Co-Signer", co_opts)

raw_data = {
    "Age": age, "Income": income, "LoanAmount": loan, "CreditScore": credit,
    "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti,
    "Education": education, "EmploymentType": employment, "MaritalStatus": marital, 
    "HasMortgage": mortgage, "HasDependents": dependents, "LoanPurpose": loan_purpose, 
    "HasCoSigner": co_signer
}

if st.button("Analyze Risk", use_container_width=True):
    
    policy_status, policy_msg = run_policy_guardrails(raw_data)
    
    if policy_status == "REJECT":
        st.error(f"🚫 **Automatic Rejection**\n\nReason: {policy_msg}")
        st.stop()
    elif policy_status == "WARN":
        st.warning(f"⚠️ **Policy Warning:** {policy_msg}")

    input_df = pd.DataFrame([raw_data])
    
    input_df["Income"] = input_df["Income"].clip(lower=0)
    input_df["Income_Loan_Ratio"] = input_df["Income"] / (input_df["LoanAmount"] + 1)
    input_df["Monthly_Obligation"] = input_df["LoanAmount"] / input_df["LoanTerm"]
    input_df["DTI_Strict"] = (input_df["Monthly_Obligation"] + 500) / (input_df["Income"] / 12 + 1)
    input_df["Tenure_Age_Ratio"] = input_df["MonthsEmployed"] / (input_df["Age"] * 12 + 1)
    input_df["Credit_Income_Interaction"] = input_df["CreditScore"] * np.log1p(input_df["Income"])
    
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-25","26-35","36-45","46-55","56-65","65+"]
    input_df["Age_Group"] = pd.cut(input_df["Age"], bins=bins, labels=labels)

    input_df["Education"] = manual_label_encoder(education, edu_opts)
    input_df["EmploymentType"] = manual_label_encoder(employment, emp_opts)
    input_df["MaritalStatus"] = manual_label_encoder(marital, mar_opts)
    input_df["HasMortgage"] = manual_label_encoder(mortgage, mort_opts)
    input_df["HasDependents"] = manual_label_encoder(dependents, dep_opts)
    input_df["LoanPurpose"] = manual_label_encoder(loan_purpose, purp_opts)
    input_df["HasCoSigner"] = manual_label_encoder(co_signer, co_opts)
    input_df["Age_Group"] = manual_label_encoder(input_df["Age_Group"].iloc[0], labels)

    input_df = input_df[training_columns]
    
    input_df_scaled = scaler.transform(input_df)

    prediction_proba = model.predict_proba(input_df_scaled)[0, 1]
    
    THRESH_LOW = 0.30
    THRESH_HIGH = 0.65
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Default Probability", value=f"{prediction_proba:.2%}")
        
        if prediction_proba < THRESH_LOW:
            st.success("## 🟢 Low Risk")
            st.caption("Auto-Approve")
        elif prediction_proba > THRESH_HIGH:
            st.error("## 🔴 High Risk")
            st.caption("Auto-Reject")
        else:
            st.warning("## 🟡 Medium Risk")
            st.caption("Manual Review Required")

    with res_col2:
        st.write("### Analysis Breakdown")
        st.progress(float(prediction_proba))
        st.caption(f"0% —— Low (<30%) —— Medium —— High (>65%) —— 100%")
        
        if credit < 600: st.write("❌ **Credit Score:** Too low")
        elif credit > 720: st.write("✅ **Credit Score:** Excellent")
        
        if dti > 0.45: st.write(f"❌ **Debt Ratio:** High ({dti:.2f})")
        else: st.write(f"✅ **Debt Ratio:** Healthy ({dti:.2f})")
        
        if input_df["Credit_Income_Interaction"].iloc[0] > 7000:
            st.write("✅ **Income/Credit Synergy:** Strong")
