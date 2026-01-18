import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sklearn
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Credit Risk Prediction", 
    page_icon="🏦",
    layout="centered"
)

# --- DEBUGGING SIDEBAR ---
with st.sidebar:
    st.header("🔍 System Audit")
    st.write(f"**SKLearn Version:** {sklearn.__version__}")
    st.write(f"**Pandas Version:** {pd.__version__}")
    debug_mode = st.checkbox("Show Debug Data", value=False)

HF_REPO_ID = "gouthamkrishna404/credit-risk-prediction" 

def download_from_hf():
    if not os.path.exists("models"):
        os.makedirs("models")
    
    files_to_download = ["credit_risk_model.pkl", "scaler.pkl", "training_columns.pkl"]
    
    for file in files_to_download:
        dest_path = os.path.join("models", file)
        if not os.path.exists(dest_path):
            with st.spinner(f"Downloading {file}..."):
                try:
                    path = hf_hub_download(repo_id=HF_REPO_ID, filename=file)
                    import shutil
                    shutil.copy(path, dest_path)
                except Exception as e:
                    st.error(f"Error downloading {file}: {e}")

@st.cache_resource
def load_artifacts():
    download_from_hf()
    try:
        model = joblib.load("models/credit_risk_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        train_cols = joblib.load("models/training_columns.pkl")
        return model, scaler, train_cols
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

model, scaler, training_columns = load_artifacts()

if model is None:
    st.error("Model artifacts could not be loaded.")
    st.stop()

def run_policy_guardrails(inputs):
    if inputs['Age'] < 18: return "REJECT", "Age < 18"
    if inputs['Age'] > 75: return "REJECT", "Age > 75"
    if not (300 <= inputs['CreditScore'] <= 850): return "REJECT", "Invalid Credit Score"
    return "PASS", ""

def manual_label_encoder(value, options_list):
    options_list = sorted([str(x) for x in options_list])
    try:
        return options_list.index(str(value))
    except ValueError:
        return 0 

st.title("🏦 Credit Risk Prediction System")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
    loan = st.number_input("Loan Amount ($)", 0, 1000000, 10000)
    loan_term = st.number_input("Loan Term (months)", 6, 360, 36)
    months_employed = st.number_input("Months Employed", 0, 600, 24)
with col2:
    credit = st.number_input("Credit Score", 300, 850, 700)
    dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.3)
    num_credit_lines = st.number_input("Credit Lines", 0, 50, 3)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 10.0)

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
    input_df = pd.DataFrame([raw_data])
    
    # Feature Engineering
    input_df["Income_Loan_Ratio"] = input_df["Income"] / (input_df["LoanAmount"] + 1)
    input_df["Monthly_Obligation"] = input_df["LoanAmount"] / input_df["LoanTerm"]
    input_df["DTI_Strict"] = (input_df["Monthly_Obligation"] + 500) / (input_df["Income"] / 12 + 1)
    input_df["Tenure_Age_Ratio"] = input_df["MonthsEmployed"] / (input_df["Age"] * 12 + 1)
    input_df["Credit_Income_Interaction"] = input_df["CreditScore"] * np.log1p(input_df["Income"])
    
    labels = ["18-25","26-35","36-45","46-55","56-65","65+"]
    input_df["Age_Group"] = pd.cut(input_df["Age"], bins=[18, 25, 35, 45, 55, 65, 100], labels=labels)

    # Encoding
    input_df["Education"] = manual_label_encoder(education, edu_opts)
    input_df["EmploymentType"] = manual_label_encoder(employment, emp_opts)
    input_df["MaritalStatus"] = manual_label_encoder(marital, mar_opts)
    input_df["HasMortgage"] = manual_label_encoder(mortgage, mort_opts)
    input_df["HasDependents"] = manual_label_encoder(dependents, dep_opts)
    input_df["LoanPurpose"] = manual_label_encoder(loan_purpose, purp_opts)
    input_df["HasCoSigner"] = manual_label_encoder(co_signer, co_opts)
    input_df["Age_Group"] = manual_label_encoder(str(input_df["Age_Group"].iloc[0]), labels)

    # Align with training columns
    input_df = input_df[training_columns]
    input_df_scaled = scaler.transform(input_df)

    # --- DEBUG SECTION ---
    if debug_mode:
        st.subheader("🛠 Debugging Output")
        st.write("**1. Column Order Match:**", list(input_df.columns) == training_columns)
        st.write("**2. Raw Processed Row:**", input_df)
        st.write("**3. Scaled Array (First 5 values):**", input_df_scaled[0][:5])

    prediction_proba = model.predict_proba(input_df_scaled)[0, 1]
    
    st.metric(label="Default Probability", value=f"{prediction_proba:.2%}")
    if prediction_proba < 0.30: st.success("Low Risk")
    elif prediction_proba > 0.65: st.error("High Risk")
    else: st.warning("Medium Risk")
