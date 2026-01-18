import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Credit Risk Prediction", 
    page_icon="🏦",
    layout="centered"
)

def train_model():
    if not os.path.exists('models'):
        os.makedirs('models')

    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Training Phase 1/5: Loading Data...")
    
    try:
        df = pd.read_csv("data/raw_loan_data.csv")
    except FileNotFoundError:
        st.error("Fatal Error: 'data/raw_loan_data.csv' not found. Cannot train model.")
        st.stop()

    if "LoanID" in df.columns:
        df.drop("LoanID", axis=1, inplace=True)
HF_REPO_ID = "gouthamkrishna404/credit-risk-prediction" 

    status_text.text("Training Phase 2/5: Feature Engineering...")
    progress_bar.progress(20)
def download_from_hf():
    """Downloads artifacts from Hugging Face if they don't exist locally."""
    if not os.path.exists("models"):
        os.makedirs("models")

    df["Income"] = df["Income"].clip(lower=0)
    df["Income_Loan_Ratio"] = df["Income"] / (df["LoanAmount"] + 1)
    df["Monthly_Obligation"] = df["LoanAmount"] / df["LoanTerm"]
    df["DTI_Strict"] = (df["Monthly_Obligation"] + 500) / (df["Income"] / 12 + 1)
    df["Tenure_Age_Ratio"] = df["MonthsEmployed"] / (df["Age"] * 12 + 1)
    df["Credit_Income_Interaction"] = df["CreditScore"] * np.log1p(df["Income"])

    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-25","26-35","36-45","46-55","56-65","65+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        lbl = LabelEncoder()
        df[col] = lbl.fit_transform(df[col].astype(str))

    df = df.fillna(0)

    X = df.drop("Default", axis=1)
    y = df["Default"]
    
    training_columns = X.columns.tolist()
    joblib.dump(training_columns, "models/training_columns.pkl")

    status_text.text("Training Phase 3/5: Balancing Data (SMOTE)...")
    progress_bar.progress(40)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    joblib.dump(scaler, "models/scaler.pkl")

    status_text.text("Training Phase 4/5: Training Stacking Ensemble...")
    progress_bar.progress(60)

    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=42))
    files_to_download = [
        "credit_risk_model.pkl",
        "scaler.pkl",
        "training_columns.pkl"
    ]

    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=3
    )

    stacking_model.fit(X_scaled, y_resampled)
    joblib.dump(stacking_model, "models/credit_risk_model.pkl")

    status_text.text("Training Phase 5/5: Complete!")
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    for file in files_to_download:
        dest_path = os.path.join("models", file)
        if not os.path.exists(dest_path):
            with st.spinner(f"Downloading {file} from Hugging Face..."):
                try:
                    path = hf_hub_download(repo_id=HF_REPO_ID, filename=file)
                    import shutil
                    shutil.copy(path, dest_path)
                except Exception as e:
                    st.error(f"Error downloading {file}: {e}")

@st.cache_resource
def load_artifacts():
    if not os.path.exists("models/credit_risk_model.pkl"):
        train_model()
        
    download_from_hf()
    try:
        model = joblib.load("models/credit_risk_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        train_cols = joblib.load("models/training_columns.pkl")
        return model, scaler, train_cols
    except FileNotFoundError:
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

model, scaler, training_columns = load_artifacts()

if model is None:
    st.error("Critical Error: Model could not be trained or loaded.")
    st.error("Model artifacts could not be loaded from Hugging Face.")
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
@@ -203,52 +141,49 @@
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
    
    age_group_labels = ["18-25","26-35","36-45","46-55","56-65","65+"]
    current_age_group = str(input_df["Age_Group"].iloc[0])
    input_df["Age_Group"] = manual_label_encoder(current_age_group, age_group_labels)
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
