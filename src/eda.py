import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('outputs/plots'):
    os.makedirs('outputs/plots')

try:
    df = pd.read_csv("data/raw_loan_data.csv")
except FileNotFoundError:
    print("Data file not found.")
    exit()

if "LoanID" in df.columns:
    df.drop("LoanID", axis=1, inplace=True)

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

try:
    model = joblib.load("models/credit_risk_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    train_cols = joblib.load("models/training_columns.pkl")
except FileNotFoundError:
    print("Error: Models not found in 'models/' folder.")
    exit()

X = df.drop("Default", axis=1)
y_true = df["Default"]
X = X[train_cols]

X_scaled = scaler.transform(X)
X_df = pd.DataFrame(X_scaled, columns=train_cols)

y_proba = model.predict_proba(X_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

with np.errstate(divide='ignore', invalid='ignore'):
    fscore = (2 * precision * recall) / (precision + recall)

ix = np.argmax(fscore)
best_thresh = thresholds[ix]
best_f1 = fscore[ix]

print(f"Default Threshold: 0.50")
print(f"Optimal Threshold: {best_thresh:.4f} (Maximizes F1-Score: {best_f1:.4f})")

threshold_configs = {
    "30": 0.30,
    "50": 0.50,
    "Optimized": best_thresh
}

for name, thresh in threshold_configs.items():
    y_pred_loop = (y_proba >= thresh).astype(int)
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_loop)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title(f"Confusion Matrix\n(Threshold = {thresh:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/confusion_matrix_{name}.png")
    plt.close()

plt.figure(figsize=(8, 6))
plt.plot(thresholds, fscore[:-1], label='F1 Score', color='black')
plt.plot(thresholds, precision[:-1], label='Precision', linestyle='--', color='blue')
plt.plot(thresholds, recall[:-1], label='Recall', linestyle='--', color='green')
plt.axvline(best_thresh, color='red', linestyle=':', label=f'Best ({best_thresh:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall-F1 Trade-off')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig("outputs/plots/threshold_tuning.png")
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Stacking Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("outputs/plots/roc_curve.png")
plt.close()

try:
    xgb_model = None
    if hasattr(model, "named_estimators_"):
        if 'XGBoost' in model.named_estimators_:
            xgb_model = model.named_estimators_['XGBoost']
        else:
            xgb_model = model.estimators_[-1]
    
    if xgb_model:
        X_sample = X_df.sample(n=min(1000, len(X_df)), random_state=42)
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure()
        plt.title("SHAP Summary (XGBoost Component)")
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig("outputs/plots/shap_summary.png", bbox_inches='tight')
        plt.close()
    else:
        print("Could not find XGBoost inside Stacking Model.")

except Exception as e:
    print(f"SHAP Skipped: {e}")
