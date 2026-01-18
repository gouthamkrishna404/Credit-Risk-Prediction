import pandas as pd
import numpy as np
import io
import os
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

if not os.path.exists('models'):
    os.makedirs('models')
    
df = pd.read_csv("data/raw_loan_data.csv")

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

X = df.drop("Default", axis=1)
y = df["Default"]

if len(df) < 50:
    X = pd.concat([X]*10, ignore_index=True)
    y = pd.concat([y]*10, ignore_index=True)

joblib.dump(list(X.columns), "models/training_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

model_configs = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
        "params": {"classifier__C": [0.01, 0.1, 1, 10]}
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "classifier__n_estimators": [100, 200], 
            "classifier__max_depth": [5, 10, None]
        }
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200], 
            "classifier__learning_rate": [0.05, 0.1]
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
        "params": {
            "classifier__learning_rate": [0.05, 0.1], 
            "classifier__max_depth": [3, 5],
            "classifier__scale_pos_weight": [1, 5]
        }
    }
]

tuned_estimators = [] 

for config in model_configs:
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3 if len(X_train) > 10 else 1)), 
        ('classifier', config["model"])
    ])
    
    search = RandomizedSearchCV(
        pipeline, 
        config["params"], 
        n_iter=5, 
        scoring='roc_auc', 
        cv=3, 
        random_state=42, 
        n_jobs=-1
    )
    
    search.fit(X_train_scaled, y_train)
    best_pipeline = search.best_estimator_
    
    tuned_model = best_pipeline.named_steps['classifier']
    tuned_estimators.append((config["name"], tuned_model))
    
    probs = best_pipeline.predict_proba(X_test_scaled)[:, 1]
    preds = best_pipeline.predict(X_test_scaled)
    
    print(config["name"])
    print(search.best_params_)
    print(classification_report(y_test, preds))

stacking_clf = StackingClassifier(
    estimators=tuned_estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=3,
    n_jobs=-1
)

stacking_clf.fit(X_train_scaled, y_train)

final_preds = stacking_clf.predict(X_test_scaled)
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))

joblib.dump(stacking_clf, "models/credit_risk_model.pkl")
