import kagglehub
import os, shutil

path = kagglehub.dataset_download("nikhil1e9/loan-default")

for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), "data/raw_loan_data.csv")

print("Dataset downloaded successfully")
