import pandas as pd
import matplotlib.pyplot as plt
from modelling import approved_applications, logreg,  model_approval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import os
import joblib 


model_LR = joblib.load("models/FMlogistic.pkl")
scaler = joblib.load("models/FMscaler.pkl")


print(approved_applications.head())


#processing positives

"""
Fannie Mai

"loan id",
    "original interest rate",
    "original UPB",
    "original loan term",
    "original LTV ratio",
    "debt to income",
    "Current Loan Delinquency Status",
    "original list price",

HMDA

"action_taken",
    "loan_type",
    "loan_amount",  X (UPB)
    "income",
    "debt_to_income_ratio",  X
    "loan_to_value_ratio",  X
    "interest_rate",
    "property_value",
    "loan_term",  X
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "derived_race",
    "derived_sex",
    "applicant_age",
    "negative_amortization",
    "rate_spread",
"""

df = approved_applications.copy()

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY2019_positives.csv")
df.to_csv(output_file_path, index=False)


print(df.columns.tolist())

X = df[['loan_amount', 'debt_to_income_ratio', 'loan_to_value_ratio', 'loan_term', 'interest_rate']]

print(X.head())



#Changing column names to match fannie mae dataset
Column_names = [
    "original UPB",  #represent original unpaid balance (i.e. loan amount)
    "debt to income",
    "original LTV ratio", #loan to value
    "original loan term",
    "original interest rate",
]

X = X.rename(columns=dict(zip(X.columns, Column_names)))

X = X.dropna()

print(X.head())


#Ensuring features are scaled in the correct order
expected_features = list(scaler.feature_names_in_)
X = X[expected_features].copy()

print(X.head())


X_scaled = scaler.transform(X)

predictions = model_LR.predict(X_scaled)

probablility = model_LR.predict_proba(X_scaled)[:, 1]   #probability of delinquency
print(predictions
      )
print(probablility)

# Note accuracy score and classification report cannot be achieved as we don't know if these applicants become delinquent or not, therefore we must go off of certainty.

print(f"Average probability: {probablility.mean():.2f}")

#False = won't be delinquent, True = will be
