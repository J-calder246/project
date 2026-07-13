import pandas as pd
import matplotlib.pyplot as plt
from modelling import approved_applications, logreg, scaler, model_approval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from FannieMae_modelling import model_FM
import os

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


"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    


#standardising features with scaler (gives attributes equal weighting and influence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





#initialising model



#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

"""