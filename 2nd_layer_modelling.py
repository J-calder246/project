"""
Second Layer Modelling
______________________-
This module uses both the model in modelling.py (for the applications) and the Fannie Mae delinquency model on the whole processed dataset
to approve applications and predict if they are delinquent or not. 
As this model works on the whole dataset, this is the one that will be primarily evaluated for bias.


"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import os
import joblib 
from sklearn.impute import SimpleImputer

#loading the logistic regression models that will be used
model_LR_fm = joblib.load("models/FMlogistic.pkl")
scaler_fm = joblib.load("models/FMscaler.pkl")
model_LR_1st = joblib.load("models/logistic.pkl")


df = pd.read_csv("processed_datasets/NY2019.csv")

print("df head")
print(df.head)


#Setting X and y(target) variables
X = df.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", 
                     "loan_purpose", "negative_amortization", "applicant_age", "occupancy_type", 
                     
       ])
X = X.drop(columns=["interest_rate", "rate_spread"])
y = df['approved']


#imputing missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)




#scaling X
scaler = StandardScaler()
X = scaler.fit_transform(X)



#modelling full dataset based upon model created and saved in modelling.py

predictions = model_LR_1st.predict(X)

probability = model_LR_1st.predict_proba(X)[:, 1]   #probability of delinquency shows us how certain the models are 
print(predictions
      )
print(probability)

# Note accuracy score and classification report cannot be achieved as we don't know if these applicants become delinquent or not, therefore we must go off of certainty.

print(f"Average probability: {probability.mean():.2f}")

#creating column for if the application is approved
df["approved"] = (predictions)

#creating column for the probability of delinquency according to the model

df["probability_of_approval"] = (probability)

#Saving non-delinquent applications for evaluation

print(df.columns.tolist())

approved_applications = df[df["approved"].isin([True])]   #keeps rows that were approved for next layer of modelling

print("approved applications")
print(approved_applications.head)

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

#cutting the dataset down to the features that are in (or can be interpreted to be in) the delinquency dataset 
X_fm = df[['loan_amount', 'debt_to_income_ratio', 'loan_to_value_ratio', 'loan_term', 'interest_rate']]


print(X_fm.head())

X_fm = X_fm.apply(pd.to_numeric, errors="coerce")  #ensures numeric values



#Changing column names to match fannie mae dataset
Column_names = [
    "original UPB",  #represent original unpaid principal balance (i.e. loan amount)
    "debt to income",
    "original LTV ratio", #loan to value
    "original loan term",
    "original interest rate",
]

X_fm = X_fm.rename(columns=dict(zip(X_fm.columns, Column_names)))

#setting up mask for filtering missing values

mask = X_fm.notna().all(axis=1)

#applying mask X and original dataset to ensure that they are the same size and can therefore be saved together

X_fm_masked = X_fm.loc[mask].copy()   #copies X values accoring the the mask
approved_applications_masked = approved_applications.loc[mask].copy()

print(len(X_fm_masked))
print(len(approved_applications_masked))

if len(X_fm_masked) == len(approved_applications_masked
                        ):
    print("lengths match, these will work")
else:
    print(":(")

"""
results
39977
39977
lengths match, these will work
"""

                    

print(X_fm_masked.head())


#Ensuring features are scaled in the correct order (the order they fit in the (scaled) fannie mae model)
expected_features = list(scaler_fm.feature_names_in_)
X_MS = X_fm_masked[expected_features].copy()


print(X_MS.head())


X_MS = scaler_fm.transform(X_MS)

predictions = model_LR_fm.predict(X_MS)

probability = model_LR_fm.predict_proba(X_MS)[:, 1]   #probability of delinquency
print(predictions
      )
print(probability)

# Note accuracy score and classification report cannot be achieved as we don't know if these applicants become delinquent or not, therefore we must go off of certainty.

print(f"Average probability: {probability.mean():.2f}")

#False = won't be delinquent, True = will be

#creating columns for these predictions

approved_applications_masked["delinquent"] = (predictions)

#creating column for the probability of delinquency according to the model

approved_applications_masked["probability_of_delinquency"] = (probability)

#Saving non-delinquent applications for evaluation

print(approved_applications_masked.columns.tolist())

non_delinquent_applications = approved_applications_masked[
    approved_applications_masked["delinquent"] == 0].copy()

delinquent_applications = approved_applications_masked[
    approved_applications_masked["delinquent"] == 1].copy()


print(non_delinquent_applications.head())


directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "approved_non_delinquent.csv")
non_delinquent_applications.to_csv(output_file_path, index=False)

directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "approved_but_delinquent.csv")
delinquent_applications.to_csv(output_file_path, index=False)

