import pandas as pd
import os
from pathlib import Path
from io import StringIO




df = pd.read_csv("Datasets/2019, NY fianl (1).xls")
print(df.columns)


print(df["Current Loan Delinquency Status"].value_counts(dropna=False))

    
"""
Input features from first dataset:
-----------------------------------
"action_taken",
    "loan_type",
    "loan_amount",   X
    "income",
    "debt_to_income_ratio",   X
    "loan_to_value_ratio",   x  
    "interest_rate",   X
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

    


Features from fannie may dataset:
-----------------------------------
refererence pool ID,
loan id,
original interest rate,  X
current interest rate,
original UPB   (similar to loan amount)  X  
original loan term,   X
original LTV (LOAN TO VALUE) ratio,  X
original combined LTV ratio,
debt to income,   X
original credit score,
state,
MSDA,
Amortization type,
prepayment penalty indicator,
Current Loan Delinquency Status,   X   (target)
original list start date,
original list price,   X  (either use this for loan amounts or UPB)
current list start date,
current list price,
credit score at issuance,
ARM baloon indicator,
HLTV refinance option indicator

"""

Columns_to_keep = [
    "loan id",
    "original interest rate",
    "original UPB",
    "original loan term",
    "original LTV ratio",
    "debt to income",
    "Current Loan Delinquency Status",
    "original list price",
]

df = df[Columns_to_keep]

print (df.head())

print(df.columns.tolist())

print(df["Current Loan Delinquency Status"].value_counts(dropna=False))
#Delinquency has monthly values for each case, for this, i need to ddivide the dataset into those who are delinquent and those who arent

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY2019.csv")
df.to_csv(output_file_path, index=False)


df = pd.read_csv("processed_datasets/NY2019.csv")

#delting XX values for delinquency

df = df[df["Current Loan Delinquency Status"] != "XX"].copy()

#Convert string numbers into numeric values for delinquency so different people can be compared

df["Current Loan Delinquency Status"] = pd.to_numeric(df["Current Loan Delinquency Status"], errors="coerce")

#Getting maximum delinquecy for each case, so I can set up a way of classifying deliquency 

print(df.shape)

Months_delinquent = (df.groupby("loan id")["Current Loan Delinquency Status"].max())

#Keep 1 row of information for each case
loan__features = (df.groupby("loan id")
                  .first()
                  .reset_index())


#Change delinquency columns to the maximum value

loan_features = loan__features.drop(columns=["Current Loan Delinquency Status"])  #drops original delinquency cols

loan_features = loan_features.merge(
    Months_delinquent, on="loan id"
)

print(df.head())

print(df["Current Loan Delinquency Status"].value_counts(dropna=False))

#Creating simple target for delinquency (>0 = deinquenct)

loan_features["delinquent"] = (loan_features["Current Loan Delinquency Status"] > 0).astype(str)

print(loan_features["delinquent"].value_counts())

"""
delinquent
False    8488
True     2171
"""
df = loan_features.copy()


directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "SingleValue_NY2019.csv")
df.to_csv(output_file_path, index=False)

print(df["delinquent"].value_counts())

"""
results

delinquent
False    8488
True     2171

classes are imbalanced, heavily weighted towards non-delinquent homeowners
"""