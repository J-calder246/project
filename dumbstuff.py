import pandas as pd
import os

df = pd.read_csv("processed_datasets/reducedcols.csv")

df = df.head()

columns_to_keep = [
    "loan_type",
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "interest_rate",
    "property_value",
    "loan_term",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "derived_race",
    "derived_sex",
    "applicant_age",
    "negative_amortization",
    "rate_spread",   #difference between APR and APOR
]

df = df[columns_to_keep]

print(df.shape)

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "5_entries.csv")
df.to_csv(output_file_path, index=False)