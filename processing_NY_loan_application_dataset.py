import pandas as pd
import os



df = pd.read_csv("raw_datasets/NY_loan_application_datasets.csv", low_memory=False)


print(df.columns.tolist())
#Simplifying dataset before uploading

df = df.drop_duplicates()

df = df[~df["action_taken"].isin(["3", "5"])]
df = df[
    (df["derived_dwelling_category"].isin(["Single Family (1-4 Units):Site-Built", "Single Family (1-4 Units):Manufactured"]))]

columns_to_keep = [
    "action_taken",
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
    "rate_spread",
]

df = df[columns_to_keep]


#Upload dataset

directory = "processed_datasets"
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY_applications_data(simplified1).csv")
df.to_csv(output_file_path, index=False)
