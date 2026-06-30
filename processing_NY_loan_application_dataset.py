import pandas as pd
import os
import numpy as np




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

#converting ranges into a single number for DTI and age

debt_to_income_mode = {
    "<20%": 10,    #low level to be generous to applicant with low levels of debt
    "20%-<30%": 25,
    "30%-<36%": 33,
    "36": 36,
    "37": 37,
    "38": 38,
    "39": 39,
    "40": 40,
    "41": 41,
    "42": 42,
    "43": 43,
    "44": 44,
    "45": 45,
    "46": 46,
    "47": 47,
    "48": 48,
    "49": 49,
    "50%-60%": 55,
    ">60%": 65,  # low level to be punishing to applicant with high levels of debt
    "Exempt": None
    

}

age_mode = {
    "<25": 23,
    "25-34": 30,
    "35-44": 40,
    "45-54": 50,
    "55-64": 60,
    "65-74": 70,
    ">74": 80,
    "8888": np.nan,
}

df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(debt_to_income_mode)

df["applicant_age"] = df["applicant_age"].map(age_mode)


#getting dummy values for race, gender etc
df = pd.get_dummies(df, columns=["derived_race"])
df = pd.get_dummies(df, columns=["derived_sex"])

#setting "exempt" value to NAN 
df = df.replace("Exempt", np.nan)





#Upload dataset

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY2024.csv")
df.to_csv(output_file_path, index=False)
