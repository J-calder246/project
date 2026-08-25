#importing necessary libraries for processing

import pandas as pd
import os
import numpy as np
from pandas import DataFrame

"""
Note

Many processing decisions are taken from the HMDA's glossary on the dataset. E.G. Removing applications irrelevant action_taken points like application withdrawn/ closed for incompleteness

Link to glossary: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields 
"""



df = pd.read_csv("raw_datasets/state_NY 2019 (1).csv", low_memory=False)



print(df.columns.tolist())
#Simplifying dataset before uploading

df = df.drop_duplicates()


df = df[df["action_taken"].isin([1, 2, 3])]  #Keeps values for loan given out (1), loan approved but not purchased (2) and loan denied (3). Other action aren't useful for what we are trying to do
df["approved"] = df["action_taken"].isin([1, 2]) #defines first two as loans that are approved (wheterh they are accepted or not)
df = df[
    (df["derived_dwelling_category"].isin(["Single Family (1-4 Units):Site-Built", "Single Family (1-4 Units):Manufactured"]))]   #focuses dataset on single family applications


#Cutting dataset down to columns I want to use. 
columns_to_keep = [
    "approved",
    "action_taken",
    "loan_type",
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "interest_rate",
    "property_value",
    "loan_term",
    "loan_purpose",
    "occupancy_type",
    "derived_race",
    "derived_sex",
    "applicant_age",
    "negative_amortization",
    "rate_spread",   #difference between APR and APOR
]

df = df[columns_to_keep]

print(df.dtypes)



#converting ranges into a single number for Debt to income so that higher levels can have more influence on models. This methods also means that DTI won't have to be encoded

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
    ">60%": 65,  # debt to income converted into single numbers so models can accordingly interpret the higher levels
    "Exempt": None
    

}

df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(debt_to_income_mode)


"""
Note: saving a dataset now before I start converting race and gender etc into dummy values. 
This is important for easy interpretation in the evaluation stage.
"""

#Removing unclear values from dataset (like 1111 or race not available)

race_cols_to_keep = [
    '2 or more minority races',
    'American Indian or Alaska Native',
    'Asian',
    'Black or African American',
    'Joint',
    'Native Hawaiian or Other Pacific Islander',
    'White'  

]

Age_cols_to_keep = [
    '25-34',
    '35-44',
    '45-54',
    '55-64',
    '65-74',
    '>74',
    '<25'

]

Gender_cols_to_keep = [
    'Joint',
    'Male',
    'Female'

]

#Getting columns with the features I want to keep
df = df[df['derived_race'].isin(race_cols_to_keep)].copy()
df = df[df['applicant_age'].isin(Age_cols_to_keep)].copy()
df = df[df['derived_sex'].isin(Gender_cols_to_keep)].copy()

print(df.value_counts('derived_race'))

df_no_dummies = df

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY2019_no_dummies.csv")
df_no_dummies.to_csv(output_file_path, index=False)  #getting a csv with no dummies for evaluation later on

#getting dummy values for race, gender etc while keeping the original values for evaluation later on
cols_to_process = ["derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization",
                    "applicant_age", "occupancy_type", ]

dummies = pd.DataFrame(pd.get_dummies(df, columns = cols_to_process,  drop_first=False)  ) #creates a dataset with the dummy values without removing the original unencoded values


print("dummy cols")
print(dummies.columns.to_list())

#removes unencoded values from the dummies dataframe to prevent these columns from being duplicated
dummies = dummies.drop(columns=['approved', 'action_taken', 'loan_amount', 'income', 
                                'debt_to_income_ratio', 'loan_to_value_ratio', 'interest_rate', 
                                'property_value', 'loan_term', 'rate_spread'])

print(dummies.columns.to_list())

#combines the two
df = pd.concat([df, dummies], axis= 1)

"""
Other items to drop/get dummies for

loan_type, amorization, loan purpose, occupancy type and age (revert age back to original ranges)
"""


print(df.columns.to_list())

#setting "exempt" value to NAN 
df = df.replace("Exempt", np.nan)

print(df.head())

print(df.columns[df.columns.str.endswith(".1")])

#Imputing missing values with iterative imputer (a multivariate imputer) 
# This means that missing values will be estimated and will be unique rather than one uniform value that can interfere with models

print(df.isnull().sum)




#Upload dataset

directory = "processed_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "NY2019.csv")
df.to_csv(output_file_path, index=False)

