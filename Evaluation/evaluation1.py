
import pandas as pd


df_original = pd.read_csv("processed_datasets/NY2019_no_dummies.csv")
print(df_original.head())

df_approved_non_delinquent = pd.read_csv("modelled_datasets/approved_non_delinquent")

"""
First some processing
______________________

Need to turn dummy values back to their original values for a clearer dataset
"""

