import pandas as pd
import os

#State = 31

df = pd.read_csv(
    "raw_datasets/2024Q1.csv",
    sep="|",
    header=None,
    dtype={105: str}
)

"""


"""
directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2024"
df.to_csv(output_file_path, index=False)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")

df = pd.read_csv("2024")

print(df.columns.tolist())


df_NY = df[df["30"] == "NY"]
print(df_NY.head)

#save dataset with just NY entries

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2024, NY"
df_NY.to_csv(output_file_path, index=False)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")

#read file
df_keep = pd.read_csv("2024, NY", header=None)
print(f"DF shape: {df.shape}")
print(f"all cols: {list(df.columns)}")


#save dataset with desired columns

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2024, NY, less cols"
df_keep.to_csv(output_file_path, index=False)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")

#applying names to columns

column_names = [
    "refererence pool ID",
    "loan id",
    "original interest rate",
    "current interest rate",
    "original UPB",
    "original loan term",
    "original LTV ratio",
    "original combined LTV ratio",
    "debt to income",
    "original credit score",
    "MSDA",
    "Amortization type",
    "prepayment penalty indicator",
    "original list start date",
    "original list price",
    "current list start date",
    "current list price",
    "credit score at issuance",
    "ARM baloon indicator",
    "HLTV refinance option indicator"
]

df_final = pd.read_csv(
    "2024, NY, less cols",
    header=None,
    names=column_names,
)

print(df_final.head())

#saving dataset with names cols

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2024, NY, less cols, named cols (the main file)"
df_final.to_csv(output_file_path, index=False)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")

#dropping duplicates

df_final = df_final.drop_duplicates(subset=["loan id"])

#saving final dataset, not duplicates


directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2024, NY, less cols, named cols (the main file)"
df_final.to_csv(output_file_path, index=False)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")
