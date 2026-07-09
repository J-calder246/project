This file contains code for data processing done on the fannie mae mortgage deliquency dataset. This code could not be done in github as the raw dataset was too large (over 3GB) to be uploaded onto github. As a result this stage of the process was done in the jupyter notebook which could handle the massive dataset and once it was scaled down it was uploaded onto github (see /2019, NY fianl (1).xls)
---------------------------------------------------

Code from jupyter listed below

import pandas as pd
import os
import csv

size_gb = os.path.getsize("datasets/2019Q1[1].csv") / (1024**3)
print(size_gb)

#State = 3



# Check raw delinquency column BEFORE filtering


# Read the CSV file in smaller chunks to avoid memory issues
chunk_size = 10000  # Adjust this number based on your available memory
chunks = []

# Read the file in chunks
for chunk in pd.read_csv(
    "datasets/2019Q1[1].csv",
    sep="|",
    header=None,
    dtype={105: str},
    chunksize=chunk_size  # Process data in smaller batches
):
    # Process each chunk if needed (filter, transform, etc.)
    # For example: chunk = chunk[chunk[some_column] > some_value]
    chunks.append(chunk)

# Combine all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)

print(df.shape)
print("Raw delinquency values:")
print(df[39].value_counts(dropna=False).head(20))


print(df[39].value_counts(dropna=False).head(20))

print(df.columns.dtype)

print(f"all cols: {list(df.columns)}")

columns_to_keep = [
    0,
    1,
    7,
    8,
    9,
    12,
    19,
    20,
    22,
    23,
    30,
    31,
    34,
    35,
    39,
    64,
    65,
    66,
    67,
    68,
    99,
    102
]

df =df[columns_to_keep].copy()

print(df.shape)
print(df.head())

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2019, less cols.csv"
df.to_csv(
    output_file_path,
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)


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
    "state",
    "MSDA",
    "Amortization type",
    "prepayment penalty indicator",
    "Current Loan Delinquency Status",
    "original list start date",
    "original list price",
    "current list start date",
    "current list price",
    "credit score at issuance",
    "ARM baloon indicator",
    "HLTV refinance option indicator"
]

df_final = pd.read_csv(
    "2019, less cols.csv",
    header=None,
    names=column_names,
)

print(f"all cols: {list(df_final.columns)}")

print(df_final["Current Loan Delinquency Status"].value_counts(dropna=False).head(20))



print(df_final.head())

print (df_final.shape)


directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2019, less cols, named cols.csv"
df_final.to_csv(
    output_file_path,
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")



df_final = pd.read_csv("2019, less cols, named cols.csv", header=0)

print(f"all cols: {list(df_final.columns)}")

print(df_final["Current Loan Delinquency Status"].value_counts(dropna=False).head(20))


df_final["Current Loan Delinquency Status"] = (
    df_final["Current Loan Delinquency Status"]
    .astype(str)
    .str.zfill(2)
)

#saving final dataset, not duplicates

#print(df["Current Loan Delinquency Status"].value_counts())

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)

output_file_path = r"2019, less cols, named cols (the main file).csv"
df_final.to_csv(
    output_file_path,
    index=False,
)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")

#Getting only new york entries



df = df_final[df_final["state"] == "NY"].copy()


#save dataset with just NY entries

directory = "project"
if not os.path.exists(directory):
    os.makedirs(directory)



output_file_path = r"2019, NY fianl.csv"
df.to_csv(
    output_file_path,
    index=False,
)

print(f"\nAmmended data (no missing values and encoded data) saved to: {output_file_path}")


#Saving final dataset
size_gb = os.path.getsize("2019, NY fianl.csv") / (1024**3)
print(size_gb)