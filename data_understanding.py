import pandas as pd

df = pd.read_csv("processed_datasets/reducedcols.csv")

df_rejects = df[df["action_taken"].isin([3])]

print(df_rejects.head)

print(df_rejects['interest_rate'].value_counts(dropna=False).head(20))


print(df_rejects['rate_spread'].value_counts(dropna=False).head(20))

#Note: All applications that are rejected don't have a value for interest rate as none is offered. 
# This could be interfering with models like random forest as the model will automatically reject any application where the interest rate has no value.
#To evade this issue, use multivariate imputation strategies


df1 = pd.read_csv("processed_datasets/NY2019.csv")

df1_rejects = df1[df1["action_taken"].isin([3])]

print(df1_rejects.head)

print(df1_rejects['interest_rate'].value_counts(dropna=False).head(20))


print(df1_rejects['rate_spread'].value_counts(dropna=False).head(20))