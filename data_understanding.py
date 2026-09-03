import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



print(df["occupancy_type"].value_counts(dropna=False).head(20))

print(df["negative_amortization"].value_counts(dropna=False).head(20))


plt.figure(figsize=(6, 4))
correlation = df1[['approved', 'loan_amount', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio', 'interest_rate', 'property_value', 'loan_term',  'rate_spread', 
                   'derived_race_2 or more minority races', 'derived_race_American Indian or Alaska Native', 'derived_race_Asian', 'derived_race_Black or African American', 'derived_race_Joint', 'derived_race_Native Hawaiian or Other Pacific Islander', 'derived_race_White', 
                   'derived_sex_Female', 'derived_sex_Joint', 'derived_sex_Male', 
                   'loan_type_1', 'loan_type_2', 'loan_type_3', 'loan_type_4', 
                   'loan_purpose_1', 'loan_purpose_2', 'loan_purpose_4', 'loan_purpose_5', 'loan_purpose_31', 'loan_purpose_32',
                     'negative_amortization_1', 'negative_amortization_2', 'negative_amortization_1111', 
                     'applicant_age_25-34', 'applicant_age_35-44', 'applicant_age_45-54', 'applicant_age_55-64', 'applicant_age_65-74', 'applicant_age_<25', 'applicant_age_>74',
                    'occupancy_type_1', 'occupancy_type_2', 'occupancy_type_3']].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("heatmap")
plt.show()

print(df["loan_purpose"].value_counts(dropna=False).head(20))


