#Information and guide: https://pypi.org/project/aequitas/
import pandas as pd
import numpy as np
from aequitas.flow.methods.preprocessing.prevalence_sample import PrevalenceSampling


from sklearn.model_selection import train_test_split


df = pd.read_csv("processed_datasets/NY2019.csv")



y = df['approved']
X = df.drop(columns=["approved", "action_taken",  'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1', 'loan_purpose', 'occupancy_type', 'negative_amortization', 'derived_race', 'derived_sex', 'applicant_age'
       ]).copy()
S = df['derived_race']   #setting race as the target variable as it is the one that suffers from imbalance the most

print(X.columns.to_list)

S = S.astype('category')  #converts sensitive columns to a categorical type so the oversampler can process it

print(y.value_counts())

resampler = PrevalenceSampling(strategy='undersample')
resampler.fit(X, y, S)
X_sampled, y_sampled, S_sampled = resampler.transform(X, y, S)

"""
#splitting the data first to avoid over sample manipulating the test data
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.2, random_state=42)

#oversampling the training data
resampler = PrevalenceSampling(strategy='oversample')
resampler.fit(X_train, y_train, S_train)
X_sample_train, y_sample_train, S_sample_train = resampler.transform(X_train, y_train, S_train)
"""


print(y_sampled.value_counts())