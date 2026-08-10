#This module will attempt to manipulate datasets and mitigate bias that way




"""
Balance between races (based on applications dataset after data cleaning processed_datasets/NY2019.csv)

White                                        287012
Race Not Available                            64177
Asian                                         34446
Black or African American                     29315
Joint                                          5945
Native Hawaiian or Other Pacific Islander      1688
American Indian or Alaska Native               1568
2 or more minority races                        750

Final approval by race

American Indian or Alaska Native             0.022321
Asian                                        0.079806
Black or African American                    0.041480
Joint                                        0.103112
Native Hawaiian or Other Pacific Islander    0.014810
Race Not Available                           0.061112
White                                        0.086763


Balance by age

35-44    103000
45-54    100145
55-64     79538
25-34     76035
65-74     39322
>74       13491
<25        7778

Final approval by age

25-34    0.116275
35-44    0.088262
45-54    0.069889
55-64    0.062209
65-74    0.057907
8888     0.004557
<25      0.108511
>74      0.034467


Gender

Joint                150411
Male                 143526
Female                93405
Sex Not Available     37673

Approval by gender

Joint                0.096901
Male                 0.069388
Female               0.071773
Sex Not Available    0.059804


"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler  # Trying oversampling first to avoid losing potential valuable training data and potentially reducing the model's accuracy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier
import os



#Oversample code found on Geeksforgeeks https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-data-for-classification/


df1 = pd.read_csv("processed_datasets/NY2019.csv")

X = df1.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", "applicant_age", "occupancy_type", 'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1'])
X = X.drop(columns=["interest_rate", "rate_spread"])
y = df1['approved']

oversample = RandomOverSampler(sampling_strategy = 'minorty')
X_over_sampled, y_over_sampled = oversample.fit_resample(X, y)
print