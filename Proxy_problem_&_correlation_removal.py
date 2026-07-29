"""
Module for investigating the proxy problem (removing sensitive features not affecting bias) and how mitigation packages can remedy this

"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier
import os


#Simple model made through removing the sensitive attributes

df = pd.read_csv("processed_datasets/NY2019.csv")

print(df.columns.to_list)

X = df.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", "applicant_age", "occupancy_type", 'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1', 'applicant_age_35-44',
       'applicant_age_45-54', 'applicant_age_55-64', 'applicant_age_65-74',
       'applicant_age_<25', 'applicant_age_>74',  'derived_race_2 or more minority races',
       'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
       'derived_race_Black or African American', 'derived_race_Joint',
       'derived_race_Native Hawaiian or Other Pacific Islander',
       'derived_race_White', 'derived_sex_Female', 'derived_sex_Joint',
       'derived_sex_Male'])

print(X.columns.to_list)
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    

test_index = X_test.index   #save X values so they arent messed up when scaled later on

#standardising features with scaler (gives attributes equal weighting and influence)

#Imputing missing values (change later if this is not suitable)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#X_train = X_train.dropna()
#X_test = X_test.dropna()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising model

logreg = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print reports
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
Original classification report (from trainingmodels/modelling.py)

Classification Report:
              precision    recall  f1-score   support

       False       0.52      0.71      0.60     15776
        True       0.88      0.76      0.82     42915

    accuracy                           0.75     58691
   macro avg       0.70      0.74      0.71     58691
weighted avg       0.78      0.75      0.76     58691

New classification report

Classification Report:
              precision    recall  f1-score   support

       False       0.52      0.70      0.59     15776
        True       0.87      0.76      0.81     42915

    accuracy                           0.74     58691
   macro avg       0.69      0.73      0.70     58691
weighted avg       0.78      0.74      0.75     58691
"""


"""
Creating metricframe for analysis
"""

test_index = X_test.index

test_data = df.loc[test_index].copy()

test_data["y_pred"] = y_pred  #puts y_pred value into the test data

print(test_data.columns.to_list)