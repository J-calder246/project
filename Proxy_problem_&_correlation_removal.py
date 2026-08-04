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

#matching original data to the predicted data


test_data = df.loc[test_index].copy()

test_data["y_pred"] = y_pred  #puts y_pred value into the test data

print(test_data.columns.to_list)

metrics_dict = {"accuracy":selection_rate, "selection rate": accuracy_score, "count": count}


sensitive_features_race=test_data["derived_race"]


Race_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_race
)


#printing difference and ratio
print("metrics frame by Race")
print(Race_MF.by_group)


print("Equalised odds difference for race")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_race
))

"""
Original model
______________
metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.194175        0.582524    103.0
American Indian or Alaska Native           0.161290        0.689516    248.0
Asian                                      0.615758        0.773152   5356.0
Black or African American                  0.400665        0.642983   5109.0
Joint                                      0.708989        0.767416    890.0
Native Hawaiian or Other Pacific Islander  0.116438        0.767123    292.0
White                                      0.669544        0.756452  46693.0
Equalised odds difference for race
0.5110852110852111


Results
_____________

derived_race                              Accuracy     Selection rate   Count                          
2 or more minority races                   0.563107        0.718447    103.0
American Indian or Alaska Native           0.391129        0.782258    248.0
Asian                                      0.656273        0.787901   5356.0
Black or African American                  0.562145        0.715600   5109.0
Joint                                      0.710112        0.764045    890.0
Native Hawaiian or Other Pacific Islander  0.280822        0.835616    292.0
White                                      0.642280        0.739554  46693.0
Equalised odds difference for race
0.28560573165436387

Big increases to accuracy and selection rate for the smallest classes and similar reduction in equalised odds difference to the First mitigation (expponentiated gradient reduction)

"""


sensitive_features_age=test_data["applicant_age"]


age_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_age)



#printing difference and ratio
print("metrics frame by age")
print(age_MF.by_group)

print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age
))

"""
Original model
______________

metrics frame by age
               accuracy  selection rate    count
applicant_age                                   
35-44          0.738452        0.792233  16995.0
45-54          0.625667        0.737815  17049.0
55-64          0.568465        0.728677  13788.0
65-74          0.562900        0.711158   7035.0
<25            0.867238        0.811563   1401.0
>74            0.455221        0.683038   2423.0

Equalised odds difference for age
0.48032064284599785

Results
________
applicant_age  Accuracy   Selection rate  Count                              
35-44          0.720741        0.778641  16995.0
45-54          0.636518        0.739164  17049.0
55-64          0.579925        0.731868  13788.0
65-74          0.534328        0.689979   7035.0
<25            0.869379        0.813704   1401.0
>74            0.483698        0.693355   2423.0
Equalised odds difference for age
0.4554337650179435

No large changes to accuracy, selection rate or equalised odds difference
"""

"""
Fairlearn correlation remover

A preprocessing technique that removes the correlation between snesitive features and non-sensitive features.

https://fairlearn.org/main/user_guide/mitigation/preprocessing.html 
"""
from fairlearn.preprocessing import CorrelationRemover



y = df['approved']
X = df.drop(columns=["approved", "action_taken",  'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1', 'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization'
       ]).copy()

print(X.columns.to_list)

Sensitive_features = ['derived_race_2 or more minority races',
       'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
       'derived_race_Black or African American', 'derived_race_Joint',
       'derived_race_Native Hawaiian or Other Pacific Islander',
       'derived_race_White',

       'derived_sex_Female', 'derived_sex_Joint',
       'derived_sex_Male',

       'applicant_age_35-44', 'applicant_age_45-54', 'applicant_age_55-64', 
       'applicant_age_65-74', 'applicant_age_<25', 'applicant_age_>74'
       ]

Sensitive_features = [feature for feature in Sensitive_features if feature in X.columns]

CR = CorrelationRemover(sensitive_feature_ids=Sensitive_features)

print(X.columns.to_list())



imputer = SimpleImputer(strategy= "median")

X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_CR = pd.DataFrame(CR.fit_transform(X))
X_CR = pd.DataFrame(X_CR, columns=X_CR.columns)
print(X_CR.columns.to_list)



#Make model with the preprcoessed data

X_train, X_test, y_train, y_test = train_test_split(X_CR, y, test_size = 0.2, random_state=69)

CR_test_index = X_test.index





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
Results
_________

Accuracy: 0.73
Confusion Matrix:
[[10769  4983]
 [10598 32341]]
Classification Report:
              precision    recall  f1-score   support

       False       0.50      0.68      0.58     15752
        True       0.87      0.75      0.81     42939

    accuracy                           0.73     58691
   macro avg       0.69      0.72      0.69     58691
weighted avg       0.77      0.73      0.75     58691
"""


"""
Evaluation with metric frame
___________________________
"""

CR_test_data = df.loc[CR_test_index].copy()

CR_test_data["y_pred"] = y_pred  #puts y_pred value into the test data

sensitive_features_race=CR_test_data["derived_race"]


CR_Race_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_race
)


#printing difference and ratio
print("metrics frame by Race")
print(CR_Race_MF.by_group)


print("Equalised odds difference for race")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_race
))

"""
Results of Correlation removal
-------------------------------

metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.672414        0.775862    116.0
American Indian or Alaska Native           0.526119        0.764925    268.0
Asian                                      0.652480        0.776933   5263.0
Black or African American                  0.648848        0.754439   5294.0
Joint                                      0.649942        0.737456    857.0
Native Hawaiian or Other Pacific Islander  0.528846        0.772436    312.0
White                                      0.633606        0.726884  46581.0
Equalised odds difference for race
0.20933226823892392

__________________________________
Results from removing race


derived_race                              Accuracy     Selection rate   Count                          
2 or more minority races                   0.563107        0.718447    103.0
American Indian or Alaska Native           0.391129        0.782258    248.0
Asian                                      0.656273        0.787901   5356.0
Black or African American                  0.562145        0.715600   5109.0
Joint                                      0.710112        0.764045    890.0
Native Hawaiian or Other Pacific Islander  0.280822        0.835616    292.0
White                                      0.642280        0.739554  46693.0
Equalised odds difference for race
0.28560573165436387
---------------------------

Original model
--------------

metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.194175        0.582524    103.0
American Indian or Alaska Native           0.161290        0.689516    248.0
Asian                                      0.615758        0.773152   5356.0
Black or African American                  0.400665        0.642983   5109.0
Joint                                      0.708989        0.767416    890.0
Native Hawaiian or Other Pacific Islander  0.116438        0.767123    292.0
White                                      0.669544        0.756452  46693.0
Equalised odds difference for race
0.5110852110852111


"""