"""
Module for investigating the proxy problem (removing sensitive features not affecting bias) and how mitigation packages can remedy this

"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  true_positive_rate, selection_rate, false_positive_rate, false_negative_rate

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os


#Simple model made through removing the sensitive attributes

df = pd.read_csv("processed_datasets/NY2019.csv")

print(df.columns.to_list)

"""
_____________________________________________________________________________________________________________________
First, we will run an LR model after removing the sensitive features to investigate the extent of the proxy problem.
This is an observed affect when removing these features can have little to no effect on bias
______________________________________________________________________________________________________________________

"""

#dropping unneeded columns and sensitive columns
X = df.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", 
                     "negative_amortization", "applicant_age", "occupancy_type", 'applicant_age_35-44',
                    'applicant_age_45-54', 'applicant_age_55-64', 'applicant_age_65-74',
                    'applicant_age_<25', 'applicant_age_>74',  'derived_race_2 or more minority races',
                     'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
                    'derived_race_Black or African American', 'derived_race_Joint',
                    'derived_race_Native Hawaiian or Other Pacific Islander',
                    'derived_race_White', 'derived_sex_Female', 'derived_sex_Joint',
                    'derived_sex_Male', 'applicant_age_25-34'])
X = X.drop(columns=["interest_rate", "rate_spread"])

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




#initialising model

RF = RandomForestClassifier(class_weight="balanced", random_state=42)

#training
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)

#Saving model
import joblib

directory = "mitigated_models" 
os.makedirs(directory, exist_ok=True)

joblib.dump(RF, "mitigated_models/removed_sensitive_features_RF.pkl")



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")



print("Classification Report:")
print(classification_report(y_test, y_pred))




"""
Accuracy: 0.83
Classification Report:
              precision    recall  f1-score   support

       False       0.66      0.65      0.66     17755
        True       0.89      0.89      0.89     54167

    accuracy                           0.83     71922
   macro avg       0.77      0.77      0.77     71922
weighted avg       0.83      0.83      0.83     71922
"""


"""
Creating metricframe for analysis
"""

#matching original data to the predicted data


test_data = df.loc[test_index]

test_data["y_pred"] = y_pred  #puts y_pred value into the test data

print(test_data.columns.to_list)

metrics_dict = {"accuracy":accuracy_score, "selection rate": selection_rate, "count": count, "true positive rate":true_positive_rate,
                "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate}


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
METRIC FRAME
_______________

metrics frame by Race
                                           accuracy  selection rate    count  true positive rate
derived_race                                                                                    
2 or more minority races                   0.824675        0.480519    154.0            0.873016
American Indian or Alaska Native           0.788396        0.481229    293.0            0.806202
Asian                                      0.854754        0.756602   6816.0            0.917619
Black or African American                  0.800339        0.674576   5900.0            0.878240
Joint                                      0.858545        0.796402   1223.0            0.913313
Native Hawaiian or Other Pacific Islander  0.833828        0.471810    337.0            0.870504
White                                      0.830749        0.765695  57199.0            0.886155
Equalised odds difference for race
0.16524011710192135
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

metrics frame by age
               accuracy  selection rate    count  true positive rate
applicant_age                                                       
25-34          0.878200        0.857079  13399.0            0.932561
35-44          0.847967        0.791754  17220.0            0.908302
45-54          0.820281        0.738823  16843.0            0.886545
55-64          0.806888        0.690933  13764.0            0.855320
65-74          0.787798        0.664162   6786.0            0.824276
<25            0.844640        0.837856   1474.0            0.920569
>74            0.768883        0.602627   2436.0            0.793640
Equalised odds difference for age
0.2082713717294321
"""

"""
Fairlearn correlation remover

A preprocessing technique that removes the correlation between snesitive features and non-sensitive features.

https://fairlearn.org/main/user_guide/mitigation/preprocessing.html 
"""
from fairlearn.preprocessing import CorrelationRemover



y = df['approved']
X = df.drop(columns=["approved", "action_taken",  'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization'
       ]).copy()
X = X.drop(columns=["interest_rate", "rate_spread"])

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

X_train, X_test, y_train, y_test = train_test_split(X_CR, y, test_size = 0.2, random_state=42, stratify=y)

CR_test_index = X_test.index



#initialising model

RF = RandomForestClassifier(class_weight="balanced", random_state=42)

#training
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)

#Saving model


joblib.dump(RF, "mitigated_models/correlation_remover_RF.pkl")



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

Classification Report:
              precision    recall  f1-score   support

       False       0.46      0.67      0.55     17675
        True       0.87      0.75      0.81     54247

    accuracy                           0.73     71922
   macro avg       0.67      0.71      0.68     71922
weighted avg       0.77      0.73      0.74     71922

"""


"""
Evaluation with metric frame
___________________________
"""

CR_test_data = df.loc[CR_test_index]

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
                                           accuracy  selection rate    count  true positive rate
derived_race                                                                                    
2 or more minority races                   0.626506        0.144578    166.0            0.228571
American Indian or Alaska Native           0.633441        0.726688    311.0            0.888889
Asian                                      0.760145        0.622109   6875.0            0.765133
Black or African American                  0.738167        0.630916   5958.0            0.795009
Joint                                      0.734498        0.658515   1145.0            0.740385
Native Hawaiian or Other Pacific Islander  0.720126        0.572327    318.0            0.849624
White                                      0.725385        0.650563  57149.0            0.742598
Equalised odds difference for race
0.6603174603174603

__________________________________
Results from removing race


metrics frame by Race
                                           accuracy  selection rate    count  true positive rate
derived_race                                                                                    
2 or more minority races                   0.144578        0.626506    166.0            0.228571
American Indian or Alaska Native           0.726688        0.633441    311.0            0.888889
Asian                                      0.622109        0.760145   6875.0            0.765133
Black or African American                  0.630916        0.738167   5958.0            0.795009
Joint                                      0.658515        0.734498   1145.0            0.740385
Native Hawaiian or Other Pacific Islander  0.572327        0.720126    318.0            0.849624
White                                      0.650563        0.725385  57149.0            0.742598
Equalised odds difference for race
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


"""
_______________
For Age
_______________
"""

sensitive_features_age=CR_test_data["applicant_age"]


CR_age_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_age
)


#printing difference and ratio
print("metrics frame by age")
print(CR_age_MF.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age
))

"""

metrics frame by age
               accuracy  selection rate    count  true positive rate
applicant_age                                                       
25-34          0.803886        0.778916  13176.0            0.843641
35-44          0.696463        0.593802  17329.0            0.685554
45-54          0.751054        0.686502  16839.0            0.801177
55-64          0.685757        0.557703  13916.0            0.673246
65-74          0.702660        0.582827   6918.0            0.703064
<25            0.750368        0.739323   1358.0            0.797140
>74            0.720453        0.624895   2386.0            0.769634
Equalised odds difference for age
0.20424838593477235
"""