"""
This module will attempt to mitigate bias of the initial LR model by first augmenting the data before it is used in modelling.

The first technique used is simply dropping the sensitive features from the dataset and then training the model without them.
The technique is here not only to try and mitigate bias this was but also to investigate the 'proxy problem' established in the literature and see if removing these features has any effect at all on bias.

After this we will try using a correlation remover (one of fairlearn's preprocessing techniques) to see if this method has any effect on reducing bias.
This technique works by reducing the influence that sensitive features have on a model's prediction and seeing the extent of bias after this technique is implemented.
"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  true_positive_rate, selection_rate, false_positive_rate, false_negative_rate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier
import os


#Simple model made through removing the sensitive attributes

df = pd.read_csv("processed_datasets/NY2019.csv")

print(df.columns.to_list)


#dropping unneeded columns and sensitive columns for experiementing on results when dropping sensitive columns
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


#Imputing missing values (change later if this is not suitable)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



#standardising features with scaler (gives attributes equal weighting and influence)

scaler = joblib.load("models/FeatureScaler.pkl")  #load scaler from original model
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising model

logreg = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

#Saving model
import joblib

directory = "mitigated_models" 
os.makedirs(directory, exist_ok=True)

joblib.dump(logreg, "mitigated_models/removed_sensitive_features.pkl")  #saves the model that has been trained on the lower number of features

logreg = joblib.load("mitigated_models/removed_sensitive_features.pkl")



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print reports

print("Classification Report:")
print(classification_report(y_test, y_pred))

#Save scaler fitted for the smaller X so it can be used in the app

joblib.dump(scaler, "models/FeatureScaler_NSF.pkl")


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

       False       0.47      0.68      0.56     17755
        True       0.88      0.75      0.81     54167

    accuracy                           0.73     71922
   macro avg       0.67      0.72      0.68     71922
weighted avg       0.78      0.73      0.75     71922

"""


"""
Creating metricframe for analysis
"""

#matching original data to the predicted data


test_data = df.loc[test_index].copy()

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
Results
_____________

metrics frame by Race
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.668831        0.402597    154.0            0.587302             0.274725             0.412698
American Indian or Alaska Native           0.720137        0.392491    293.0            0.627907             0.207317             0.372093
Asian                                      0.787999        0.676643   6816.0            0.817398             0.292282             0.182602
Black or African American                  0.715085        0.578136   5900.0            0.733531             0.316029             0.266469
Joint                                      0.771055        0.726901   1223.0            0.814241             0.393701             0.185759
Native Hawaiian or Other Pacific Islander  0.756677        0.305638    337.0            0.575540             0.116162             0.424460
White                                      0.726656        0.645204  57199.0            0.740752             0.321152             0.259248
Equalised odds difference for race
0.27753917123995864

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

Results
________
metrics frame by age
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.837898        0.832152  13399.0            0.894187             0.480855             0.105813
35-44          0.763240        0.699361  17220.0            0.795256             0.352547             0.204744
45-54          0.707891        0.600190  16843.0            0.712996             0.305448             0.287004
55-64          0.673278        0.528553  13764.0            0.644057             0.258370             0.355943
65-74          0.642057        0.476864   6786.0            0.585325             0.227516             0.414675
<25            0.840570        0.881275   1474.0            0.944816             0.607914             0.055184
>74            0.619869        0.404351   2436.0            0.519143             0.206704             0.480857
Equalised odds difference for age
0.4256726401437628


"""

"""
Fairlearn correlation remover

A preprocessing technique that removes the correlation between snesitive features and non-sensitive features.

https://fairlearn.org/main/user_guide/mitigation/preprocessing.html 
"""
from fairlearn.preprocessing import CorrelationRemover



y = df['approved']
X = df.drop(columns=["approved", "action_taken",  'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization', 'loan_type'
       ]).copy()
X = X.drop(columns=["interest_rate", "rate_spread"])

print(X.columns.to_list)

#defining the sensitive features that the correlation remover will target
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





scaler = joblib.load("models/FeatureScaler.pkl")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising model

logreg = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

#Saving model


joblib.dump(CR, "mitigated_models/correlation_remover.pkl") #saves the tool that transformed the data
joblib.dump(logreg, "mitigated_models/correlation_remover_LR_model.pkl")  #save the model trained on the transformed data


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
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.636364        0.123377    154.0            0.206349             0.065934             0.793651
American Indian or Alaska Native           0.641638        0.696246    293.0            0.883721             0.548780             0.116279
Asian                                      0.754108        0.623680   6816.0            0.758068             0.256705             0.241932
Black or African American                  0.736441        0.642203   5900.0            0.801566             0.373406             0.198434
Joint                                      0.721177        0.662306   1223.0            0.742002             0.358268             0.257998
Native Hawaiian or Other Pacific Islander  0.735905        0.545994    337.0            0.841727             0.338384             0.158273
White                                      0.722687        0.647284  57199.0            0.739530             0.334434             0.260470
Equalised odds difference for race
0.6773717238833519

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
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.798194        0.776327  13399.0            0.837987             0.427151             0.162013
35-44          0.694425        0.594077  17220.0            0.684136             0.268365             0.315864
45-54          0.749154        0.690376  16843.0            0.803875             0.393822             0.196125
55-64          0.683522        0.551584  13764.0            0.667808             0.279719             0.332192
65-74          0.690687        0.565576   6786.0            0.683866             0.293632             0.316134
<25            0.737449        0.744233   1474.0            0.796823             0.517986             0.203177
>74            0.732759        0.613300   2436.0            0.773524             0.337430             0.226476
Equalised odds difference for age
0.24962100025073775
"""

"""
Correlation remover did moderately well on age but terribly for race

"""