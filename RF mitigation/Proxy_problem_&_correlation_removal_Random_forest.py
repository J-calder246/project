"""
This module will attempt to mitigate bias of the initial RF model by first augmenting the data before it is used in modelling.

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
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race
2 or more minority races                   0.824675        0.480519    154.0            0.873016             0.208791             0.126984
American Indian or Alaska Native           0.788396        0.481229    293.0            0.806202             0.225610             0.193798
Asian                                      0.854754        0.756602   6816.0            0.917619             0.316913             0.082381
Black or African American                  0.800339        0.674576   5900.0            0.878240             0.331056             0.121760
Joint                                      0.858545        0.796402   1223.0            0.913313             0.350394             0.086687
Native Hawaiian or Other Pacific Islander  0.833828        0.471810    337.0            0.870504             0.191919             0.129496
White                                      0.830749        0.765695  57199.0            0.886155             0.357159             0.113845
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
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.878200        0.857079  13399.0            0.932561             0.429637             0.067439
35-44          0.847967        0.791754  17220.0            0.908302             0.370241             0.091698
45-54          0.820281        0.738823  16843.0            0.886545             0.352853             0.113455
55-64          0.806888        0.690933  13764.0            0.855320             0.306405             0.144680
65-74          0.787798        0.664162   6786.0            0.824276             0.296062             0.175724
<25            0.844640        0.837856   1474.0            0.920569             0.482014             0.079431
>74            0.768883        0.602627   2436.0            0.793640             0.273743             0.206360
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


Classification Report:
              precision    recall  f1-score   support

       False       0.65      0.65      0.65     17755
        True       0.89      0.88      0.88     54167

    accuracy                           0.83     71922
   macro avg       0.77      0.77      0.77     71922
weighted avg       0.83      0.83      0.83     71922


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
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.714286        0.318182    154.0            0.539683             0.164835             0.460317
American Indian or Alaska Native           0.720137        0.331058    293.0            0.558140             0.152439             0.441860
Asian                                      0.835094        0.721097   6816.0            0.879936             0.287356             0.120064
Black or African American                  0.760508        0.585593   5900.0            0.775648             0.265027             0.224352
Joint                                      0.830744        0.776778   1223.0            0.883385             0.370079             0.116615
Native Hawaiian or Other Pacific Islander  0.747774        0.267062    337.0            0.517986             0.090909             0.482014
White                                      0.832987        0.778772  57199.0            0.896070             0.380960             0.103930
Equalised odds difference for race
0.3780844749654163
metrics frame by age


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

               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.880439        0.864691  13399.0            0.938356             0.447539             0.061644
35-44          0.839954        0.783508  17220.0            0.897924             0.369705             0.102076
45-54          0.809654        0.722140  16843.0            0.867663             0.341913             0.132337
55-64          0.801293        0.690279  13764.0            0.850861             0.314653             0.149139
65-74          0.793693        0.682729   6786.0            0.841827             0.316966             0.158173
<25            0.846676        0.849389   1474.0            0.928930             0.507194             0.071070
>74            0.760673        0.605090   2436.0            0.789098             0.288268             0.210902
Equalised odds difference for age
0.21892608817973552
"""