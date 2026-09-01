"""
File for various post-processing techniques to remove correlation between features and sensitive attributes.

Fairlearn: Threshold optimiser
-------------------------------
A technique that adjusts the decisions exactly according to preset fairness criteria. 
This technique is simple and effective for reducing bias, however it can cause significantly lower accuracy from models.
Guide and description: https://fairlearn.org/main/user_guide/mitigation/postprocessing.html 
Fairness constraints can be used to target true positive/negative rate, false positive/negative rate, demographic parity or equalised odds.
Objectives include: accuracy, balanced accuracy score, selection rate and true positive/negative rate
"""

from sklearn.impute import SimpleImputer
import json #for formatting output
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("processed_datasets/NY2019.csv")

y = df['approved']
X = df.drop(columns=["approved", "action_taken", 'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization'
       ])
X = X.drop(columns=["interest_rate", "rate_spread"])

imputer = SimpleImputer(strategy= "median")




"""
Setting target by race

"""
A = df['derived_race']   #setting race as the sensitive variable in this case


#imputing missing values to median

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)



X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42, stratify=y)



#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser = ThresholdOptimizer(estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100), #setting amount of estimators so it doesn't use so much memory
                                           constraints="true_positive_rate_parity",
                                           objective="balanced_accuracy_score",  #specifies a accuracy prioity from the model
                                           predict_method='predict_proba',
                                           prefit=False
)

#fitting data to threshold optimiser
thresholded_optimiser.fit(X_train, y_train, sensitive_features=A_train)  #sets up the model to post-process the predictions in favour of race (A_train)

#making predicitons
y_pred =thresholded_optimiser.predict(X_test, sensitive_features=A_test, random_state=42)  
threshold_rules_by_group = thresholded_optimiser.interpolated_thresholder_.interpolation_dict    #sets up the threshold target for each group

#Saving model
import joblib

joblib.dump(thresholded_optimiser, "mitigated_models/threshold_optimiser_race_RF.pkl")



accuracy = accuracy_score(y_test, y_pred)
classification_report_race = classification_report(y_test, y_pred)

print("accuracy score race: ", accuracy)
print("classification report: ", classification_report_race)






"""
Same process for age
------------------
"""

B = df['applicant_age']

X_train, X_test, y_train, y_test_age, B_train, B_test = train_test_split(X, y, B, test_size=0.2, random_state=42, stratify=y)

#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser_age = ThresholdOptimizer(estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100, max_depth=5),
                                           constraints="true_positive_rate_parity",
                                           objective="balanced_accuracy_score",  #specifies a accuracy prioity from the model
                                           predict_method='predict_proba',
                                           prefit=False
)

#fitting data to threshold optimiser
thresholded_optimiser_age.fit(X_train, y_train, sensitive_features=B_train)  #sets up the model to post-process the predictions in favour of race (A_train)

#making predicitons
y_pred_age =thresholded_optimiser_age.predict(X_test, sensitive_features=B_test, random_state=42)  
threshold_rules_by_age_group = thresholded_optimiser_age.interpolated_thresholder_.interpolation_dict    #sets up the threshold target for each group



joblib.dump(thresholded_optimiser_age, "mitigated_models/thresholded_optimiser_age_RF.pkl")


accuracy = accuracy_score(y_test_age, y_pred_age)
classification_report_age = classification_report(y_test_age, y_pred_age)

print("accuracy score age: ", accuracy)
print("classification report: ", classification_report_age)


print("Fairness after prediction with threshold optimiser:")
print(json.dumps(threshold_rules_by_age_group, indent=4, default=str))

"""
Results for age
----------------------
accuracy score age:  0.6505085958664872
classification report:                precision    recall  f1-score   support

       False       0.41      0.64      0.50     15885
        True       0.83      0.66      0.73     42806

    accuracy                           0.65     58691
   macro avg       0.62      0.65      0.61     58691
weighted avg       0.71      0.65      0.67     58691


"""


"""
-------------
Metric frame evaluation
--------------


"""

import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, true_positive_rate, selection_rate, true_negative_rate, false_positive_rate 
from sklearn.metrics import accuracy_score  #accuracy score for original model


sensitive_features_race_mitigated = df['derived_race'].iloc[y_test.index]  #sets sensitive feature to have same shape as the test data
sensitive_features_age_mitigated = df['applicant_age'].iloc[y_test.index]  #sets sensitive feature to have same shape as the test data

metrics_dict = {"accuracy": accuracy_score,
                "true_positive_rate": true_positive_rate,  #getting true positive as that was the target or the algorithm
                "selection_rate": selection_rate, "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate
                }



Race_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values for race
    y_pred=y_pred,   #models prediction of approval for race
    sensitive_features=sensitive_features_race_mitigated
)


#printing difference and ratio
print("metrics frame by Race")
print(Race_MF_mitigated.by_group)


print("Equalised odds difference for race")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_race_mitigated
))


age_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values for race
    y_pred=y_pred,   #models prediction of approval for race
    sensitive_features=sensitive_features_age_mitigated
)


#printing difference and ratio
print("metrics frame by age")
print(age_MF_mitigated.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age_mitigated
))

"""
Results

metrics frame by Race
                                           accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                     
2 or more minority races                   0.798701            0.952381        0.571429             0.307692             0.047619
American Indian or Alaska Native           0.781570            0.798450        0.481229             0.231707             0.201550
Asian                                      0.864877            0.945079        0.786678             0.354132             0.054921
Black or African American                  0.793220            0.888769        0.694915             0.367942             0.111231
Joint                                      0.860180            0.924665        0.812756             0.385827             0.075335
Native Hawaiian or Other Pacific Islander  0.836795            0.906475        0.498516             0.212121             0.093525
White                                      0.854403            0.943564        0.830714             0.447985             0.056436
Equalised odds difference for race
0.23586343279241553
metrics frame by age
               accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
applicant_age                                                                                        
25-34          0.896410            0.968124        0.899321             0.509697             0.031876
35-44          0.863705            0.947443        0.837340             0.439142             0.052557
45-54          0.834293            0.932518        0.791308             0.422351             0.067482
55-64          0.826649            0.915993        0.756176             0.382339             0.084007
65-74          0.826260            0.919645        0.758621             0.388430             0.080355
<25            0.870421            0.972408        0.896201             0.568345             0.027592
>74            0.791461            0.885140        0.695813             0.369832             0.114860
Equalised odds difference for age
0.19851292150637034
metrics frame by age
"""


#metric frame for age




metrics_dict = {"accuracy": accuracy_score,
                "true_positive_rate": true_positive_rate,  #getting true positive as that was the target or the algorithm
                "selection_rate": selection_rate, "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate
                }



age_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test_age  , #original approval values for race
    y_pred=y_pred_age,   #models prediction of approval for race
    sensitive_features=sensitive_features_age_mitigated
)


#printing difference and ratio
print("metrics frame by age")
print(age_MF_mitigated.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test_age  , #original approval values
        y_pred=y_pred_age,   #models prediction of approval
        sensitive_features=sensitive_features_age_mitigated
))



Race_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test_age  , #original approval values for race
    y_pred=y_pred_age,   #models prediction of approval for race
    sensitive_features=sensitive_features_race_mitigated
)


#printing difference and ratio
print("metrics frame by Race")
print(Race_MF_mitigated.by_group)


print("Equalised odds difference for race")

print(equalized_odds_difference(
    y_true=y_test_age  , #original approval values
        y_pred=y_pred_age,   #models prediction of approval
        sensitive_features=sensitive_features_race_mitigated
))

"""
Results

metrics frame by age
               accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
applicant_age                                                                                        
25-34          0.807001            0.829382        0.752892             0.319741             0.170618
35-44          0.801858            0.831875        0.718118             0.306702             0.168125
45-54          0.788755            0.842295        0.706347             0.351137             0.157705
55-64          0.785237            0.834785        0.683813             0.330665             0.165215
65-74          0.786030            0.836329        0.682729             0.329606             0.163671
<25            0.782904            0.834448        0.759837             0.438849             0.165552
>74            0.769294            0.843608        0.665435             0.358659             0.156392
Equalised odds difference for age
0.13214650799467664
metrics frame by Race
                                           accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                     
2 or more minority races                   0.720779            0.650794        0.402597             0.230769             0.349206
American Indian or Alaska Native           0.733788            0.604651        0.358362             0.164634             0.395349
Asian                                      0.822477            0.859491        0.703785             0.278599             0.140509
Black or African American                  0.753898            0.769438        0.584407             0.272313             0.230562
Joint                                      0.807850            0.852425        0.750613             0.362205             0.147575
Native Hawaiian or Other Pacific Islander  0.783383            0.633094        0.326409             0.111111             0.366906
White                                      0.794489            0.838910        0.728981             0.356161             0.161090
Equalised odds difference for race
0.2548397171451613

"""