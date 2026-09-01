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
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("processed_datasets/NY2019.csv")



y = df['approved']
X = df.drop(columns=["approved", "action_taken", 'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization', 'loan_type'
       ])
X = X.drop(columns=["interest_rate", "rate_spread"])

imputer = SimpleImputer(strategy= "median")


print(X.columns.to_list)




"""
Setting target by race

"""
A = df['derived_race']   #setting race as the sensitive variable in this case


#imputing missing values to median

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)



X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42, stratify=y)



#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser = ThresholdOptimizer(estimator=LogisticRegression(max_iter=500, solver='lbfgs'),
                                           constraints="true_positive_rate_parity",
                                           objective="balanced_accuracy_score",  #specifies a accuracy prioity from the model
                                           predict_method='predict_proba',
                                           prefit=False
)

#fitting data to threshold optimiser
thresholded_optimiser_race =thresholded_optimiser.fit(X_train, y_train, sensitive_features=A_train)  #sets up the model to post-process the predictions in favour of race (A_train)

#making predicitons
y_pred =thresholded_optimiser_race.predict(X_test, sensitive_features=A_test, random_state=42)  
threshold_rules_by_group = thresholded_optimiser_race.interpolated_thresholder_.interpolation_dict    #sets up the threshold target for each group

#Saving model
import joblib

joblib.dump(thresholded_optimiser_race, "mitigated_models/threshold_optimiser_race_LR.pkl")

thresholded_optimiser_race = joblib.load("mitigated_models/threshold_optimiser_race_LR.pkl")

accuracy = accuracy_score(y_test, y_pred)
classification_report_race = classification_report(y_test, y_pred)

print("accuracy score race: ", accuracy)
print("classification report: ", classification_report_race)





"""
ORIGINAL
----------




OUTPUT   
----------
accuracy score race:  0.660595321258796
classification report:                precision    recall  f1-score   support

       False       0.42      0.64      0.50     15885
        True       0.83      0.67      0.74     42806

    accuracy                           0.66     58691
   macro avg       0.63      0.65      0.62     58691
weighted avg       0.72      0.66      0.68     58691


"""

"""
Same process for age
------------------
"""

B = df['applicant_age']

X_train, X_test, y_train, y_test_age, B_train, B_test = train_test_split(X, y, B, test_size=0.2, random_state=42, stratify=y)

#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser_age = ThresholdOptimizer(estimator=LogisticRegression(max_iter=500, solver='lbfgs'),
                                           constraints="true_positive_rate_parity",
                                           objective="balanced_accuracy_score",  #specifies a accuracy prioity from the model
                                           predict_method='predict_proba',
                                           prefit=False
)

#fitting data to threshold optimiser
thresholded_optimiser_age = thresholded_optimiser_age.fit(X_train, y_train, sensitive_features=B_train)  #sets up the model to post-process the predictions in favour of race (A_train)

#making predicitons
y_pred_age =thresholded_optimiser_age.predict(X_test, sensitive_features=B_test, random_state=42)  
threshold_rules_by_age_group = thresholded_optimiser_age.interpolated_thresholder_.interpolation_dict    #sets up the threshold target for each group



joblib.dump(thresholded_optimiser_age, "mitigated_models/thresholded_optimiser_age_LR.pkl")

thresholded_optimiser_age = joblib.load("mitigated_models/thresholded_optimiser_age_LR.pkl")

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


metrics_dict = {"accuracy":accuracy_score, "selection rate": selection_rate,
                 "count": count, "true positive rate":true_positive_rate,
                 "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate}




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
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.688312        0.435065    154.0            0.650794             0.285714             0.349206
American Indian or Alaska Native           0.730375        0.511945    293.0            0.775194             0.304878             0.224806
Asian                                      0.773621        0.653462   6816.0            0.791742             0.275862             0.208258
Black or African American                  0.734915        0.632881   5900.0            0.792927             0.362933             0.207073
Joint                                      0.751431        0.695830   1223.0            0.782250             0.366142             0.217750
Native Hawaiian or Other Pacific Islander  0.768546        0.495549    337.0            0.820144             0.267677             0.179856
White                                      0.754733        0.698142  57199.0            0.793204             0.375739             0.206796
Equalised odds difference for race
0.16935023409843553
metrics frame by age
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.858870        0.883723  13399.0            0.936863             0.582795             0.063137
35-44          0.786237        0.748606  17220.0            0.841364             0.413137             0.158636
45-54          0.720655        0.633735  16843.0            0.745013             0.342986             0.254987
55-64          0.699070        0.570764  13764.0            0.692595             0.285784             0.307405
65-74          0.684645        0.545682   6786.0            0.665257             0.270783             0.334743
<25            0.831072        0.881275   1474.0            0.938963             0.633094             0.061037
>74            0.656404        0.448276   2436.0            0.582738             0.216760             0.417262
Equalised odds difference for age
0.4163337486435432

"""



#metric frame for age



metrics_dict = {"accuracy": accuracy_score,
                "true_positive_rate": true_positive_rate,  #getting true positive as that was the target or the algorithm
                "selection_rate": selection_rate, "false_positive_rate": false_positive_rate, 
                "false_negative_rate": false_negative_rate
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
25-34          0.782148            0.814805        0.752967             0.402785             0.185195
35-44          0.772880            0.816308        0.722706             0.384182             0.183692
45-54          0.761741            0.825384        0.708900             0.404547             0.174616
55-64          0.758646            0.820266        0.690061             0.385492             0.179734
65-74          0.756263            0.812011        0.678603             0.371901             0.187989
<25            0.748304            0.813545        0.760516             0.532374             0.186455
>74            0.748358            0.826736        0.665025             0.386592             0.173264
Equalised odds difference for age
0.1604732742731435
metrics frame by Race
                                           accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                     
2 or more minority races                   0.668831            0.476190        0.311688             0.197802             0.523810
American Indian or Alaska Native           0.682594            0.488372        0.307167             0.164634             0.511628
Asian                                      0.733862            0.733414        0.607835             0.264915             0.266586
Black or African American                  0.643898            0.572624        0.447288             0.235883             0.427376
Joint                                      0.763696            0.805986        0.721177             0.397638             0.194014
Native Hawaiian or Other Pacific Islander  0.727003            0.496403        0.270030             0.111111             0.503597
White                                      0.783860            0.851587        0.759192             0.445835             0.148413
Equalised odds difference for race
0.3753964301345114
"""


