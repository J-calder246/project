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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("processed_datasets/NY2019.csv")



y = df['approved']
X = df.drop(columns=["approved", "action_taken", 'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization', 'loan_type'
       ])
X = X.drop(columns=["interest_rate", "rate_spread"])  #dropping these columns to prevent data leakage, as unapproved candidates have no interest rate or rate spread set in the dataset

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




"""
classification report:                precision    recall  f1-score   support

       False       0.50      0.62      0.55     17755
        True       0.87      0.80      0.83     54167

    accuracy                           0.75     71922
   macro avg       0.68      0.71      0.69     71922
weighted avg       0.78      0.75      0.76     71922

#technique has maintained accuracy but not improved upon like the other models
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
metrics frame by Race
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.655844        0.467532    154.0            0.650794             0.340659             0.349206
American Indian or Alaska Native           0.716724        0.518771    293.0            0.767442             0.323171             0.232558
Asian                                      0.770393        0.650235   6816.0            0.787332             0.275862             0.212668
Black or African American                  0.733051        0.637458   5900.0            0.795086             0.371585             0.204914
Joint                                      0.756337        0.715454   1223.0            0.797730             0.401575             0.202270
Native Hawaiian or Other Pacific Islander  0.780415        0.507418    337.0            0.848921             0.267677             0.151079
White                                      0.753265        0.702093  57199.0            0.794811             0.387639             0.205189
Equalised odds difference for race
0.19812721251570176
metrics frame by age
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.858049        0.891708  13399.0            0.941078             0.612133             0.058922
35-44          0.786353        0.755691  17220.0            0.845960             0.429223             0.154040
45-54          0.719646        0.638307  16843.0            0.747476             0.353067             0.252524
55-64          0.696455        0.567858  13764.0            0.688654             0.285298             0.311346
65-74          0.680961        0.544356   6786.0            0.661662             0.274672             0.338338
<25            0.830393        0.888738   1474.0            0.943144             0.654676             0.056856
>74            0.644910        0.447455   2436.0            0.573005             0.231285             0.426995
Equalised odds difference for age
0.42339134279168844

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
metrics frame by age
               accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
applicant_age                                                                                        
25-34          0.769311            0.796013        0.733861             0.381900             0.203987
35-44          0.760918            0.796961        0.704355             0.369437             0.203039
45-54          0.749332            0.804367        0.690910             0.394466             0.195633
55-64          0.749637            0.801597        0.672915             0.371907             0.198403
65-74          0.742706            0.790653        0.662393             0.367526             0.209347
<25            0.738806            0.790134        0.732022             0.482014             0.209866
>74            0.732348            0.796885        0.643268             0.378771             0.203115
Equalised odds difference for age
0.11448886588347212
metrics frame by Race
                                           accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                     
2 or more minority races                   0.662338            0.460317        0.305195             0.197802             0.539683
American Indian or Alaska Native           0.665529            0.434109        0.276451             0.152439             0.565891
Asian                                      0.734302            0.734817        0.609448             0.267105             0.265183
Black or African American                  0.637627            0.562635        0.441017             0.235883             0.437365
Joint                                      0.730989            0.760578        0.681930             0.381890             0.239422
Native Hawaiian or Other Pacific Islander  0.727003            0.474820        0.252226             0.095960             0.525180
White                                      0.770293            0.828927        0.737758             0.428560             0.171073
Equalised odds difference for race
0.3948179907294024

"""

"""
In both case, the group that is targetted (race or age) sees large improvements to equalised odds while the other sees little improvement


"""


