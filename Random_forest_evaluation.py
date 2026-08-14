import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate, true_positive_rate, true_negative_rate

from sklearn.metrics import accuracy_score  #accuracy score for original model


df = pd.read_csv("modelled_datasets/RF_test_Data.csv")

#Setting up 'MetricFrame'  https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html 

metrics_dict = {"accuracy": accuracy_score,
                 "selection rate": selection_rate, "count": count, "true_positive_rate": true_positive_rate, "false_negative_rate": false_negative_rate,
                 "true_negative_rate": false_negative_rate
}


"""
Fairness metrics for gender
"""


#file imports dataset with data for the true value, predicted value and the evaulation values (gender age race) for those predictions
y_true=df["approved"]  , #original approval values
y_pred=df["y_pred"],   #models prediction of approval
sensitive_features_gender=df["derived_sex"]


gender_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_true  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_gender
)


#printing difference and ratio
print("metrics frame by gender")
print(gender_MF.by_group)

print("Equalised odds difference for gender")

print(equalized_odds_difference(
    y_true=y_true  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_gender
))




"""
Evaluating race
"""
sensitive_features_race=df["derived_race"]


Race_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_true  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_race
)


#printing difference and ratio
print("metrics frame by Race")
print(Race_MF.by_group)


print("Equalised odds difference for race")

print(equalized_odds_difference(
    y_true=y_true  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_race
))


"""
Evaluating age
______________
"""

sensitive_features_age=df["applicant_age"]


age_MF = MetricFrame(
    metrics=metrics_dict,
    y_true=y_true  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_age
)


#printing difference and ratio
print("metrics frame by age")
print(age_MF.by_group)

print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_true  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age
))
