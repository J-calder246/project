"""
Script that uses fairlearns evaluation package to evaluate the fairness of the original random forest model

"""

import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate, true_positive_rate, true_negative_rate

from sklearn.metrics import accuracy_score  #accuracy score for original model


df = pd.read_csv("modelled_datasets/RF_test_Data.csv")

#Setting up 'MetricFrame'  https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html 

metrics_dict = {"accuracy": accuracy_score,
                 "selection rate": selection_rate, "count": count, "true_positive_rate": true_positive_rate, "false_negative_rate": false_negative_rate,
                 "true_negative_rate": true_negative_rate
}


"""
Fairness metrics for gender
"""


#file imports dataset with data for the true value, predicted value and the evaulation values (gender age race) for those predictions
y_true=df["approved"]  #original approval values
y_pred=df["y_pred"]   #models prediction of approval
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


"""
Results
________________________________________-

metrics frame by gender
             accuracy  selection rate    count  true_positive_rate  false_negative_rate  true_negative_rate
derived_sex                                                                                                
Female       0.824176        0.719144  17290.0            0.875940             0.124060            0.689324
Joint        0.856986        0.826357  28144.0            0.924285             0.075715            0.578650
Male         0.822976        0.719647  26488.0            0.878058             0.121942            0.682979
Equalised odds difference for gender
0.11067480199436269
metrics frame by Race
                                           accuracy  selection rate    count  true_positive_rate  false_negative_rate  true_negative_rate
derived_race                                                                                                                             
2 or more minority races                   0.740260        0.422078    154.0            0.698413             0.301587            0.769231
American Indian or Alaska Native           0.761092        0.372014    293.0            0.651163             0.348837            0.847561
Asian                                      0.847271        0.732101   6816.0            0.895771             0.104229            0.714833
Black or African American                  0.774915        0.607458   5900.0            0.804536             0.195464            0.724954
Joint                                      0.852003        0.794767   1223.0            0.908153             0.091847            0.637795
Native Hawaiian or Other Pacific Islander  0.842730        0.373887    337.0            0.762590             0.237410            0.898990
White                                      0.841938        0.785101  57199.0            0.905963             0.094037            0.624798
Equalised odds difference for race
0.2741914344985362
metrics frame by age
               accuracy  selection rate    count  true_positive_rate  false_negative_rate  true_negative_rate
applicant_age                                                                                                
25-34          0.888201        0.870513  13399.0            0.946347             0.053653            0.558926
35-44          0.849477        0.791405  17220.0            0.909044             0.090956            0.634048
45-54          0.820756        0.732530  16843.0            0.882522             0.117478            0.659374
55-64          0.811755        0.699143  13764.0            0.864655             0.135345            0.688016
65-74          0.808282        0.694960   6786.0            0.861070             0.138930            0.686923
<25            0.856852        0.860923   1474.0            0.942308             0.057692            0.489209
>74            0.777504        0.621921   2436.0            0.815704             0.184296            0.711732
Equalised odds difference for age
0.22252321048189388

"""