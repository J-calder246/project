"""
Evaluation First Layer
________________________
This section will first use fairlearn's metric frame to evaluate the initial LR model's bias using metrics such as equalised odds difference, false negative rate, selection rate, etc.

Then an in-processing mitigation algorithm will be implemented to find if this technique can make any improvements upon the initial LR model and if there are any tradeoffs in accuracy.

Exponentiated gradient (the inprocessing technique used) involes setting up 'fairness constraints for models to target and then repeating trying new classifiers that satisfy theose constraints and select the one as those with minimal error while satisfying fairness.
"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate,true_positive_rate

from sklearn.metrics import accuracy_score  #accuracy score for original model


df = pd.read_csv("modelled_datasets/LR_test_data.csv")

#Setting up 'MetricFrame'  https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html 


#Setting a dictionary with many of fairlearn's metrics
metrics_dict = {"accuracy": accuracy_score, "selection rate": selection_rate, "count": count, "true_positive_rate": true_positive_rate, "false_negative_rate": false_negative_rate, "false_positive_rate": false_positive_rate}


"""
Fairness metrics for gender
"""


#file imports dataset with data for the true value, predicted value and the evaulation values (gender age race) for those predictions
y_true=df["approved"] #original approval values
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
Results
_______
metrics frame by gender
             accuracy  selection rate    count  true_positive_rate  false_negative_rate  false_positive_rate
derived_sex                                                                                                 
Female       0.714691        0.579121  17290.0            0.703298             0.296702             0.255630
Joint        0.770040        0.735148  28144.0            0.813669             0.186331             0.410401
Male         0.707528        0.576261  26488.0            0.697722             0.302278             0.267549
Equalised odds difference for gender
0.15477176844450669

Very low Equalised odds difference with most of it coming from the joint applications and not differences between male and female applications as shown by their very similar selection rates
"""

"""
Race calculations
___________________

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
Results
_________

metrics frame by Race
                                           accuracy  selection rate    count  true_positive_rate  false_negative_rate  false_positive_rate
derived_race                                                                                                                              
2 or more minority races                   0.623377        0.175325    154.0            0.253968             0.746032             0.120879
American Indian or Alaska Native           0.617747        0.160410    293.0            0.248062             0.751938             0.091463
Asian                                      0.775381        0.642606   6816.0            0.785528             0.214472             0.252326
Black or African American                  0.621695        0.386780   5900.0            0.506749             0.493251             0.184426
Joint                                      0.767784        0.720360   1223.0            0.808050             0.191950             0.385827
Native Hawaiian or Other Pacific Islander  0.667656        0.127596    337.0            0.251799             0.748201             0.040404
White                                      0.740852        0.669715  57199.0            0.765812             0.234188             0.343800
Equalised odds difference for race
0.5599875200998392

High equalised odds ratio with the smallest groups and black applicants being worst affected
"""

"""
Age calculations
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

print("Equalised odds difference for age in original model")

print(equalized_odds_difference(
    y_true=y_true  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age
))


"""
Results
_______

metrics frame by age
               accuracy  selection rate    count  true_positive_rate  false_negative_rate  false_positive_rate
applicant_age                                                                                                 
25-34          0.848123        0.853720  13399.0            0.912891             0.087109             0.518647
35-44          0.766086        0.705575  17220.0            0.801038             0.198962             0.360322
45-54          0.698094        0.580300  16843.0            0.692472             0.307528             0.287216
55-64          0.676765        0.516202  13764.0            0.637731             0.362269             0.231926
65-74          0.661067        0.498526   6786.0            0.614506             0.385494             0.231891
<25            0.829037        0.860244   1474.0            0.924749             0.075251             0.582734
>74            0.588259        0.348112   2436.0            0.449708             0.550292             0.173184
Equalised odds difference for age in original model
0.4750411820496182
"""

