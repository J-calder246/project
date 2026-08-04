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
X = df.drop(columns=["approved", "action_taken",  'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1', 'loan_purpose', 'occupancy_type', 'derived_race', 'derived_sex',
       'applicant_age', 'negative_amortization'
       ])

imputer = SimpleImputer(strategy= "median")

X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
"""
Setting target by race

"""
A = df['derived_race']   #setting race as the sensitive variable in this case

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42)

#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser = ThresholdOptimizer(estimator=LogisticRegression(),
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



accuracy = accuracy_score(y_test, y_pred)
classification_report_race = classification_report(y_test, y_pred)

print("accuracy score: ", accuracy)
print("classification report: ", classification_report_race)


print("Fairness after prediction with threshold optimiser:")
print(json.dumps(threshold_rules_by_group, indent=4, default=str))



"""
ORIGINAL
----------




OUTPUT   *****************FIND OUT WHAT THIS SHIT MEANS********************
----------
accuracy score:  0.6564720314869401
classification report:                precision    recall  f1-score   support

       False       0.41      0.64      0.50     15885
        True       0.83      0.66      0.74     42806

    accuracy                           0.66     58691
   macro avg       0.62      0.65      0.62     58691
weighted avg       0.72      0.66      0.67     58691

Fairness after prediction with threshold optimiser:
{
    "2 or more minority races": {
        "p0": 0.6606249999999997,
        "operation0": "[>0.7513811048416068]",
        "p1": 0.3393750000000003,
        "operation1": "[>0.6881073478793891]"
    },
    "American Indian or Alaska Native": {
        "p0": 0.9051904761904764,
        "operation0": "[>0.7173910626572055]",
        "p1": 0.09480952380952357,
        "operation1": "[>0.6907645379499201]"
    },
    "Asian": {
        "p0": 0.3259548611111108,
        "operation0": "[>0.7745687430130659]",
        "p1": 0.6740451388888892,
        "operation1": "[>0.7692113650740764]"
    },
    "Black or African American": {
        "p0": 0.9648384146341444,
        "operation0": "[>0.7547605707306595]",
        "p1": 0.03516158536585556,
        "operation1": "[>0.7478000914771941]"
    },
    "Joint": {
        "p0": 0.6944436619718303,
        "operation0": "[>0.7720165795535647]",
        "p1": 0.3055563380281697,
        "operation1": "[>0.7544940237319475]"
    },
    "Native Hawaiian or Other Pacific Islander": {
        "p0": 0.358230769230769,
        "operation0": "[>0.7463560763326227]",
        "p1": 0.641769230769231,
        "operation1": "[>0.7136934990691183]"
    },
    "White": {
        "p0": 0.8784310544611805,
        "operation0": "[>0.7270168059900999]",
        "p1": 0.12156894553881947,
        "operation1": "[>0.7244996522109624]"
    }
}
"""

"""
Same process for age
------------------
"""

B = df['applicant_age']

X_train, X_test, y_train, y_test_age, B_train, B_test = train_test_split(X, y, B, test_size=0.2, random_state=42)

#setting up threshold optimiser (true positive rate)  (can be used to improve disparate impact score with this method)

thresholded_optimiser_age = ThresholdOptimizer(estimator=LogisticRegression(max_iter=100, solver='lbfgs'),
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



accuracy = accuracy_score(y_test_age, y_pred_age)
classification_report_age = classification_report(y_test_age, y_pred_age)

print("accuracy score: ", accuracy)
print("classification report: ", classification_report_age)


print("Fairness after prediction with threshold optimiser:")
print(json.dumps(threshold_rules_by_age_group, indent=4, default=str))

"""
Results for age
----------------------

accuracy score:  0.658192908623128
classification report:                precision    recall  f1-score   support

       False       0.41      0.62      0.49     15885
        True       0.83      0.67      0.74     42806

    accuracy                           0.66     58691
   macro avg       0.62      0.64      0.62     58691
weighted avg       0.71      0.66      0.67     58691

Fairness after prediction with threshold optimiser:
{
    "35-44": {
        "p0": 0.28584156570363056,
        "operation0": "[>0.7723472136068059]",
        "p1": 0.7141584342963694,
        "operation1": "[>0.7668032228309302]"
    },
    "45-54": {
        "p0": 0.5381718213058367,
        "operation0": "[>0.7268976613833137]",
        "p1": 0.46182817869416326,
        "operation1": "[>0.724418941237627]"
    },
    "55-64": {
        "p0": 0.9969776951672858,
        "operation0": "[>0.6893263006498664]",
        "p1": 0.003022304832714151,
        "operation1": "[>0.6704056195401928]"
    },
    "65-74": {
        "p0": 0.24285714285714144,
        "operation0": "[>0.6900905468485822]",
        "p1": 0.7571428571428586,
        "operation1": "[>0.6694077363772339]"
    },
    "<25": {
        "p0": 0.7011025641025633,
        "operation0": "[>0.7722657563394972]",
        "p1": 0.29889743589743667,
        "operation1": "[>0.7595896293750831]"
    },
    ">74": {
        "p0": 0.44445508982035786,
        "operation0": "[>0.6791877137107909]",
        "p1": 0.5555449101796421,
        "operation1": "[>0.6640249271350256]"
    }

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


metrics_dict = {"accuracy": accuracy_score,
                "true_positive_rate": true_positive_rate,  #getting true positive as that was the target or the algorithm
                "selection_rate": selection_rate,
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

"""
Original model
----------------

metrics frame by Race
                                           accuracy  selection rate    count  true_positive_rate
derived_race
2 or more minority races                   0.194175        0.582524    103.0            0.290909
American Indian or Alaska Native           0.161290        0.689516    248.0            0.313131
Asian                                      0.615758        0.773152   5356.0            0.776333
Black or African American                  0.400665        0.642983   5109.0            0.535318
Joint                                      0.708989        0.767416    890.0            0.801994
Native Hawaiian or Other Pacific Islander  0.116438        0.767123    292.0            0.322917
White                                      0.669544        0.756452  46693.0            0.783857
Equalised odds difference for race
0.5110852110852111


Results
--------

metrics frame by Race
                                           accuracy  true_positive_rate  selection_rate
derived_race                                                                           
2 or more minority races                   0.690647            0.600000        0.388489
American Indian or Alaska Native           0.719844            0.712871        0.447471
Asian                                      0.672443            0.651679        0.543605
Black or African American                  0.681809            0.672152        0.526223
Joint                                      0.685225            0.691563        0.611349
Native Hawaiian or Other Pacific Islander  0.748299            0.650943        0.360544
White                                      0.650261            0.660399        0.590126
Equalised odds difference for race
0.18323940175047695


Low selection rate compared to other algorithms due to it targetting true positives

"""