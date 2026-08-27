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

thresholded_optimiser = ThresholdOptimizer(estimator=LogisticRegression(max_iter=500, solver='lbfgs'),
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

joblib.dump(thresholded_optimiser, "mitigated_models/threshold_optimiser_race_LR.pkl")

thresholded_optimiser= joblib.load("mitigated_models/threshold_optimiser_race_LR.pkl")

accuracy = accuracy_score(y_test, y_pred)
classification_report_race = classification_report(y_test, y_pred)

print("accuracy score race: ", accuracy)
print("classification report: ", classification_report_race)


print("Fairness after prediction with threshold optimiser:")
print(json.dumps(threshold_rules_by_group, indent=4, default=str))



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

Fairness after prediction with threshold optimiser:
{
    "2 or more minority races": {
        "p0": 0.30258064516129,
        "operation0": "[>0.7454059943997791]",
        "p1": 0.69741935483871,
        "operation1": "[>0.7039550180191518]"
    },
    "American Indian or Alaska Native": {
        "p0": 0.817981132075471,
        "operation0": "[>0.7001817461200499]",
        "p1": 0.18201886792452904,
        "operation1": "[>0.6206982427132273]"
    },
    "Asian": {
        "p0": 0.6968201754385949,
        "operation0": "[>0.7616763762999366]",
        "p1": 0.30317982456140513,
        "operation1": "[>0.7569031310146412]"
    },
    "Black or African American": {
        "p0": 0.18742724458204277,
        "operation0": "[>0.7442102201114098]",
        "p1": 0.8125727554179573,
        "operation1": "[>0.7379503502101671]"
    },
    "Joint": {
        "p0": 0.7488545454545451,
        "operation0": "[>0.7502663522790776]",
        "p1": 0.2511454545454549,
        "operation1": "[>0.7424226814423645]"
    },
    "Native Hawaiian or Other Pacific Islander": {
        "p0": 0.6618113207547166,
        "operation0": "[>0.7184808674461409]",
        "p1": 0.3381886792452834,
        "operation1": "[>0.681123060155343]"
    },
    "White": {
        "p0": 0.016409003831414415,
        "operation0": "[>0.7069509623702396]",
        "p1": 0.9835909961685856,
        "operation1": "[>0.7008896300472085]"
    }
}
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
thresholded_optimiser_age.fit(X_train, y_train, sensitive_features=B_train)  #sets up the model to post-process the predictions in favour of race (A_train)

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

Fairness after prediction with threshold optimiser:
{
    "35-44": {
        "p0": 0.1997153439153421,
        "operation0": "[>0.7570948254612533]",
        "p1": 0.8002846560846579,
        "operation1": "[>0.7523378429667906]"
    },
    "45-54": {
        "p0": 0.0034795221843005363,
        "operation0": "[>0.7304004127271055]",
        "p1": 0.9965204778156994,
        "operation1": "[>0.7122080519385579]"
    },
    "55-64": {
        "p0": 0.1768053571428525,
        "operation0": "[>0.6899955308946297]",
        "p1": 0.8231946428571475,
        "operation1": "[>0.6837880730476227]"
    },
    "65-74": {
        "p0": 0.8934579439252237,
        "operation0": "[>0.6765173815767052]",
        "p1": 0.10654205607477629,
        "operation1": "[>0.6738983589896522]"
    },
    "<25": {
        "p0": 0.12226460481099664,
        "operation0": "[>0.7614373737302991]",
        "p1": 0.8777353951890033,
        "operation1": "[>0.7411145749802928]"
    },
    ">74": {
        "p0": 0.9667027027026907,
        "operation0": "[>0.6746973818626758]",
        "p1": 0.03329729729730935,
        "operation1": "[>0.6714295052488348]"
    }
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
2 or more minority races                   0.676259            0.581818        0.388489
American Indian or Alaska Native           0.696498            0.673267        0.439689
Asian                                      0.676022            0.655382        0.545300
Black or African American                  0.690388            0.679897        0.527003
Joint                                      0.679872            0.681881        0.601713
Native Hawaiian or Other Pacific Islander  0.741497            0.679245        0.387755
White                                      0.654421            0.668584        0.598233
Equalised odds difference for race
0.16451497808263066


Low selection rate compared to other algorithms due to it targetting true positives

"""


#metric frame for age

sensitive_features_age_mitigated = df['applicant_age'].iloc[y_test.index]  #sets sensitive feature to have same shape as the test data


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

"""
metrics frame by age
               accuracy  true_positive_rate  selection_rate  false_positive_rate  false_negative_rate
applicant_age                                                                                        
25-34          0.687215            0.695656        0.646523             0.361713             0.304344
35-44          0.677697            0.685228        0.613408             0.349933             0.314772
45-54          0.660347            0.681505        0.601077             0.394101             0.318495
55-64          0.655682            0.685832        0.602340             0.412773             0.314168
65-74          0.656554            0.687165        0.606764             0.416055             0.312835
<25            0.687151            0.705686        0.656425             0.406780             0.294314
>74            0.663252            0.681486        0.570209             0.369617             0.318514
Equalised odds difference for age
0.06612175574544876

"""

"""
Exponentiated gradient on all three
____________________________________

"""


