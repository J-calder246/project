"""
Evaluation First Layer
________________________
This module evlauate the bias propensity of the first model (model focussing on application not delinqunecy)
"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate
from training_models.modelling import y_pred, y_test 
from sklearn.metrics import accuracy_score  #accuracy score for original model


df = pd.read_csv("modelled_datasets/LR_test_data.csv")

#Setting up 'MetricFrame'  https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html 

metrics_dict = {"accuracy":selection_rate, "selection rate": accuracy_score, "count": count}


"""
Fairness metrics for gender
"""

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
Results
_______
metrics frame by gender
             accuracy  selection rate    count
derived_sex                                   
Female       0.577671        0.731771  14510.0
Joint        0.736025        0.780736  23059.0
Male         0.567134        0.722659  21122.0
Equalised odds difference for gender
0.14780810866746355
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
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.194175        0.582524    103.0
American Indian or Alaska Native           0.161290        0.689516    248.0
Asian                                      0.615758        0.773152   5356.0
Black or African American                  0.400665        0.642983   5109.0
Joint                                      0.708989        0.767416    890.0
Native Hawaiian or Other Pacific Islander  0.116438        0.767123    292.0
White                                      0.669544        0.756452  46693.0
Equalised odds difference for race
0.5110852110852111
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

print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_true  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age
))


"""
Results
_______

metrics frame by age
               accuracy  selection rate    count
applicant_age                                   
35-44          0.738452        0.792233  16995.0
45-54          0.625667        0.737815  17049.0
55-64          0.568465        0.728677  13788.0
65-74          0.562900        0.711158   7035.0
<25            0.867238        0.811563   1401.0
>74            0.455221        0.683038   2423.0

Equalised odds difference for age
0.48032064284599785
"""

"""
Mitigation section

Fairlearn contains 5 mitigation strategies suitable for classification problems

Exponentiated Gradient
Gridsearch
Threshold optimiser
correlation remover (removes correlations that can come up between sensitive and non-sensitive features)
Adversarial fairness classifier


"""
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#Setting up for modelling with mitigation strategies


df_modelling = pd.read_csv("processed_datasets/NY2019.csv")



"""
Attempting mitigation with three protected attributes at a time


"""

y = df_modelling['approved']
X = df_modelling.drop(columns=["approved", "action_taken", "loan_type", "loan_purpose", "negative_amortization", "occupancy_type", 'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1'])
A = X[['applicant_age', 'derived_race']]  #feature for sensitive feature


X = X.drop(columns=["derived_race", 'applicant_age', 'derived_sex'])
#ssetting up exponentiated gradient reducer




X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#setting up logreg pipeline

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            solver="lbfgs", #improves model's efficiency when memory is low
            random_state=69
        )
        )
        ]
    )

#using exponentiated gradient with the pipeline

exponentiated_gradient = ExponentiatedGradient(
    estimator=model_pipeline,
    constraints=DemographicParity(),
    sample_weight_name="logreg__sample_weight" #allows for more weight for underrepresented samples

)

exponentiated_gradient.fit(X_train, y_train, sensitive_features=A_train)

y_pred = exponentiated_gradient.predict(X_test, random_state=69)

print("Classification Report:")
print(classification_report(y_test, y_pred))


#Saving model
import joblib

joblib.dump(exponentiated_gradient, "mitigated_models/exponentiated_gradient_LR.pkl")

exponentiated_gradient_LR = joblib.load("mitigated_models/exponentiated_gradient_LR.pkl")
"""
precision    recall  f1-score   support

       False       0.57      0.47      0.51     15885
        True       0.81      0.87      0.84     42806

    accuracy                           0.76     58691
   macro avg       0.69      0.67      0.68     58691
weighted avg       0.75      0.76      0.75     58691

"""


#Using metric frame for evaluations

sensitive_features_age_mitigated = A_test["applicant_age"]



Age_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_age_mitigated
)


#printing difference and ratio
print("metrics frame by Race")
print(Age_MF_mitigated.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age_mitigated
))

"""
Original results

metrics frame by age
               accuracy  selection rate    count
applicant_age                                   
35-44          0.738452        0.792233  16995.0
45-54          0.625667        0.737815  17049.0
55-64          0.568465        0.728677  13788.0
65-74          0.562900        0.711158   7035.0
<25            0.867238        0.811563   1401.0
>74            0.455221        0.683038   2423.0
Equalised odds difference for age
0.48032064284599785

_________________

Mitigated results


metrics frame by Race
               accuracy  selection rate    count
applicant_age                                   
35-44          0.777334        0.776639  17277.0
45-54          0.775459        0.761114  16870.0
55-64          0.776277        0.755430  13767.0
65-74          0.790125        0.754426   6947.0
<25            0.767684        0.740638   1442.0
>74            0.783920        0.687186   2388.0
Equalised odds difference for age
0.12717942834955576

"""

"""
Race
_____
"""

sensitive_features_race=A_test["derived_race"]



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
Original results

metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.194175        0.582524    103.0
American Indian or Alaska Native           0.161290        0.689516    248.0
Asian                                      0.615758        0.773152   5356.0
Black or African American                  0.400665        0.642983   5109.0
Joint                                      0.708989        0.767416    890.0
Native Hawaiian or Other Pacific Islander  0.116438        0.767123    292.0
White                                      0.669544        0.756452  46693.0
Equalised odds difference for race
0.5110852110852111


_________________

Mitigated results

metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.733813        0.431655    139.0
American Indian or Alaska Native           0.766537        0.455253    257.0
Asian                                      0.785270        0.767565   5309.0
Black or African American                  0.781244        0.710275   5129.0
Joint                                      0.784797        0.683084    934.0
Native Hawaiian or Other Pacific Islander  0.792517        0.493197    294.0
White                                      0.776898        0.770551  46629.0
Equalised odds difference for race
0.24374769651977046
"""


