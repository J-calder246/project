import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate
from training_models.modelling import y_pred, y_test 
from sklearn.metrics import accuracy_score  #accuracy score for original model
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#Setting up for modelling with mitigation strategies



df_modelling = pd.read_csv("processed_datasets/NY2019.csv")

y = df_modelling['approved']
X = df_modelling.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", "applicant_age", "occupancy_type", 'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1'])
A = df_modelling["derived_race"]  #feature for sensitive feature
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

"""
classification report

 precision    recall  f1-score   support

       False       0.75      0.45      0.56     15885
        True       0.82      0.94      0.88     42806

    accuracy                           0.81     58691
   macro avg       0.79      0.70      0.72     58691
weighted avg       0.80      0.81      0.79     58691

"""

#Using metric frame for evaluations

sensitive_features_race=A_test



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
Pre-mitigation Results
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


_______________________________________________

Post-mitigation reslts

metrics frame by Race
                                           accuracy  selection rate    count
derived_race                                                                
2 or more minority races                   0.741007        0.568345    139.0
American Indian or Alaska Native           0.836576        0.540856    257.0
Asian                                      0.838388        0.820682   5309.0
Black or African American                  0.824137        0.738351   5129.0
Joint                                      0.865096        0.851178    934.0
Native Hawaiian or Other Pacific Islander  0.833333        0.520408    294.0
White                                      0.839263        0.819919  46629.0
Equalised odds difference for race
0.21392168876016482

Assessment:

Big improvements to accuracy for all groups especially the smallest ones
Selection rate on marginally improved and only for the largest groups.
Significant drop in equalised odds

Improvement ideas: Use models on more balanced classes to see if rates improve for the smallest groups

"""



"""
Mitigation for age
___________________
"""

y = df_modelling['approved']
X = df_modelling.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", "applicant_age", "occupancy_type", 'loan_type.1', 'approved.1', 'action_taken.1', 'loan_amount.1',
       'income.1', 'debt_to_income_ratio.1', 'loan_to_value_ratio.1',
       'interest_rate.1', 'property_value.1', 'loan_term.1', 'rate_spread.1',
       'loan_type_1.1', 'loan_type_2.1', 'loan_type_3.1', 'loan_type_4.1'])
A = df_modelling["applicant_age"]  #feature for sensitive feature
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

"""
Classification report

 precision    recall  f1-score   support

       False       0.76      0.45      0.57     15885
        True       0.82      0.95      0.88     42806

    accuracy                           0.81     58691
   macro avg       0.79      0.70      0.72     58691
weighted avg       0.81      0.81      0.80     58691

"""


#Using metric frame for evaluations

sensitive_features_age_mitigated = A_test



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

Pre-mitigation Results age
_________

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

_________

Post mitigation results

metrics frame by Race
               accuracy  selection rate    count
applicant_age                                   
35-44          0.853157        0.838340  17277.0
45-54          0.836870        0.810551  16870.0
55-64          0.829738        0.801191  13767.0
65-74          0.832014        0.798042   6947.0
<25            0.837032        0.822469   1442.0
>74            0.821189        0.740369   2388.0
Equalised odds difference for age
0.07380773790088913

Assessment
Small increases to accuracy and selection rate accross the board showing model has effectively reduced bias while maintaining accuracy
Large drop in the equalised odds aswell showing mitigation strategy has been effective
"""