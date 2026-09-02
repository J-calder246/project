"""
this section usesan in-processing mitigation algorithm with a random forest classifier will be implemented to find if this technique can make any improvements upon the initial LR model and if there are any tradeoffs in accuracy.

Exponentiated gradient (the inprocessing technique used) involes setting up 'fairness constraints for models to target and then repeating trying new classifiers that satisfy theose constraints and select the one as those with minimal error while satisfying fairness.

"""


import pandas as pd
"""evaluation packages""" 
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate,true_positive_rate
from sklearn.metrics import accuracy_score  #accuracy score for original model
from fairlearn.reductions import ExponentiatedGradient, DemographicParity  #exponentiated gradient is an inprocessing technique
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("processed_datasets/NY2019.csv")

y = df['approved']
X = df.drop(columns=["approved", "action_taken", "loan_type", 
                               "loan_purpose", "negative_amortization", "occupancy_type"
       ])
X = X.drop(columns=["interest_rate", "rate_spread"])
A = X[['applicant_age', 'derived_race']]  #feature for sensitive feature


X = X.drop(columns=["derived_race", 'applicant_age', 'derived_sex'])
#ssetting up exponentiated gradient reducer




X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#setting up RF pipeline
print("First")  #tests to see where it is at

model_pipeline = Pipeline([
        ("RF", RandomForestClassifier(
            class_weight="balanced", random_state=42, n_estimators=25, max_depth=5   #some parameters to reduce the memory it needs 
        )
        )
        ]
    )


print("Second")

#using exponentiated gradient with the pipeline

exponentiated_gradient = ExponentiatedGradient(
    estimator=model_pipeline,
    max_iter=10,
    constraints=DemographicParity(),
    sample_weight_name="RF__sample_weight" #allows for more weight for underrepresented samples

)
print("third")
exponentiated_gradient.fit(X_train, y_train, sensitive_features=A_train)

print("fourth")

y_pred = exponentiated_gradient.predict(X_test, random_state=69)

print("Classification Report Exponentiated gradient:")
print(classification_report(y_test, y_pred))


#Saving model
import joblib

joblib.dump(exponentiated_gradient, "mitigated_models/exponentiated_gradient_RF.pkl")


"""
                precision    recall  f1-score   support

       False       0.48      0.67      0.56     17755
        True       0.88      0.77      0.82     54167

    accuracy                           0.74     71922
   macro avg       0.68      0.72      0.69     71922
weighted avg       0.78      0.74      0.75     71922

"""


#Using metric frame for evaluations

sensitive_features_age_mitigated = A_test["applicant_age"]

sensitive_features_race_mitigated = A_test["derived_race"]

metrics_dict = {"accuracy":accuracy_score, "selection rate": selection_rate,
                 "count": count, "true positive rate":true_positive_rate,
                 "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate}



Age_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_age_mitigated
)


#printing difference and ratio
print("metrics frame age")
print(Age_MF_mitigated.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_age_mitigated
))


race_MF_mitigated = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test  , #original approval values
    y_pred=y_pred,   #models prediction of approval
    sensitive_features=sensitive_features_race_mitigated
)


#printing difference and ratio
print("metrics frame race")
print(race_MF_mitigated.by_group)


print("Equalised odds difference for age")

print(equalized_odds_difference(
    y_true=y_test  , #original approval values
        y_pred=y_pred,   #models prediction of approval
        sensitive_features=sensitive_features_race_mitigated
))


"""
Results

metrics frame age
               accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
applicant_age                                                                                                 
25-34          0.799313        0.771326  13399.0            0.835704             0.406763             0.164296
35-44          0.758653        0.675029  17220.0            0.776798             0.306971             0.223202
45-54          0.734430        0.635872  16843.0            0.756013             0.321965             0.243987
55-64          0.716362        0.597355  13764.0            0.723916             0.301310             0.276084
65-74          0.694371        0.578397   6786.0            0.695707             0.308702             0.304293
<25            0.639077        0.633650   1474.0            0.668060             0.485612             0.331940
>74            0.680624        0.646552   2436.0            0.758598             0.453631             0.241402
Equalised odds difference for age
0.18430146712324463
metrics frame race
                                           accuracy  selection rate    count  true positive rate  false_positive_rate  false_negative_rate
derived_race                                                                                                                              
2 or more minority races                   0.597403        0.551948    154.0            0.682540             0.461538             0.317460
American Indian or Alaska Native           0.580205        0.607509    293.0            0.713178             0.524390             0.286822
Asian                                      0.789173        0.675763   6816.0            0.817599             0.288451             0.182401
Black or African American                  0.738983        0.645763   5900.0            0.806425             0.374772             0.193575
Joint                                      0.624693        0.596893   1223.0            0.639835             0.433071             0.360165
Native Hawaiian or Other Pacific Islander  0.578635        0.560831    337.0            0.669065             0.484848             0.330935
White                                      0.740502        0.659574  57199.0            0.759021             0.322303             0.240979
Equalised odds difference for age
0.23593923131349542

"""