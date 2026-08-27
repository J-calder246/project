
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

       False       0.57      0.47      0.51     15885
        True       0.81      0.87      0.84     42806

    accuracy                           0.76     58691
   macro avg       0.69      0.67      0.68     58691
weighted avg       0.75      0.76      0.75     58691

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