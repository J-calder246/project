from fairlearn.reductions import ExponentiatedGradient, DemographicParity  #exponentiated gradient is an inprocessing technique
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#Setting up for modelling with mitigation strategies
import pandas as pd
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate,true_positive_rate

from sklearn.metrics import accuracy_score 


df_modelling = pd.read_csv("processed_datasets/NY2019.csv")



"""
Attempting mitigation with three protected attributes at a time

This section will use Exponentiated gradient. An in-processing technique which creates many classifier subjected to set fairness constraints, 
it then chooses the one with the least error and selects that one


"""

y = df_modelling['approved']
X = df_modelling.drop(columns=["approved", "action_taken", "loan_type", 
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
    constraints=DemographicParity(),  #targets equal selection rate by demographic
    sample_weight_name="logreg__sample_weight" #allows for more weight for underrepresented samples

)

exponentiated_gradient.fit(X_train, y_train, sensitive_features=A_train)

y_pred = exponentiated_gradient.predict(X_test, random_state=69)

print("Classification Report Exponentiated gradient:")
print(classification_report(y_test, y_pred))


#Saving model
import joblib

joblib.dump(exponentiated_gradient, "mitigated_models/exponentiated_gradient_LR.pkl")

"""
precision    recall  f1-score   support

       False       0.54      0.41      0.47     17726
        True       0.82      0.89      0.85     54196

    accuracy                           0.77     71922
   macro avg       0.68      0.65      0.66     71922
weighted avg       0.75      0.77      0.76     71922

"""


#Using metric frame for evaluations

sensitive_features_age_mitigated = A_test["applicant_age"]

#Setting a dictionary with many of fairlearn's metrics
metrics_dict = {"accuracy": accuracy_score, "selection rate": selection_rate, "count": count, "true_positive_rate": true_positive_rate, "false_negative_rate": false_negative_rate, "false_positive_rate": false_positive_rate}


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
_________________

Mitigated results
____________________


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
 results of exponentiated gradient


metrics frame by age
               accuracy  selection rate    count  true_positive_rate  false_negative_rate  false_positive_rate
applicant_age                                                                                                 
25-34          0.788417        0.802075  13399.0            0.847383             0.152617             0.545500
35-44          0.787921        0.808130  17220.0            0.880430             0.119570             0.546649
45-54          0.752538        0.805142  16843.0            0.885559             0.114441             0.595024
55-64          0.745278        0.805870  13764.0            0.893383             0.106617             0.601164
65-74          0.730622        0.812408   6786.0            0.889617             0.110383             0.634905
<25            0.729986        0.816825   1474.0            0.836957             0.163043             0.730216
>74            0.690476        0.824713   2436.0            0.907203             0.092797             0.682682
Equalised odds difference for age
0.18471607597065065


metrics frame by Race
                                           accuracy  selection rate    count  true_positive_rate  false_negative_rate  false_positive_rate
derived_race                                                                                                                              
2 or more minority races                   0.441558        0.746753    154.0            0.730159             0.269841             0.758242
American Indian or Alaska Native           0.511945        0.771331    293.0            0.821705             0.178295             0.731707
Asian                                      0.764525        0.807218   6816.0            0.890559             0.109441             0.579639
Black or African American                  0.705254        0.803559   5900.0            0.905238             0.094762             0.632058
Joint                                      0.704824        0.809485   1223.0            0.824561             0.175439             0.751969
Native Hawaiian or Other Pacific Islander  0.465875        0.786350    337.0            0.805755             0.194245             0.772727
White                                      0.772251        0.807759  57199.0            0.875515             0.124485             0.577965
Equalised odds difference for race
0.1947618216716105

Large improvements to equalised odds for both age and race.
Models also have very low false negative rates which is very disirable as the models aren't selecting unqualified candidates

"""


