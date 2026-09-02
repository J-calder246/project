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

"""
Mitigation section

"""
from fairlearn.reductions import ExponentiatedGradient, DemographicParity  #exponentiated gradient is an inprocessing technique
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
    constraints=DemographicParity(),
    sample_weight_name="logreg__sample_weight" #allows for more weight for underrepresented samples

)

exponentiated_gradient.fit(X_train, y_train, sensitive_features=A_train)

y_pred = exponentiated_gradient.predict(X_test, random_state=69)

print("Classification Report Exponentiated gradient:")
print(classification_report(y_test, y_pred))


#Saving model
import joblib

joblib.dump(exponentiated_gradient, "mitigated_models/exponentiated_gradient_LR.pkl")

exponentiated_gradient_LR = joblib.load("mitigated_models/exponentiated_gradient_LR.pkl")
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
               accuracy  selection rate    count  true_positive_rate
applicant_age                                                       
25-34          0.797829        0.809976  13172.0            0.856329
35-44          0.790847        0.817859  17437.0            0.887316
45-54          0.765191        0.798505  16720.0            0.891371
55-64          0.750664        0.813339  13929.0            0.906205
65-74          0.740311        0.826100   6889.0            0.902600
<25            0.726955        0.800978   1432.0            0.816054
>74            0.697397        0.815194   2343.0            0.898474
Equalised odds difference for age
0.18329660142379878

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

Classification Report Exponentiated gradient:
              precision    recall  f1-score   support

       False       0.52      0.41      0.46     17755
        True       0.82      0.88      0.85     54167

    accuracy                           0.76     71922
   macro avg       0.67      0.64      0.65     71922
weighted avg       0.75      0.76      0.75     71922



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

"""


