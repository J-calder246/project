#This module will use fairlearn to evaluate the models and mitigate bias

import modelling.modelling
import pandas as pd
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate
from modelling.modelling
from sklearn.metrics import accuracy_score  #accuracy score for original model

df_modelled = pd.read_csv("modelled_datasets/approved_application_1st_model_LR.csv")

print(df_modelled.columns.to_list)

#Note: Equalised odds in the difference between true positives and true negatives by groups

#Evaluation  on first model
"""
race = X_test_scaled_RF["derived_race"]
#logistic regression model
MF = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    Discriminatory_features = race

)

print(MF.by_group)

"""