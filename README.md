# project
Mortgage and bias repo

Introduction

This project aims to create models for predicting if mortgage applications will be granted or rejected. The research point of this project is investigating if these models can be susceptible to bias against age, racial or gender based groups. It also includes the use of Microsoft Fairlearns' bias mitigation packages and assessments of how effective these techniques are at reducing bias and whether these methods come with various trade-offs against accuracy.

Research Questions

-       To what extent do various machine learning and AI models processing automated loan approval decisions exhibit bias against certain groups?

-	What mitigation techniques can be used to reduce bias resulting from machine learning, and is there a significant trade-off between predictive performance and bias?
-	Can an equitable balance between predictive performance and fairness be achieved?


Raw Datasets

-       HMDA datasets counting mortgage applications from the state of New york in 2019 with finanacial metrics and demographic features from each of the applicant. This dataset is used to predict approval. "raw_datasets/state_NY 2019 (1).csv"
-       A Fannie mae dataset from 2019, displaying various mortgages and if those borrowers failed to make payments on their mortgages. This dataset is used to predict if the applicants would be delinquent. "raw_datasets/2019, NY fianl (1).xls"

Data processing

Section found in the 'processing' folder with seperate files for the HMDA dataset and the Fannie Mae dataset. Includes processes for encoding categorical data and removing unecessary features from these datasets.

Training Baseline models

Trained two models for each dataset (random forest classifier and logistic regression classifier) for predicting application approval and delinquency. 

Baseline fairness

Includes measures of statistical parity on the baseline models after both have been used together (see 2nd layer modelling.py) to get an idea of the proportion of applicants each model accepts.

Also evaluates equalised odds, selection rate etc for the basic applications models to get a more clear view of impact by demographic that uses true and false positives and negatives

Mitigation

_________________________________
HMDA Mortgage Application dataset
            |
            |
            |
            V
    Train-Test-Split
            |   
            |
            |
            V
Logistic regression/ Random Forest
            |
            |
            |
            V
    Approved Applications
            |
            |
            |
            V
Delinquency Models (Random Forest/ Logistic regression)
            |
            |
            |
            |
            V
Applications predicted to be approved and non-delinquent.
__________________________________________


