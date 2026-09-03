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


____________________________________________________

Training Baseline models

Trained two models for each dataset (random forest classifier and logistic regression classifier) for predicting application approval and delinquency. 


___________________________________________

Baseline fairness

Includes measures of statistical parity on the baseline models after both have been used together (see 2nd layer modelling.py) to get an idea of the proportion of applicants each model accepts.

Also evaluates equalised odds, selection rate etc for the basic applications models to get a more clear view of impact by demographic that uses true and false positives and negatives

_______________________________________________________________-

Mitigation

To answer the 2nd research question:  'To what extent do various machine learning and AI models processing automated loan approval decisions exhibit bias against certain groups?' This artifact has used various types of mitigation techniques on the data

This project has used various pre, in and post processing techniques to mitigate bias

Pre processing
 includes retraining the models without using the sensitive values and seeing how that effect fairness as well as using Fairlearn's correlation remover which remove the correlation between sensitive features and final results that could be causing bias from the model.

In- processing includes an Exponentiated gradient model. This technique creates many different classifiers that are constrained by a fairness target. Expponentiated gradient then selects the classifier with the least error to ensure accuracy and fairness.

Post-processing techniques include a threshold optimiser technique which alters a model prediction to target fairness while setting limits to maintain accuracy.


These models are evaluated using metric frame to find accuracy, equalised odd, selection rate etc to answer how effective these models are anfd to find if these techniques result in any trade-offs, losing accuracy when reducing bias is attempted.


____________________________________________


2 layer modelling

The data from this project at times passes through two models.
- First it'll go through a model to predict if the application will be approved.
- If this application passes this stage and is approved, it gets saved onto a dataset containing the applications predicted to be approved.
- This data is then fed into a delinquency dataset after it's features have been cut down to those that apply to the delinquency model.
- Those that have been approved and are predicted not to be delinquent are evaluating in Evaluation/evaluation1.py and evalauate_RF_models_(Statistical_parity).py to see the statistical parity of those that have gone through.

- The data of the mitigated models haven't gone through this process as answering this is not really relevant for the research process but the files can be copied and this process can be applied to this data.


This process is here so that a framework exists that can filter applications that have gotten through the application process but would be predicted to be delinquent
This prevent applicants who are likely to default on their debt getting through the process reducing the amount of unqualified candidates being approved.
This process when applied to the mitigated models also help to prevent potentially unqualified candidates that were elevated by the mitigation processes from being approved to mortgages that they are unsuited for.
Although this hasn't been applied to the full dataset due to time constraint, the application includes this in a way as you can see if an enterred application is likely to be delinquent giving more certainty to the models predicting applications.

This approach has some issues however. The main of these being the limited amount of features that cross over between the HMDA dataset and the Fannie mae dataset. This means that very few features could be used to predict delinquency as we have to use features associated with the HMDA dataset, limiting it's accuracy.
As a result, the delinquency models should only be used as suggestions when looking at applications, providing more certainty to the approved applications rather than be used as a final hurdle that the applications have to follow.

The data flow diagram for this is shown below
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


App
___

The final component of this project is the application for deploying the models.
This app allows for a user to enter an application and then run the models and mitigation techniques on this data to predict if the application will be approved.

Unfortunately, this app mostly includes just the logistic regression models as the Random Forest models were too large to be properly deployed and lead to the app running very slowly and crashing.




____________________________-

artefact resusability and further development

_____________________________

This project is focuesed on loan applications specifically from the state of New york and in the year of 2019.
Due to the scope of this artefact focused here, the models developed will be best used on applications from this year and state and shouldn't be used in different circumstances due to the differences in the lending market by time and place
However, similar process will be applicable to other states, so long as the data used is appropriate for what you are targetting (ie; using a dataset from california in 2014 but following the same process as shown here)
This artefact could also be used as a bedrock to do something similar but for the whole Country. 
For this, as the HMDA dataset is divided by states, datasets would have to be sampled and combined to create a dataset that proporitonally includes the whole of the united states.

I would also recommend using different mitigation process from different packages to further evaluate the effects of bias mitigation.
These packages weren't used here to ensure that the project has a balanced and limited scope but processes from AIF36o or Aequitas could be used in a similar way to see if these packages produce more effective bias mitigation techniques on this dataset.