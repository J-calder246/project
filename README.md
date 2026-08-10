# project
Mortgage and bias repo

This project aims to create models for predicting if mortgage applications will be granted or rejected. The research point of this project is investigating if these models can be susceptible to bias against age, racial or gender based groups. It also includes the use of Microsoft Fairlearns' bias mitigation packages and assessments of how effective these techniques are at reducing bias and whether these methods come with various trade-offs against accuracy.

For predicting applications, the data follows a pipeline as shown below.

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


User uploads a datasets on front end

Model is trained for granting mortgages

Models predictions are assessed by the delinquency dataset