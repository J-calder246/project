"""
Evaluation model 1
__________________

This module will evaluate the bias that results when the data goes through both of the original models (application model ---> Delinquency model ---> prediction)

"""


import pandas as pd


df_original = pd.read_csv("processed_datasets/NY2019.csv")   #This dataset must be used and not a test data file like in the other statistical parity file as in the 2 layer section the data isn't split into train and test
print(df_original.head())

df_approved_non_delinquent = pd.read_csv("modelled_datasets/approved_non_delinquent.csv")  #applications from df_original that were predicted to be approved and then predicted to be non-delinquent

"""
First some processing
______________________

Removing dummy values for clarity



"""

print(df_approved_non_delinquent.columns.to_list)



relevant_columns = ['approved', 'action_taken', 'loan_type', 'loan_amount', 'income',
       'debt_to_income_ratio', 'loan_to_value_ratio', 'interest_rate',
       'property_value', 'loan_term', 'loan_purpose',
       'occupancy_type', 'derived_race', 'derived_sex', 'applicant_age',
       'negative_amortization', 'rate_spread', 'delinquent',
       'probability_of_delinquency']

df_approved_non_delinquent = df_approved_non_delinquent[relevant_columns]

print(df_approved_non_delinquent.columns.to_list)

"""
Finding counts for race
________________________

"""


"""
Evaluation of race
"""

print("Race of entire sample:")
print(df_original['derived_race'].value_counts(dropna=False).head(20))

print("Race of approved applications:")
print(df_approved_non_delinquent['derived_race'].value_counts(dropna=False).head(20))

#prints what ratio of applications by race are approved when sent through both models
print("Ratio of applications approved/applications by race")
print(((df_approved_non_delinquent['derived_race'].value_counts(dropna=False).head(20))/(df_original['derived_race'].value_counts(dropna=False).head(20))))



"""
Results

(advantaged groups starred)
_______
Ratio of applications approved by race
derived_race
2 or more minority races                     0.119143
American Indian or Alaska Native             0.094412
Asian                                        0.418170
Black or African American                    0.226476
Joint                                        0.520925
Native Hawaiian or Other Pacific Islander    0.085068
White                                        0.457714
"""


"""
Evaluation of gender

"""

print("Gender of entire sample:")
print(df_original['derived_sex'].value_counts(dropna=False).head(20))

print("Gender of approved applications:")
print(df_approved_non_delinquent['derived_sex'].value_counts(dropna=False).head(20))

#prints what ratio of applications by race are approved when sent through both models
print("Ratio of applications approved/total applications by Gender")
print(((df_approved_non_delinquent['derived_sex'].value_counts(dropna=False).head(20))/(df_original['derived_sex'].value_counts(dropna=False).head(20))))

"""
Results  (no significant differences)

Joint     0.510543
Male      0.378722
Female    0.386617
"""
"""
Age
______
"""

print("Age groups of entire sample:")
print(df_original['applicant_age'].value_counts(dropna=False).head(20))

print("age groups of approved applications:")
print(df_approved_non_delinquent['applicant_age'].value_counts(dropna=False).head(20))

#prints what ratio of applications by race are approved when sent through both models
print("Ratio of applications approved/applications by age")
print(((df_approved_non_delinquent['applicant_age'].value_counts(dropna=False).head(20))/(df_original['applicant_age'].value_counts(dropna=False).head(20))))

"""
Results
_______

25-34    0.606669
35-44    0.482382
45-54    0.381869
55-64    0.340438
65-74    0.326447
<25      0.583733
>74      0.199562
"""



"""
Race Calculations
_________________

Favoured 
"""
"""
Disparate imapct
________________

Positive outcome% (Non-favoured group)/Positive outcome% (Favoured groups)

Remember 80% Rule


"""

"""
Section Finding approved ratio for the favoured races  (asian, joint and white)
_____________________________-_

Positives(asian+joint+white)/Applications(asian+joint+white)   Working it out by dividing the sums pevents smaller groups have the same impact as larger groups

"""

#Positives

Asian_approved = (df_approved_non_delinquent["derived_race"] == "Asian").value_counts(dropna=True)

Joint_approved = (df_approved_non_delinquent["derived_race"] == "Joint").value_counts(dropna=True)

White_approved = (df_approved_non_delinquent["derived_race"] == "White").value_counts(dropna=True)
#applications

Asian_applications = (df_original["derived_race"] == "Asian").value_counts(dropna=True)

Joint_applications = (df_original["derived_race"] == "Joint").value_counts(dropna=True)

White_applications = (df_original["derived_race"] == "White").value_counts(dropna=True)


#Total Ratio calculation

Sum_approved_favoured = (Asian_approved + Joint_approved + White_approved)   
print(Sum_approved_favoured)

Sum_applications_approved = (Asian_applications + Joint_applications + White_applications)  
print(Sum_applications_approved)

Ratio_approved_favoured = ((Sum_approved_favoured)/ (Sum_applications_approved))              
print("Ratio for favoured groups")
print(Ratio_approved_favoured)  

#Ratio of favoured racial groups: 0.454704

"""
Calculating statistical parity ratio for the unfavoured groups
Equation
________
Statistical parity ratio = (Probability of favourable outcome for underpriveleged group) / (probability of positive outcome for advantaged group)
In this case we take probability as the percentage of applicants selected out of the whole sample size for that group

Favoured groups defined as those with high approval rates. Here we select white asian and joint as those groups have high approval rates (over 60%)

"""

Amerindian_approved = (df_approved_non_delinquent["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_approved)

Amerindian_applications = (df_original["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_applications)

Amerindian_ratio = (Amerindian_approved)/(Amerindian_applications)

print(Amerindian_ratio)

#Ratio American indian(alaskan native)/Favoured

Ratio_AI_Favoured = ((Amerindian_ratio)/(Ratio_approved_favoured))

print("Ratio of American Indian / Alaskan Native : Favoured groups")
print(Ratio_AI_Favoured)

#Results:  0.207635


#two or more minority

Two_or_more_approved = (df_approved_non_delinquent["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_Favoured = ((Two_or_more_ratio)/(Ratio_approved_favoured))

print("Ratio of two or more minority races : Favoured groups")
print(Ratio_Two_or_more_Favoured)

#Results  0.262024

#Black or African American 

Black_approved = (df_approved_non_delinquent["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_Favoured = ((Black_ratio)/(Ratio_approved_favoured))

print("Black or African american : Favoured groups")
print(Ratio_Black_Favoured)

#Results 0.498073

#Native Hawaiian or Other Pacific Islander       0.085068

Hawaiian_or_PI_approved = (df_approved_non_delinquent["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_Favoured = ((Hawaiian_or_PI_ratio)/(Ratio_approved_favoured))

print("Ratio of Hawaiian/Pacific Islander : Favoured groups")
print(Hawaiian_or_PI_Favoured)

#Results:  0.187085



"""
Age section
___________

Evaluating bias against different age groups


"""



