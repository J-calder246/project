"""
Module for evaluation of the results of '2nd layer modelling' using statistical parity
    -   Statistical parity is the ratio of cases selected positive by the model compared to the favoured group
    -   Measures such as equalised odds or disparate impact cannot be used here as 'True Positives' cannot be found through the two layer moddling system.
    -   Assessment of true positive cases are assessed in the 1st layer evaluation module
"""

import pandas as pd


df_original = pd.read_csv("processed_datasets/NY2019_no_dummies.csv")
print(df_original.head())

df_approved_non_delinquent = pd.read_csv("modelled_datasets/approved_non_delinquent.csv")

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

print("Ratio of applications approved/applications by race")
print(((df_approved_non_delinquent['derived_race'].value_counts(dropna=False).head(20))/(df_original['derived_race'].value_counts(dropna=False).head(20))))



"""
Results

(advantaged groups starred)
_______
Ratio of applications approved by race
derived_race
2 or more minority races                     0.013953
American Indian or Alaska Native             0.010550
Asian                                        0.073886
Black or African American                    0.038214
Joint                                        0.090909
Native Hawaiian or Other Pacific Islander    0.009485
White                                        0.080901 *
"""


"""
Evaluation of gender

"""

print("Gender of entire sample:")
print(df_original['derived_sex'].value_counts(dropna=False).head(20))

print("Gender of approved applications:")
print(df_approved_non_delinquent['derived_sex'].value_counts(dropna=False).head(20))

print("Ratio of applications approved/total applications by Gender")
print(((df_approved_non_delinquent['derived_sex'].value_counts(dropna=False).head(20))/(df_original['derived_sex'].value_counts(dropna=False).head(20))))

"""
Results  (no significant differences)

Joint     0.091710
Male      0.064405
Female    0.067302
"""
"""
Age
______
"""

print("Age groups of entire sample:")
print(df_original['applicant_age'].value_counts(dropna=False).head(20))

print("age groups of approved applications:")
print(df_approved_non_delinquent['applicant_age'].value_counts(dropna=False).head(20))

print("Ratio of applications approved/applications by age")
print(((df_approved_non_delinquent['applicant_age'].value_counts(dropna=False).head(20))/(df_original['applicant_age'].value_counts(dropna=False).head(20))))

"""
Results
_______

35-44    0.092013
45-54    0.073900
55-64    0.066830
65-74    0.062647
<25      0.112489
>74      0.040789
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
Section Finding favoured races
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

Sum_approved_favoured = (Asian_approved + Joint_approved + White_approved)   #Should be : 28264
print(Sum_approved_favoured)

Sum_applications_approved = (Asian_applications + Joint_applications + White_applications)  # Should be 327403
print(Sum_applications_approved)

Ratio_approved_favoured = ((Sum_approved_favoured)/ (Sum_applications_approved))              #Should be: 0.086327
print("Ratio for favoured groups")
print(Ratio_approved_favoured)   #Result = True     0.086328    (correct)



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

#Results:  0.258566


#2 or more minority races                     0.034667

Two_or_more_approved = (df_approved_non_delinquent["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_Favoured = ((Two_or_more_ratio)/(Ratio_approved_favoured))

print("Ratio of two or more minority races : Favoured groups")
print(Ratio_Two_or_more_Favoured)

#Results  0.401570

#Black or African American                    0.041480

Black_approved = (df_approved_non_delinquent["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_Favoured = ((Black_ratio)/(Ratio_approved_favoured))

print("Black or African american : Favoured groups")
print(Ratio_Black_Favoured)

#Results 0.480499

#Native Hawaiian or Other Pacific Islander    0.014810

Hawaiian_or_PI_approved = (df_approved_non_delinquent["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_Favoured = ((Hawaiian_or_PI_ratio)/(Ratio_approved_favoured))

print("Ratio of Hawaiian/Pacific Islander : Favoured groups")
print(Hawaiian_or_PI_Favoured)
