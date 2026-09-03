"""
Module for evaluation of the results of '2nd layer modelling' using statistical parity
    -   Statistical parity is the ratio of cases selected positive by the model compared to the reference group
    -   Measures such as equalised odds or disparate impact cannot be used here as 'True Positives' cannot be found through the two layer moddling system.
    -   Assessment of true positive cases are assessed in the 1st layer evaluation module
"""

import pandas as pd


df_original = pd.read_csv("modelled_datasets/RF_test_Data.csv")
print(df_original.head())

df_approved_non_delinquent = pd.read_csv("modelled_datasets/approved_non_delinquent(RF).csv")

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
Ratio of applications approved/applications by race
derived_race
White                                        0.633997
Asian                                        0.581719
Black or African American                    0.450339
Joint                                        0.664759
Native Hawaiian or Other Pacific Islander    0.290801
American Indian or Alaska Native             0.232082
2 or more minority races                     0.266234
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

Ratio of applications approved/total applications by Gender
derived_sex
Joint     0.679861
Male      0.562557
Female    0.570908
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

applicant_age
25-34    0.726547
35-44    0.648374
45-54    0.581369
55-64    0.551003
65-74    0.530946
<25      0.681818
>74      0.419540
"""



"""
Race Calculations
_________________

reference 
"""
"""
Disparate imapct
________________

Positive outcome% (Non-reference group)/Positive outcome% (reference groups)

Remember 80% Rule


"""

"""
Section Finding reference races
_____________________________-_

Positives(asian+joint+white)/Applications(asian+joint+white)   Working it out by dividing the sums pevents smaller groups have the same impact as larger groups

"""

#Positives

White_approved = (df_approved_non_delinquent["derived_race"] == "White").value_counts(dropna=True)
#applications



White_applications = (df_original["derived_race"] == "White").value_counts(dropna=True)


#Using white as the reference groups as they are the largest race of all applicants and have a relatively high approval rate

Sum_approved_reference = (White_approved)  
print(Sum_approved_reference)

Sum_applications_approved = (White_applications)  
print(Sum_applications_approved)

Ratio_approved_reference = ((Sum_approved_reference)/ (Sum_applications_approved))             
print("Ratio for reference groups")
print(Ratio_approved_reference)     

"""

Results
_______

Ratio for reference groups

0.633997

"""

#Ratio for 2 or more races ratio
Two_or_more_approved = (df_approved_non_delinquent["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_reference = ((Two_or_more_ratio)/(Ratio_approved_reference))

print("Ratio of two or more minority races : reference groups")
print(Ratio_Two_or_more_reference)

#Result    0.419929


#Hawaiian or pacific Islander ratio

Hawaiian_or_PI_approved = (df_approved_non_delinquent["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_reference = ((Hawaiian_or_PI_ratio)/(Ratio_approved_reference))

print("Ratio of Hawaiian/Pacific Islander : reference groups")
print(Hawaiian_or_PI_reference)

# Results from this :   0.458679


#Black or African American ratio

Black_approved = (df_approved_non_delinquent["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_reference = ((Black_ratio)/(Ratio_approved_reference))

print("Black or African american : reference groups")
print(Ratio_Black_reference)


# True     0.710317



Amerindian_approved = (df_approved_non_delinquent["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_approved)

Amerindian_applications = (df_original["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_applications)

Amerindian_ratio = (Amerindian_approved)/(Amerindian_applications)

print(Amerindian_ratio)

#Ratio American indian(alaskan native)/reference

Ratio_AI_reference = ((Amerindian_ratio)/(Ratio_approved_reference))

print("Ratio of American Indian / Alaskan Native : reference groups")
print(Ratio_AI_reference)

#results from this: 0.366061

#Asian

Asian_approved = (df_approved_non_delinquent["derived_race"] == "Asian").value_counts(dropna=True)
print(Amerindian_approved)

Asian_applications = (df_original["derived_race"] == "Asian").value_counts(dropna=True)
print(Asian_applications)

Asian_ratio = (Asian_approved)/(Asian_applications)

print(Asian_ratio)
Ratio_Asian_reference = ((Asian_ratio)/(Ratio_approved_reference))

print("Ratio of Asian : reference groups")
print(Ratio_Asian_reference)

# Results:  0.917543

#Joint
Joint_approved = (df_approved_non_delinquent["derived_race"] == "Joint").value_counts(dropna=True)
print(Joint_approved)

Joint_applications = (df_original["derived_race"] == "Joint").value_counts(dropna=True)
print(Joint_applications)

Joint_ratio = (Joint_approved)/(Joint_applications)

print(Joint_ratio)

Ratio_Joint_reference = ((Joint_ratio)/(Ratio_approved_reference))

print("Ratio of Joint : reference groups")
print(Ratio_Joint_reference)

# Results:  1.04852

"""
Evaluation for gender
"""

woman_approved = (df_approved_non_delinquent["derived_sex"] == "Female").value_counts(dropna=True)
#applications



woman_applications = (df_original["derived_sex"] == "Female").value_counts(dropna=True)





Ratio_approved_gender_reference = ((woman_approved)/ (woman_applications))             
print("Ratio for reference groups")
print(Ratio_approved_reference)  


#Working out the statistical parity ratio between men and women

men_approved = (df_approved_non_delinquent["derived_sex"] == "Male").value_counts(dropna=True)
print(Joint_approved)

men_applications = (df_original["derived_sex"] == "Male").value_counts(dropna=True)
print(Joint_applications)

men_ratio = (men_approved)/(men_applications)

print(men_ratio)

Ratio_men_reference = ((men_ratio)/(Ratio_approved_gender_reference))

print("Ratio of men : reference groups")
print(Ratio_men_reference)

#Result 0.985372

"""
Evaluation for age

Going to use 35-44 as the reference group as it is the largest and has high rates of approval

"""

#Positives

approved_35_44 = (df_approved_non_delinquent["applicant_age"] == "35-44").value_counts(dropna=True)
#applications

applications_35_44 = (df_original["applicant_age"] == "35-44").value_counts(dropna=True)


Ratio_approved_age_reference = ((approved_35_44)/ (applications_35_44))             
print("Ratio for reference groups")
print(Ratio_approved_age_reference)     

# 0.648374

#Ratio <25
approved_25 = (df_approved_non_delinquent["applicant_age"] == "<25").value_counts(dropna=True)


applications_25 = (df_original["applicant_age"] == "<25").value_counts(dropna=True)

ratio_25 = (approved_25)/(applications_25)

print(ratio_25)

Ratio_less_than_25 = ((ratio_25)/(Ratio_approved_age_reference))

print("Ratio of age group <25 : reference group")
print(Ratio_less_than_25)

#Result    1.051582


#25 - 34 ratio

approved_25_34 = (df_approved_non_delinquent["applicant_age"] == "25-34").value_counts(dropna=True)


applications_25_34 = (df_original["applicant_age"] == "25-34").value_counts(dropna=True)


ratio_25_34 = (approved_25_34)/(applications_25_34)

print(ratio_25_34)

reference_25_34 = ((ratio_25_34)/(Ratio_approved_age_reference))

print("Ratio of 25-34 : reference groups")
print(reference_25_34)

# Results from this :  1.120567


#45 - 54 ratio

approved_45_54 = (df_approved_non_delinquent["applicant_age"] == "45-54").value_counts(dropna=True)


applications_45_54 = (df_original["applicant_age"] == "45-54").value_counts(dropna=True)


ratio_45_54 = (approved_45_54)/(applications_45_54)

print(ratio_45_54)

Ratio_45_54_reference = ((ratio_45_54)/(Ratio_approved_age_reference))

print("45-54 : reference groups")
print(Ratio_45_54_reference)


# Results: 0.896657

#55-64

approved_55_64 = (df_approved_non_delinquent["applicant_age"] == "55-64").value_counts(dropna=True)

applications_55_64 = (df_original["applicant_age"] == "55-64").value_counts(dropna=True)

ratio_55_64 = (approved_55_64)/(applications_55_64)

print(ratio_55_64)

#Ratio American indian(alaskan native)/reference

Ratio_55_64_reference = ((ratio_55_64)/(Ratio_approved_age_reference))

print("Ratio of 55 - 64 : reference groups")
print(Ratio_55_64_reference)

#results from this: 0.849822

#65-74

approved_65_74_ = (df_approved_non_delinquent["applicant_age"] == "65-74").value_counts(dropna=True)
print(approved_65_74_)

applications_65_74_ = (df_original["applicant_age"] == "65-74").value_counts(dropna=True)
print(applications_65_74_)

ratio_65_74_ = (approved_65_74_)/(applications_65_74_)

print(ratio_65_74_ )
Ratio_65_74_reference = ((ratio_65_74_ )/(Ratio_approved_age_reference))

print("Ratio of 65 - 74 : reference groups")
print(Ratio_65_74_reference)

# Results: 0.818889

#over 74
over_74_approved = (df_approved_non_delinquent["applicant_age"] == ">74").value_counts(dropna=True)
print(over_74_approved)

over_74_applications = (df_original["applicant_age"] == ">74").value_counts(dropna=True)
print(over_74_applications)

over_74_ratio = (over_74_approved)/(over_74_applications)

print(over_74_ratio)

Ratio_over_74_reference = ((over_74_ratio)/(Ratio_approved_age_reference))

print("Ratio of over 74 : reference groups")
print(Ratio_over_74_reference)

#Results: 0.647065