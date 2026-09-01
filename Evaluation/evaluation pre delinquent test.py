"""
Evaluation pre-delinquency stage
_______________________________

This section will evlauate the bias of the first model which models the approval of mortgage application to investigate to what extent that thing is suseptable to bias

The metric used in this section is statistical parity, the ratio of approved application between the selected group and a reference group from each demographic. 
"""


import pandas as pd

df_original = pd.read_csv("modelled_datasets/LR_test_data.csv")


df_modelled = pd.read_csv("modelled_datasets/approved_application_1st_model_LR.csv")


print(df_modelled.head())

relevant_columns = ['approved', 'action_taken', 'loan_type', 'loan_amount', 'income',
       'debt_to_income_ratio', 'loan_to_value_ratio', 'interest_rate',
       'property_value', 'loan_term', 'loan_purpose',
       'occupancy_type', 'derived_race', 'derived_sex', 'applicant_age',
       'negative_amortization', 'rate_spread','y_pred']

df_modelled = df_modelled[relevant_columns]

print(df_modelled.columns.to_list)


"""
Evaluation
___________

"""

"""
Evaluation of race
"""

print("Race of entire sample:")
print(df_original['derived_race'].value_counts(dropna=False).head(20))

print("Race of approved applications:")
print(df_modelled['derived_race'].value_counts(dropna=False).head(20))


#finding the ratio of approved applications by dividing the total approved applications by race by the total number of applications from the testing data
print("Ratio of applications approved/applications by race")
print(((df_modelled['derived_race'].value_counts(dropna=False).head(20))/(df_original['derived_race'].value_counts(dropna=False).head(20))))

"""
Results

Race of entire sample:
derived_race
White                                        57199
Asian                                         6816
Black or African American                     5900
Joint                                         1223
Native Hawaiian or Other Pacific Islander      337
American Indian or Alaska Native               293
2 or more minority races                       154


race of positives
derived_race
White                                        38307
Asian                                         4380
Black or African American                     2282
Joint                                          881
American Indian or Alaska Native                47
Native Hawaiian or Other Pacific Islander       43
2 or more minority races                        27

Ratio of applications approved/applications by race
derived_race
2 or more minority races                     0.175325
American Indian or Alaska Native             0.160410
Asian                                        0.642606
Black or African American                    0.386780
Joint                                        0.720360
Native Hawaiian or Other Pacific Islander    0.127596
White                                        0.669715

"""

#Getting percentage accepted by age

print("Age groups of entire sample:")
print(df_original['applicant_age'].value_counts(dropna=False).head(20))

print("age groups of approved applications:")
print(df_modelled['applicant_age'].value_counts(dropna=False).head(20))


#doing the division process but for age
print("Ratio of applications approved/applications by age")
print(((df_modelled['applicant_age'].value_counts(dropna=False).head(20))/(df_original['applicant_age'].value_counts(dropna=False).head(20))))

"""
Results
________

Age groups of entire sample:
applicant_age
35-44    17220
45-54    16843
55-64    13764
25-34    13399
65-74     6786
>74       2436
<25       1474
Name: count, dtype: int64
age groups of approved applications:
applicant_age
35-44    12150
25-34    11439
45-54     9774
55-64     7105
65-74     3383
<25       1268
>74        848
Name: count, dtype: int64
Ratio of applications approved/applications by age
applicant_age
25-34    0.853720
35-44    0.705575
45-54    0.580300
55-64    0.516202
65-74    0.498526
<25      0.860244
>74      0.348112
"""
"""
Percentage accepted by sex

"""
print("Gender of entire sample:")
print(df_original['derived_sex'].value_counts(dropna=False).head(20))

print("Gender of approved applications:")
print(df_modelled['derived_sex'].value_counts(dropna=False).head(20))


#division process by gender
print("Ratio of applications approved/total applications by Gender")
print(((df_modelled['derived_sex'].value_counts(dropna=False).head(20))/(df_original['derived_sex'].value_counts(dropna=False).head(20))))

"""
Results
_______

Gender of entire sample:

derived_sex
Joint     28144
Male      26488
Female    17290
Name: count, dtype: int64

Gender of approved applications:

derived_sex
Joint     20690
Male      15264
Female    10013
Name: count, dtype: int64

Ratio of applications approved/total applications by Gender

derived_sex
Joint     0.735148
Male      0.576261
Female    0.579121

"""

"""
Calculating statistical parity ratio (ratio of positives between unreference and reference groups)

Statistical parity = (Probability of favourable outcome for underpriveleged group) / (probability of positive outcome for advantaged group)
In this case we take probability as the percentage of applicants selected out of the whole sample size for that group

reference groups defined as those with high approval rates. Here we select white asian and joint as those groups have high approval rates (over 60%)

reference: Asian, Joint, White


Unreference groups generally have much lower approval rates than the reference groups 

Unreference: 2 or more minority races, American Indian or Alaska Native, Asian, 
Black or African American,  Joint, Native Hawaiian or Other Pacific Islander

"""


#getting the ratio of positives for the combined advantaged groups



White_approved = (df_modelled["derived_race"] == "White").value_counts(dropna=True)
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

0.669715

"""

#Ratio for 2 or more races ratio
Two_or_more_approved = (df_modelled["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_reference = ((Two_or_more_ratio)/(Ratio_approved_reference))

print("Ratio of two or more minority races : reference groups")
print(Ratio_Two_or_more_reference)

#Result    0.261790


#Hawaiian or pacific Islander ratio

Hawaiian_or_PI_approved = (df_modelled["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_reference = ((Hawaiian_or_PI_ratio)/(Ratio_approved_reference))

print("Ratio of Hawaiian/Pacific Islander : reference groups")
print(Hawaiian_or_PI_reference)

# Results from this :   0.190524


#Black or African American ratio

Black_approved = (df_modelled["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_reference = ((Black_ratio)/(Ratio_approved_reference))

print("Black or African american : reference groups")
print(Ratio_Black_reference)


# True     0.577529



Amerindian_approved = (df_modelled["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_approved)

Amerindian_applications = (df_original["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_applications)

Amerindian_ratio = (Amerindian_approved)/(Amerindian_applications)

print(Amerindian_ratio)

#Ratio American indian(alaskan native)/reference

Ratio_AI_reference = ((Amerindian_ratio)/(Ratio_approved_reference))

print("Ratio of American Indian / Alaskan Native : reference groups")
print(Ratio_AI_reference)

#results from this: 0.239519

#Asian

Asian_approved = (df_modelled["derived_race"] == "Asian").value_counts(dropna=True)
print(Amerindian_approved)

Asian_applications = (df_original["derived_race"] == "Asian").value_counts(dropna=True)
print(Asian_applications)

Asian_ratio = (Asian_approved)/(Asian_applications)

print(Asian_ratio)
Ratio_Asian_reference = ((Asian_ratio)/(Ratio_approved_reference))

print("Ratio of Asian : reference groups")
print(Ratio_Asian_reference)

# Results: 0.959522

#Joint
Joint_approved = (df_modelled["derived_race"] == "Joint").value_counts(dropna=True)
print(Joint_approved)

Joint_applications = (df_original["derived_race"] == "Joint").value_counts(dropna=True)
print(Joint_applications)

Joint_ratio = (Joint_approved)/(Joint_applications)

print(Joint_ratio)

Ratio_Joint_reference = ((Joint_ratio)/(Ratio_approved_reference))

print("Ratio of Joint : reference groups")
print(Ratio_Joint_reference)

#Results: 1.075622


"""
Evaluation for Gender
"""

woman_approved = (df_modelled["derived_sex"] == "Female").value_counts(dropna=True)
#applications



woman_applications = (df_original["derived_sex"] == "Female").value_counts(dropna=True)


#Using white as the reference groups as they are the largest race of all applicants and have a relatively high approval rate


Ratio_approved_gender_reference = ((woman_approved)/ (woman_applications))             
print("Ratio for reference groups")
print(Ratio_approved_reference)  


#Working out the statistical parity ratio between men and women

men_approved = (df_modelled["derived_sex"] == "Male").value_counts(dropna=True)
print(Joint_approved)

men_applications = (df_original["derived_sex"] == "Male").value_counts(dropna=True)
print(Joint_applications)

men_ratio = (men_approved)/(men_applications)

print(men_ratio)

Ratio_men_reference = ((men_ratio)/(Ratio_approved_gender_reference))

print("Ratio of men : reference groups")
print(Ratio_men_reference)

#Results: 0.995062
# Very little bias between genders showing models are extremely fair for this demographic


"""
Statistical parity evaluation for age

Using 35-44 as the reference group due to it's size and high approval rates
"""

"""
Evaluation for age

Going to use 35-44 as the reference group as it is the largest and has high rates of approval

"""

#Positives

approved_35_44 = (df_modelled["applicant_age"] == "35-44").value_counts(dropna=True)
#applications

applications_35_44 = (df_original["applicant_age"] == "35-44").value_counts(dropna=True)


Ratio_approved_age_reference = ((approved_35_44)/ (applications_35_44))             
print("Ratio for reference groups")
print(Ratio_approved_age_reference) 

#________________________________________________________________
#Ratio of applications accepted for 35-44 age group: 0.705575
# dont use in spreadsheet
#________________________________________________________________




#Ratio <25
approved_25 = (df_modelled["applicant_age"] == "<25").value_counts(dropna=True)


applications_25 = (df_original["applicant_age"] == "<25").value_counts(dropna=True)

ratio_25 = (approved_25)/(applications_25)

print(ratio_25)

Ratio_less_than_25 = ((ratio_25)/(Ratio_approved_age_reference))

print("Ratio of age group <25 : reference group")
print(Ratio_less_than_25)

#Result    1.219210


#25 - 34 ratio

approved_25_34 = (df_modelled["applicant_age"] == "25-34").value_counts(dropna=True)


applications_25_34 = (df_original["applicant_age"] == "25-34").value_counts(dropna=True)


ratio_25_34 = (approved_25_34)/(applications_25_34)

print(ratio_25_34)

reference_25_34 = ((ratio_25_34)/(Ratio_approved_age_reference))

print("Ratio of 25-34 : reference groups")
print(reference_25_34)

# Results from this :   1.209964


#45 - 54 ratio

approved_45_54 = (df_modelled["applicant_age"] == "45-54").value_counts(dropna=True)


applications_45_54 = (df_original["applicant_age"] == "45-54").value_counts(dropna=True)


ratio_45_54 = (approved_45_54)/(applications_45_54)

print(ratio_45_54)

Ratio_45_54_reference = ((ratio_45_54)/(Ratio_approved_age_reference))

print("45-54 : reference groups")
print(Ratio_45_54_reference)


# Results: 0.822450

#55-64

approved_55_64 = (df_modelled["applicant_age"] == "55-64").value_counts(dropna=True)

applications_55_64 = (df_original["applicant_age"] == "55-64").value_counts(dropna=True)

ratio_55_64 = (approved_55_64)/(applications_55_64)

print(ratio_55_64)

#Ratio American indian(alaskan native)/reference

Ratio_55_64_reference = ((ratio_55_64)/(Ratio_approved_age_reference))

print("Ratio of 55 - 64 : reference groups")
print(Ratio_55_64_reference)

#results from this: 0.731604

#65-74

approved_65_74_ = (df_modelled["applicant_age"] == "65-74").value_counts(dropna=True)
print(approved_65_74_)

applications_65_74_ = (df_original["applicant_age"] == "65-74").value_counts(dropna=True)
print(applications_65_74_)

ratio_65_74_ = (approved_65_74_)/(applications_65_74_)

print(ratio_65_74_ )
Ratio_65_74_reference = ((ratio_65_74_ )/(Ratio_approved_age_reference))

print("Ratio of 65 - 74 : reference groups")
print(Ratio_65_74_reference)

# Results: 0.706553

#over 74
over_74_approved = (df_modelled["applicant_age"] == ">74").value_counts(dropna=True)
print(over_74_approved)

over_74_applications = (df_original["applicant_age"] == ">74").value_counts(dropna=True)
print(over_74_applications)

over_74_ratio = (over_74_approved)/(over_74_applications)

print(over_74_ratio)

Ratio_over_74_reference = ((over_74_ratio)/(Ratio_approved_age_reference))

print("Ratio of over 74 : reference groups")
print(Ratio_over_74_reference)

#Results: 0.493373