"""
Evaluation pre-delinquency stage
_______________________________

This section will evlauate the bias of the first model which models the approval of mortgage application to investigate to what extent that thing is suseptable to bias

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
Calculating statistical parity ratio (ratio of positives between unfavoured and favoured groups)

Statistical parity = (Probability of favourable outcome for underpriveleged group) / (probability of positive outcome for advantaged group)
In this case we take probability as the percentage of applicants selected out of the whole sample size for that group

Favoured groups defined as those with high approval rates. Here we select white asian and joint as those groups have high approval rates (over 60%)

Favoured: Asian, Joint, White


Unfavoured groups generally have much lower approval rates than the favoured groups 

Unfavoured: 2 or more minority races, American Indian or Alaska Native, Asian, 
Black or African American,  Joint, Native Hawaiian or Other Pacific Islander

"""


#getting the ratio of positives for the combined advantaged groups

Asian_approved = (df_modelled["derived_race"] == "Asian").value_counts(dropna=True)

Joint_approved = (df_modelled["derived_race"] == "Joint").value_counts(dropna=True)

White_approved = (df_modelled["derived_race"] == "White").value_counts(dropna=True)
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

"""

Results
_______

Ratio for favoured groups

0.667832

"""

#Ratio for 2 or more races ratio
Two_or_more_approved = (df_modelled["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_Favoured = ((Two_or_more_ratio)/(Ratio_approved_favoured))

print("Ratio of two or more minority races : Favoured groups")
print(Ratio_Two_or_more_Favoured)

#Result    0.262528


#Hawaiian or pacific Islander ratio

Hawaiian_or_PI_approved = (df_modelled["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_Favoured = ((Hawaiian_or_PI_ratio)/(Ratio_approved_favoured))

print("Ratio of Hawaiian/Pacific Islander : Favoured groups")
print(Hawaiian_or_PI_Favoured)

# Results from this :   0.191061


#Black or African American ratio

Black_approved = (df_modelled["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_Favoured = ((Black_ratio)/(Ratio_approved_favoured))

print("Black or African american : Favoured groups")
print(Ratio_Black_Favoured)


# True     0.579157



Amerindian_approved = (df_modelled["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_approved)

Amerindian_applications = (df_original["derived_race"] == "American Indian or Alaska Native").value_counts(dropna=True)
print(Amerindian_applications)

Amerindian_ratio = (Amerindian_approved)/(Amerindian_applications)

print(Amerindian_ratio)

#Ratio American indian(alaskan native)/Favoured

Ratio_AI_Favoured = ((Amerindian_ratio)/(Ratio_approved_favoured))

print("Ratio of American Indian / Alaskan Native : Favoured groups")
print(Ratio_AI_Favoured)

#results from this: 0.240195