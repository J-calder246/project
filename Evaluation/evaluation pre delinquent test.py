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

print("Ratio of applications approved/applications by race")
print(((df_modelled['derived_race'].value_counts(dropna=False).head(20))/(df_original['derived_race'].value_counts(dropna=False).head(20))))

"""
Results

derived_race
White                                        287012
Race Not Available                            64177
Asian                                         34446
Black or African American                     29315
Joint                                          5945
Native Hawaiian or Other Pacific Islander      1688
American Indian or Alaska Native               1568
2 or more minority races                        750


derived_race
White                                        39813
Race Not Available                            6900
Asian                                         4450
Black or African American                     2463
Joint                                          898
American Indian or Alaska Native                69
Native Hawaiian or Other Pacific Islander       59
2 or more minority races                        39
Free Form Text Only                              4

Ratio of applications approved/applications by race
derived_race
2 or more minority races                     0.052000
American Indian or Alaska Native             0.044005
Asian                                        0.129188
Black or African American                    0.084018
Free Form Text Only                          0.035088
Joint                                        0.151051
Native Hawaiian or Other Pacific Islander    0.034953
Race Not Available                           0.107515
White                                        0.138715

"""

#Getting percentage accepted by age

print("Age groups of entire sample:")
print(df_original['applicant_age'].value_counts(dropna=False).head(20))

print("age groups of approved applications:")
print(df_modelled['applicant_age'].value_counts(dropna=False).head(20))

print("Ratio of applications approved/applications by age")
print(((df_modelled['applicant_age'].value_counts(dropna=False).head(20))/(df_original['applicant_age'].value_counts(dropna=False).head(20))))

"""
Results
________

Age groups of entire sample:
applicant_age
35-44    103000
45-54    100145
55-64     79538
25-34     76035
65-74     39322
>74       13491
<25        7778

Name: count, dtype: int64
age groups of approved applications:
applicant_age
35-44    14391
25-34    12674
45-54    11853
55-64     8483
65-74     4117
<25       1268
>74       1097
Name: count, dtype: int64
Ratio of applications approved/applications by age
applicant_age
25-34    0.166686
35-44    0.139718
45-54    0.118358
55-64    0.106653
65-74    0.104700
<25      0.163024
>74      0.081313

"""
"""
Percentage accepted by sex

"""
print("Gender of entire sample:")
print(df_original['derived_sex'].value_counts(dropna=False).head(20))

print("Gender of approved applications:")
print(df_modelled['derived_sex'].value_counts(dropna=False).head(20))

print("Ratio of applications approved/total applications by Gender")
print(((df_modelled['derived_sex'].value_counts(dropna=False).head(20))/(df_original['derived_sex'].value_counts(dropna=False).head(20))))

"""
Results
_______

Gender of entire sample:
derived_sex
Joint                150411
Male                 143526
Female                93405
Sex Not Available     37673
Name: count, dtype: int64
Gender of approved applications:
derived_sex
Joint                22650
Male                 16783
Female               10966
Sex Not Available     4296
Name: count, dtype: int64
Ratio of applications approved/total applications by Gender
derived_sex
Joint                0.150587
Male                 0.116934
Female               0.117403
Sex Not Available    0.114034

"""

"""
Calculating ratios of favoured groups

"""


#Positives

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
print(Ratio_approved_favoured)   #Result = True     0.086328    (correct)

"""

Results
_______

Ratio for favoured groups

0.137937

(from combined models:  0.086328)
"""

#Ratio for 2 or more races
Two_or_more_approved = (df_modelled["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_applications = (df_original["derived_race"] == "2 or more minority races").value_counts(dropna=True)


Two_or_more_ratio = (Two_or_more_approved)/(Two_or_more_applications)

print(Two_or_more_ratio)

Ratio_Two_or_more_Favoured = ((Two_or_more_ratio)/(Ratio_approved_favoured))

print("Ratio of two or more minority races : Favoured groups")
print(Ratio_Two_or_more_Favoured)

#Result    0.376984
#From the two combined models: 0.401570

#Hawaiian or pacific Islander

Hawaiian_or_PI_approved = (df_modelled["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_applications = (df_original["derived_race"] == "Native Hawaiian or Other Pacific Islander").value_counts(dropna=True)


Hawaiian_or_PI_ratio = (Hawaiian_or_PI_approved)/(Hawaiian_or_PI_applications)

print(Hawaiian_or_PI_ratio)

Hawaiian_or_PI_Favoured = ((Hawaiian_or_PI_ratio)/(Ratio_approved_favoured))

print("Ratio of Hawaiian/Pacific Islander : Favoured groups")
print(Hawaiian_or_PI_Favoured)

# Results from this :   0.253395
# Results from combined model:  0.171560


#Black or African American                    0.041480

Black_approved = (df_modelled["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_applications = (df_original["derived_race"] == "Black or African American").value_counts(dropna=True)


Black_ratio = (Black_approved)/(Black_applications)

print(Black_ratio)

Ratio_Black_Favoured = ((Black_ratio)/(Ratio_approved_favoured))

print("Black or African american : Favoured groups")
print(Ratio_Black_Favoured)


# True     0.044005
#Results from combined model: 0.480499



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

#results from this: True     0.319023
#Results from combined models:  0.258566