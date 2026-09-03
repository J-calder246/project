import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
from fairlearn.metrics import MetricFrame, count, equalized_odds_difference, equalized_odds_ratio,  false_negative_rate, false_positive_rate, selection_rate




#Preparing a logistic regression classification model

df1 = pd.read_csv("processed_datasets/NY2019.csv")

print(df1.isnull().sum())

print(df1.columns.to_list())



print(df1.shape)
X = df1.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", 
                      "applicant_age", "occupancy_type"
       ])  #dropping the aproved values and values that have been encoded


#removing interest rate and rate spread to prevent data leakage as models are matching empty values here to unapproved applications
X = X.drop(columns=["interest_rate", "rate_spread"])  
y = df1['approved']

print(X.columns.to_list())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    #stratify ensures that the test and train data have a proportional amount of target results (approved)

test_index = X_test.index   #save X values so they arent messed up when scaled later on and we can save this index and use it for the evaluation



#Imputing missing values (change later if this is not suitable)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


#implementing a scaler so features have more proportional influence in the predictions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising and fitting LR model

logreg = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)


#getting accuracy score and reports
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print reports
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
Accuracy: 0.73
Confusion Matrix:
[[12279  5476]
 [13676 40491]]
Classification Report:
              precision    recall  f1-score   support

       False       0.47      0.69      0.56     17755
        True       0.88      0.75      0.81     54167

    accuracy                           0.73     71922
   macro avg       0.68      0.72      0.69     71922
weighted avg       0.78      0.73      0.75     71922
"""

#Getting positive predictions from the first model for use later on, when bias is evaluated

positive_predictions = df1.loc[test_index].copy()

positive_predictions["y_pred"] = y_pred  #creates a column for the predicted value


approved_applications = positive_predictions[positive_predictions["y_pred"] == 1].copy() #isolates values predicted to be approved

print("number of approved applications:",len(approved_applications))



#Saving the model for use in the Flask app and in the two layer model stage

joblib.dump(logreg, "models/logistic.pkl")

model_approval = joblib.load("models/logistic.pkl")

Predicted_df = pd.read_csv("/workspaces/project/processed_datasets/NY2019.csv")








"""
___________________________________________________________
Now attempting Random Forest classification on dataset
___________________________________________________________

"""
print(X.columns.to_list())

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Splitting X and y for an RF test   

test_index_RF = X_test_RF.index

imputer = SimpleImputer(strategy="median")
X_train_RF = pd.DataFrame(imputer.fit_transform(X_train_RF))
X_test_RF = pd.DataFrame(imputer.transform(X_test_RF))


Model_RF = RandomForestClassifier(class_weight="balanced", random_state=42)

Model_RF.fit(X_train_RF, y_train_RF)

y_pred_RF = Model_RF.predict(X_test_RF)

accuracy = accuracy_score(y_test_RF, y_pred_RF)

print(f"accuracy: {accuracy:.4f}")


print("classification report:")
print(classification_report(y_test_RF, y_pred_RF) )

feature_importance = pd.DataFrame({
    "feature": X_train_RF.columns,
    "importance": Model_RF.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importance.head(20))

joblib.dump(Model_RF, "models/Applications_RF.pkl")


#Saving RF data
positive_predictions_RF = df1.loc[test_index_RF]

positive_predictions_RF["y_pred"] = y_pred_RF  #makes new column for predicted value

approved_applications_RF = positive_predictions_RF[positive_predictions_RF["y_pred"] == 1].copy() #isolates values predicted to be approved

print("number of approved applications:",len(approved_applications_RF))


directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "RF_test_Data.csv")
positive_predictions_RF.to_csv(output_file_path, index=False)

output_file_path = os.path.join(directory, "RF_approved_applications.csv")
approved_applications_RF.to_csv(output_file_path, index=False)


#saving approved predictions to dataset

directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

#Saving the testing data from the LR model for use when evaluating the model
output_file_path = os.path.join(directory, "approved_application_1st_model_LR.csv")
approved_applications.to_csv(output_file_path, index=False)


directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

#Saving the applicants predicted to be approved from the logistic regression model
output_file_path = os.path.join(directory, "LR_test_data.csv")
positive_predictions.to_csv(output_file_path, index=False)


joblib.dump(scaler, "models/FeatureScaler.pkl") #saving the scaler for use in the application
