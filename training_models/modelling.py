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

#produces duplicate columns that end with .1

print(df1.columns[df1.columns.str.endswith(".1")])



#duplicate columns removed from X, if possible find a better solution later
print(df1.shape)
X = df1.drop(columns=["approved", "action_taken", "derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization", 
                      "applicant_age", "occupancy_type"
       ])


#removing interest rate and rate spread to prevent data leakage as models are matching empty values here to unapproved applications
X = X.drop(columns=["interest_rate", "rate_spread"])
y = df1['approved']

print(X.columns.to_list())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    

test_index = X_test.index   #save X values so they arent messed up when scaled later on

#standardising features with scaler (gives attributes equal weighting and influence)

#Imputing missing values (change later if this is not suitable)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#X_train = X_train.dropna()
#X_test = X_test.dropna()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising model

logreg = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print reports
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))


#Getting positive predictions from the first model for use in another

positive_predictions = df1.loc[test_index].copy()

positive_predictions["y_pred"] = y_pred


approved_applications = positive_predictions[positive_predictions["y_pred"] == 1].copy() 

print("number of approved applications:",len(approved_applications))

#X_positive = X_test.iloc[approved_applications].copy()
#y_positive =y_test.iloc[approved_applications].copy() #locates and copies data from rows


# Saving predictions into new dataset

joblib.dump(logreg, "models/logistic.pkl")

model_approval = joblib.load("models/logistic.pkl")

Predicted_df = pd.read_csv("/workspaces/project/processed_datasets/NY2019.csv")

#prediction = model_approval.predict(Predicted_df.drop(columns=['action_taken']))






"""
___________________________________________________________
Now attempting Random Forest classification on dataset
___________________________________________________________

"""
print(X.columns.to_list())

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    

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

model_approval_RF = joblib.load("models/Applications_RF.pkl")

#Saving RF data
positive_predictions_RF = df1.loc[test_index_RF]

positive_predictions_RF["y_pred"] = y_pred_RF  #makes new column for predicted value


directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "RF_test_Data.csv")
positive_predictions_RF.to_csv(output_file_path, index=False)


#saving approved predictions to dataset

directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "approved_application_1st_model_LR.csv")
approved_applications.to_csv(output_file_path, index=False)


directory = "modelled_datasets" 
os.makedirs(directory, exist_ok=True)

output_file_path = os.path.join(directory, "LR_test_data.csv")
positive_predictions.to_csv(output_file_path, index=False)


