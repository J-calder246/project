import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib



#Preparing a logistic regression classification model

df1 = pd.read_csv("processed_datasets/NY2019.csv")






print(df1.shape)
X = df1.drop(columns=['action_taken'])
y = df1['action_taken']







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
