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

#Imputing missing values (change later if this is not suitable)

df1 = df1.dropna()




print(df1.shape)
X = df1.drop(columns=['action_taken'])
y = df1['action_taken']

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    


#standardising features with scaler (gives attributes equal weighting and influence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialising model

logreg = LogisticRegression(max_iter=1000)

#training
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


#Getting positive predictions from the first model for use in another

predictions = logreg.predict(X)

model_positives = predictions == 1

print(len(model_positives)  )

X_positive = X[model_positives]
y_positive =y[model_positives]


# Saving predictions into new dataset

joblib.dump(logreg, "models/logistic.pkl")

model = joblib.load("models/logistic.pkl")

Predicted_df = pd.read_csv("/workspaces/project/processed_datasets/NY2019.csv")

prediction = model.predict(Predicted_df.drop(columns=['action_taken']))
