import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer



#Preparing a logistic regression classification model

df1 = pd.read_csv("/workspaces/project/processed_datasets/NY2024.csv")

#Imputing missing values (change later if this is not suitable)






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