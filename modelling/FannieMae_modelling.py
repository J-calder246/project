import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib





df = pd.read_csv("processed_datasets/SingleValue_NY2019.csv")

print(df.head())




print(df.isna().sum().sort_values(ascending=False))


df = df.drop(columns=["original list price"])
#dropping nas (change later if appropriate)
df = df.dropna()

print(df.head())

X = df.drop(columns=['Current Loan Delinquency Status', 'delinquent'])  #list price dropped due to missing values (ammend later if possible)
y = df['delinquent']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   #stratify  prevent overfitting and helps ensure that the data is representative and balanced   

#scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#inistialising

logreg = LogisticRegression(
    class_weight="balanced",   # handles imbalances
    random_state=42)

logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)



#getting accuracy  metrics
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
Results


Classification Report:
              precision    recall  f1-score   support
       False       0.87      0.58      0.70      1716
        True       0.27      0.64      0.38       416

    accuracy                           0.59      2132
   macro avg       0.57      0.61      0.54      2132
weighted avg       0.75      0.59      0.63      2132

Model performs well on non-delinquent cases but poorly on delinquent ones

ratio -   4:1 (false-true)
"""



joblib.dump(logreg, "models/FMlogistic.pkl")

model_FM = joblib.load("models/FMlogistic.pkl")