import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier




df = pd.read_csv("processed_datasets/SingleValue_NY2019.csv")

print(df.head())




print(df.isna().sum().sort_values(ascending=False))

#drop original list price and used UPB as a stand in for loan amount
df = df.drop(columns=["original list price"])
#dropping nas (change later if appropriate)
df = df.dropna()

print(df.head())

X = df.drop(columns=['loan id', 'Current Loan Delinquency Status', 'delinquent'])  #list price dropped due to missing values (ammend later if possible)
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

       False       0.86      0.57      0.69      1698
        True       0.27      0.63      0.38       434

    accuracy                           0.58      2132
   macro avg       0.56      0.60      0.53      2132
weighted avg       0.74      0.58      0.62      2132

Accuracy: 0.58

Model performs well on non-delinquent cases but poorly on delinquent ones

ratio -   4:1 (false-true)
"""

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

"""
Results
---------------
accuracy: 0.7111
classification report:
              precision    recall  f1-score   support

       False       0.81      0.83      0.82      1698
        True       0.26      0.24      0.25       434

    accuracy                           0.71      2132
   macro avg       0.54      0.53      0.53      2132
weighted avg       0.70      0.71      0.70      2132

Evaluation
Random forest, despite being more accurate overall struggled even more with mortgages that were delinquent than the logistic regression model. 

"""

feature_importance = pd.DataFrame({
    "feature": X_train_RF.columns,
    "importance": Model_RF.feature_importances_
}).sort_values("importance", ascending=False)

print(feature_importance.head(20))

joblib.dump(Model_RF, "models/Fannie_mae_RF.pkl")




joblib.dump(logreg, "models/FMlogistic.pkl")
joblib.dump(scaler, "models/FMscaler.pkl")




"""
Random Forest
"""