"""

Section that defines the routes and functions in the app where the models developed are deployed
"""

import flask  # type: ignore
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import pandas as pd
from config import Config
from databases import input_collection, input_collection_encoded
import joblib
from bson.objectid import ObjectId
import os
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



"""
Logistic regression models
___________________________

Note: Random forest models are too large to load here and therefore they cannot be used in the Flask app
"""




app = Flask(__name__, template_folder='Templates')
app.secret_key = 'my_secret_key'

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('Home.html')

@app.route("/models_home")
def models_home():
     return render_template("models_home.html")

@app.route("/datalist")
def datalist():
     data = list(input_collection.find())
     return render_template("datalist.html", records=data)


@app.route("/inputdata", methods=["GET"])
def input_data():
    return render_template('input_data.html')


@app.route('/input_data', methods=['POST'])
def input_data_post():
    try:
        loan_type = request.form.get("loan_type")
        loan_amount = request.form.get("loan_amount")
        income = request.form.get("income")
        debt_to_income_ratio = request.form.get("debt_to_income_ratio")
        loan_to_value_ratio = request.form.get("loan_to_value_ratio")
        interest_rate = request.form.get("interest_rate")
        property_value = request.form.get("property_value")
        loan_term = request.form.get("loan_term")
        loan_purpose = request.form.get("loan_purpose")
        occupancy_type = request.form.get("occupancy_type")
        derived_race = request.form.get("derived_race")
        derived_sex = request.form.get("derived_sex")
        applicant_age = request.form.get("applicant_age")
        negative_amortization = request.form.get("negative_amortization")
        rate_spread = request.form.get("rate_spread")

        application_data = {
            "loan_type": ((loan_type)),
            "loan_amount": ((loan_amount)),
            "income": ((income)),
            "debt_to_income_ratio": (debt_to_income_ratio),
            "loan_to_value_ratio": (loan_to_value_ratio),
            "interest_rate": (interest_rate),
            "property_value": (property_value),
            "loan_term": (loan_term),
            "loan_purpose": ((loan_purpose)),
            "occupancy_type": ((occupancy_type)),
            "derived_race": (derived_race),
            "derived_sex": (derived_sex),
            "applicant_age": (applicant_age),
            "negative_amortization": ((negative_amortization)),
            "rate_spread": (rate_spread)
                    }

                    #inserting data
        input_collection.insert_one(application_data)
        flash("Application successfully stored.", "success")

        return redirect(url_for("datalist"))
    except Exception as error:
        print(type(error).__name__, error)
        flash(f"Insert failed: {error}", "danger")
        return redirect(url_for("home"))


@app.route("/delete/<string:select_id>", methods=["POST"])
def delete(select_id):
    try:
         oid = ObjectId(select_id)
         input_collection.delete_one({"_id": oid})
         flash("Application deleted", "success")
         return redirect(url_for("datalist"))
    except Exception as error:
        flash("user has not been deleted", "danger")
        return redirect(url_for("datalist"))




@app.route("/select_id", methods=['POST'])
def select_id():
    selected_id = request.form.get("select_id")
    session["selected_application"] = selected_id
    return redirect(url_for("selected_application"))

@app.route("/selected_application")
def selected_application():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)

    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application

    if not applicant_data:
        return "applicant not found"
    
    

    return render_template("selected_application.html", applicant_data = applicant_data)




#defining a process for processing the data into a form the models are used to

def process_data(applicant_data):
    
    df = pd.DataFrame([applicant_data])
    
    print(df.head())

    #drop id
    df = df.drop(columns=["_id"])
    
    #processing dataframe

    
    
    columns_names = [
                "loan_type",
                "loan_amount",
                "income",
                "debt_to_income_ratio",
                "loan_to_value_ratio",
                "interest_rate",
                "property_value",
                "loan_term",
                "loan_purpose",
                "occupancy_type",
                "derived_race",
                "derived_sex",
                "applicant_age",
                "negative_amortization",
                "rate_spread"]
    print(df.dtypes)

    df = pd.DataFrame(df, columns= columns_names)

    #converting occupancy type into int values

    occupancy_dict = {'Principal residence': 1,
    'Second residence': 2,
    'Investment property': 3}

    df["occupancy_type"] = df["occupancy_type"].map(occupancy_dict)

    #doing the same for negative amortization

    amortisation_dict = {'Negative amortization': 1,
        'No negative amortization': 2,
        'Exempt': 1111}
    
    df["negative_amortization"] = df["negative_amortization"].map(amortisation_dict)


    """
Current datatypes

[1 rows x 16 columns]
loan_type                object
loan_amount              object
income                   object
debt_to_income_ratio     object
loan_to_value_ratio      object
interest_rate            object
property_value           object
loan_term                object
loan_purpose             object
occupancy_type           object
derived_race             object
derived_sex              object
applicant_age            object
negative_amortization    object
rate_spread              object


        Datatypes the models are used to
        
    
    
    loan_type                  int64
    loan_amount              float64
    income                   float64
    debt_to_income_ratio      object
    loan_to_value_ratio       object
    interest_rate             object
    property_value            object
    loan_term                 object
    loan_purpose               int64
    occupancy_type             int64
    derived_race              object
    derived_sex               object
    applicant_age             object
    negative_amortization      int64
    rate_spread               object
    
        """
    #converting datatypes

    data_dict = {'loan_type': int,
        'loan_amount':              float,
        'income'      :             float,
        'debt_to_income_ratio':      object,
        'loan_to_value_ratio':       object,
        'interest_rate'       :      object,
        'property_value'       :     object,
        'loan_term'             :    object,
        'loan_purpose'           :    int,
        'occupancy_type'          :   int,
        'derived_race'            : object,
        'derived_sex'               :object,
        'applicant_age'      :      object,
        'negative_amortization':      int,
        'rate_spread'           :    object}
    df = df.astype(data_dict)

    print(df.dtypes)

    #convert string values into numeric values where appropriate

    numeric_columns = [
        "loan_type",
        "loan_amount",
        "income",
        "loan_to_value_ratio",
        "interest_rate",
        "property_value",
        "loan_term",
        "loan_purpose",
        "occupancy_type",
        "negative_amortization",
        "rate_spread"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

            
    
        
    debt_to_income_mode = {
            "<20%": 10,   
            "20%-<30%": 25,
            "30%-<36%": 33,   
            "36%": 36,
            "37%": 37,
            "38%": 38,
            "39%": 39,
            "40%": 40,
            "41%": 41,
            "42%": 42,
            "43%": 43,
            "44%": 44,
            "45%": 45,
            "46%": 46,
            "47%": 47,
            "48%": 48,
            "49%": 49,
            "50%-60%": 55,
            ">60%": 65,  
            "Exempt": None
            }
        
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(debt_to_income_mode)


        #as we are encoding a single row which won't have all the datapoints needed to encode, we can use "all categories" to still create these columns
    all_categories_race = ['2 or more minority races',
                           'American Indian or Alaska Native', 'Asian',
                           'Black or African American', 'Joint',
                           'Native Hawaiian or Other Pacific Islander', 'White']
    all_categories_gender = ['Female', 'Joint', 'Male']
    all_categories_loan_type = ['1', '2', '3', '4']
    all_categories_loan_purpose = ['1', '2', '4', "5", '31', '32']
    all_categories_negative_amortiziation = ['1', '2', '1111']
    all_categories_applicant_age = ['25-34', '35-44', '45-54', '55-64', '65-74', '<25', '>74',]
    all_categories_occupancy_type = ['1', '2', '3']


    df['derived_race'] = pd.Categorical(df['derived_race'], categories=all_categories_race)
    df['derived_sex'] = pd.Categorical(df['derived_sex'], categories=all_categories_gender)
    df['loan_type'] = pd.Categorical(df['loan_type'], categories=all_categories_loan_type)
    df['loan_purpose'] = pd.Categorical(df['loan_purpose'], categories=all_categories_loan_purpose)
    df['negative_amortization'] = pd.Categorical(df['negative_amortization'], categories=all_categories_negative_amortiziation)
    df['applicant_age'] = pd.Categorical(df['applicant_age'], categories=all_categories_applicant_age)
    df['occupancy_type'] = pd.Categorical(df['occupancy_type'], categories=all_categories_occupancy_type)
    
        
    cols_to_encode = ["derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization",
                            "applicant_age", "occupancy_type"]
    
    #combining the dummies with the original data like in the original processing file so we can use these features later
    dummies = pd.DataFrame(pd.get_dummies(df, columns=cols_to_encode, drop_first=False))

    dummies = dummies.drop(columns=['loan_amount', 'income', 
                                'debt_to_income_ratio', 'loan_to_value_ratio', 'interest_rate', 
                                'property_value', 'loan_term', 'rate_spread'])

    #combines the two
    df = pd.concat([df, dummies], axis= 1)

    print(df.head())
    print(df.columns.to_list)

    df = pd.DataFrame(df)


    

#useful bit for checking the data
    directory = "processed_datasets" 
    os.makedirs(directory, exist_ok=True)
    
    output_file_path = os.path.join(directory, "experimental_flask_df")
    df.to_csv(output_file_path, index=False)

       
    return df




"""
Modelling with Flask
_____________________

Guides: https://mljourney.com/how-to-deploy-machine-learning-models-using-flask/
https://www.geeksforgeeks.org/machine-learning/deploy-machine-learning-model-using-flask/ 

"""

#define the copied dummies so it's easier to drop them
cols_to_encode = ["derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization",
                            "applicant_age", "occupancy_type"]  
@app.route("/logistic")
def logistic():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)
        
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
        
    if not applicant_data:
        return "applicant not found"

    model_LR_1st = joblib.load("models/logistic.pkl")   #original logistic regression model for applications.

    df = process_data(applicant_data)
    X = pd.DataFrame(df)

    X = X.drop(columns=["interest_rate", "rate_spread"])
    X = X.drop(columns=cols_to_encode)

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    prediction = model_LR_1st.predict(X)[0]
    if prediction == 1:
        prediction = "approved"
    else:
        prediction = "denied"
    probability = model_LR_1st.predict_proba(X)[0, 1]
    return render_template("logistic.html", prediction_text= f"model prediction: {prediction}",
                           probability_text=f"prediction probability: {probability}")



    
@app.route("/mitigation") 
def mitigation():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)
                
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
                
    if not applicant_data:
            return "applicant not found"
    return render_template("mitigation.html")  

@app.route("/preprocessing")   #NEED TO CHANGE THIS SECTION. AT THE MOMENT IT IS INCORRECT AS THESE MODELS ARE TIED TO THE DATA NOT THE MODEL ITSELF
def preprocessing():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)
            
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
            
    if not applicant_data:
            return "applicant not found"

    sensitive_features = ['applicant_age_35-44',
                    'applicant_age_45-54', 'applicant_age_55-64', 'applicant_age_65-74',
                    'applicant_age_<25', 'applicant_age_>74',  'derived_race_2 or more minority races',
                     'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
                    'derived_race_Black or African American', 'derived_race_Joint',
                    'derived_race_Native Hawaiian or Other Pacific Islander',
                    'derived_race_White', 'derived_sex_Female', 'derived_sex_Joint',
                    'derived_sex_Male', 'applicant_age_25-34']

    CR = joblib.load("mitigated_models/correlation_remover.pkl") #this is used to transform the data, not to acutally model
    model_LR_1st = joblib.load("models/logistic.pkl")  #going to use this to model because it already exists
    no_sensitive_features = joblib.load("mitigated_models/removed_sensitive_features.pkl")

    
    df = process_data(applicant_data)
    X = pd.DataFrame(df)
    
    X = X.drop(columns=["interest_rate", "rate_spread"])
    X = X.drop(columns=cols_to_encode)
    X = X.drop(columns=sensitive_features)

    X = np.reshape(X, (22, 1)) #reshape to 2d array
    X = pd.DataFrame(X)

    print(X.head())

    X = scaler.fit_transform(X)

    X = np.reshape(X, (22, 1))
        
    X = pd.DataFrame(X)
    X_CR = pd.DataFrame(CR.fit_transform(X))
    X_CR = pd.DataFrame(X_CR, columns=X_CR.columns)
    prediction_correlation_remover = model_LR_1st.predict(X_CR)[0]
    prob_correlation_remover = model_LR_1st.predict_proba(X_CR)[0, 1]


    
    prediction_no_sensitive_features = no_sensitive_features.predict(X)[0]
    probability_no_sensitive_features = no_sensitive_features.predict_proba(X)[0, 1]
    return render_template("preprocessing.html", prediction_text_CR=f"model prediction: {prediction_correlation_remover}",
                               probability_text_CR=f"prediction probability: {prob_correlation_remover}",
                               prediction_textNSF=f"model prediction: {prediction_no_sensitive_features}",
                               probability_text_NSF=f"prediction probability: {probability_no_sensitive_features}")





@app.route("/inprocessing")
def inprocessing():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)
            
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
            
    if not applicant_data:
            return "applicant not found"

    #loading the model
    exponentiated_gradient = joblib.load("mitigated_models/exponentiated_gradient_LR.pkl")
    
    df = process_data(applicant_data)
    X = pd.DataFrame(df)
    
    X = X.drop(columns=[ "interest_rate", "rate_spread"]) #model not trained on these
    X = X.drop(columns= cols_to_encode)

    X = np.reshape(X, (39, 1)) #reshape to 2d array
    X = pd.DataFrame(X)
    print(X.head)
    X = scaler.fit_transform(X)
        
    
    prediction_exponentiated_gradient = exponentiated_gradient.predict(X)[0]
    if prediction_exponentiated_gradient == 1:
        prediction_exponentiated_gradient = "approved"
    else:
        prediction_exponentiated_gradient = "denied"
    return render_template("in_processing.html", prediction_XG_text=f"model prediction: {prediction_exponentiated_gradient}")


@app.route("/postprocessing")
def postprocessing():
    selected_id = session.get("selected_application")
    print("selected ID:", selected_id)
            
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
            
    if not applicant_data:
        return "applicant not found"

    threshold_optimiser_race = joblib.load("mitigated_models/threshold_optimiser_race_LR.pkl")
    threshold_optimiser_age = joblib.load("mitigated_models/thresholded_optimiser_age_LR.pkl")
    
    df = process_data(applicant_data)
    X = pd.DataFrame(df)
    print(X.columns.to_list)

    A = df['derived_race']
    B = df['applicant_age']
    
    X = X.drop(columns=["interest_rate", "rate_spread"])
    X = X.drop(columns= cols_to_encode)

    X = scaler.fit_transform(X)
        
    
    prediction_race = threshold_optimiser_race.predict(X, sensitive_features=A, random_state=42)[0]
    if prediction_race == 1:
        prediction_race = "approved"
    else:
        prediction_race = "denied"

    prediction_age = threshold_optimiser_age.predict(X, sensitive_features=B, random_state=42)[0]
    if prediction_age == 1:
        prediction_age = "approved"
    else:
        prediction_age = "denied"

    print(prediction_race)
    print(prediction_age)
    return render_template("post_processing.html", prediction_text_TOR=f"model prediction: {prediction_race}",
                               prediction_text_TOA=f"model prediction: {prediction_age}")

                



#processing for delinquency with a similar process as to in the 2nd layer modelling file

def process_FM(applicant_data):
     df = pd.DataFrame([applicant_data])

     print(df.head())

    #processing dataframe

    
     columns_names = [
                "loan_type",
                "loan_amount",
                "income",
                "debt_to_income_ratio",
                "loan_to_value_ratio",
                "interest_rate",
                "property_value",
                "loan_term",
                "loan_purpose",
                "occupancy_type",
                "derived_race",
                "derived_sex",
                "applicant_age",
                "negative_amortization",
                "rate_spread"]
     print(df.dtypes)

     df = (df.rename(columns=dict(zip(df.columns,columns_names))))

     df = pd.DataFrame(df)

     print(df.columns.to_list)
     
     df = df[['loan_amount', 'debt_to_income_ratio', 'loan_to_value_ratio', 'loan_term', 'interest_rate']]

     debt_to_income_mode = {
                 "<20%": 10,   
                 "20%-<30%": 25,
                 "30%-<36%": 33,   
                 "36%": 36,
                 "37%": 37,
                 "38%": 38,
                 "39%": 39,
                 "40%": 40,
                 "41%": 41,
                 "42%": 42,
                 "43%": 43,
                 "44%": 44,
                 "45%": 45,
                 "46%": 46,
                 "47%": 47,
                 "48%": 48,
                 "49%": 49,
                 "50%-60%": 55,
                 ">60%": 65,  
                 "Exempt": None
                 }
             
     df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(debt_to_income_mode)

     print(df["debt_to_income_ratio"])


    
    #converting datatypes

     data_dict = {
        'loan_amount':              float,
        'debt_to_income_ratio':      object,
        'loan_to_value_ratio':       object,
        'interest_rate'       :      object,
        'loan_term'             :    object,}
     df = df.astype(data_dict)

     print(df.dtypes)

            
     df_fm = pd.DataFrame(df, columns= columns_names)

     numeric_columns = [
             "loan_amount",
             "loan_to_value_ratio",
             "interest_rate",
             "loan_term",
         ]
     
     for col in numeric_columns:
             df[col] = pd.to_numeric(
                 df[col],
                 errors="coerce"
             )
    

     Column_names = [
    "original UPB",  #represent original unpaid principal balance (i.e. loan amount)
    "debt to income",
    "original LTV ratio", #loan to value
    "original loan term",
    "original interest rate",
]
     df_fm = df_fm.rename(columns=dict(zip(df_fm.columns,Column_names)))



        
@app.route("/delinquency")
def delinquency():
     selected_id = session.get("selected_application")
     print("selected ID:", selected_id)
        
     applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})  #finds data according to application
        
     if not applicant_data:
        return "applicant not found"

     X = process_FM(applicant_data)

     


     #loading model and original scaler

     model_LR_fm = joblib.load("models/FMlogistic.pkl")  # original fannie mae model for delinquency.
     scaler_fm = joblib.load("models/FMscaler.pkl")
     model_fm_RF = joblib.load("models/Fannie_mae_RF.pkl")  # original fannie mae model for delinquency.

     X = scaler_fm.fit_transform(X)
     
     prediction_delinquency_LR = model_LR_fm.predict(X)[0]
     probability_delinquency_LR = model_LR_fm.predict_proba(X)[:, 1]

     
     
     prediction_delinquency_RF = model_fm_RF.predict(X)[0]
     probability_delinquency_RF = model_fm_RF.predict_proba(X)[:, 1]
    
     return render_template("delinquency.html", prediction_LR_text= f"model prediction: {prediction_delinquency_LR}",
                                    probability_LR_text= f"prediction probability: {probability_delinquency_LR}", 
                                    prediction_text_RF= f"model prediction: {prediction_delinquency_RF}",
                                         probability_text_RF= f"prediction probability: {probability_delinquency_RF}")




@app.route("/all_predictions")
def all_predictions(prediction, probability, prediction_correlation_remover, prob_correlation_remover, prediction_no_sensitive_features, probability_no_sensitive_features, prediction_exponentiated_gradient, prediction_delinquency_LR, probability_delinquency_LR, prediction_race, prediction_age,  prediction_delinquency,
                    probability_delinquency, prediction_delinquency_RF, probability_delinquency_RF):
    selected_id = session.get("selected_application")
    applicant_data = input_collection.find_one({"_id": ObjectId(selected_id)})

    prediction = logistic(applicant_data)

    if not applicant_data:
        return "applicant not found"
    return render_template("all_predictions.html", prediction_text= f"model prediction: {prediction}",
                           probability_text=f"prediction probability: {probability}",  prediction_text_CR=f"model prediction: {prediction_correlation_remover}",
                               probability_text_CR=f"prediction probability: {prob_correlation_remover}",
                               prediction_textNSF=f"model prediction: {prediction_no_sensitive_features}",
                               probability_text_NSF=f"prediction probability: {probability_no_sensitive_features}",prediction_XG_text=f"model prediction: {prediction_exponentiated_gradient}", prediction_LR_text= f"model prediction: {prediction_delinquency_LR}",
                                    probability_LR_text= f"prediction probability: {probability_delinquency_LR}", prediction_text_TOR=f"model prediction: {prediction_race}",
                               prediction_text_TOA=f"model prediction: {prediction_age}"
                                    )
