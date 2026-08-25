import flask  # type: ignore
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import pandas as pd
from config import Config
from databases import input_collection
import joblib
from bson.objectid import ObjectId


"""
Models for application
-----------------------
"""

#original models

model_LR_1st = joblib.load("models/logistic.pkl")   #original logistic regression model for applications.

model_LR_fm = joblib.load("models/FMlogistic.pkl")  # original fannie mae model for delinquency.


#mitigated models


#pre-processing
correlation_remover = joblib.load("mitigated_models/correlation_remover.pkl")
no_sensitive_features = joblib.load("mitigated_models/removed_sensitive_features.pkl")

#in-processing
exponentiated_gradient = joblib.load("mitigated_models/exponentiated_gradient_LR.pkl")

#post processing
threshold_optimiser_race = joblib.load("mitigated_models/threshold_optimiser_race_LR.pkl")
threshold_optimiser_age = joblib.load("mitigated_models/thresholded_optimiser_age_LR.pkl")



"""
UPLOAD DATASETS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method --'POST':
        file = request.files['file']
        if file and filename.endswith('.csv'):
            df = pd.read_csv(file)
            records = df.to_dict(orient='records')
            collection.insert_many(records)
            flash ("Data successfully uploaded", "success")
            return redirect('view')

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


@app.route("/input_data", methods=["GET"])
def input_data():
    return render_template('input_data.html')


@app.route('/input_data', methods=['POST'])
def input_data_post():
    try:
        loan_type = request.form.get("loan_type"),
        loan_amount = request.form.get("loan_amount"),
        income = request.form.get("income"),
        debt_to_income_ratio = request.form.get("debt_to_income_ratio"),
        loan_to_value_ratio = request.form.get("loan_to_value_ratio"),
        interest_rate = request.form.get("interest_rate"),
        property_value = request.form.get("property_value"),
        loan_term = request.form.get("loan_term"),
        loan_purpose = request.form.get("loan_purpose"),
        occupancy_type = request.form.get("occupancy_type"),
        derived_race = request.form.get("derived_race"),
        derived_sex = request.form.get("derived_sex"),
        applicant_age = request.form.get("applicant_age"),
        negative_amortization = request.form.get("negative_amortization"),
        rate_spread = request.form.get("rate_spread")

        application_data = {
            "loan_type": float((loan_type)),
            "loan_amount": float((loan_amount)),
            "income": float((income)),
            "debt_to_income_ratio": (debt_to_income_ratio),
            "loan_to_value_ratio": (loan_to_value_ratio),
            "interest_rate": (interest_rate),
            "property_value": (property_value),
            "loan_term": (loan_term),
            "loan_purpose": int((loan_purpose)),
            "occupancy_type": int((occupancy_type)),
            "derived_race": (derived_race),
            "derived_sex": (derived_sex),
            "applicant_age": (applicant_age),
            "negative_amortization": int((negative_amortization)),
            "rate_spread": (rate_spread)
                    }

                    #inserting data
        input_collection.insert_one(application_data)
        flash("Application successfully stored.", "success")
        return redirect(url_for("datalist"))
    except Exception as error:
        print(type(error).__name__, error)
        flash(f"Insert failed: {error}", "danger")
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

@app.route("/process_application", methods=["POST"])
def process_data():
    df= pd.read_html(selected_application)
    #Preprocessing data with a similar process as in processing_NY_loan_application 
    #but without unecessary part
    columns_names = [
        "_id",
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
    df = pd.read_html(df,
        header='columns_names'
        )

    cols_to_encode = ["derived_race", "derived_sex", "loan_type", "loan_purpose", "negative_amortization",
                    "applicant_age", "occupancy_type"]
    df = pd.get_dummies(df, cols_to_encode, drop_first=True)
    print(df.head)
    return render_template("models_home.html")
    
    

                   
                
            

"""
approved                    bool
action_taken               int64
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


"""
Modelling with Flask
_____________________

Guides: https://mljourney.com/how-to-deploy-machine-learning-models-using-flask/
https://www.geeksforgeeks.org/machine-learning/deploy-machine-learning-model-using-flask/ 

"""
"""
Section which converts input data into the correct values and gets dummies so the models can be used
"""




#request for (data) by ID, call it features and run model in app
@app.route('/modelling_home/<applicant_id>', methods=['POST'])
def convert_data(applicant_id):
    id_to_model = session.get

    
        

@app.route('/logistic')
def LR_page():
    return render_template('logistic.html')

