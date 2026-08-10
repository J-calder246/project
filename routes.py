import flask  # type: ignore
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import pandas as pd
from config import Config
from databases import input_collection, result
import joblib



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


@app.route("/input_data", methods=["GET", "POST"])
def input_data():
    return render_template('input_data.html')


@app.route('/input_data', methods=['POST'])
def input_data_post():
                    "applicant_id" == request.form.get("id"),
                    "loan_type" == request.form.get("loan_type"),
                    "loan_amount" == request.form.get("loan_amount"),
                    "income" == request.form.get("income"),
                    "debt_to_income_ratio" == request.form.get("debt_to_income_ratio"),
                    "loan_to_value_ratio" == request.form.get("loan_to_value_ratio"),
                    "interest_rate" == request.form.get("interest_rate"),
                    "property_value" == request.form.get("property_value"),
                    "loan_term" == request.form.get("loan_term"),
                    "loan_purpose" == request.form.get("loan_purpose"),
                    "occupancy_type" == request.form.get("occupancy_type"),
                    "derived_race" == request.form.get("derived_race"),
                    "derived_sex" == request.form.get("derived_sex"),
                    "applicant_age" == request.form.get("applicant_age"),
                    "negative_amortization" == request.form.get("negative_amortization"),
                    "rate_spread" == request.form.get("rate_spread")

                    application_data = {
                        "applicant_id": str("id"),
                        "loan_type": str(("loan_type")),
                        "loan_amount": str(("loan_amount")),
                        "income": str(("income")),
                        "debt_to_income_ratio": str("debt_to_income_ratio"),
                        "loan_to_value_ratio": str("loan_to_value_ratio"),
                        "interest_rate": str("interest_rate"),
                        "property_value": str("property_value"),
                        "loan_term": str("loan_term"),
                        "loan_purpose": str(("loan_purpose")),
                        "occupancy_type": str(("occupancy_type")),
                        "derived_race": str("derived_race"),
                        "derived_sex": str("derived_sex"),
                        "applicant_age": str("applicant_age"),
                        "negative_amortization": str(("negative_amortization")),
                        "rate_spread": str("rate_spread")
                    }      
                    #inserting data
                    input_collection.insert_one(application_data)
                    flash("Application successfully stored.", "success")
                    return redirect(url_for('datalist'))

             
                   
                
            

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
loan_type                  int64
loan_purpose               int64
occupancy_type             int64
derived_race              object
derived_sex               object
applicant_age             object
negative_amortization      int64
rate_spread               object
"""

