import flask  # type: ignore
from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import pandas as pd
from config import Config
from databases import input_collection


app = Flask(__name__)
app.secret_key = 'your_secret_key'


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
app = Flask(__name__)
app.secret_key = 'my_secret_key'

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/input_data")
def add_data():
    return render_template('input_data.html')

@app.route('/input_data', methods=['POST'])
def add_application():
        id = request.form.get("id")
        loan_type = request.form.get("customer_id")
        loan_amount = request.form.get("loan_amount")
        income = request.form.get("income")
        debt_to_income_ratio = request.form.get("debt_to_income_ratio")
        loan_to_value_ratio = request.form.get("loan_to_value_ratio")
        interest_rate = request.form.get("interest_rate")
        property_value = request.form.get("property_value")
        loan_term = request.form.get("loan_term")
        loan_type = request.form.get("loan_type")
        loan_purpose = request.form.get("loan_purpose")
        occupancy_type = request.form.get("occupancy_type")
        derived_race  = request.form.get("derived_race")
        derived_sex = request.form.get("derived_sex")
        applicant_age = request.form.get("applicant_age")
        negative_amortization = request.form.get("negative_amortization")
        rate_spread = request.form.get("rate_spread")

        application_data = {
            id : id,
            loan_type: loan_type,
            loan_amount: loan_amount,
            income: income,
            debt_to_income_ratio: debt_to_income_ratio,
            loan_to_value_ratio: loan_to_value_ratio,
            interest_rate: interest_rate,
            property_value: property_value,
            loan_term: loan_term,
            loan_type: loan_type,
            loan_purpose: loan_purpose,
            occupancy_type: occupancy_type,
            derived_race: derived_race,
            derived_sex: derived_sex,
            applicant_age: applicant_age,
            negative_amortization: negative_amortization,
            rate_spread: rate_spread,


        }

        #inserting data
        input_collection.insert_one(application_data)
        flash("Application successfully stored.", "success")




