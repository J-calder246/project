import flask  # type: ignore
from flask import Flask, render_template, request, redirect, flash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

"""
UPLOAD DATASETS
"""

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
