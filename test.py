from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd
import joblib


"""
_______________________________
Flask app tests
_______________________________

"""
from routes import app
import unittest
from pathlib import Path




class FlaskTests(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()


# testing homepage
    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<h1>Mortgage Approval Model Demonstration</h1>", response.data)


    def test_invalid_application(self):
        response_input = self.client.post(
            '/input_data',
            data= {
                'loan_type': '1',
                'loan_amount': '-535000.0',  # invalid loan amount
                'income': '42000.0',
                'debt_to_income_ratio': '<20%',
                'interest_rate': '6.8',
                'property_value': '840500.0',
                'loan_term': '180',
                'loan_purpose': '1',
                'occupancy_type': 'Principal residence',	
                'derived_race': 'American Indian or Alaska Native',	
                'derived_sex': 'Joint',
                'applicant_age': '<25',	
                'negative_amortization': 'Negative amortization',	
                'rate_spread': '1'

            },
            follow_redirects=True
        )
        self.assertIn(
            response_input.status_code, [200, 400]
        )

    def test_missing_input(self):
        response_missing = self.client.post(
            '/input_data',
            data= {
                    'loan_type': '1',
                    'loan_amount': '535000.0',  # invalid loan amount
                    'income': '42000.0',
                    'debt_to_income_ratio': '<20%',
                    'interest_rate': '',
                    'property_value': '840500.0',
                    'loan_term': '180',
                    'loan_purpose': '1',
                     'occupancy_type': 'Principal residence',	
                    'derived_race': 'American Indian or Alaska Native',	
                    'derived_sex': 'Joint',
                    'applicant_age': '<25',	
                    'negative_amortization': 'Negative amortization',	
                    'rate_spread': '1'
            
            },
            follow_redirects=True
            )
        self.assertEqual(
        response_missing.status_code, [200, 400])

    def test_valid_input(self):
        response_valid = self.client.post(
            '/input_data',
            data= {
                            'loan_type': '1',
                            'loan_amount': '535000.0',  # invalid loan amount
                            'income': '42000.0',
                            'debt_to_income_ratio': '<20%',
                            'interest_rate': '5.4',
                            'property_value': '840500.0',
                            'loan_term': '180',
                            'loan_purpose': '1',
                             'occupancy_type': 'Principal residence',	
                            'derived_race': 'American Indian or Alaska Native',	
                            'derived_sex': 'Joint',
                            'applicant_age': '<25',	
                            'negative_amortization': 'Negative amortization',	
                            'rate_spread': '1'
                    
                    },
            follow_redirects=True
                    )
        self.assertIn(
        response_valid.status_code, [200, 400])
        



    





unittest.main(argv=[''], verbosity=2, exit=False)

model_LR_1st = joblib.load("models/logistic.pkl")   #original logistic regression model for applications.

df = pd.read_csv("processed_datasets/NY2019.csv")
