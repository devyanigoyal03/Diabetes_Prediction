from wsgiref import simple_server
from flask import Flask, request, app, render_template
from flask import Response

import pickle
import bz2
import numpy as np
import pandas as pd

app = Flask(__name__)

app.config['DEBUG'] = True

scalar1 = bz2.BZ2File("Model\standardScalar.pkl", "rb")
scaler = pickle.load(scalar1)
pred = bz2.BZ2File("Model\modelForPrediction.pkl", "rb")
model = pickle.load(pred)


# Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

# Route for Single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':

        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = int(request.form.get('Glucose'))
        BloodPressure = int(request.form.get('BloodPressure'))
        SkinThickness = int(request.form.get('SkinThickness'))
        Insulin = int(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        new_data = scaler.transform(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        return render_template('single_prediction.html', result=result)

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")