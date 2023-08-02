from flask import Flask, request, render_template, jsonify
from flask import Response
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__)
app= application

scaler= pickle.load(open("/config/workspace/pickled_models/standard_scaler.pkl", "rb"))
model= pickle.load(open("/config/workspace/pickled_models/model_for_prediction.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

# for the prediction of single new datapoint
@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="POST":

        # we retrieve all the data inputed in form and store in respective variables
        Pregnancies= int(request.form.get("Pregnancies"))
        Glucose= int(request.form.get("Glucose"))
        BloodPressure= int(request.form.get("BloodPressure"))
        SkinThickness= int(request.form.get("SkinThickness"))
        Insulin= int(request.form.get("Insulin"))
        BMI= float(request.form.get("BMI"))
        DiabetesPedigreeFunction= float(request.form.get("DiabetesPedigreeFunction"))
        Age= int(request.form.get("Age"))

        X_new= [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        X_new_scaled= scaler.transform(X_new)
        y_new_pred= model.predict(X_new)         # it will come as an array of either 0 or 1

        if y_new_pred[0]== 0:
            result="Non-diabetic"
        else:
            result="Diabetic"

        return render_template("result.html", results=result)

        

    else:
        return render_template("home.html")



if __name__=="__main__":
    app.run(host="0.0.0.0")
