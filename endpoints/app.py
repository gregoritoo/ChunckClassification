from flask import Flask, render_template, request, jsonify
import numpy as np
import sys

sys.path.append("../")
import os
from io import BytesIO
import pandas as pd
import requests
import xgboost as xgb
from core.pipeline import Pipeline

app = Flask(__name__)


PRETRAINED_SCALER = "../models/scaler.pkl"
PRETRAINED_CLASSIFIER = "../models/xgb_model.joblib"
PRETRAINED_ROBERTA = "../models/roberta_sentiment_classification"


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")

    if uploaded_file:
        csv_content = uploaded_file.read()
        csv_bytes_io = BytesIO(csv_content)
        df = pd.read_csv(csv_bytes_io)
        model = Pipeline(PRETRAINED_SCALER, PRETRAINED_CLASSIFIER, PRETRAINED_ROBERTA)
        prediction = model.predict(df)
        interpretations = model.interpret_bert()
        shap_values = model.interpret_shap()
        return jsonify(
            {
                "msg": "success",
                "data": model.data.iloc[0].tolist(),
                "data_name": list(model.data.keys()),
                "predictions": prediction.tolist(),
                "interpretations": interpretations,
                "shap_values": shap_values.tolist(),
            }
        )
    else:
        return jsonify({"msg": "error"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
