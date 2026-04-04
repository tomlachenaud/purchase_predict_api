import pandas as pd
from flask import Flask, request, jsonify

from src.model import Model

app = Flask(__name__)
model = Model()


@app.route("/", methods=["GET"])
def home():
    # At beginning, we load model from MLflow
    return ("OK !", 200)


from io import StringIO
import json

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
        
    df = pd.read_json(StringIO(body))
    results = [int(x) for x in model.predict(df).flatten()]
    return (jsonify(results), 200)


if __name__ == "__main__":
    app.run(port=5000)
