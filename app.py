from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("anomaly_model.h5", compile=False)

@app.route("/")
def home():
    return "Electricity Anomaly Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)
    result = int(prediction[0] > 0.5)

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


