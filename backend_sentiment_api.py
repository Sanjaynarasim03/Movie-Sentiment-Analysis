'''
backend_sentiment_api.py

Provides a Flask-based REST API to predict sentiment (positive/negative)
for a given movie review using the trained sentiment analysis model.
'''

from flask import Flask, request, jsonify
import joblib
import os
import traceback

# Flask app
app = Flask(__name__)

# Load saved pipeline (TF-IDF + trained classifier)
MODEL_PATH = 'sentiment_model.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model 'sentiment_model.pkl' not found. Please train and save it first.")

model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        review = data.get("review", "")

        if not review:
            return jsonify({"error": "Empty review provided."}), 400

        prediction = model.predict([review])[0]
        label = "positive" if prediction == 1 else "negative"

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/', methods=['GET'])
def index():
    return "Sentiment Analysis API is running. Use POST /predict with JSON {'review': 'your text'}"

if __name__ == '__main__':
    app.run(debug=True)
