# Fullstack Sentiment Analysis on IMDB Movie Reviews

This project performs **sentiment analysis** on the [IMDB Movie Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) using machine learning and provides a complete **web application** using **Flask**.

---

## 📁 Dataset Setup

Download and extract the dataset from:
[http://ai.stanford.edu/\~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

Your folder structure should look like:

```
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

---

## 🛠️ Dependencies

Install the required Python packages:

```bash
pip install flask scikit-learn matplotlib seaborn joblib
```

---

## 🔧 Step 1: Train and Save the Model

Use the provided script to train the model and save it as `sentiment_model.pkl`:

```bash
python train_and_save_model.py
```

This will:

* Load and process the dataset
* Train a Naive Bayes classifier with hyperparameter tuning
* Save the trained model as `sentiment_model.pkl`

---

## 🚀 Step 2: Launch the Fullstack Web App

Use the Flask app to serve both the frontend UI and prediction API:

```bash
python app.py
```

* Visit [http://localhost:5000](http://localhost:5000) to use the web UI.
* You can also make API calls to `POST /predict` with a JSON body like:

```json
{
  "review": "This movie was fantastic!"
}
```

---

## 📊 Output

* Live UI to enter reviews and get predictions
* REST API for backend integration
* Model accuracy and best parameters printed in terminal during training

---

## 💡 Future Improvements

* Upgrade to deep learning (LSTM, BERT)
* Add login & user feedback system
* Deploy on Render, Railway, or Streamlit Cloud

---

## 📌 Author

Sanjay Narasimhan

---



