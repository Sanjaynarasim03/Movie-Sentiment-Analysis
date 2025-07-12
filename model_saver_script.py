'''
train_and_save_model.py

Trains the best sentiment classifier on IMDB reviews and saves the pipeline as sentiment_model.pkl
'''

import os
import joblib
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Load IMDB dataset
DATA_DIR = "aclImdb"
train_data = load_files(os.path.join(DATA_DIR, "train"), categories=["pos", "neg"], encoding="utf-8")

X_train, y_train = train_data.data, train_data.target

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.95, min_df=5, stop_words="english", ngram_range=(1, 2))),
    ("clf", MultinomialNB())
])

# Hyperparameter tuning
param_grid = {
    "clf__alpha": [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Save the best model
print("Best parameters:", grid.best_params_)
joblib.dump(grid.best_estimator_, "sentiment_model.pkl")
print("Model saved to 'sentiment_model.pkl'")
