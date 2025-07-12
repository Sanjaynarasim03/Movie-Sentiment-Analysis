'''
sentiment_analysis_imdb_advanced.py

Performs advanced sentiment analysis on the IMDB movie review dataset using TF-IDF features,
GridSearchCV for hyperparameter tuning, multiple classifiers (Naive Bayes, SVM, Logistic Regression),
and visualizes performance with confusion matrix heatmaps.

Requirements:
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

Usage:
  python sentiment_analysis_imdb_advanced.py --data_dir path/to/aclImdb
'''

import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_imdb_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print("Loading training data from", train_dir)
    train_data = load_files(train_dir, categories=['pos', 'neg'], encoding='utf-8')
    print(f"Loaded {len(train_data.data)} training samples")

    print("Loading test data from", test_dir)
    test_data = load_files(test_dir, categories=['pos', 'neg'], encoding='utf-8')
    print(f"Loaded {len(test_data.data)} test samples")

    return train_data.data, train_data.target, test_data.data, test_data.target


def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_train, y_train, X_test, y_test, labels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, title=f'{model.named_steps["clf"].__class__.__name__} Confusion Matrix')


def main(args):
    # Load and split data
    X_train_texts, y_train, X_test_texts, y_test = load_imdb_data(args.data_dir)

    # Common pipeline base
    tfidf = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english', ngram_range=(1, 2))

    # Models to try
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    for name, clf in models.items():
        print(f"\nTraining and Evaluating: {name}")
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])

        if name == 'Naive Bayes':
            param_grid = {'clf__alpha': [0.1, 1.0, 10.0]}
        elif name == 'SVM':
            param_grid = {'clf__C': [0.1, 1.0, 10.0]}
        else:  # Logistic Regression
            param_grid = {'clf__C': [0.1, 1.0, 10.0]}

        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_texts, y_train)
        best_model = grid.best_estimator_

        print("Best Parameters:", grid.best_params_)
        evaluate_model(best_model, X_train_texts, y_train, X_test_texts, y_test, labels=['neg', 'pos'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced IMDB Sentiment Analysis')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to aclImdb dataset root')
    args = parser.parse_args()
    main(args)
