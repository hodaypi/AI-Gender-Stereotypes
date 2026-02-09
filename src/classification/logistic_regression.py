import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import os
from collections import Counter
import itertools
import numpy as np

def train_optimized_baseline(data_dir, model_dir):
    print("--- 1. Loading Training Data ---")
    train_df = pd.read_pickle(os.path.join(data_dir, "processed/train_data.pkl"))
    X_train_text = train_df['text_logreg']
    y_train = train_df['label']

    print("\n--- Analyzing Vocabulary ---")
    # check unique words in the voabulary 
    all_words = list(itertools.chain.from_iterable([text.split() for text in X_train_text]))
    word_counts = Counter(all_words)
    total_unique_words = len(word_counts)
    
    print(f"Total unique words in training set: {total_unique_words}")
    
    # decide the dictionary length
    MAX_FEATURES = 10000 
    print(f"Selecting top {MAX_FEATURES} features for TF-IDF.")

    # ==========================================
    # TF-IDF
    # ==========================================
    print("\n--- 2. Feature Extraction (TF-IDF) ---")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=5, max_df=0.95)
    X_train_vec = tfidf.fit_transform(X_train_text)
    
    print("TF-IDF Vectorization complete.")
    
    # ==========================================
    # Logistic regression
    # ==========================================
    print("\n--- 3. Hyperparameter Tuning with Cross-Validation ---")
    base_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='liblinear')


    param_grid = {
        'penalty': ['l1', 'l2'],    
        'C': [0.1, 1, 5, 10]           
    }

    # define Grid Search
    # cv=5 -> for Cross Validation
    # scoring='f1_macro' 
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5, 
        scoring='f1', 
        verbose=1,
        n_jobs=-1
    )

    print(f"Starting Grid Search (testing {len(param_grid['penalty']) * len(param_grid['C'])} combinations)...")
    grid_search.fit(X_train_vec, y_train)
    
    # results
    print("\n--- Optimization Results ---")
    print(f"Best Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score (F1): {grid_search.best_score_:.4f}")

    # find the best model
    best_model = grid_search.best_estimator_

    # save model
    print("\n--- 4. Saving Model & Vectorizer ---")
    joblib.dump(best_model, os.path.join(model_dir, "logistic_regression.pkl"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    print("Saved successfully.")

def main():
    DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"

    train_optimized_baseline(DATA_DIR,MODEL_DIR)