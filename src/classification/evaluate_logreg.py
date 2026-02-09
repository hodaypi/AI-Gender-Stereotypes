import pandas as pd
import joblib
import os
import sys
sys.path.append('/content') 
from src.classification.metrics import (
    print_metrics_report, 
    plot_confusion_matrix, 
    plot_advanced_graphs, 
    analyze_errors
)

def evaluate_logreg(data_dir,model_dir, data_type='val'):
    # load data and model
    print("Loading LogReg Data...")
    df = pd.read_pickle(os.path.join(data_dir, f"processed/{data_type}_data.pkl"))
    model = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    
    X_text = df['text_logreg'] 
    y_true = df['label']
    
    # prediction
    X_vec = tfidf.transform(X_text)
    y_pred = model.predict(X_vec)
    y_prob = model.predict_proba(X_vec)[:, 1]
    
    # call metric to evaluate prediction results
    print_metrics_report(y_true, y_pred, y_prob)
    plot_confusion_matrix(y_true, y_pred)
    plot_advanced_graphs(y_true, y_prob)
    analyze_errors(df, y_true, y_pred, y_prob, text_col='text_logreg')

def main():
  DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
  MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"
  evaluate_logreg(DATA_DIR, MODEL_DIR)