import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

sys.path.append('/content')

from src.classification.metrics import (
    print_metrics_report,
    plot_confusion_matrix,
    plot_advanced_graphs,
    analyze_errors
)

# save results of final evaluation to json so we can use them for visualization
def save_results_to_json(base_dir, model_name, metrics):
    results_path = os.path.join(base_dir, "results", "model_comparison.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    print(f">> Saving results for '{model_name}' to JSON...")
    data[model_name] = metrics

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=4)

#  eval BERT (DistilBERT + Large)
def evaluate_bert_model(data_dir, model_dir, model_folder_name, max_len=128, data_type='test'):
    print(f"\n=======================================================")
    print(f"   FINAL EVALUATION: {model_folder_name} ({data_type.upper()})")
    print(f"=======================================================")

    model_path = os.path.join(model_dir, model_folder_name)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_path = os.path.join(data_dir, f"processed/{data_type}_data.pkl")
    df = pd.read_pickle(data_path)

    texts = df['text_bert'].tolist()
    y_true = df['label']

    print(f"Predicting on {len(texts)} samples from {data_type} set...")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        def __len__(self):
            return len(self.encodings.input_ids)

    encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len)
    dataset = SimpleDataset(encodings)

    trainer = Trainer(model=model)
    raw_preds = trainer.predict(dataset)

    logits = torch.tensor(raw_preds.predictions)
    probs = softmax(logits, dim=1)[:, 1].numpy()
    y_pred = np.argmax(raw_preds.predictions, axis=1)

    # Metrics & Save
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, probs)
    except: auc = 0.5

    metrics_to_save = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "AUC": float(auc)
    }

    pretty_name = "BERT Large" if "large" in model_folder_name else "DistilBERT"
    save_results_to_json(data_dir, pretty_name, metrics_to_save)

    # reports
    print_metrics_report(y_true, y_pred, probs)
    plot_confusion_matrix(y_true, y_pred)

    print(f"\n--- Error Analysis for {model_folder_name} ---")
    analyze_errors(df, y_true, y_pred, probs, text_col='text_bert')

# Logistic Regression
def evaluate_logreg_model(data_dir, model_dir, data_type='test'):
    print(f"\n=======================================================")
    print(f"   FINAL EVALUATION: Logistic Regression ({data_type.upper()})")
    print(f"=======================================================")

    df = pd.read_pickle(os.path.join(data_dir, f"processed/{data_type}_data.pkl"))
    model = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    X_text = df['text_logreg']
    y_true = df['label']

    X_vec = tfidf.transform(X_text)
    y_pred = model.predict(X_vec)
    y_prob = model.predict_proba(X_vec)[:, 1]

    # Metrics & Save
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5

    metrics_to_save = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "AUC": float(auc)
    }

    save_results_to_json(data_dir, "Logistic Regression", metrics_to_save)

    print_metrics_report(y_true, y_pred, y_prob)
    plot_confusion_matrix(y_true, y_pred)
    analyze_errors(df, y_true, y_pred, y_prob, text_col='text_logreg')

def main():

    DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_DIR = "/content/gdrive/MyDrive/Data mining/text mining/models"

    print("🚀 STARTING FINAL PROJECT EVALUATION ON TEST SET 🚀")

    # 1. Run Logistic Regression
    evaluate_logreg_model(DATA_DIR, MODEL_DIR, data_type='test')

    # 2. Run DistilBERT (Small)
    evaluate_bert_model(DATA_DIR, MODEL_DIR, "bert_finetuned", max_len=128, data_type='test')

    # 3. Run BERT Large (The Main Event)
    evaluate_bert_model(DATA_DIR, MODEL_DIR, "bert_large_finetuned", max_len=512, data_type='test')

    print("\n\n DONE! All results saved to data/results/model_comparison.json")

if __name__ == "__main__":
    main()