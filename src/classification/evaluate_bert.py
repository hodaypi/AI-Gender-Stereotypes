import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.nn.functional import softmax

sys.path.append('/content') 

from src.classification.metrics import (
    print_metrics_report, 
    plot_confusion_matrix, 
    plot_advanced_graphs, 
    analyze_errors
)

def evaluate_bert(data_dir, model_dir, model_folder_name, max_len=128, data_type='val'):
    print(f"\n=======================================================")
    print(f"   EVALUATING MODEL: {model_folder_name} (Max Len: {max_len})")
    print(f"=======================================================")
    
    # load model and Tokenizer
    model_path = os.path.join(model_dir, model_folder_name)
    
    try:
        # AutoModel to recognize DistilBERT or BERT Large
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # load data
    data_path = os.path.join(data_dir, f"processed/{data_type}_data.pkl")
    df = pd.read_pickle(data_path)
    
    texts = df['text_bert'].tolist()
    y_true = df['label']

    print(f"Predicting on {len(texts)} texts...")

    # prepare text
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

    # results
    logits = torch.tensor(raw_preds.predictions)
    probs = softmax(logits, dim=1)[:, 1].numpy()
    y_pred = np.argmax(raw_preds.predictions, axis=1)

    print("--- Generating Report ---")
    
    # call metric to evaluate prediction results
    print_metrics_report(y_true, y_pred, probs)
    plot_confusion_matrix(y_true, y_pred)
    plot_advanced_graphs(y_true, probs)
    
    print(f"\n--- Error Analysis for {model_folder_name} ---")
    analyze_errors(df, y_true, y_pred, probs, text_col='text_bert')

def main():
    DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"
  
    # DistilBERT
    evaluate_bert(DATA_DIR, MODEL_DIR, model_folder_name="bert_finetuned", max_len=128)
    
    #BERT Large
    evaluate_bert(DATA_DIR, MODEL_DIR, model_folder_name="bert_large_finetuned", max_len=512)