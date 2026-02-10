import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import os

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert_large(data_dir, model_dir):
    print("--- 1. Loading Data ---")
    train_df = pd.read_pickle(os.path.join(data_dir, "processed/train_data.pkl"))
    val_df = pd.read_pickle(os.path.join(data_dir, "processed/val_data.pkl"))
    
    print("Using 'text_bert' column (full text).")
    train_texts = train_df['text_bert'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['text_bert'].tolist()
    val_labels = val_df['label'].tolist()

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    print("\n--- 2. Tokenization (BERT Large + 512 Length) ---")
    
    # --- CHANGE 1: Using BERT Large instead of DistilBERT ---
    model_name = "bert-large-uncased" 
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(texts):
        # --- CHANGE 2: Increasing Max Length to 512 ---
        # This captures much more context from long Reddit posts
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    print("Tokenizing... (This might take a moment due to length)")
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    class GenderDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = GenderDataset(train_encodings, train_labels)
    val_dataset = GenderDataset(val_encodings, val_labels)

    print("\n--- 3. Initializing BERT Large Model ---")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device) 

    # --- CHANGE 3: Advanced Training Arguments ---
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "bert_large_results"), 
        
        # Epochs & Batch Size
        num_train_epochs=5,              
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=16,   
        gradient_accumulation_steps=2,   
        
        # Optimization
        learning_rate=2e-5,              
        warmup_steps=500,
        weight_decay=0.01,
        
        # Logging & Saving
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,     
        metric_for_best_model="f1",
        
        # System
        fp16=True,                       
    )

    print("\n--- 4. Training with Early Stopping ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # --- CHANGE 4: Early Stopping ---
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )

    trainer.train()

    print("\n--- 5. Saving Final Large Model ---")
    save_path = os.path.join(model_dir, "bert_large_finetuned")
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"BERT Large Model saved successfully to {save_path}")

def main():
  DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
  MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"
  
  torch.cuda.empty_cache()
  
  train_bert_large(DATA_DIR, MODEL_DIR)