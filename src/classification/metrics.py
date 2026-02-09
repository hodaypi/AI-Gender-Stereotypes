import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

def print_metrics_report(y_true, y_pred, y_prob=None, target_names=['Hard AI', 'Soft AI']):
    
    print("\n" + "="*40)
    print("   METRICS REPORT")
    print("="*40)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
            print(f"AUC-ROC:  {auc_score:.4f}")
        except:
            print("AUC-ROC:  N/A (Needs probabilities)")
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

def plot_confusion_matrix(y_true, y_pred, labels=['Hard AI', 'Soft AI']):
   
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_advanced_graphs(y_true, y_prob):
    
    # ROC & PR Curve
    if y_prob is None:
        print("Skipping advanced graphs (no probabilities provided)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR (area = {pr_auc:.2f})')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_errors(df, y_true, y_pred, y_prob, text_col, num_examples=5):
    
    print("\n--- Error Analysis ---")
    df_res = df.copy()
    df_res['true'] = y_true.values
    df_res['pred'] = y_pred
    if y_prob is not None:
        df_res['prob'] = y_prob
    
    # False Positives (Hard classified as Soft)
    fp = df_res[(df_res['true'] == 0) & (df_res['pred'] == 1)]
    print(f"False Positives: {len(fp)}")
    if not fp.empty:
        for i, row in fp.head(num_examples).iterrows():
            prob_str = f"(Prob: {row['prob']:.2f})" if 'prob' in df_res else ""
            print(f" - {prob_str} {str(row[text_col])[:100]}...")
            
    # False Negatives (Soft classified as Hard)
    fn = df_res[(df_res['true'] == 1) & (df_res['pred'] == 0)]
    print(f"False Negatives: {len(fn)}")
    if not fn.empty:
        for i, row in fn.head(num_examples).iterrows():
            prob_str = f"(Prob: {row['prob']:.2f})" if 'prob' in df_res else ""
            print(f" - {prob_str} {str(row[text_col])[:100]}...")