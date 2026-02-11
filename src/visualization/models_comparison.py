import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from math import pi

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['font.family'] = 'sans-serif'

def clean_model_names(df):
    name_map = {}
    for name in df['Model'].unique():
        if 'Logistic' in name:
            name_map[name] = 'Logistic Regression'
        elif 'bert_finetuned' in name or 'Distil' in name:
            name_map[name] = 'DistilBERT'
        elif 'large' in name or 'Large' in name:
            name_map[name] = 'BERT Large'
        else:
            name_map[name] = name
    return df.replace(name_map)

def plot_performance_comparison(df, output_dir):
    """ compare models results"""
    print("Generating Performance Bar Chart...")
    
    metrics_to_show = ['Accuracy', 'F1 Score', 'AUC']
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    df_melted = df_melted[df_melted['Metric'].isin(metrics_to_show)]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_melted, 
        x="Metric", 
        y="Score", 
        hue="Model", 
        palette="viridis"
    )

    plt.title("Model Performance Comparison (Test Set)", fontsize=16, fontweight='bold')
    plt.ylim(0.85, 1.0) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_performance_comparison.png"))
    plt.show()

def plot_error_analysis(output_dir):
    """Error analysis """
    print("Generating Error Analysis Chart...")

    error_data = {
        'Model': [
            'Logistic Regression', 'Logistic Regression',
            'DistilBERT', 'DistilBERT',
            'BERT Large', 'BERT Large'
        ],
        'Error Type': [
            'False Positives (Soft instead of Hard)', 'False Negatives (Hard instead of Soft)',
            'False Positives (Soft instead of Hard)', 'False Negatives (Hard instead of Soft)',
            'False Positives (Soft instead of Hard)', 'False Negatives (Hard instead of Soft)'
        ],
        'Count': [
            55, 38,   # LR Results
            265, 106, # DistilBERT Results
            49, 69    # BERT Large Results
        ]
    }
    
    df_errors = pd.DataFrame(error_data)

    plt.figure(figsize=(12, 6))
    
    custom_palette = sns.color_palette("Paired")
    
    ax = sns.barplot(
        data=df_errors,
        x="Model",
        y="Count",
        hue="Error Type",
        palette="Reds_r" 
    )

    plt.title("Error Analysis: Bias Reduction in BERT Large", fontsize=16, fontweight='bold')
    plt.ylabel("Number of Errors", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.legend(title="Error Type")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_error_bias_analysis.png"))
    plt.show()

def plot_radar_chart(df, output_dir):
    print("Generating Radar Chart...")
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # סגירת המעגל
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, fontsize=10)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.90, 0.95, 1.0], ["0.90", "0.95", "1.0"], color="grey", size=8)
    plt.ylim(0.85, 1.0)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71'] 
    models = df['Model'].unique()
    
    for i, model in enumerate(models):
        values = df[df['Model'] == model][categories].values.flatten().tolist()
        values += values[:1] 
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.title("Holistic Model Comparison", size=16, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_radar_chart.png"))
    plt.show()

def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    RESULTS_FILE = os.path.join(BASE_DIR, "results", "model_comparison.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "visualization")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(RESULTS_FILE):
        print("JSON file not found! Please run the evaluation script first.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'Model'})
    df = clean_model_names(df)
    
    desired_order = ['DistilBERT', 'BERT Large', 'Logistic Regression']
    df['Model'] = pd.Categorical(df['Model'], categories=desired_order, ordered=True)
    df = df.sort_values('Model')

    plot_performance_comparison(df, OUTPUT_DIR)
    plot_error_analysis(OUTPUT_DIR)
    plot_radar_chart(df, OUTPUT_DIR)
    
    print(f"\n All 3 plots saved successfully to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()