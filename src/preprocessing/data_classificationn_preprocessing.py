import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_split(posts_pkl_path):
    print("Loading posts data...")
    df = pd.read_pickle(posts_pkl_path)
    
    # we want to do classification between hard_ai and soft_ai only
    df_model = df[df['ai_category'].isin(['hard_ai', 'soft_ai'])].copy()
    
    # binary labeling
    label_map = {'hard_ai': 0, 'soft_ai': 1}
    df_model['label'] = df_model['ai_category'].map(label_map)
    
    # for LR we need cleaned text, for bert we need original text
    df_model['text_logreg'] = df_model['cleaned_tokens'].apply(lambda x: ' '.join(x))
    df_model['text_bert'] = df_model['full_text'].astype(str)
    
    print(f"Total Filtered Data: {len(df_model)}")
    
    # ==========================================
    # According to project requirment, split the data to 70/15/15
    # ==========================================
    
    # test_size=0.3
    train_df, temp_df = train_test_split(
        df_model, 
        test_size=0.3, 
        random_state=42, 
        stratify=df_model['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label']
    )
    
    # Sanity Check
    total = len(df_model)
    print(f"\n--- Data Split Statistics ---")
    print(f"Train:      {len(train_df)} ({len(train_df)/total:.1%}) -> For Training + CV")
    print(f"Validation: {len(val_df)}  ({len(val_df)/total:.1%}) -> For Validation")
    print(f"Test:       {len(test_df)}  ({len(test_df)/total:.1%}) -> For Final Evaluation")
    
    return train_df, val_df, test_df

def plot_split_distribution(df, column_name, set_name):
    """
    Plots both Bar chart (counts) and Pie chart (percentages) side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar Chart
    sns.countplot(x=column_name, data=df, ax=axes[0], palette="viridis", order=df[column_name].value_counts().index)
    axes[0].set_title(f'{set_name} Set - Counts')
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")
    
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].text(p.get_x() + p.get_width() / 2., height + 5, f'{int(height)}', ha="center")

    # Pie Chart
    counts = df[column_name].value_counts()
    axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(counts)))
    axes[1].set_title(f'{set_name} Set - Percentage')

    plt.tight_layout()
    plt.show()

def main():
  BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
  POSTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")

  #data splitting
  train_df, val_df, test_df = prepare_split(POSTS_PKL)
  
  #visualizing data distribution
  plot_split_distribution(train_df, 'ai_category', "Train (70%)")
  plot_split_distribution(val_df, 'ai_category', "Validation (15%)")
  plot_split_distribution(test_df, 'ai_category', "Test (15%)")

  #save to 3 files
  train_df.to_pickle(os.path.join(BASE_DIR, "processed/train_data.pkl"))
  val_df.to_pickle(os.path.join(BASE_DIR, "processed/val_data.pkl"))
  test_df.to_pickle(os.path.join(BASE_DIR, "processed/test_data.pkl"))

  print("\nFiles saved: train_data.pkl, val_data.pkl, test_data.pkl")