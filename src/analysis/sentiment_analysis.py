import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

def analyze_rq2_sentiment():

    base_dir = "/content/gdrive/MyDrive/Data mining/text mining"
    data_path = os.path.join(base_dir, "data/processed/final_comments_dataset.csv")
    results_dir = os.path.join(base_dir, "visualization")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Loading comments dataset...")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=['body', 'post_id'])
    
    palette_all = {
        'female_hard_ai': '#d5a6bd', 
        'male_hard_ai':   '#a4c2f4',
        'female_soft_ai': '#f9cb9c', 
        'male_soft_ai':   '#a2c4c9'
    }

    # --- 1-Sentiment Distribution in Comments by Research Group (All Groups)---
    all_groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    df_all = df[df['parent_research_group'].isin(all_groups)].copy()
    
    print(f"Total relevant comments (All Groups): {len(df_all)}")

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x='parent_research_group', y='sentiment_vader', data=df_all, 
                palette=palette_all, order=all_groups)
    
    plt.title('Sentiment Distribution in Comments by Research Group (All Groups)')
    plt.ylabel('VADER Sentiment Score (-1 to 1)')
    plt.xlabel('Post Context (Parent Group)')
    plt.grid(axis='y', alpha=0.3)
    
    save_path_all = os.path.join(results_dir, "RQ2_Sentiment_Boxplot_All.png")
    plt.savefig(save_path_all, bbox_inches='tight', dpi=300)
    print(f"Saved All-Groups Graph to: {save_path_all}")
    plt.show()

    # --- Focused Comparison: Sentiment in Hard AI (Female vs Male) ---
  
    plt.figure(figsize=(7, 6))
    
    hard_ai_groups = ['female_hard_ai', 'male_hard_ai']
    df_hard = df[df['parent_research_group'].isin(hard_ai_groups)].copy()
    
    palette_focused = {k: palette_all[k] for k in hard_ai_groups}
    
    sns.boxplot(x='parent_research_group', y='sentiment_vader', data=df_hard, 
                palette=palette_focused, order=hard_ai_groups, width=0.5)
    
    plt.title('Focused Comparison: Sentiment in Hard AI (Female vs Male)')
    plt.ylabel('VADER Sentiment Score (-1 to 1)')
    plt.xlabel('Gender Context')
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    
    plt.xticks(ticks=[0, 1], labels=['Female Hard AI', 'Male Hard AI'])
    
    save_path_focused = os.path.join(results_dir, "RQ2_Sentiment_Boxplot_Focused.png")
    plt.savefig(save_path_focused, bbox_inches='tight', dpi=300)
    print(f"Saved Focused Graph to: {save_path_focused}")
    plt.show()

    # --- STATISTICAL TEST ---
    
    alpha= 0.05

    print("\n" + "="*50)
    print("STATISTICAL TEST: Hostility in Hard AI (Female vs Male)")
    print("="*50)
    
    group_female = df_all[df_all['parent_research_group'] == 'female_hard_ai']['sentiment_vader']
    group_male = df_all[df_all['parent_research_group'] == 'male_hard_ai']['sentiment_vader']
    
    t_stat, p_val = ttest_ind(group_female, group_male, equal_var=False)
    
    print(f"Female Hard AI (Mean): {group_female.mean():.4f}")
    print(f"Male Hard AI   (Mean): {group_male.mean():.4f}")
    print(f"Difference:            {group_female.mean() - group_male.mean():.4f}")
    print("-" * 30)
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_val:.4e}")

    if p_val < alpha:
      print("\n=> RESULT: SIGNIFICANT DIFFERENCE FOUND.")
      if t_stat < 0:
          print("Women received significantly MORE NEGATIVE (lower) sentiment than men.")
      else:
        print("Women received significantly MORE POSITIVE (higher) sentiment than men (Surprisingly!).")
    else:
      print("\n=> RESULT: NO SIGNIFICANT DIFFERENCE.")
      print("We cannot prove that women receive different sentiment in Hard AI topics.")

      inspect_comments(df_clean, 'female_hard_ai')
      inspect_comments(df_clean, 'male_hard_ai')

def inspect_comments(df_clean, group_name, n=5):
  
    print(f"\n=== Top {n} MOST POSITIVE comments for: {group_name} ===")
    subset = df_clean[df_clean['parent_research_group'] == group_name]
    
    # top positive
    top_pos = subset.nlargest(n, 'sentiment_vader')
    for i, row in top_pos.iterrows():
        print(f"[Score: {row['sentiment_vader']:.2f}] {row['body'][:200]}...") # מציג רק תחילת הטקסט
        
    print(f"\n=== Top {n} MOST NEGATIVE comments for: {group_name} ===")
    
    # top negative
    top_neg = subset.nsmallest(n, 'sentiment_vader')
    for i, row in top_neg.iterrows():
        print(f"[Score: {row['sentiment_vader']:.2f}] {row['body'][:200]}...")

def main():
    print("=" * 60)
    print("STARTING Sentiment Analysis (Original + Focused)")
    print("=" * 60)
    analyze_rq2_sentiment()

if __name__ == "__main__":
    main()