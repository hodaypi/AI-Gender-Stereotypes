import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path

COMMENTS_PATH = Path("/content/gdrive/MyDrive/Data mining/text mining/data/processed/final_comments_dataset.csv")

def analyze_rq2_sentiment():
    print("Loading comments dataset...")
    df = pd.read_csv(COMMENTS_PATH)
    original_count = len(df)
    
    #remove duplicated comments
    df = df.drop_duplicates(subset=['body', 'post_id'])
    
    new_count = len(df)
    duplicates_removed = original_count - new_count
    print(f"Found and removed {duplicates_removed} duplicate comments!")
    print(f"New dataset size: {new_count}")

    #Filter research groups
    target_groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    df_clean = df[df['parent_research_group'].isin(target_groups)].copy()
    
    print(f"Total relevant comments: {len(df_clean)}")
    
    # check average sentiment
    print("\n--- Average Sentiment (VADER) by Group ---")
    print(df_clean.groupby('parent_research_group')['sentiment_vader'].mean().sort_values())

    # visualisation
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='parent_research_group', y='sentiment_vader', data=df_clean, palette='coolwarm', order=target_groups)
    plt.title('Sentiment Distribution in Comments by Research Group')
    plt.ylabel('VADER Sentiment Score (-1 to 1)')
    plt.xlabel('Post Context (Parent Group)')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    #statistical test
    print("\n" + "="*50)
    print("STATISTICAL TEST: Hostility in Hard AI (Female vs Male)")
    print("="*50)
    
    group_female = df_clean[df_clean['parent_research_group'] == 'female_hard_ai']['sentiment_vader']
    group_male = df_clean[df_clean['parent_research_group'] == 'male_hard_ai']['sentiment_vader']
    
    t_stat, p_val = ttest_ind(group_female, group_male, equal_var=False)
    
    print(f"Female Hard AI (Mean): {group_female.mean():.4f}")
    print(f"Male Hard AI   (Mean): {group_male.mean():.4f}")
    print(f"Difference:            {group_female.mean() - group_male.mean():.4f}")
    print("-" * 30)
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_val:.4e}")
    
    alpha = 0.05
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
    print("STARTING Sentiment analysis")
    print("=" * 60)
    analyze_rq2_sentiment()
    


