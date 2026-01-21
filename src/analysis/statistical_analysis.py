import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_chi_square_research_group(TARGET_GROUPS,processed_dir,posts_file):
    # load data
    posts_path = os.path.join(processed_dir, posts_file)
    df = pd.read_csv(posts_path)
    print(f"Total posts loaded: {len(df)}")

    # filter only research groups
    df_clean = df[df['research_group'].isin(TARGET_GROUPS)].copy()
    print(f"Posts in target groups: {len(df_clean)}")

    df_clean['stat_gender'] = df_clean['research_group'].apply(
        lambda x: 'Female' if x.startswith('female') else 'Male'
    )

    df_clean['stat_topic'] = df_clean['research_group'].apply(
        lambda x: 'Hard AI' if 'hard_ai' in x else 'Soft AI'
    )

    # Contingency Table
    contingency_table = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'])
    
    print("\n--- Contingency Table (Counts) ---")
    print(contingency_table)

    # Distribution Percentages
    contingency_pct = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'], normalize='index') * 100
    print("\n--- Distribution Percentages ---")
    print(contingency_pct.round(2).astype(str) + '%')

    # Chi-Square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n" + "="*40)
    print("RQ1 RESULTS: The Gender-Topic Split")
    print("="*40)
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value:        {p:.4e}") 
    
    if p < 0.05:
        print("\n=> RESULT: SIGNIFICANT BIAS DETECTED.")
        print("There is a statistically significant dependency between Gender and AI Topic.")
        print("(This proves that women and men appear in different frequencies across Hard vs Soft AI).")
    else:
        print("\n=> RESULT: NO SIGNIFICANT BIAS.")

    # visualization
    plt.figure(figsize=(8, 6))
    ax = contingency_pct.plot(kind='bar', stacked=True, color=['#2c3e50', '#e67e22'], alpha=0.9, width=0.6)
    
    plt.title("The Split: AI Topic Distribution by Gender Context")
    plt.xlabel("Gender")
    plt.ylabel("Percentage (%)")
    plt.legend(title='AI Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

    plt.tight_layout()
    plt.show()

def main():
  processed_dir = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
  posts_file = "final_posts_dataset.csv"

  TARGET_GROUPS = [
      'female_hard_ai', 
      'female_soft_ai', 
      'male_hard_ai', 
      'male_soft_ai'
  ]
  run_chi_square_research_group(TARGET_GROUPS,processed_dir,posts_file)

if __name__ == "__main__":
    main()
    