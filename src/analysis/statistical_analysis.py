"""
STATISTICAL TEST: Chi-Square Test of Independence
Check if 'Gender' and 'AI Topic' are related or independent.
"""
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import six
import numpy as np
import os

def run_chi_square_research_group(TARGET_GROUPS, processed_dir, results_dir, posts_file):
    # --- 1. Load Data ---
    posts_path = os.path.join(processed_dir, posts_file)
    if not os.path.exists(posts_path):
        print(f"Error: File not found at {posts_path}")
        return

    df = pd.read_csv(posts_path)
    
    # --- 2. Filter & Prepare ---
    df_clean = df[df['research_group'].isin(TARGET_GROUPS)].copy()
    
    # Labeling
    df_clean['stat_gender'] = df_clean['research_group'].apply(
        lambda x: 'Female' if x.startswith('female') else 'Male'
    )
    df_clean['stat_topic'] = df_clean['research_group'].apply(
        lambda x: 'Hard AI' if 'hard_ai' in x else 'Soft AI'
    )

    # --- 3. Calculate Statistics ---
    contingency_table = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'])
    # Normalize by index (row) to get percentages summing to 100% per gender
    contingency_pct = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'], normalize='index') * 100
    
    contingency_pct_formatted = contingency_pct.applymap(lambda x: f"{x:.1f}%")

    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # --- 4. VISUALIZATION: STACKED BAR CHART (NEW) ---
    print("Generating Stacked Bar Chart...")
    
    # Colors from your Palette: Hard AI (#a2c4c9), Soft AI (#f9cb9c)
    # Note: Columns are alphabetical: 'Hard AI', 'Soft AI'
    my_colors = ['#a2c4c9', '#f9cb9c'] 
    
    plt.figure(figsize=(8, 6))
    
    # Plot using the calculated percentages
    ax = contingency_pct.plot(kind='bar', stacked=True, color=my_colors, alpha=0.9, width=0.6, figsize=(8,6))

    plt.title("The Split: AI Topic Distribution by Gender Context", fontsize=15, fontweight='bold')
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(title='AI Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)

    # Add percentage labels inside the bars
    for c in ax.containers:
        # Using black text for better contrast on light pastel colors
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='black', weight='bold', fontsize=11)

    plt.tight_layout()
    
    save_path_chart = os.path.join(results_dir, "RQ1_Stacked_Bar_Distribution.png")
    plt.savefig(save_path_chart, bbox_inches='tight', dpi=300)
    print(f"Saved Stacked Bar Chart to: {save_path_chart}")
    plt.show()

    # --- Print Stats ---
    print(f"Chi-Square P-Value: {p:.4e}")
    if p < 0.05:
      print("\n=> RESULT: SIGNIFICANT BIAS DETECTED.")
      print("There is a statistically significant dependency between Gender and AI Topic.")
    else:
      print("\n=> RESULT: NO SIGNIFICANT BIAS.")

def main():
    processed_dir = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
    results_dir = "/content/gdrive/MyDrive/Data mining/text mining/visualization"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    posts_file = "final_posts_dataset.csv"
    TARGET_GROUPS = ['female_hard_ai', 'female_soft_ai', 'male_hard_ai', 'male_soft_ai']
    
    run_chi_square_research_group(TARGET_GROUPS, processed_dir, results_dir, posts_file)

if __name__ == "__main__":
    main()