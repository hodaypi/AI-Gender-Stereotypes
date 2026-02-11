import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import six
import numpy as np
import os

UNIFORM_HEADER_COLOR = '#6d9eeb'  
CORNER_HEADER_COLOR = '#9f9e9e'    
ROW_COLORS = ['w']                

def render_unified_table(data, col_width=3.5, row_height=0.7, font_size=14,
                         header_color=UNIFORM_HEADER_COLOR, 
                         corner_color=CORNER_HEADER_COLOR,
                         row_colors=ROW_COLORS, edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=1, 
                         ax=None, **kwargs):
  
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k == (0, 0):
            cell.set_facecolor(corner_color)
            cell.set_text_props(weight='bold', color='white')
        elif k[0] == 0 or k[1] < header_columns:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            cell.set_text_props(color='black')

    return ax

def run_chi_square_research_group(TARGET_GROUPS, processed_dir, results_dir, posts_file):
    # --- 1. Load Data ---
    posts_path = os.path.join(processed_dir, posts_file)
    if not os.path.exists(posts_path):
        print(f"Error: File not found at {posts_path}")
        return

    df = pd.read_csv(posts_path)
    
    # --- 2. Filter & Prepare ---
    df_clean = df[df['research_group'].isin(TARGET_GROUPS)].copy()
    
    df_clean['stat_gender'] = df_clean['research_group'].apply(
        lambda x: 'Female' if x.startswith('female') else 'Male'
    )
    df_clean['stat_topic'] = df_clean['research_group'].apply(
        lambda x: 'Hard AI' if 'hard_ai' in x else 'Soft AI'
    )

    # --- 3. Calculate Statistics ---
    contingency_table = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'])
    contingency_pct = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'], normalize='index') * 100
    contingency_pct_formatted = contingency_pct.applymap(lambda x: f"{x:.1f}%")

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Chi-Square P-Value: {p:.4e}")

    # --- 4. SAVE PRETTY TABLES AS IMAGES ---

    # --- Table 1: Absolute Counts ---
    df_counts_display = contingency_table.reset_index()
    df_counts_display.rename(columns={'stat_gender': 'Gender / Domain'}, inplace=True)
    
    ax1 = render_unified_table(df_counts_display, header_columns=1, col_width=4.0)
    plt.title(f"Table 1: Frequency of Posts by Gender and AI Domain\n(N={len(df_clean)})", fontsize=16, y=1.1)
    
    save_path_counts = os.path.join(results_dir, "Table_RQ1_Counts.png")
    plt.savefig(save_path_counts, bbox_inches='tight', dpi=300)
    print(f"Saved Count Table to: {save_path_counts}")
    plt.show()

    # --- Table 2: Percentages ---
    df_pct_display = contingency_pct_formatted.reset_index()
    df_pct_display.rename(columns={'stat_gender': 'Gender / Domain'}, inplace=True)

    ax2 = render_unified_table(df_pct_display, header_columns=1, col_width=4.0)
    plt.title(f"Table 2: Distribution of AI Topics within Gender Groups\n(Chi-Square p={p:.2e})", fontsize=16, y=1.1)
    
    save_path_pct = os.path.join(results_dir, "Table_RQ1_Percentages.png")
    plt.savefig(save_path_pct, bbox_inches='tight', dpi=300)
    print(f"Saved Percentage Table to: {save_path_pct}")
    plt.show()

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