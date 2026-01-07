import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path

POSTS_PATH = Path("/content/gdrive/MyDrive/Data mining/text mining/data/processed/final_posts_dataset.csv")

# LIWC categories
LIWC_COLS = [
    "LIWC_power_achievement",
    "LIWC_cognitive_processes",
    "LIWC_negative_emotions",
    "LIWC_certainty",
    "LIWC_affiliation_social",
    "LIWC_perceptual_processes"
]

def analyze_rq3_liwc():
    print("Loading posts dataset...")
    df = pd.read_csv(POSTS_PATH)
    
    # filter by gender and self-talk
    df_self = df[
        (df['gender_context'].isin(['female', 'male'])) & 
        (df['is_self_talk'] == True)
    ].copy()
    
    print(f"Total posts analyzing (Self-Talk only): {len(df_self)}")
    print(f"Female posts: {len(df_self[df_self['gender_context']=='female'])}")
    print(f"Male posts:   {len(df_self[df_self['gender_context']=='male'])}")

    df_long = df_self.melt(
        id_vars=['gender_context'], 
        value_vars=LIWC_COLS, 
        var_name='LIWC_Category', 
        value_name='Score'
    )
    
    df_long['LIWC_Category'] = df_long['LIWC_Category'].str.replace('LIWC_', '').str.replace('_', ' ').str.title()

    # visualisation
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_long, 
        x='LIWC_Category', 
        y='Score', 
        hue='gender_context', 
        palette={'female': '#e74c3c', 'male': '#3498db'}, 
        ci=95, 
        capsize=0.1
    )
    plt.title('Psychological Analysis (LIWC) of Self-Talk Posts: Female vs. Male')
    plt.ylabel('Percentage of Words used (%)')
    plt.xlabel('Psychological Category')
    plt.legend(title='Gender')
    plt.grid(axis='y', alpha=0.2)
    plt.show()

    # check each creiteria
    print("\n" + "="*60)
    print(f"{'CATEGORY':<25} | {'FEMALE MEAN':<12} | {'MALE MEAN':<12} | {'P-VALUE':<10} | {'RESULT'}")
    print("="*60)
    
    for col in LIWC_COLS:
        group_f = df_self[df_self['gender_context'] == 'female'][col]
        group_m = df_self[df_self['gender_context'] == 'male'][col]
        
        # T-Test
        t_stat, p_val = ttest_ind(group_f, group_m, equal_var=False)
        
        res_str = ""
        if p_val < 0.05:
            if group_f.mean() > group_m.mean():
                res_str = "Female > Male (SIG)"
            else:
                res_str = "Male > Female (SIG)"
        else:
            res_str = "No Diff"
            
        clean_name = col.replace("LIWC_", "")
        print(f"{clean_name:<25} | {group_f.mean():.4f}       | {group_m.mean():.4f}       | {p_val:.4f}     | {res_str}")

def main():
    print("=" * 60)
    print("STARTING LIWC analysis")
    print("=" * 60)
    analyze_rq3_liwc()