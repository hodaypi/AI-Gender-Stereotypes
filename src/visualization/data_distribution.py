import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re

# --- Configuration & Palette ---
BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining"
POSTS_PATH = os.path.join(BASE_DIR, "data/processed/final_posts_dataset.csv")
COMMENTS_PATH = os.path.join(BASE_DIR, "data/processed/final_comments_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "visualization")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# צבעים אחידים
PALETTE = {
    'female': '#d5a6bd', 'Female': '#d5a6bd',
    'male': '#a4c2f4',   'Male': '#a4c2f4',
    'Undetermined': '#909090', 
    'Other/Unlabeled': '#909090',
    'mixed':'#909090',
    'general':'#909090',  
    
    # research_group
    'female_hard_ai': '#d5a6bd', 
    'male_hard_ai':   '#a4c2f4',
    'female_soft_ai': '#f9cb9c', 
    'male_soft_ai':   '#a2c4c9',

    # ai_category
    'hard_ai': '#a2c4c9',
    'soft_ai': '#f9cb9c'
}

# רשימות סינון קשיחות
VALID_GENDERS = ['male', 'female']
VALID_CATEGORIES = ['hard_ai', 'soft_ai']
VALID_GROUPS = ['male_hard_ai', 'female_hard_ai', 'male_soft_ai', 'female_soft_ai']

def add_labels(ax):
    """הוספת מספרים מעל העמודות"""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + (height*0.01), 
                    f'{int(height)}', 
                    ha="center", fontsize=10, fontweight='bold')

def save_and_show(filename):
    """שומר ומציג את הגרף"""
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

# ==========================================
# 1. GENERAL PLOTTING FUNCTIONS
# ==========================================

def plot_filtered_count(df, col, valid_values, title, filename):
    """גרף עמודות שמציג רק ערכים מתוך רשימת valid_values"""
    plt.figure(figsize=(8, 6))
    
    # סינון הדאטה רק לערכים הרצויים
    df_clean = df[df[col].isin(valid_values)].copy()
    
    if df_clean.empty:
        print(f"Skipping {title}: No data found for specified categories.")
        return

    # שימוש ב-Order כדי לשמור על סדר קבוע
    ax = sns.countplot(x=col, data=df_clean, palette=PALETTE, order=valid_values)
    
    plt.title(title, fontsize=15)
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Count')
    add_labels(ax)
    save_and_show(filename)

def plot_vader_kde_filtered(df, gender_col, title, filename):
    """גרף התפלגות סנטימנט (KDE) רק לגברים ונשים"""
    plt.figure(figsize=(10, 6))
    
    # סינון: רק גברים ונשים (בלי undetermined)
    df_clean = df[df[gender_col].isin(['male', 'female'])].copy()
    
    if df_clean.empty:
        print(f"Skipping VADER plot {title}: No gender data found.")
        return

    sns.kdeplot(data=df_clean, x='sentiment_vader', hue=gender_col, 
                palette=PALETTE, fill=True, alpha=0.3, linewidth=2, common_norm=False)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Sentiment Score (-1 = Negative, 1 = Positive)")
    plt.xlim(-1, 1)
    
    # קו האפס
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.text(0.05, plt.gca().get_ylim()[1]*0.9, 'Neutral', color='grey')
    
    save_and_show(filename)

# ==========================================
# 2. WORD CLOUD & TEXT ANALYSIS FUNCTIONS
# ==========================================

def plot_group_wordclouds(df):
    """Generates 4 Word Clouds - one for each research group."""
    print("Generating Word Clouds per Group...")
    
    # מסננים רק את 4 הקבוצות
    df = df[df['research_group'].isin(VALID_GROUPS)]
    
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post", "deleted", "removed"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, group in enumerate(VALID_GROUPS):
        subset = df[df['research_group'] == group]
        
        text_data = " ".join(subset['title'].fillna('').astype(str) + " " + subset['selftext'].fillna('').astype(str))
        
        if not text_data.strip():
            axes[i].text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        wc = WordCloud(
            stopwords=stopwords,
            background_color="white",
            colormap="ocean", 
            width=800, height=500,
            max_words=50
        ).generate(text_data)

        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"{group} ({len(subset)} posts)", fontsize=14, fontweight='bold')
        axes[i].axis("off")

    plt.suptitle("Word Clouds by Research Group", fontsize=20)
    save_and_show("WordClouds_By_Group.png")

def plot_top_words_comparison(df):
    """Shows the top 10 most frequent words for each group."""
    print("Generating Top 10 Words Analysis...")
    
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post", "deleted", "removed", "ai", "one", "would", "like", "amp", "know", "think"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, group in enumerate(VALID_GROUPS):
        subset = df[df['research_group'] == group]
        text_data = " ".join(subset['title'].fillna('').astype(str) + " " + subset['selftext'].fillna('').astype(str))
        
        words = re.findall(r'\b\w+\b', text_data.lower())
        words = [w for w in words if w not in stopwords and len(w) > 2] 
        
        common_words = Counter(words).most_common(10)
        
        if not common_words:
            continue
            
        words_list, counts_list = zip(*common_words)
        color = PALETTE.get(group, 'grey')
        
        sns.barplot(x=list(counts_list), y=list(words_list), ax=axes[i], color=color)
        axes[i].set_title(f"Top 10 Words: {group}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Frequency")

    plt.suptitle("Most Frequent Words Analysis", fontsize=18)
    save_and_show("Top_10_Words_By_Group.png")

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================

def main():
    
    # --- PART A: POSTS ANALYSIS ---
    print("\n" + "="*40)
    print("STARTING POSTS ANALYSIS (Self-Talk Data)")
    print("="*40)
    
    if os.path.exists(POSTS_PATH):
        df_posts = pd.read_csv(POSTS_PATH)
        
        # 1. Self-Talk by Gender (רק גברים ונשים)
        plot_filtered_count(df_posts, 'gender_context', VALID_GENDERS,
                          "Self-Talk by Gender", 
                          "Posts_SelfTalk_Gender.png")
        
        # 2. AI Category Distribution (רק hard ו-soft)
        plot_filtered_count(df_posts, 'ai_category', VALID_CATEGORIES,
                          "Posts Distribution by AI Category", 
                          "Posts_Dist_Category.png")
        
        # 3. Research Groups Distribution (רק ה-4 קבוצות)
        plot_filtered_count(df_posts, 'research_group', VALID_GROUPS,
                          "Posts Distribution by Research Group", 
                          "Posts_Dist_ResearchGroup.png")
        
        # 4. VADER Sentiment Distribution
        plot_vader_kde_filtered(df_posts, 'gender_context', 
                       "Sentiment Distribution: Male vs Female Authors (Posts)", 
                       "Posts_VADER_Dist.png")
        
        # 5. Text Analysis
        if 'title' in df_posts.columns and 'selftext' in df_posts.columns:
            plot_group_wordclouds(df_posts)
            plot_top_words_comparison(df_posts)
            
    else:
        print(f"File not found: {POSTS_PATH}")

    # --- PART B: COMMENTS ANALYSIS ---
    print("\n" + "="*40)
    print("STARTING COMMENTS ANALYSIS")
    print("="*40)
    
    if os.path.exists(COMMENTS_PATH):
        df_comments = pd.read_csv(COMMENTS_PATH)
        
        if 'parent_research_group' in df_comments.columns:
            
            # Helper logic to extract target info
            def get_target_gender(group):
                if pd.isna(group): return None
                if 'female' in str(group).lower(): return 'female'
                if 'male' in str(group).lower(): return 'male'
                return None # Exclude everything else

            def get_target_category(group):
                if pd.isna(group): return None
                if 'hard' in str(group).lower(): return 'hard_ai'
                if 'soft' in str(group).lower(): return 'soft_ai'
                return None

            df_comments['target_gender'] = df_comments['parent_research_group'].apply(get_target_gender)
            df_comments['target_category'] = df_comments['parent_research_group'].apply(get_target_category)
            
            # 1. Gender Distribution (Replies TO Men vs TO Women)
            plot_filtered_count(df_comments, 'target_gender', VALID_GENDERS,
                              "Distribution of Comments by Target Gender", 
                              "Comments_Dist_TargetGender.png")
            
            # 2. AI Category Distribution
            plot_filtered_count(df_comments, 'target_category', VALID_CATEGORIES,
                              "Distribution of Comments by AI Category", 
                              "Comments_Dist_Category.png")
            
            # 3. Research Groups Distribution
            plot_filtered_count(df_comments, 'parent_research_group', VALID_GROUPS,
                              "Distribution of Comments by Parent Group", 
                              "Comments_Dist_ParentGroup.png")
            
            # 4. VADER Sentiment Distribution
            plot_vader_kde_filtered(df_comments, 'target_gender', 
                           "Sentiment Distribution: Replies to Men vs Women", 
                           "Comments_VADER_Dist.png")
            
        else:
            print("Skipping detailed comment analysis: 'parent_research_group' missing.")
            
    else:
        print(f"File not found: {COMMENTS_PATH}")

    print("\nAll visualizations completed.")

if __name__ == "__main__":
    main()