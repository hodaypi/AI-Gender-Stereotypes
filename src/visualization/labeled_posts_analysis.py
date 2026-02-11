import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re

PALETTE = {
      'female': '#d5a6bd',      
      'male': '#a4c2f4',        
      'Female': '#d5a6bd',
      'Male': '#a4c2f4',
      'Undetermined': '#909090', 
      'Other/Unlabeled': '#909090',
      
      # research_group
      'female_hard_ai': '#d5a6bd', 
      'male_hard_ai':   '#a4c2f4',
      'female_soft_ai': '#f9cb9c', 
      'male_soft_ai':   '#a2c4c9',

      # ai_category
      'hard_ai': '#a2c4c9',
      'soft_ai': '#f9cb9c'
}

def add_labels(ax, total_count=None):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + (height*0.01), 
                    f'{int(height)}', 
                    ha="center", fontsize=11, fontweight='bold')
            if total_count:
                percentage = (height / total_count) * 100
                if percentage > 4: 
                    ax.text(p.get_x() + p.get_width()/2., height/2, 
                            f'{percentage:.1f}%', 
                            ha="center", fontsize=9, color='white', fontweight='bold')

# ==========================================
#  plot graphs functions
# ==========================================

def plot_gender_distribution(df):
    """1. male VS female"""
    plt.figure(figsize=(7, 5))
    df_gender = df[df['gender_context'].isin(['male', 'female'])]
    ax = sns.countplot(data=df_gender, x='gender_context', palette=PALETTE, order=['male', 'female'])
    plt.title("1. Gender Distribution (Identified Only)", fontsize=14)
    add_labels(ax)
    plt.tight_layout()
    plt.show()

def plot_ai_category_distribution(df):
    """2. hard VS soft"""
    plt.figure(figsize=(7, 5))
    df_ai = df[df['ai_category'].isin(['hard_ai', 'soft_ai'])]
    ax = sns.countplot(data=df_ai, x='ai_category', palette=PALETTE, order=['hard_ai', 'soft_ai'])
    plt.title("3. AI Category Distribution (Identified Only)", fontsize=14)
    add_labels(ax)
    plt.tight_layout()
    plt.show()

def plot_ai_category_distribution_all(df):
    """3. hard VS soft VS other"""
    plt.figure(figsize=(9, 6))
    df['ai_plot'] = df['ai_category'].apply(lambda x: x if x in ['hard_ai', 'soft_ai'] else 'Other/Unlabeled')
    ax = sns.countplot(data=df, x='ai_plot', order=['hard_ai', 'soft_ai', 'Other/Unlabeled'], palette=PALETTE)
    plt.title("4. AI Category Coverage (Full Data)", fontsize=14)
    add_labels(ax, len(df))
    plt.tight_layout()
    plt.show()

def plot_research_groups(df):
    """4. 4 research groups"""
    plt.figure(figsize=(10, 6))
    groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    df_groups = df[df['research_group'].isin(groups)]
    ax = sns.countplot(data=df_groups, x='research_group', order=groups, palette=PALETTE)
    plt.title("5. Research Groups Distribution", fontsize=14)
    plt.xticks(rotation=15)
    add_labels(ax)
    plt.tight_layout()
    plt.show()

def plot_research_groups_all(df):
    """5. 4 research_groups VS Unlabeled """
    plt.figure(figsize=(11, 6))
    groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    df['group_plot'] = df['research_group'].apply(lambda x: x if x in groups else 'Other/Unlabeled')
    order = groups + ['Other/Unlabeled']
    ax = sns.countplot(data=df, x='group_plot', order=order, palette=PALETTE)
    plt.title("6. Research Groups Coverage (Full Data)", fontsize=14)
    plt.xticks(rotation=15)
    add_labels(ax, len(df))
    plt.tight_layout()
    plt.show()

def plot_selftalk_distribution(df):
    """6. self talk male VS female """
    plt.figure(figsize=(7, 5))
    df_self = df[(df['is_self_talk'] == True) & (df['gender_context'].isin(['male', 'female']))]
    ax = sns.countplot(data=df_self, x='gender_context', order=['male', 'female'], palette=PALETTE)
    plt.title("7. Self-Talk by Gender", fontsize=14)
    add_labels(ax)
    plt.tight_layout()
    plt.show()
def plot_group_wordclouds(df):
    """
    Generates 4 Word Clouds - one for each research group.
    """
    groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    
    # remove Stopwords 
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        # filter research group
        subset = df[df['research_group'] == group]
        
        text_data = " ".join(subset['title'].fillna('') + " " + subset['selftext'].fillna(''))
        
        if not text_data.strip():
            axes[i].text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        # create wordcloud
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
    plt.tight_layout()
    plt.show()

def plot_top_words_comparison(df, palette):
    """
    Shows the top 10 most frequent words for each group using Bar Charts.
    """
    groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post", "deleted", "removed", "ai", "one", "would", "like"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        subset = df[df['research_group'] == group]
        text_data = " ".join(subset['title'].fillna('') + " " + subset['selftext'].fillna(''))
        
        words = re.findall(r'\b\w+\b', text_data.lower())
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        common_words = Counter(words).most_common(10)
        
        if not common_words:
            continue
            
        words_list, counts_list = zip(*common_words)
        
        color = palette.get(group, 'grey')
        
        sns.barplot(x=list(counts_list), y=list(words_list), ax=axes[i], color=color)
        axes[i].set_title(f"Top 10 Words: {group}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Frequency")

    plt.suptitle("Most Frequent Words Analysis", fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_post_length_distribution(df, palette):
    """
    Boxplot showing the length of posts (word count) across groups.
    Answers: Do Hard AI posts tend to be longer?
    """
    df['word_count'] = (df['title'].fillna('') + " " + df['selftext'].fillna('')).apply(lambda x: len(str(x).split()))
    
    groups = ['female_hard_ai', 'male_hard_ai', 'female_soft_ai', 'male_soft_ai']
    df_groups = df[df['research_group'].isin(groups)]

    plt.figure(figsize=(12, 6))
    
    sns.boxplot(data=df_groups, x='research_group', y='word_count', order=groups, palette=palette, showfliers=False)
    
    plt.title("Post Length Distribution (Word Count) by Research Group", fontsize=16)
    plt.ylabel("Number of Words per Post")
    plt.xlabel("Research Group")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
# ==========================================
#  MAIN FUNCTION
# ==========================================
def main():

  PROCESSED_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
  POSTS_FILE = "labeled_all_gender_posts.csv"

  #colors
  PALETTE = {
        'female': '#d5a6bd',      
        'male': '#a4c2f4',        
        'Female': '#d5a6bd',
        'Male': '#a4c2f4',
        'Undetermined': '#909090', 
        'Other/Unlabeled': '#909090',
        
        # research_group
        'female_hard_ai': '#d5a6bd', 
        'male_hard_ai':   '#a4c2f4',
        'female_soft_ai': '#f9cb9c', 
        'male_soft_ai':   '#a2c4c9',

        # ai_category
        'hard_ai': '#a2c4c9',
        'soft_ai': '#f9cb9c'
  }

  path = os.path.join(PROCESSED_DIR, POSTS_FILE)
  
  if not os.path.exists(path):
      print(f"Error: File not found at {path}")
      return

  print("Loading data...")
  df = pd.read_csv(path)
  print(f"Data loaded: {len(df)} posts.")
  sns.set_style("whitegrid")

  
  #1
  print("Generating Graph 1...")
  plot_gender_distribution(df)
  #2
  print("Generating Graph 2...")
  plot_ai_category_distribution(df)
  #3
  print("Generating Graph 3...")
  plot_ai_category_distribution_all(df)
  #4
  print("Generating Graph 4...")
  plot_research_groups(df)
  #5
  print("Generating Graph 5...")
  plot_research_groups_all(df)
  #6
  print("Generating Graph 6...")
  plot_selftalk_distribution(df)
  # 7. Word Clouds
  print("Generating Graph 7 (Word Clouds)...")
  plot_group_wordclouds(df)

  # 8. Top 10 Words
  print("Generating Graph 8 (Top Words)...")
  plot_top_words_comparison(df, PALETTE)

  # 9. Post Length
  print("Generating Graph 9 (Post Length)...")
  plot_post_length_distribution(df, PALETTE)  
  

if __name__ == "__main__":
    main()