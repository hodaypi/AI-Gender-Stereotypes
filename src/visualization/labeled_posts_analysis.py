import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def add_labels(ax, total_count=None):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            # המספר המוחלט
            ax.text(p.get_x() + p.get_width()/2., height + (height*0.01), 
                    f'{int(height)}', 
                    ha="center", fontsize=11, fontweight='bold')
            # אחוזים (אם נשלח סך הכל)
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

# ==========================================
#  MAIN FUNCTION
# ==========================================
def main():

  PROCESSED_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
  POSTS_FILE = "labeled_all_gender_posts.csv"

  #colors
  PALETTE = {
      'female': '#e74c3c',      
      'male': '#2c3e50',        
      'Female': '#e74c3c',
      'Male': '#2c3e50',
      'Undetermined': '#95a5a6', 
      'Other/Unlabeled': '#95a5a6',
      
      # research_group
      'female_hard_ai': '#c0392b',
      'male_hard_ai': '#2c3e50',
      'female_soft_ai': '#e67e22',
      'male_soft_ai': '#3498db',
      
      # ai_category
      'hard_ai': '#2c3e50',
      'soft_ai': '#e67e22'
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
  

if __name__ == "__main__":
    main()