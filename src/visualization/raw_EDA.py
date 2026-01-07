import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import math


DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data/raw" 
INPUT_FILE = "all_gender_posts.txt"  

def analyze_raw_data():
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    print(f"Loading raw data from: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    df['title'] = df['title'].fillna('')
    df['selftext'] = df['selftext'].fillna('')
    
    df['full_text'] = df['title'] + " " + df['selftext']

    # general statistics
    total_posts = len(df)
    unique_subs = df['subreddit'].nunique()
    
    print("\n" + "="*40)
    print("RAW DATA SUMMARY")
    print("="*40)
    print(f"Total Posts Collected: {total_posts}")
    print(f"Number of Subreddits:  {unique_subs}")
    print("="*40)

    # Subreddit distribution
    plt.figure(figsize=(12, 6))
    sub_counts = df['subreddit'].value_counts()
    
    ax = sns.barplot(x=sub_counts.index, y=sub_counts.values, palette="viridis")
    
    plt.title(f"Distribution of Collected Posts by Subreddit (Total: {total_posts})", fontsize=16)
    plt.xlabel("Subreddit", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.xticks(rotation=45, ha='right') 
    
    for i, v in enumerate(sub_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

    # Subreddit word clouds
    print("\nGenerating Word Clouds per Subreddit...")
    
    # remove Stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post", "deleted", "removed", "amp"])

    # ניקח את ה-Subreddits המובילים (למשל ה-6 הגדולים) כדי לא להעמיס על הגרף
    #top_subs = sub_counts.head(6).index.tolist()
    
    #dynamic graph size
    num_subs = len(sub_counts)
    cols = 3 
    rows = math.ceil(num_subs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() 

    for i, sub_name in enumerate(sub_counts.index):
        subset = df[df['subreddit'] == sub_name]
        text_data = " ".join(subset['full_text'].astype(str))
        
        if not text_data.strip():
            axes[i].text(0.5, 0.5, "Not enough text", ha='center', va='center')
            axes[i].set_title(f"r/{sub_name}", fontsize=14, fontweight='bold')
            axes[i].axis("off")
            continue

        try:
            wordcloud = WordCloud(
                stopwords=stopwords,
                background_color="white",
                width=800, 
                height=500,
                colormap="ocean", 
                max_words=80
            ).generate(text_data)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f"r/{sub_name} ({len(subset)} posts)", fontsize=14, fontweight='bold')
            axes[i].axis("off")
        except ValueError:
            axes[i].text(0.5, 0.5, "No valid words", ha='center', va='center')
            axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Word Clouds by Subreddit (Raw Data)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    plt.show()

def main():

  analyze_raw_data()

if __name__ == "__main__":
    main()