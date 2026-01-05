import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os

# --- הגדרות נתיבים ---
# אנא ודאי שהנתיב והשם של הקובץ מדויקים
DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data/raw" 
INPUT_FILE = "all_gender_posts.txt"  # שימי לב שזה קובץ ה-CSV הגולמי

def analyze_raw_data():
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    # 1. טעינת הדאטה הגולמי
    print(f"Loading raw data from: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    # מילוי ערכים חסרים בטקסט (למניעת שגיאות בענן המילים)
    df['title'] = df['title'].fillna('')
    df['selftext'] = df['selftext'].fillna('')
    
    # יצירת עמודת טקסט מלא לניתוח
    df['full_text'] = df['title'] + " " + df['selftext']

    # --- חלק א': סטטיסטיקה כללית ---
    total_posts = len(df)
    unique_subs = df['subreddit'].nunique()
    
    print("\n" + "="*40)
    print("RAW DATA SUMMARY")
    print("="*40)
    print(f"Total Posts Collected: {total_posts}")
    print(f"Number of Subreddits:  {unique_subs}")
    print("="*40)

    # --- חלק ב': גרף התפלגות לפי Subreddit ---
    plt.figure(figsize=(12, 6))
    
    # סופרים כמה פוסטים יש בכל סאב-רדיט
    sub_counts = df['subreddit'].value_counts()
    
    # יצירת הגרף
    ax = sns.barplot(x=sub_counts.index, y=sub_counts.values, palette="viridis")
    
    plt.title(f"Distribution of Collected Posts by Subreddit (Total: {total_posts})", fontsize=16)
    plt.xlabel("Subreddit", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.xticks(rotation=45, ha='right') # סיבוב השמות כדי שיהיה קריא
    
    # הוספת המספרים על העמודות
    for i, v in enumerate(sub_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

    # --- חלק ג': ענני מילים לכל Subreddit ---
    print("\nGenerating Word Clouds per Subreddit...")
    
    # הגדרת מילים להסרה (Stopwords) - מילים נפוצות שלא נותנות מידע
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "com", "www", "reddit", "post", "deleted", "removed", "amp", "people", "think"])

    # ניקח את ה-Subreddits המובילים (למשל ה-6 הגדולים) כדי לא להעמיס על הגרף
    top_subs = sub_counts.head(6).index.tolist()
    
    # הגדרת הגריד של הגרפים (2 שורות, 3 עמודות)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # משטיח את המערך כדי שיהיה קל לרוץ עליו בלולאה

    for i, sub_name in enumerate(top_subs):
        # סינון הטקסטים ששייכים רק לסאב-רדיט הנוכחי
        subset = df[df['subreddit'] == sub_name]
        text_data = " ".join(subset['full_text'].astype(str))
        
        # יצירת ענן המילים
        wordcloud = WordCloud(
            stopwords=stopwords,
            background_color="white",
            width=800, 
            height=500,
            colormap="ocean", # צבעים יפים
            max_words=80
        ).generate(text_data)
        
        # הצגה בתוך הגריד
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f"r/{sub_name} ({len(subset)} posts)", fontsize=14, fontweight='bold')
        axes[i].axis("off") # הסרת צירים

    # אם יש פחות מ-6 סאב-רדיטים, נסתיר את הגרפים הריקים שנשארו
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Word Clouds by Subreddit (Raw Data)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # כדי שהכותרת הראשית לא תסתיר
    plt.show()

def main():

  analyze_raw_data()

if __name__ == "__main__":
    main()