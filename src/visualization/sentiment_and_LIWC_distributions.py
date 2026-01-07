import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from math import pi


# columns
VADER_COL = 'sentiment_vader'
LIWC_COLS = [
    'LIWC_power_achievement', 
    'LIWC_negative_emotions', 
    'LIWC_cognitive_processes', 
    'LIWC_certainty', 
    'LIWC_affiliation_social', 
    'LIWC_perceptual_processes'
]

# colors
PALETTE = {'female': '#e74c3c', 'male': '#2c3e50'}

def plot_vader_distribution(df):
    """1. התפלגות סנטימנט VADER (צפיפות)"""
    plt.figure(figsize=(10, 6))
    
    # סינון: רק גברים ונשים מזוהים
    df_clean = df[df['gender_context'].isin(['male', 'female'])]
    
    # גרף KDE (צפיפות)
    sns.kdeplot(data=df_clean, x=VADER_COL, hue='gender_context', 
                palette=PALETTE, fill=True, alpha=0.3, linewidth=2)
    
    plt.title("Distribution of VADER Sentiment Scores", fontsize=15)
    plt.xlabel("Sentiment Score (-1 = Negative, 1 = Positive)")
    plt.xlim(-1, 1)
    
    # הוספת קו האפס (ניטרלי)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.text(0.05, plt.gca().get_ylim()[1]*0.9, 'Neutral', color='grey')
    
    plt.tight_layout()
    plt.show()

def plot_liwc_comparison(df):
    """2. השוואת כל קטגוריות LIWC בין גברים לנשים (Boxplot)"""
    plt.figure(figsize=(14, 8))
    
    df_clean = df[df['gender_context'].isin(['male', 'female'])]
    
    # כדי להציג את כל העמודות בגרף אחד, צריך לעשות להן "Melt"
    # זה הופך את הדאטה מרחב לארוך (מתאים לסיברון)
    df_melted = df_clean.melt(id_vars=['gender_context'], 
                              value_vars=LIWC_COLS, 
                              var_name='LIWC_Category', 
                              value_name='Score')
    
    # ניקוי השם של הקטגוריה לתצוגה יפה (מוריד את ה-LIWC_)
    df_melted['LIWC_Category'] = df_melted['LIWC_Category'].str.replace('LIWC_', '').str.replace('_', ' ').str.title()
    
    # יצירת Boxplot משולב
    sns.boxplot(data=df_melted, x='LIWC_Category', y='Score', hue='gender_context', 
                palette=PALETTE, showfliers=False) # showfliers=False מסתיר חריגים קיצוניים כדי שהגרף יהיה קריא
    
    plt.title("Psychological Features Comparison (LIWC Scores)", fontsize=16)
    plt.xticks(rotation=15)
    plt.ylabel("Score / Frequency")
    plt.xlabel("Psychological Category")
    plt.legend(title='Gender')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """3. מפת חום - קורלציות בין כל המשתנים"""
    plt.figure(figsize=(10, 8))
    
    # רשימת כל העמודות המספריות שמעניינות אותנו
    cols_to_check = [VADER_COL] + LIWC_COLS
    
    # חישוב קורלציה
    corr_matrix = df[cols_to_check].corr()
    
    # קיצור שמות לתצוגה בגרף
    labels = [c.replace('LIWC_', '').replace('sentiment_', '') for c in cols_to_check]
    
    # יצירת המפה
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title("Correlation Heatmap: Sentiment vs. Psychology", fontsize=16)
    plt.tight_layout()
    plt.show()
def plot_radar_chart(df):
    """4. גרף רדאר - פרופיל פסיכולוגי ממוצע (גברים מול נשים)"""
    # חישוב ממוצעים לכל מגדר בכל קטגוריה
    df_clean = df[df['gender_context'].isin(['male', 'female'])]
    means = df_clean.groupby('gender_context')[LIWC_COLS].mean()
    
    # הכנת הנתונים לגרף
    categories = [c.replace('LIWC_', '').replace('_', ' ').title() for c in LIWC_COLS]
    N = len(categories)
    
    # חישוב זוויות (יוצרים מעגל)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # סגירת המעגל
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # ציר ה-X (הקטגוריות)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # ציור הקווים עבור כל מגדר
    for gender in ['male', 'female']:
        values = means.loc[gender].tolist()
        values += values[:1] # סגירת המעגל
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=gender, color=PALETTE[gender])
        ax.fill(angles, values, color=PALETTE[gender], alpha=0.1)
    
    plt.title("Average Psychological Profile (Radar Chart)", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def plot_sentiment_by_topic_violin(df):
    """5. גרף כינור - סנטימנט לפי מגדר וסוג ה-AI"""
    plt.figure(figsize=(10, 6))
    
    # סינון רלוונטי
    df_clean = df[
        (df['gender_context'].isin(['male', 'female'])) & 
        (df['ai_category'].isin(['hard_ai', 'soft_ai']))
    ]
    
    # גרף כינור (שילוב של Boxplot וצפיפות)
    sns.violinplot(data=df_clean, x='ai_category', y=VADER_COL, hue='gender_context',
                   split=True, inner="quart", palette=PALETTE)
    
    plt.title("Sentiment Distribution by AI Topic & Gender", fontsize=15)
    plt.xlabel("AI Category")
    plt.ylabel("Sentiment Score (VADER)")
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5) # קו האפס
    
    plt.tight_layout()
    plt.show()

def plot_certainty_vs_sentiment(df):
    """6. גרף פיזור - הקשר בין ביטחון (Certainty) לסנטימנט"""
    plt.figure(figsize=(10, 6))
    
    df_clean = df[df['gender_context'].isin(['male', 'female'])]
    
    # שימוש ב-lmplot שמוסיף גם קו מגמה (רגרסיה)
    # אנחנו רוצים לראות אם הקו עולה (קשר חיובי) או יורד
    sns.regplot(data=df_clean[df_clean['gender_context']=='male'], 
                x='LIWC_certainty', y=VADER_COL, 
                scatter_kws={'alpha':0.1}, line_kws={'color': PALETTE['male']}, label='Male')
                
    sns.regplot(data=df_clean[df_clean['gender_context']=='female'], 
                x='LIWC_certainty', y=VADER_COL, 
                scatter_kws={'alpha':0.1, 'color': PALETTE['female']}, line_kws={'color': PALETTE['female']}, label='Female')

    plt.title("Correlation: Confidence (Certainty) vs. Sentiment", fontsize=15)
    plt.xlabel("LIWC Certainty Score")
    plt.ylabel("VADER Sentiment Score")
    plt.legend()
    plt.xlim(0, 20) # מגביל את הציר כדי לא לראות חריגים רחוקים מדי
    
    plt.tight_layout()
    plt.show()


def main():
  
  PROCESSED_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
  POSTS_FILE = "final_posts_dataset.csv"

  path = os.path.join(PROCESSED_DIR, POSTS_FILE)
  print(f"Loading data from: {path}")
  
  try:
      df = pd.read_csv(path)
      print(f"Loaded {len(df)} posts.")
      sns.set_style("whitegrid")
      
      
      print("1. Plotting VADER Distribution...")
      plot_vader_distribution(df)
      
      print("2. Plotting LIWC Comparison (Big Graph)...")
      plot_liwc_comparison(df)
      
      #print("3. Plotting Correlations...")
      #plot_correlation_matrix(df)
      
      print("4. Plotting Radar Chart...")
      plot_radar_chart(df)

      print("5. Plotting Sentiment by Topic (Violin)...")
      plot_sentiment_by_topic_violin(df)

      print("6. Plotting Certainty vs Sentiment Scatter...")
      plot_certainty_vs_sentiment(df)
      
  except FileNotFoundError:
      print("Error: File not found.")
  except KeyError as e:
      print(f"Error: Missing column in dataset - {e}")

if __name__ == "__main__":
    main()