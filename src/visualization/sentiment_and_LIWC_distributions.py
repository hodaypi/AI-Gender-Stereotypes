import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


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
      
      print("3. Plotting Correlations...")
      plot_correlation_matrix(df)
      
  except FileNotFoundError:
      print("Error: File not found.")
  except KeyError as e:
      print(f"Error: Missing column in dataset - {e}")

if __name__ == "__main__":
    main()