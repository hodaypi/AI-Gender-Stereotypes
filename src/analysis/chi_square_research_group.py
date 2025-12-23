import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_chi_square_research_group(TARGET_GROUPS,processed_dir,posts_file):
    # 1. טעינה
    posts_path = os.path.join(processed_dir, posts_file)
    df = pd.read_csv(posts_path)
    print(f"Total posts loaded: {len(df)}")

    # 2. סינון: משאירים רק שורות שה-research_group שלהן הוא אחד מ-4 הקבוצות
    df_clean = df[df['research_group'].isin(TARGET_GROUPS)].copy()
    print(f"Posts in target groups: {len(df_clean)}")

    # 3. פירוק העמודה research_group למשתנים לצורך הטבלה הסטטיסטית
    # אנחנו צריכים להפריד חזרה למגדר ולנושא כדי לבנות את המטריצה
    
    # מיפוי למגדר
    df_clean['stat_gender'] = df_clean['research_group'].apply(
        lambda x: 'Female' if x.startswith('female') else 'Male'
    )
    
    # מיפוי לנושא (Hard vs Soft)
    df_clean['stat_topic'] = df_clean['research_group'].apply(
        lambda x: 'Hard AI' if 'hard_ai' in x else 'Soft AI'
    )

    # 4. יצירת טבלת שכיחות (Contingency Table)
    # שורות: Female/Male, עמודות: Hard/Soft
    contingency_table = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'])
    
    print("\n--- Contingency Table (Counts) ---")
    print(contingency_table)

    # 5. חישוב אחוזים (להבנה ויזואלית)
    contingency_pct = pd.crosstab(df_clean['stat_gender'], df_clean['stat_topic'], normalize='index') * 100
    print("\n--- Distribution Percentages ---")
    print(contingency_pct.round(2).astype(str) + '%')

    # 6. מבחן Chi-Square
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n" + "="*40)
    print("RQ1 RESULTS: The Gender-Topic Split")
    print("="*40)
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value:        {p:.4e}") 
    
    if p < 0.05:
        print("\n=> RESULT: SIGNIFICANT BIAS DETECTED.")
        print("There is a statistically significant dependency between Gender and AI Topic.")
        print("(This proves that women and men appear in different frequencies across Hard vs Soft AI).")
    else:
        print("\n=> RESULT: NO SIGNIFICANT BIAS.")

    # 7. ויזואליזציה
    plt.figure(figsize=(8, 6))
    
    # צבעים: Hard=כחול כהה, Soft=אדום/כתום
    ax = contingency_pct.plot(kind='bar', stacked=True, color=['#2c3e50', '#e67e22'], alpha=0.9, width=0.6)
    
    plt.title("The Split: AI Topic Distribution by Gender Context")
    plt.xlabel("Gender")
    plt.ylabel("Percentage (%)")
    plt.legend(title='AI Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # הוספת מספרים על הגרף
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

    plt.tight_layout()
    plt.show()

def main():
  # נתיבים
  processed_dir = "/content/gdrive/MyDrive/Data mining/text mining/data/processed"
  posts_file = "final_posts_dataset.csv"

  # הגדרת הקבוצות המדויקות שאנחנו רוצים להשוות
  TARGET_GROUPS = [
      'female_hard_ai', 
      'female_soft_ai', 
      'male_hard_ai', 
      'male_soft_ai'
  ]
  run_chi_square_research_group(TARGET_GROUPS,processed_dir,posts_file)

if __name__ == "__main__":
    main()
    