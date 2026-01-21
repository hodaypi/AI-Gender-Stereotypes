import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

#NLTK
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextCleaner:
    def __init__(self):
        # Stopwords
        self.stop_words = set(stopwords.words('english'))
        custom_stops = [
            'https', 'http', 'com', 'www', 'reddit', 'post', 'comment', 
            'deleted', 'removed', 'would', 'could', 'should', 'like', 'one', 
            'get', 'know', 'think', 'people', 'amp', 'x200b'
        ]
        self.stop_words.update(custom_stops)
        
        # init Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
    
    # lowercasing
    def to_lowercase(self, text):
        return str(text).lower()
    
    # Tokenization
    def tokenize_text(self, text):
        return word_tokenize(text)

    # StopWords
    def remove_stopwords(self, tokens):
        return [w for w in tokens if w not in self.stop_words and len(w) > 2]

    #Lemmatization
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(w, pos='v') for w in tokens]

    def preprocess_full_pipeline(self, text):
        if pd.isna(text) or text == '':
            return []
        
        text = self.to_lowercase(text)
        tokens = self.tokenize_text(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        
        return tokens

# ==========================================
# remove duplications
# ==========================================
def remove_duplicates(df):
    """
    Removes rows with identical text content.
    we saw there are many duplicated comments.
    """
    initial_count = len(df)
    
    df_clean = df.drop_duplicates(subset=['full_text'], keep='first').copy()
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    print(f"\n--- Deduplication Report ---")
    print(f"Original size: {initial_count}")
    print(f"Duplicates removed: {removed_count}")
    print(f"Clean size: {final_count}")
    print(f"----------------------------\n")
    
    return df_clean

def run_preprocessing(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # check if posts or comments
    if 'title' in df.columns and 'selftext' in df.columns:
        print("Detected POSTS dataset. Merging title + selftext...")
        df['full_text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
        
    elif 'body' in df.columns:
        print("Detected COMMENTS dataset. Using 'body' column...")
        df['full_text'] = df['body'].fillna('')
        
    else:
        print("Error: Could not find text columns (title/selftext or body).")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # remove duplication before cleaning
    df = remove_duplicates(df)
    
  
    cleaner = TextCleaner()
    
    print("Starting preprocessing pipeline (tokenization, cleaning, lemmatization)...")
    df['cleaned_tokens'] = df['full_text'].apply(cleaner.preprocess_full_pipeline)
    
    initial_len = len(df)
    df = df[df['cleaned_tokens'].map(len) > 0]
    print(f"Removed {initial_len - len(df)} empty rows after cleaning.")
    
    print(f"Saving processed data (Pickle) to {output_path}...")
    df.to_pickle(output_path)
    print("Done!")
   
def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    
    print("\n--- Processing POSTS ---")
    run_preprocessing(
        os.path.join(BASE_DIR, "processed/final_posts_dataset.csv"),
        os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")
    )
    
    print("\n--- Processing COMMENTS ---")
    run_preprocessing(
        os.path.join(BASE_DIR, "processed/final_comments_dataset.csv"), 
        os.path.join(BASE_DIR, "processed/word2vec_comments.pkl")
    )

if __name__ == "__main__":
    main()