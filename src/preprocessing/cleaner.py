"""
Text Cleaner for Reddit Data
Prepares data for VADER sentiment analysis and LIWC.
Preserves punctuation, case, and emojis.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

#Hardcoded Paths
DATA_DIR = Path("/content/gdrive/MyDrive/Data mining/text mining/data")

RAW_DIR = DATA_DIR / "raw"       
INTERIM_DIR = DATA_DIR / "interim" 

def clean_text_for_vader(text):
    """
    Cleans text while preserving features needed for sentiment analysis.
    Keeps: Punctuation (!, ?), Capitalization (CAPS), Emojis for for VADER.
    Removes: URLs, Markdown links, HTML entities, User mentions.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs (http/https)
    text = re.sub(r'http\S+', '', text)
    
    # 2. Remove Markdown links [text](url) -> keep only "text"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 3. Remove Reddit user mentions (u/username) & Subreddit links (r/subreddit)
    text = re.sub(r'u/\S+', '', text)
    text = re.sub(r'r/\S+', '', text)
    
    # 4. Remove HTML entities (like &amp; &gt;)
    text = re.sub(r'&[a-z]+;', ' ', text)

    # 5. Remove Newlines and extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_file(input_filename, output_filename, data_type='posts'):
    """
    Reads a raw JSON file, cleans the text fields, and saves to interim folder.
    """
    input_path = RAW_DIR / input_filename
    output_path = INTERIM_DIR / output_filename

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_path}")
        return

    cleaned_data = []
    
    for item in tqdm(data, desc=f"Cleaning {input_filename}"):
        clean_item = item.copy()
        
        if data_type == 'posts':
            if 'title' in clean_item:
                clean_item['title'] = clean_text_for_vader(clean_item['title'])
            if 'selftext' in clean_item:
                clean_item['selftext'] = clean_text_for_vader(clean_item['selftext'])
                
        elif data_type == 'comments':
            if 'body' in clean_item:
                clean_item['body'] = clean_text_for_vader(clean_item['body'])
        
        cleaned_data.append(clean_item)

    print(f"Saving cleaned data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

def main():
    print("=" * 60)
    print("STARTING DATA CLEANING (PRESERVING SENTIMENT FEATURES)")
    print("=" * 60)
    
    #clean posts
    process_file("all_gender_posts.json", "all_posts_clean.json", data_type='posts')

    #clean comments
    process_file("comments.json", "comments_clean.json", data_type='comments')

    print("\nDone! Cleaned files are in the 'data/interim' folder.")

if __name__ == "__main__":
    main()