"""
Sentiment & Psychological Analyzer (VADER + Dictionary-based LIWC)
------------------------------------------------------------------
1. Merges posts and comments to inherit context (Research Group).
2. Calculates Sentiment (VADER Compound score).
3. Calculates Psychological scores (LIWC) based on generated lexicons.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Set
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Paths
DATA_DIR = Path("/content/gdrive/MyDrive/Data mining/text mining/data")
PROCESSED_DIR = DATA_DIR / "processed" 
LEXICONS_DIR = DATA_DIR / "lexicons"

#LIWC files created by lexical_collector
LIWC_CATEGORIES = [
    "power_achievement",
    "cognitive_processes",
    "negative_emotions",
    "certainty",
    "affiliation_social",
    "perceptual_processes"
]

def load_liwc_lexicons() -> Dict[str, Set[str]]:
    """
    Loads LIWC lexicon files from the disk into a dictionary of sets.
    Returns: {'power_achievement': {'win', 'success', ...}, ...}
    """
    print(f"Loading LIWC lexicons from: {LEXICONS_DIR}")
    lexicons = {}
    
    for category in LIWC_CATEGORIES:
        file_path = LEXICONS_DIR / f"{category}.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = set(line.strip().lower() for line in f if line.strip())
                lexicons[category] = words
                print(f"   > Loaded '{category}': {len(words)} words")
        except FileNotFoundError:
            print(f"   > WARNING: Lexicon file not found: {category}.txt (Score will be 0)")
            lexicons[category] = set()
            
    return lexicons

def get_vader_score(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculates VADER compound score (-1 to 1)."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def get_liwc_scores(text: str, lexicons: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Calculates percentage of words belonging to each LIWC category.
    """
    if not isinstance(text, str) or not text.strip():
        return {f"LIWC_{cat}": 0.0 for cat in lexicons}

    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    if total_words == 0:
        return {f"LIWC_{cat}": 0.0 for cat in lexicons}

    scores = {}
    for category, lexicon_words in lexicons.items():
        match_count = 0
        for word in words:
            if word in lexicon_words:
                match_count += 1
        
        # חישוב אחוזים: (מספר המילים מהקטגוריה / סה"כ מילים) * 100
        scores[f"LIWC_{category}"] = (match_count / total_words) * 100

    return scores

def process_data_enrichment(posts_file: str, comments_file: str):
    """
    Main orchestration function.
    """
    # init
    print("Initializing VADER and Loading Lexicons...")
    analyzer = SentimentIntensityAnalyzer()
    liwc_lexicons = load_liwc_lexicons()
    
    # load data
    posts_path = PROCESSED_DIR / posts_file
    comments_path = PROCESSED_DIR / comments_file
    
    print(f"Loading posts from {posts_path}...")
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
        
    print(f"Loading comments from {comments_path}...")
    with open(comments_path, 'r', encoding='utf-8') as f:
        comments_data = json.load(f)

    # -------------------------------------------------------
    # labeling (VADER + LIWC)
    # -------------------------------------------------------
    enriched_posts = []
    for post in tqdm(posts_data, desc="Analyzing Posts"):
        full_text = f"{post.get('title', '')} {post.get('selftext', '')}"
        
        # vader
        post['sentiment_vader'] = get_vader_score(full_text, analyzer)
        
        # LIWC 
        liwc_scores = get_liwc_scores(full_text, liwc_lexicons)
        post.update(liwc_scores) 
        
        enriched_posts.append(post)

    # save to final data file
    df_posts = pd.DataFrame(enriched_posts)
    final_posts_path = PROCESSED_DIR / "final_posts_dataset.csv"
    df_posts.to_csv(final_posts_path, index=False, encoding='utf-8')
    print(f"Saved enriched posts to: {final_posts_path}")

    # -------------------------------------------------------
    # merge for comments
    # we  want to know foreach comment the reasearch group of its post 
    # -------------------------------------------------------
    post_context_map = df_posts.set_index('id')[['research_group', 'gender_context', 'ai_category']].to_dict('index')

    enriched_comments = []
    for comment in tqdm(comments_data, desc="Analyzing Comments"):
        text = comment.get('body', '')
        parent_id = comment.get('post_id')
        
        # Merge
        parent_info = post_context_map.get(parent_id)
        if parent_info:
            comment['parent_research_group'] = parent_info['research_group']
            comment['parent_gender'] = parent_info['gender_context']
            comment['parent_ai_category'] = parent_info['ai_category']
        else:
            comment['parent_research_group'] = 'unknown'
            comment['parent_gender'] = 'unknown'
            comment['parent_ai_category'] = 'unknown'

        # vader
        comment['sentiment_vader'] = get_vader_score(text, analyzer)
        
        # LIWC
        liwc_scores = get_liwc_scores(text, liwc_lexicons)
        comment.update(liwc_scores)
        
        enriched_comments.append(comment)

    # save to final data file
    df_comments = pd.DataFrame(enriched_comments)
    final_comments_path = PROCESSED_DIR / "final_comments_dataset.csv"
    df_comments.to_csv(final_comments_path, index=False, encoding='utf-8')
    print(f"Saved enriched comments to: {final_comments_path}")

def main():
    print("=" * 60)
    print("STARTING FULL ANALYSIS (MERGE + VADER + FULL LIWC)")
    print("=" * 60)
    
    process_data_enrichment("labeled_all_gender_posts.json", "labeled_comments.json")
    
    print("\nAnalysis Complete! Final CSV files are ready in 'data/processed'.")

if __name__ == "__main__":
    main()