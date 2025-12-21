"""
Data Labeler for Gender Stereotypes in AI Research

This script labels collected Reddit data with:
- Gender categories (female/male mentions)
- AI category (Hard AI / Soft AI)
- Self-talk identification (first-person posts)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# Data paths
DATA_DIR = Path("/content/gdrive/MyDrive/Data mining/text mining/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LEXICONS_DIR = DATA_DIR / "lexicons"


def load_lexicon(filename: str) -> List[str]:
    """Load a lexicon file and return list of terms."""
    filepath = LEXICONS_DIR / filename
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_all_lexicons() -> Dict[str, List[str]]:
    """Load all lexicon files."""
    return {
        "female": load_lexicon("gender_female.txt"),
        "male": load_lexicon("gender_male.txt"),
        "hard_ai": load_lexicon("hard_ai.txt"),
        "soft_ai": load_lexicon("soft_ai.txt")
    }


def count_term_occurrences(text: str, terms: List[str]) -> int:
    """Count how many times terms from a list appear in text."""
    text_lower = text.lower()
    count = 0
    for term in terms:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(term) + r'\b'
        count += len(re.findall(pattern, text_lower))
    return count


def detect_gender_context(text: str, lexicons: Dict[str, List[str]]) -> Tuple[str, int, int]:
    """
    Detect gender context in text.

    Returns:
        Tuple of (primary_gender, female_count, male_count)
    """
    female_count = count_term_occurrences(text, lexicons["female"])
    male_count = count_term_occurrences(text, lexicons["male"])

    if female_count > male_count:
        primary = "female"
    elif male_count > female_count:
        primary = "male"
    elif female_count > 0:  # Both equal and non-zero
        primary = "both"
    else:
        primary = "none"

    return primary, female_count, male_count


def detect_ai_category(text: str, lexicons: Dict[str, List[str]]) -> Tuple[str, int, int]:
    """
    Detect AI category in text (Hard AI vs Soft AI).

    Returns:
        Tuple of (primary_category, hard_count, soft_count)
    """
    hard_count = count_term_occurrences(text, lexicons["hard_ai"])
    soft_count = count_term_occurrences(text, lexicons["soft_ai"])

    if hard_count > soft_count:
        primary = "hard_ai"
    elif soft_count > hard_count:
        primary = "soft_ai"
    elif hard_count > 0:
        primary = "mixed"
    else:
        primary = "general"

    return primary, hard_count, soft_count


def detect_self_talk(text: str) -> Tuple[bool, int]:
    """
    Detect if text is self-referential (first-person).

    Returns:
        Tuple of (is_self_talk, first_person_count)
    """
    first_person_patterns = [
        r'\bi\b', r'\bmy\b', r'\bme\b', r'\bmyself\b', r'\bmine\b',
        r"\bi'm\b", r"\bi've\b", r"\bi'll\b", r"\bi'd\b"
    ]

    text_lower = text.lower()
    count = 0
    for pattern in first_person_patterns:
        count += len(re.findall(pattern, text_lower))

    # Consider it self-talk if there are multiple first-person references
    is_self_talk = count >= 2

    return is_self_talk, count


def label_post(post: Dict, lexicons: Dict[str, List[str]]) -> Dict:
    """Add labels to a single post."""
    # Combine title and selftext for analysis
    full_text = f"{post.get('title', '')} {post.get('selftext', '')}"

    # Gender detection
    gender, female_count, male_count = detect_gender_context(full_text, lexicons)
    post["gender_context"] = gender
    post["female_term_count"] = female_count
    post["male_term_count"] = male_count

    # AI category detection
    ai_cat, hard_count, soft_count = detect_ai_category(full_text, lexicons)
    post["ai_category"] = ai_cat
    post["hard_ai_count"] = hard_count
    post["soft_ai_count"] = soft_count

    # Self-talk detection
    is_self_talk, first_person_count = detect_self_talk(full_text)
    post["is_self_talk"] = is_self_talk
    post["first_person_count"] = first_person_count

    # Combined label for research groups
    if gender in ["female", "male"]:
        post["research_group"] = f"{gender}_{ai_cat}"
    else:
        post["research_group"] = "control"

    return post


def label_comment(comment: Dict, lexicons: Dict[str, List[str]]) -> Dict:
    """Add labels to a single comment."""
    text = comment.get("body", "")

    # Gender detection
    gender, female_count, male_count = detect_gender_context(text, lexicons)
    comment["gender_context"] = gender
    comment["female_term_count"] = female_count
    comment["male_term_count"] = male_count

    # AI category detection
    ai_cat, hard_count, soft_count = detect_ai_category(text, lexicons)
    comment["ai_category"] = ai_cat
    comment["hard_ai_count"] = hard_count
    comment["soft_ai_count"] = soft_count

    # Self-talk detection
    is_self_talk, first_person_count = detect_self_talk(text)
    comment["is_self_talk"] = is_self_talk
    comment["first_person_count"] = first_person_count

    return comment


def process_posts(input_file: str, output_file: str):
    """Process and label all posts."""
    input_path = RAW_DIR / input_file
    output_path = PROCESSED_DIR / output_file

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    # Load lexicons
    lexicons = load_all_lexicons()

    # Label posts
    labeled_posts = []
    for post in tqdm(posts, desc="Labeling posts"):
        labeled_post = label_post(post, lexicons)
        labeled_posts.append(labeled_post)

    # Save labeled data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_posts, f, ensure_ascii=False, indent=2)

    # Also save as CSV for easier analysis
    df = pd.DataFrame(labeled_posts)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Processed {len(labeled_posts)} posts")
    print(f"Saved to {output_path} and {csv_path}")

    # Print summary statistics
    print_label_summary(df)


def process_comments(input_file: str, output_file: str):
    """Process and label all comments."""
    input_path = RAW_DIR / input_file
    output_path = PROCESSED_DIR / output_file

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    # Load lexicons
    lexicons = load_all_lexicons()

    # Label comments
    labeled_comments = []
    for comment in tqdm(comments, desc="Labeling comments"):
        labeled_comment = label_comment(comment, lexicons)
        labeled_comments.append(labeled_comment)

    # Save labeled data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_comments, f, ensure_ascii=False, indent=2)

    # Also save as CSV
    df = pd.DataFrame(labeled_comments)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Processed {len(labeled_comments)} comments")
    print(f"Saved to {output_path} and {csv_path}")


def print_label_summary(df: pd.DataFrame):
    """Print summary statistics for labeled data."""
    print("\n" + "=" * 50)
    print("Label Summary Statistics")
    print("=" * 50)

    print("\nGender Context Distribution:")
    print(df['gender_context'].value_counts())

    print("\nAI Category Distribution:")
    print(df['ai_category'].value_counts())

    print("\nResearch Group Distribution:")
    print(df['research_group'].value_counts())

    print(f"\nSelf-Talk Posts: {df['is_self_talk'].sum()} ({df['is_self_talk'].mean()*100:.1f}%)")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Data Labeler for Gender Stereotypes in AI Research")
    print("=" * 60)

    # Process posts
    print("\nProcessing posts...")
    process_posts("all_posts.json", "labeled_posts.json")

    # Process gender-specific posts
    print("\nProcessing gender-related posts...")
    process_posts("gender_posts.json", "labeled_gender_posts.json")

    # Process comments
    print("\nProcessing comments...")
    process_comments("comments.json", "labeled_comments.json")


if __name__ == "__main__":
    main()
