"""
Reddit Data Collector for Gender Stereotypes in AI Research

This script collects posts and comments from specified subreddits
using keyword search for gender-related terms in AI discourse.

Features:
- Checkpoint/resume: Saves progress after each subreddit/query
- Can continue from where it left off if interrupted
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

import praw
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUBREDDITS = {
    "technical": ["MachineLearning", "OpenAI"],
    "creative": ["Midjourney", "StableDiffusion"],
    "career": ["CSCareerQuestions"],
    "ethics_soft_ai": ["artificial", "singularity", "ChatGPT", "AIethics"]
}

# Flatten subreddit list
ALL_SUBREDDITS = [sub for category in SUBREDDITS.values() for sub in category]

# Search queries for gender terms (optimized for Reddit search)
FEMALE_QUERIES = ["woman", "women", "female", "she", "her", "girl"]
MALE_QUERIES = ["man", "men", "male", "he", "his", "guy"]

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
TMP_DIR = DATA_DIR / "tmp"
LEXICONS_DIR = DATA_DIR / "lexicons"
CHECKPOINT_FILE = TMP_DIR / "checkpoint.json"

# Rate limiting configuration (Reddit allows ~60 requests/minute for OAuth)
# Being conservative to avoid bans
RATE_LIMIT = {
    "between_queries": 2,      # seconds between search queries
    "between_subreddits": 5,   # seconds between subreddits
    "between_comments": 1,     # seconds between comment fetches
    "on_error": 30,            # seconds to wait on error (backoff)
}


def load_lexicon(filename: str) -> List[str]:
    """Load a lexicon file and return list of terms."""
    filepath = LEXICONS_DIR / filename
    with open(filepath, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def init_reddit() -> praw.Reddit:
    """Initialize Reddit API client with rate limit handling."""
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "GenderAIResearch/1.0"),
        ratelimit_seconds=300  # Wait up to 5 min if rate limited
    )
    return reddit


def rate_limited_request(func, *args, max_retries: int = 3, **kwargs):
    """Execute a function with retry logic for rate limiting."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                wait_time = RATE_LIMIT["on_error"] * (attempt + 1)
                print(f"\n  Rate limited! Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(f"Max retries ({max_retries}) exceeded")


# ============================================================
# Checkpoint/Resume Functions
# ============================================================

def load_checkpoint() -> Dict:
    """Load checkpoint data if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "completed_searches": [],  # List of "subreddit:query:gender" strings
        "completed_subreddits": [],
        "comments_collected_for": [],  # Post IDs with comments collected
        "stage": "posts"  # "posts" or "comments"
    }


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint data."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)


def clear_checkpoint():
    """Clear checkpoint after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    print("Checkpoint cleared.")


def load_tmp_posts(subreddit: str, gender: str) -> List[Dict]:
    """Load temporary posts for a subreddit/gender combination."""
    filepath = TMP_DIR / f"{subreddit}_{gender}_posts.json"
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_tmp_posts(posts: List[Dict], subreddit: str, gender: str):
    """Save temporary posts for a subreddit/gender combination."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TMP_DIR / f"{subreddit}_{gender}_posts.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"  Checkpoint: Saved {len(posts)} {gender} posts from r/{subreddit}")


def load_tmp_comments() -> List[Dict]:
    """Load temporary comments."""
    filepath = TMP_DIR / "comments_tmp.json"
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_tmp_comments(comments: List[Dict]):
    """Save temporary comments."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TMP_DIR / "comments_tmp.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)


# ============================================================
# Data Collection Functions
# ============================================================

def post_to_dict(post, subreddit_name: str, search_query: str, gender_category: str) -> Dict:
    """Convert a PRAW submission to a dictionary."""
    return {
        "id": post.id,
        "subreddit": subreddit_name,
        "title": post.title,
        "selftext": post.selftext,
        "author": str(post.author) if post.author else "[deleted]",
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "created_date": datetime.fromtimestamp(post.created_utc).isoformat(),
        "url": post.url,
        "permalink": post.permalink,
        "is_self": post.is_self,
        "flair": post.link_flair_text,
        "search_query": search_query,
        "gender_category": gender_category
    }


def search_posts_by_keyword(
    reddit: praw.Reddit,
    subreddit_name: str,
    query: str,
    gender_category: str,
    limit: int = 500,
    time_filter: str = "all",
    sort: str = "relevance"
) -> List[Dict]:
    """
    Search posts in a subreddit by keyword.

    Args:
        reddit: PRAW Reddit instance
        subreddit_name: Name of the subreddit
        query: Search query string
        gender_category: 'female' or 'male' for labeling
        limit: Maximum number of posts to collect
        time_filter: Time filter (hour, day, week, month, year, all)
        sort: Sort method (relevance, hot, top, new, comments)

    Returns:
        List of post dictionaries
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    try:
        results = subreddit.search(
            query=query,
            time_filter=time_filter,
            limit=limit,
            sort=sort
        )

        for post in results:
            post_data = post_to_dict(post, subreddit_name, query, gender_category)
            posts.append(post_data)

    except Exception as e:
        print(f"Error searching '{query}' in r/{subreddit_name}: {e}")

    return posts


def collect_gender_posts_with_checkpoint(
    reddit: praw.Reddit,
    subreddit_name: str,
    checkpoint: Dict,
    limit_per_query: int = 200,
    time_filter: str = "all"
) -> Dict[str, List[Dict]]:
    """
    Collect posts related to gender terms from a subreddit with checkpoint support.

    Returns:
        Dictionary with 'female' and 'male' post lists
    """
    # Load existing posts from tmp if any
    female_posts = load_tmp_posts(subreddit_name, "female")
    male_posts = load_tmp_posts(subreddit_name, "male")

    seen_ids: Set[str] = set(p["id"] for p in female_posts + male_posts)

    # Search for female-related posts
    print(f"  Searching female terms in r/{subreddit_name}...")
    for query in FEMALE_QUERIES:
        search_key = f"{subreddit_name}:{query}:female"

        if search_key in checkpoint["completed_searches"]:
            print(f"    Skipping '{query}' (already completed)")
            continue

        posts = search_posts_by_keyword(
            reddit, subreddit_name, query, "female",
            limit=limit_per_query, time_filter=time_filter
        )

        new_posts = 0
        for post in posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                female_posts.append(post)
                new_posts += 1

        print(f"    '{query}': {new_posts} new posts")

        # Save checkpoint after each query
        checkpoint["completed_searches"].append(search_key)
        save_checkpoint(checkpoint)
        save_tmp_posts(female_posts, subreddit_name, "female")

        time.sleep(RATE_LIMIT["between_queries"])  # Rate limiting

    # Search for male-related posts
    print(f"  Searching male terms in r/{subreddit_name}...")
    for query in MALE_QUERIES:
        search_key = f"{subreddit_name}:{query}:male"

        if search_key in checkpoint["completed_searches"]:
            print(f"    Skipping '{query}' (already completed)")
            continue

        posts = search_posts_by_keyword(
            reddit, subreddit_name, query, "male",
            limit=limit_per_query, time_filter=time_filter
        )

        new_posts = 0
        for post in posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                male_posts.append(post)
                new_posts += 1

        print(f"    '{query}': {new_posts} new posts")

        # Save checkpoint after each query
        checkpoint["completed_searches"].append(search_key)
        save_checkpoint(checkpoint)
        save_tmp_posts(male_posts, subreddit_name, "male")

        time.sleep(RATE_LIMIT["between_queries"])  # Rate limiting

    return {"female": female_posts, "male": male_posts}


def collect_comments(
    reddit: praw.Reddit,
    post_id: str,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Collect comments from a post.

    Args:
        reddit: PRAW Reddit instance
        post_id: Reddit post ID
        limit: Maximum number of 'MoreComments' to expand (None for all)

    Returns:
        List of comment dictionaries
    """
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=limit)

    comments = []
    for comment in submission.comments.list():
        try:
            comment_data = {
                "id": comment.id,
                "post_id": post_id,
                "parent_id": comment.parent_id,
                "body": comment.body,
                "author": str(comment.author) if comment.author else "[deleted]",
                "score": comment.score,
                "created_utc": comment.created_utc,
                "created_date": datetime.fromtimestamp(comment.created_utc).isoformat(),
                "is_submitter": comment.is_submitter,
                "depth": comment.depth
            }
            comments.append(comment_data)
        except Exception as e:
            print(f"Error processing comment {comment.id}: {e}")
            continue

    return comments


def save_data(data: List[Dict], filename: str, format: str = "json"):
    """Save collected data to file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DIR / filename

    if format == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif format == "csv":
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')

    print(f"Saved {len(data)} items to {filepath}")


def main():
    """Main data collection pipeline with checkpoint/resume support."""
    print("=" * 60)
    print("Reddit Data Collector for Gender Stereotypes in AI Research")
    print("=" * 60)

    # Load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint["completed_searches"]:
        print(f"\nResuming from checkpoint...")
        print(f"  Completed searches: {len(checkpoint['completed_searches'])}")
        print(f"  Stage: {checkpoint['stage']}")

    # Initialize Reddit client
    reddit = init_reddit()
    print("Connected to Reddit API")

    # ============================================================
    # Stage 1: Collect Posts
    # ============================================================
    if checkpoint["stage"] == "posts":
        all_female_posts = []
        all_male_posts = []

        for subreddit in ALL_SUBREDDITS:
            if subreddit in checkpoint["completed_subreddits"]:
                print(f"\nSkipping r/{subreddit} (already completed)")
                # Load from tmp
                all_female_posts.extend(load_tmp_posts(subreddit, "female"))
                all_male_posts.extend(load_tmp_posts(subreddit, "male"))
                continue

            print(f"\nCollecting from r/{subreddit}...")
            results = collect_gender_posts_with_checkpoint(
                reddit, subreddit, checkpoint,
                limit_per_query=200,
                time_filter="all"
            )
            all_female_posts.extend(results["female"])
            all_male_posts.extend(results["male"])

            print(f"  Total: {len(results['female'])} female, {len(results['male'])} male posts")

            # Mark subreddit as completed
            checkpoint["completed_subreddits"].append(subreddit)
            save_checkpoint(checkpoint)

            time.sleep(RATE_LIMIT["between_subreddits"])  # Rate limiting between subreddits

        # Combine all posts
        all_posts = all_female_posts + all_male_posts

        print(f"\n{'=' * 60}")
        print("Post Collection Summary")
        print("=" * 60)
        print(f"Total female-related posts: {len(all_female_posts)}")
        print(f"Total male-related posts: {len(all_male_posts)}")
        print(f"Total posts: {len(all_posts)}")

        # Save final data
        save_data(all_female_posts, "female_posts.json")
        save_data(all_male_posts, "male_posts.json")
        save_data(all_posts, "all_gender_posts.json")

        # Also save as CSV
        save_data(all_posts, "all_gender_posts.csv", format="csv")

        # Create summary DataFrame
        if all_posts:
            df = pd.DataFrame(all_posts)
            print(f"\nPosts per subreddit:")
            print(df.groupby(['subreddit', 'gender_category']).size().unstack(fill_value=0))

        # Update checkpoint for comments stage
        checkpoint["stage"] = "comments"
        save_checkpoint(checkpoint)

    # ============================================================
    # Stage 2: Collect Comments
    # ============================================================
    print("\n" + "=" * 60)
    print("Collecting comments for high-engagement posts...")
    print("=" * 60)

    # Load all posts
    all_posts_file = RAW_DIR / "all_gender_posts.json"
    if not all_posts_file.exists():
        print("Error: all_gender_posts.json not found. Run post collection first.")
        return

    with open(all_posts_file, 'r', encoding='utf-8') as f:
        all_posts = json.load(f)

    df = pd.DataFrame(all_posts)

    # Sort by num_comments and take top posts
    df_sorted = df.sort_values('num_comments', ascending=False)
    top_posts = df_sorted.head(200)  # Top 200 posts by comment count

    # Load existing comments
    all_comments = load_tmp_comments()
    collected_post_ids = set(checkpoint.get("comments_collected_for", []))

    print(f"Already collected comments for {len(collected_post_ids)} posts")

    for _, post in tqdm(top_posts.iterrows(), total=len(top_posts), desc="Collecting comments"):
        if post["id"] in collected_post_ids:
            continue

        try:
            comments = collect_comments(reddit, post["id"], limit=10)
            # Add post metadata to comments
            for comment in comments:
                comment["post_gender_category"] = post["gender_category"]
                comment["post_subreddit"] = post["subreddit"]
            all_comments.extend(comments)

            # Save checkpoint
            collected_post_ids.add(post["id"])
            checkpoint["comments_collected_for"] = list(collected_post_ids)
            save_checkpoint(checkpoint)
            save_tmp_comments(all_comments)

        except Exception as e:
            print(f"\nError collecting comments for post {post['id']}: {e}")
            time.sleep(RATE_LIMIT["on_error"])  # Wait longer on error

        time.sleep(RATE_LIMIT["between_comments"])  # Rate limiting

    print(f"\nTotal comments collected: {len(all_comments)}")
    save_data(all_comments, "comments.json")
    save_data(all_comments, "comments.csv", format="csv")

    # Clear checkpoint after successful completion
    clear_checkpoint()

    # Final summary
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print(f"Files saved to: {RAW_DIR}")
    print(f"  - female_posts.json")
    print(f"  - male_posts.json")
    print(f"  - all_gender_posts.json / .csv")
    print(f"  - comments.json / .csv")


if __name__ == "__main__":
    main()
