import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_lexicon(path):
    """Loads words from a text file into a list."""
    if not os.path.exists(path):
        print(f"Warning: Lexicon not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def get_avg_vector(model, words):
    """Calculates the average vector for a list of words."""
    valid_vectors = [model.wv[w] for w in words if w in model.wv]
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(valid_vectors, axis=0)

# ==========================================
# TRAINING MODEL FUNCTION 
# ==========================================
def train_model(posts_path, comments_path, output_model_path):
    print("Loading preprocessed datasets...")
    
    sentences = []
    
    # load posts
    if os.path.exists(posts_path):
        df_posts = pd.read_pickle(posts_path)
        posts_list = df_posts['cleaned_tokens'].tolist()
        sentences.extend(posts_list)
        print(f"Loaded {len(posts_list)} posts.")
    else:
        print(f"Warning: Posts file not found at {posts_path}")

    # load comments
    if os.path.exists(comments_path):
        df_comments = pd.read_pickle(comments_path)
        comments_list = df_comments['cleaned_tokens'].tolist()
        sentences.extend(comments_list)
        print(f"Loaded {len(comments_list)} comments.")
    else:
        print(f"Warning: Comments file not found at {comments_path}")

    print(f"Total sentences for training: {len(sentences)}")

    # model parameters
    # ---------------------------------------------------------
    VECTOR_SIZE = 100   
    WINDOW = 5          
    MIN_COUNT = 5       
    WORKERS = multiprocessing.cpu_count() 
    SG = 1              # Skip-Gram 
    EPOCHS = 15         
    # ---------------------------------------------------------

    print(f"Training Word2Vec model...")
    print(f"Params: Size={VECTOR_SIZE}, Window={WINDOW}, MinCount={MIN_COUNT}, Epochs={EPOCHS}")
    
    # model training
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        epochs=EPOCHS
    )
    
    # save model
    print(f"Saving model to {output_model_path}...")
    model.save(output_model_path)
    
    print(f"Done! Vocabulary size: {len(model.wv.index_to_key)} unique words.")
    return model

# ==========================================
# ANALYSIS FUNCTION 
# ==========================================
def calculate_cosine_distances(model, hard_words, soft_words, female_words, male_words):
    """
    Core Function: Calculates average distance between Gender groups and AI Topics.
    """
    print("\n--- A. Quantitative: Cosine Distance Analysis ---")
    
    # calculate avg vectors
    hard_vec = get_avg_vector(model, hard_words)
    soft_vec = get_avg_vector(model, soft_words)
    
    gender_groups = {'Female': female_words, 'Male': male_words}
    
    for gender_name, words in gender_groups.items():
        gender_vec = get_avg_vector(model, words)
        
        sim_hard = model.wv.cosine_similarities(gender_vec, [hard_vec])[0]
        sim_soft = model.wv.cosine_similarities(gender_vec, [soft_vec])[0]
        
        print(f"[{gender_name} Group]")
        print(f"   Dist to Hard AI: {sim_hard:.4f}")
        print(f"   Dist to Soft AI: {sim_soft:.4f}")
        
        gap = sim_soft - sim_hard
        direction = "SOFT" if gap > 0 else "HARD"
        print(f"   => Result: Closer to {direction} AI (Gap: {abs(gap):.4f})")
        print("-" * 30)

def analyze_closest_neighbors(model):
    """
    Bonus: Prints the top 5 most similar words to key gender terms.
    """
    print("\n--- B. Qualitative: Most Similar Words (Context) ---")
    target_terms = ['woman', 'man', 'she', 'he', 'female', 'male']
    
    for term in target_terms:
        if term in model.wv:
            similar = model.wv.most_similar(term, topn=5)
            words_only = [w[0] for w in similar]
            print(f"Closest to '{term}': {words_only}")

def analyze_bias_leaderboard(model, hard_words, soft_words, female_words, male_words):
    """
    Bonus: Projects words onto a 'Gender Axis' to find the most Gendered technical terms.
    """
    print("\n--- C. The 'Bias Leaderboard' (Gender Axis Projection) ---")
    
    male_vec = get_avg_vector(model, male_words)
    female_vec = get_avg_vector(model, female_words)
    gender_axis = male_vec - female_vec
    
    def get_gender_score(word):
        # Dot Product (positive=male, negative=female)
        return np.dot(model.wv[word], gender_axis)

    # Hard AI
    hard_scores = [(w, get_gender_score(w)) for w in hard_words if w in model.wv]
    hard_scores.sort(key=lambda x: x[1], reverse=True) 
    
    print("\n[Hard AI] Top 5 'Masculine' terms:")
    for w, score in hard_scores[:5]: print(f"   {w} ({score:.4f})")
        
    print("\n[Hard AI] Top 5 'Feminine' terms:")
    for w, score in hard_scores[-5:]: print(f"   {w} ({score:.4f})")

    # Soft AI
    soft_scores = [(w, get_gender_score(w)) for w in soft_words if w in model.wv]
    soft_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n[Soft AI] Top 5 'Masculine' terms:")
    for w, score in soft_scores[:5]: print(f"   {w} ({score:.4f})")
        
    print("\n[Soft AI] Top 5 'Feminine' terms:")
    for w, score in soft_scores[-5:]: print(f"   {w} ({score:.4f})")

def run_semantic_analysis(model, hard_lex_path, soft_lex_path, female_lex_path, male_lex_path):
    """
    Loads lexicons and runs all analysis sub-functions.
    """
    print("\n" + "="*60)
    print("      PHASE 2: FULL SEMANTIC ANALYSIS      ")
    print("="*60)
    
    # load lexicons
    hard_words = load_lexicon(hard_lex_path)
    soft_words = load_lexicon(soft_lex_path)
    female_words = load_lexicon(female_lex_path)
    male_words = load_lexicon(male_lex_path)
    
    # Lexicon Coverage
    print(f"Lexicon Coverage (Found in Model):")
    print(f" - Hard AI: {[w for w in hard_words if w in model.wv].__len__()} / {len(hard_words)}")
    print(f" - Soft AI: {[w for w in soft_words if w in model.wv].__len__()} / {len(soft_words)}")

    # calculations and analysis
    calculate_cosine_distances(model, hard_words, soft_words, female_words, male_words)
    analyze_closest_neighbors(model)
    analyze_bias_leaderboard(model, hard_words, soft_words, female_words, male_words)
    
    print("\nAnalysis Complete.")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    
    # routes
    POSTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")
    COMMENTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_comments.pkl")
    MODEL_OUTPUT = os.path.join(BASE_DIR, "processed/gender_ai_w2v.model")
    
    LEX_HARD = os.path.join(BASE_DIR, "lexicons/hard_ai.txt")
    LEX_SOFT = os.path.join(BASE_DIR, "lexicons/soft_ai.txt")
    LEX_FEMALE = os.path.join(BASE_DIR, "lexicons/gender_female.txt") 
    LEX_MALE = os.path.join(BASE_DIR, "lexicons/gender_male.txt")     

    # model ini and training
    trained_model = train_model(POSTS_PKL, COMMENTS_PKL, MODEL_OUTPUT)
    
    # analysis
    run_semantic_analysis(trained_model, LEX_HARD, LEX_SOFT, LEX_FEMALE, LEX_MALE)