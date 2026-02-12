import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def load_lexicon(path):
    """Loads words from a text file into a list."""
    if not os.path.exists(path):
        print(f"Warning: Lexicon not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def get_avg_vector(model, words):
    """Calculates the average vector for a list of words."""
    # filter words that dont exist in the model
    valid_vectors = [model.wv[w] for w in words if w in model.wv]
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    
    # calculate centroid
    return np.mean(valid_vectors, axis=0)

# ==========================================
# 2. TRAINING MODEL FUNCTION
# ==========================================
def train_model(posts_path, comments_path, output_model_path):
    print("Loading preprocessed datasets...")
    
    sentences = []
    
    # Load Posts
    if os.path.exists(posts_path):
        df_posts = pd.read_pickle(posts_path)
        posts_list = df_posts['cleaned_tokens'].tolist()
        sentences.extend(posts_list)
        print(f"Loaded {len(posts_list)} posts.")
    else:
        print(f"Warning: Posts file not found at {posts_path}")

    # Load Comments
    if os.path.exists(comments_path):
        df_comments = pd.read_pickle(comments_path)
        comments_list = df_comments['cleaned_tokens'].tolist()
        sentences.extend(comments_list)
        print(f"Loaded {len(comments_list)} comments.")
    else:
        print(f"Warning: Comments file not found at {comments_path}")

    print(f"Total sentences for training: {len(sentences)}")

    # ---------------------------------------------------------
    # UPDATED MODEL PARAMETERS
    # ---------------------------------------------------------
    VECTOR_SIZE = 300       # bigger size 100 ->300
    WINDOW = 10             # increased to catch context 5->10
    MIN_COUNT = 10          # increased to filter noise and rare words
    WORKERS = multiprocessing.cpu_count() 
    SG = 1                  # Skip-Gram 
    EPOCHS = 30             # increased 15->30
    SAMPLE = 1e-4           # Down-sampling for overly common words
    # ---------------------------------------------------------

    print(f"Training Word2Vec model...")
    print(f"Params: Size={VECTOR_SIZE}, Window={WINDOW}, MinCount={MIN_COUNT}, Epochs={EPOCHS}")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        epochs=EPOCHS,
        sample=SAMPLE
    )
    
    print(f"Saving model to {output_model_path}...")
    model.save(output_model_path)
    
    print(f"Done! Vocabulary size: {len(model.wv.index_to_key)} unique words.")
    return model

# ==========================================
# 3. ANALYSIS FUNCTIONS
# ==========================================

def calculate_cosine_distances(model, hard_words, soft_words, female_words, male_words):
    """
    A. Quantitative Analysis: Group Distances
    Calculates average distance between Gender groups and AI Topics.
    """
    print("\n" + "="*50)
    print("--- A. Quantitative: Group Cosine Distance Analysis ---")
    print("="*50)
    
    # calculate avg vectors
    hard_vec = get_avg_vector(model, hard_words)
    soft_vec = get_avg_vector(model, soft_words)
    
    female_vec = get_avg_vector(model, female_words)
    male_vec = get_avg_vector(model, male_words)

    # calculate Similarity foreach group
    # Female Analysis
    f_sim_hard = model.wv.cosine_similarities(female_vec, [hard_vec])[0]
    f_sim_soft = model.wv.cosine_similarities(female_vec, [soft_vec])[0]
    
    print(f"\n[Female Group Average]")
    print(f"   Similarity to Hard AI: {f_sim_hard:.4f}")
    print(f"   Similarity to Soft AI: {f_sim_soft:.4f}")
    gap_f = f_sim_soft - f_sim_hard
    print(f"   => Closer to: {'SOFT' if gap_f > 0 else 'HARD'} (Gap: {abs(gap_f):.4f})")

    # Male Analysis
    m_sim_hard = model.wv.cosine_similarities(male_vec, [hard_vec])[0]
    m_sim_soft = model.wv.cosine_similarities(male_vec, [soft_vec])[0]

    print(f"\n[Male Group Average]")
    print(f"   Similarity to Hard AI: {m_sim_hard:.4f}")
    print(f"   Similarity to Soft AI: {m_sim_soft:.4f}")
    gap_m = m_sim_soft - m_sim_hard
    print(f"   => Closer to: {'SOFT' if gap_m > 0 else 'HARD'} (Gap: {abs(gap_m):.4f})")

    print("\n[Direct Comparison: Who is closer to Hard AI?]")
    hard_bias = m_sim_hard - f_sim_hard
    
    if hard_bias > 0:
        print(f"   RESULT: MEN are closer to Hard AI by {hard_bias:.4f}")
    else:
        print(f"   RESULT: WOMEN are closer to Hard AI by {abs(hard_bias):.4f}")
    

    print("\n[Direct Comparison: Who is closer to Soft AI?]")
    soft_bias = m_sim_soft - f_sim_soft
    
    if soft_bias > 0:
        print(f"   RESULT: MEN are closer to Soft AI by {soft_bias:.4f}")
    else:
        print(f"   RESULT: WOMEN are closer to Soft AI by {abs(soft_bias):.4f}")



def calculate_specific_terms_distance(model, hard_words, soft_words):
    """
    Specific Requirement Check: 'Man' vs 'Woman' individual words.
    """
    print("\n" + "="*50)
    print("--- Specific Requirement: 'Woman' vs 'Man' Analysis ---")
    print("="*50)
    
    if 'woman' not in model.wv or 'man' not in model.wv:
        print("Error: 'woman' or 'man' not found in vocabulary (check min_count)!")
        return

    hard_vec = get_avg_vector(model, hard_words)
    soft_vec = get_avg_vector(model, soft_words)

    targets = ['woman', 'man']
    
    for term in targets:
        term_vec = model.wv[term]
        
        sim_hard = model.wv.cosine_similarities(term_vec, [hard_vec])[0]
        sim_soft = model.wv.cosine_similarities(term_vec, [soft_vec])[0]
        
        print(f"['{term}']")
        print(f"   Similarity to Hard AI: {sim_hard:.4f}")
        print(f"   Similarity to Soft AI: {sim_soft:.4f}")
        
        gap = sim_soft - sim_hard
        direction = "SOFT" if gap > 0 else "HARD"
        print(f"   => Result: Closer to {direction} AI (Gap: {abs(gap):.4f})")
        print("-" * 30)


def analyze_closest_neighbors_filtered(model, female_words, male_words):
    """
    Qualitative: Context Analysis (Filtered)
    Finds similar words but removes gender-specific terms to see the underlying context.
    """
    print("\n" + "="*50)
    print("--- B. Qualitative: Context Analysis (Filtered) ---")
    print("="*50)
    
    gender_stopwords = set(female_words + male_words)
    target_terms = ['woman', 'man', 'female', 'male', 'she', 'he']
    gender_stopwords.update(target_terms)

    for term in target_terms:
        if term in model.wv:
            similar_candidates = model.wv.most_similar(term, topn=100)
            
            filtered_results = []
            for word, score in similar_candidates:
                if word.lower() not in gender_stopwords:
                    filtered_results.append(word)
                
                if len(filtered_results) == 10:
                    break
            
            print(f"Closest to '{term}' (Non-Gender): {filtered_results}")


def analyze_bias_leaderboard(model, hard_words, soft_words, female_words, male_words):
    """
    C. The 'Bias Leaderboard' (Gender Axis Projection)
    """
    print("\n" + "="*50)
    print("--- C. The 'Bias Leaderboard' (Gender Axis Projection) ---")
    print("="*50)
    
    male_vec = get_avg_vector(model, male_words)
    female_vec = get_avg_vector(model, female_words)
    
    # create gender vector (Male - Female)
    gender_axis = male_vec - female_vec
    
    def get_gender_score(word):
        # Dot Product: (positive=male, negative=female)
        return np.dot(model.wv[word], gender_axis)

    # --- Hard AI Analysis ---
    hard_scores = [(w, get_gender_score(w)) for w in hard_words if w in model.wv]
    hard_scores.sort(key=lambda x: x[1], reverse=True) # Sort descending
    
    print("\n[Hard AI Lexicon Analysis]")
    print("Top 5 'Masculine' terms (High Score):")
    for w, score in hard_scores[:5]: print(f"   {w} ({score:.4f})")
        
    print("Top 5 'Feminine' terms (Low Score):")
    for w, score in hard_scores[-5:]: print(f"   {w} ({score:.4f})")

    # --- Soft AI Analysis ---
    soft_scores = [(w, get_gender_score(w)) for w in soft_words if w in model.wv]
    soft_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n[Soft AI Lexicon Analysis]")
    print("Top 5 'Masculine' terms (High Score):")
    for w, score in soft_scores[:5]: print(f"   {w} ({score:.4f})")
        
    print("Top 5 'Feminine' terms (Low Score):")
    for w, score in soft_scores[-5:]: print(f"   {w} ({score:.4f})")


def run_semantic_analysis(model, hard_lex_path, soft_lex_path, female_lex_path, male_lex_path):
    """
    Main controller for all analysis steps.
    """
    # Load Lexicons
    hard_words = load_lexicon(hard_lex_path)
    soft_words = load_lexicon(soft_lex_path)
    female_words = load_lexicon(female_lex_path)
    male_words = load_lexicon(male_lex_path)
    
    # Print Coverage Info
    print(f"\nLexicon Coverage check:")
    print(f" - Hard AI: {[w for w in hard_words if w in model.wv].__len__()} found / {len(hard_words)} total")
    print(f" - Soft AI: {[w for w in soft_words if w in model.wv].__len__()} found / {len(soft_words)} total")

    # 1. Group Analysis (Main Research Question)
    calculate_cosine_distances(model, hard_words, soft_words, female_words, male_words)
    
    # 2. Specific Terms Analysis (Project Requirement)
    calculate_specific_terms_distance(model, hard_words, soft_words)
    
    # 3. Context Analysis (Filtered)
    analyze_closest_neighbors_filtered(model, female_words, male_words)
    
    # 4. Gender Axis Projection
    analyze_bias_leaderboard(model, hard_words, soft_words, female_words, male_words)
    
    print("\nAnalysis Complete.")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    #routes
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    DATA_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_DIR= "/content/gdrive/MyDrive/Data mining/text mining/models"

    POSTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_posts.pkl")
    COMMENTS_PKL = os.path.join(BASE_DIR, "processed/word2vec_comments.pkl")
    MODEL_OUTPUT = os.path.join(MODEL_DIR, "gender_ai_w2v_optimized.model")
    
    LEX_HARD = os.path.join(BASE_DIR, "lexicons/hard_ai.txt")
    LEX_SOFT = os.path.join(BASE_DIR, "lexicons/soft_ai.txt")
    LEX_FEMALE = os.path.join(BASE_DIR, "lexicons/gender_female.txt") 
    LEX_MALE = os.path.join(BASE_DIR, "lexicons/gender_male.txt")      

    # model training
    trained_model = train_model(POSTS_PKL, COMMENTS_PKL, MODEL_OUTPUT)
    
    # Analysis
    run_semantic_analysis(trained_model, LEX_HARD, LEX_SOFT, LEX_FEMALE, LEX_MALE)

if __name__ == "__main__":
    main()