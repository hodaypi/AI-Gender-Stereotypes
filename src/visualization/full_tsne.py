import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import os

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

plt.style.use('dark_background')

COLORS = {
    'Male Term': '#00BFFF',    # DeepSkyBlue
    'Female Term': '#FF1493',  # DeepPink
    'Hard AI': '#32CD32',      # LimeGreen
    'Soft AI': '#FFA500',      # Orange
    'Background': '#808080'    # DimGray
}

def load_lexicon(path):
    if not os.path.exists(path):
        print(f"Error: Lexicon not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def visualize_full_semantic_space(model_path, hard_lex_path, soft_lex_path, female_lex_path, male_lex_path):
    print(f"Loading model from {model_path}...")
    model = Word2Vec.load(model_path)
    
    # load lexicons
    print("Loading full lexicons from files...")
    hard_words = [w for w in load_lexicon(hard_lex_path) if w in model.wv]
    soft_words = [w for w in load_lexicon(soft_lex_path) if w in model.wv]
    male_words = [w for w in load_lexicon(male_lex_path) if w in model.wv]
    female_words = [w for w in load_lexicon(female_lex_path) if w in model.wv]
    
    target_words_set = set(hard_words + soft_words + male_words + female_words)
    
    # load all the other words in the model
    all_vocab = list(model.wv.index_to_key) 
    background_words = [w for w in all_vocab if w not in target_words_set]
    
    print(f"\n--- Plotting Configuration ---")
    print(f"Target Words (Highlighted): {len(target_words_set)}")
    print(f" - Hard AI: {len(hard_words)}")
    print(f" - Soft AI: {len(soft_words)}")
    print(f" - Male Terms: {len(male_words)}")
    print(f" - Female Terms: {len(female_words)}")
    print(f"Background Words (Gray): {len(background_words)}")
    print("------------------------------")

    vectors = []
    labels = []
    words = []
    sizes = []
    alphas = []
    colors = []
    
    # add background
    for w in background_words:
        vectors.append(model.wv[w])
        labels.append('Background')
        words.append(w)
        sizes.append(5)         
        alphas.append(0.3)     
        colors.append(COLORS['Background'])

    # add Hard AI
    for w in hard_words:
        vectors.append(model.wv[w])
        labels.append('Hard AI')
        words.append(w)
        sizes.append(80)
        alphas.append(0.9)
        colors.append(COLORS['Hard AI'])
        
    # add Soft AI
    for w in soft_words:
        vectors.append(model.wv[w])
        labels.append('Soft AI')
        words.append(w)
        sizes.append(80)
        alphas.append(0.9)
        colors.append(COLORS['Soft AI'])

    # add Male Terms
    for w in male_words:
        vectors.append(model.wv[w])
        labels.append('Male Term')
        words.append(w)
        sizes.append(100)
        alphas.append(1.0)
        colors.append(COLORS['Male Term'])

    # add Female Terms
    for w in female_words:
        vectors.append(model.wv[w])
        labels.append('Female Term')
        words.append(w)
        sizes.append(100)
        alphas.append(1.0)
        colors.append(COLORS['Female Term'])

    X = np.array(vectors)
    
    # t-SNE
    print("Running t-SNE on full vocabulary (Pro+ Mode)...")
    tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, random_state=42, init='pca', learning_rate='auto', n_jobs=-1)
    X_embedded = tsne.fit_transform(X)
    
    x_coords = X_embedded[:, 0]
    y_coords = X_embedded[:, 1]
    
    # plot graph
    plt.figure(figsize=(24, 18)) 
    
    print("Rendering plot...")
    plt.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=alphas, edgecolors='none')

    # add text for lexicons only
    texts = []
    for i, label in enumerate(labels):
        if label != 'Background':
            is_gender = label in ['Male Term', 'Female Term']
            
            texts.append(plt.text(
                x_coords[i], y_coords[i], 
                words[i],
                fontsize=8,
                color='white'
            ))

    if adjust_text:
        print("Optimizing text placement (this helps avoid overlap)...")
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='white', lw=0.5))

    plt.title("The Semantic Universe: Gendered Lexicons within AI Discourse", fontsize=24, weight='bold', color='white')
    
    # Legend
    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in list(COLORS.values())[:-1]]
    plt.legend(markers, list(COLORS.keys())[:-1], title='Category', loc='upper right', fontsize=14, frameon=True, facecolor='black', labelcolor='white')
    
    plt.axis('off')
    plt.show()

def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    
    MODEL_PATH = "/content/gdrive/MyDrive/Data mining/text mining/models/gender_ai_w2v_optimized.model"
    
    # lexicons routes
    LEX_HARD = os.path.join(BASE_DIR, "lexicons/hard_ai.txt")
    LEX_SOFT = os.path.join(BASE_DIR, "lexicons/soft_ai.txt")
    LEX_FEMALE = os.path.join(BASE_DIR, "lexicons/gender_female.txt") 
    LEX_MALE = os.path.join(BASE_DIR, "lexicons/gender_male.txt")

    visualize_full_semantic_space(MODEL_PATH, LEX_HARD, LEX_SOFT, LEX_FEMALE, LEX_MALE)

if __name__ == "__main__":
    main()