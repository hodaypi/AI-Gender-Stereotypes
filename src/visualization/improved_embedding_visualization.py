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

sns.set_style("whitegrid")

COLORS = {
    'Male Term': '#3c78d8',    
    'Female Term': '#a64d79',  
    'Hard AI': '#6aa84f',      
    'Soft AI': '#f1c232'       
}

FILTERED_MALE = [
    'man', 'male', 'boy', 'he', 'him', 'his', 'men', 'males', 
    'guy', 'guys', 'dude', 'dudes', 'gentleman'
]

FILTERED_FEMALE = [
    'woman', 'female', 'girl', 'she', 'her', 'hers', 'women', 'females', 
    'lady', 'ladies', 'gal', 'gals'
]

def load_lexicon(path):
    """Loads words from a text file into a list."""
    if not os.path.exists(path):
        print(f"Warning: Lexicon not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def visualize_tsne_full(model_path, hard_lex_path, soft_lex_path):
    print(f"Loading OPTIMIZED model from {model_path}...")
    model = Word2Vec.load(model_path)
    
    # load lexicons
    print("Loading full lexicons...")
    hard_words_full = load_lexicon(hard_lex_path)
    soft_words_full = load_lexicon(soft_lex_path)
    
    hard_words = [w for w in hard_words_full if w in model.wv]
    soft_words = [w for w in soft_words_full if w in model.wv]
    
    # gender lexicons
    male_words = [w for w in FILTERED_MALE if w in model.wv]
    female_words = [w for w in FILTERED_FEMALE if w in model.wv]

    plot_data = []
    
    def collect_vectors(words_list, category_label):
        for w in words_list:
            plot_data.append({
                'word': w, 
                'vector': model.wv[w], 
                'category': category_label
            })

    collect_vectors(hard_words, 'Hard AI')
    collect_vectors(soft_words, 'Soft AI')
    collect_vectors(male_words, 'Male Term')
    collect_vectors(female_words, 'Female Term')

    df = pd.DataFrame(plot_data)
    
    X = np.array(df['vector'].tolist())
    
    print(f"Total words to plot: {len(df)}")
    print(f" - Hard AI: {len(hard_words)}")
    print(f" - Soft AI: {len(soft_words)}")

    # t-SNE
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    
    # plot graph
    print("Generating plot...")
    plt.figure(figsize=(16, 12))
    
    sns.scatterplot(data=df, x='x', y='y', hue='category', palette=COLORS, 
                    s=100, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # add labels
    texts = []
    for i in range(df.shape[0]):
        is_gender = df.iloc[i]['category'] in ['Male Term', 'Female Term']
        
        texts.append(plt.text(
            df.iloc[i]['x'], df.iloc[i]['y'], 
            df.iloc[i]['word'],
            fontsize=10 if is_gender else 8,
            weight='bold' if is_gender else 'normal',
            color='black',
            alpha=0.9
        ))

    if adjust_text:
        print("Optimizing text placement (this might take a few seconds)...")
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title("t-SNE Visualization: The Gendered Landscape of AI (Full Lexicon)", fontsize=20, weight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title='Category', loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.show()

def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    MODEL_PATH = "/content/gdrive/MyDrive/Data mining/text mining/models/gender_ai_w2v_optimized.model"
    
    LEX_HARD = os.path.join(BASE_DIR, "lexicons/hard_ai.txt")
    LEX_SOFT = os.path.join(BASE_DIR, "lexicons/soft_ai.txt")

    visualize_tsne_full(MODEL_PATH, LEX_HARD, LEX_SOFT)

if __name__ == "__main__":
    main()