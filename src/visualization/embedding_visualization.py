import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

sns.set_style("whitegrid")

COLORS = {
    'Male Term': '#0000FF',    # Blue
    'Female Term': '#FF1493',  # Deep Pink
    'Hard AI': '#808080',      # Grey
    'Soft AI': '#DAA520'       # Goldenrod
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

def visualize_tsne(model_path, hard_lex_path, soft_lex_path, female_lex_path, male_lex_path):
    print(f"Loading model from {model_path}...")
    model = Word2Vec.load(model_path)
    
    # load lexicons
    print("Loading lexicons...")
    hard_words = load_lexicon(hard_lex_path)
    soft_words = load_lexicon(soft_lex_path)
    #female_words = load_lexicon(female_lex_path)
    #male_words = load_lexicon(male_lex_path)
    
    print("Using FILTERED gender lists...")
    male_words = FILTERED_MALE
    female_words = FILTERED_FEMALE

    # collect words
    plot_data = []
    
    def add_words(word_list, label):
        count = 0
        for w in word_list:
            if w in model.wv:
                plot_data.append({
                    'word': w,
                    'vector': model.wv[w],
                    'category': label
                })
                count += 1
        return count

    n_hard = add_words(hard_words, 'Hard AI')
    n_soft = add_words(soft_words, 'Soft AI')
    n_fem = add_words(female_words, 'Female Term')
    n_male = add_words(male_words, 'Male Term')
    
    print(f"Words to plot: Hard={n_hard}, Soft={n_soft}, Female={n_fem}, Male={n_male}")
    print(f"Total points: {len(plot_data)}")

    # data preparation for t-SNE
    df = pd.DataFrame(plot_data)
    X = np.array(df['vector'].tolist())
    
    # t-SNE
    # 2 components X and Y
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, n_iter=5000, perplexity=50, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    
    # plot
    print("Generating plot...")
    plt.figure(figsize=(16, 12)) 
    
    plot = sns.scatterplot(
        data=df, x='x', y='y', hue='category', 
        palette=COLORS, s=100, alpha=0.8, edgecolor='k'
    )
    
    # add labels
    for i in range(df.shape[0]):
        plt.text(
            df.x[i]+0.2, 
            df.y[i]+0.2, 
            df.word[i], 
            fontsize=9, 
            alpha=0.75,
            weight='bold' if df.category[i] in ['Female Term', 'Male Term'] else 'normal'
        )

    plt.title("t-SNE Visualization of Semantic Space: Gender & AI Topics", fontsize=20, weight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title='Category', loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.show()

def visualize_pca(model_path, hard_lex_path, soft_lex_path):
    print("\n--- Starting PCA Visualization ---")
    model = Word2Vec.load(model_path)
    
    # load lexicons
    hard_words = load_lexicon(hard_lex_path)
    soft_words = load_lexicon(soft_lex_path)
    male_words = FILTERED_MALE
    female_words = FILTERED_FEMALE
    
    plot_data = []
    
    def add_words(word_list, label):
        for w in word_list:
            if w in model.wv:
                plot_data.append({'word': w, 'vector': model.wv[w], 'category': label})

    add_words(hard_words, 'Hard AI')
    add_words(soft_words, 'Soft AI')
    add_words(female_words, 'Female Term')
    add_words(male_words, 'Male Term')
    
    if not plot_data:
        print("Error: No data for PCA.")
        return

    df = pd.DataFrame(plot_data)
    X = np.array(df['vector'].tolist())
    
    # PCA
    print("Running PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X)
    
    var_ratio = pca.explained_variance_ratio_
    print(f"PCA Explained Variance Ratio: {var_ratio} (Total: {sum(var_ratio):.2f})")
    
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    
    # plot
    plt.figure(figsize=(16, 12))
    
    category_order = ['Hard AI', 'Soft AI', 'Male Term', 'Female Term']
    df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)
    df = df.sort_values('category')

    sns.scatterplot(data=df, x='x', y='y', hue='category', palette=COLORS, s=150, alpha=0.85, edgecolor='black', linewidth=0.6)
    
    for i in range(df.shape[0]):
        is_gender = df.iloc[i]['category'] in ['Male Term', 'Female Term']
        plt.text(
            df.iloc[i]['x']+0.02, df.iloc[i]['y']+0.02, df.iloc[i]['word'], 
            fontsize=12 if is_gender else 9, 
            weight='bold' if is_gender else 'normal', 
            alpha=0.9, color='black'
        )

    plt.title("PCA Visualization: Global Semantic Structure", fontsize=20, weight='bold')
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]:.1%} variance)", fontsize=12)
    plt.legend(title='Category', loc='best', fontsize=12)
    plt.grid(True, alpha=0.2)
    
    plt.show()
    
def main():
    BASE_DIR = "/content/gdrive/MyDrive/Data mining/text mining/data"
    
    MODEL_PATH = os.path.join(BASE_DIR, "processed/gender_ai_w2v.model")
    HARD_PATH = os.path.join(BASE_DIR, "lexicons/hard_ai.txt")
    SOFT_PATH = os.path.join(BASE_DIR, "lexicons/soft_ai.txt")
    
    visualize_tsne(
        model_path=os.path.join(BASE_DIR, "processed/gender_ai_w2v.model"),
        hard_lex_path=os.path.join(BASE_DIR, "lexicons/hard_ai.txt"),
        soft_lex_path=os.path.join(BASE_DIR, "lexicons/soft_ai.txt"),
        female_lex_path=os.path.join(BASE_DIR, "lexicons/gender_female.txt"), 
        male_lex_path=os.path.join(BASE_DIR, "lexicons/gender_male.txt")      
    )
    visualize_pca(MODEL_PATH, HARD_PATH, SOFT_PATH)