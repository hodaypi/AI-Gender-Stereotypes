# Gender Stereotypes in AI Discourse on Reddit

## Overview

This research project investigates how gender stereotypes manifest in AI-related discussions on Reddit, examining the semantic division between "Hard AI" (technical/core) and "Soft AI" (ethics/UX) and their gender associations.

## Research Questions

1. **Division and Bias**: Is the mention of women associated with "Soft AI" terminology at a significantly higher frequency than men?
2. **Sentiment and Climate**: Are responses to posts about women in "Hard AI" contexts more hostile or negative compared to men?
3. **Self-Talk and Internalization**: Do women discussing themselves in AI contexts show psychological patterns (via LIWC) indicating stereotype internalization?

## Hypotheses

- **H1**: Women are mentioned more frequently with "Soft AI" words than men (statistically significant)
- **H2**: Comments on women's posts in "Hard AI" contexts exhibit more negative sentiment
- **H3**: Women's self-referential posts show lower power/certainty scores in LIWC analysis

## Data Sources

### Subreddits
| Category | Subreddits |
|----------|------------|
| Technical Core | r/MachineLearning, r/OpenAI |
| Creative/Usage | r/Midjourney, r/StableDiffusion |
| Career | r/CSCareerQuestions |
| Ethics & Soft AI | r/artificial, r/singularity, r/ChatGPT, r/AIethics |

## Dataset Summary

### Posts by Subreddit
| Subreddit | Female | Male | Total | Category |
|-----------|--------|------|-------|----------|
| MachineLearning | 550 | 704 | 1,254 | Technical |
| OpenAI | 703 | 797 | 1,500 | Technical |
| Midjourney | 1,144 | 1,101 | 2,245 | Creative |
| StableDiffusion | 1,102 | 1,026 | 2,128 | Creative |
| CSCareerQuestions | 964 | 865 | 1,829 | Career |
| artificial | 659 | 751 | 1,410 | Ethics/Soft AI |
| singularity | 764 | 859 | 1,623 | Ethics/Soft AI |
| ChatGPT | 1,059 | 912 | 1,971 | Ethics/Soft AI |
| AIethics | 3 | 14 | 17 | Ethics/Soft AI |
| **Total** | **6,948** | **7,029** | **13,977** | |

### Collection Totals
| Data Type | Count |
|-----------|-------|
| Total Posts | 13,977 |
| Female-related Posts | 6,948 |
| Male-related Posts | 7,029 |
| Comments | 309,124 |

### Output Files
- `data/raw/female_posts.json` - Posts mentioning female terms
- `data/raw/male_posts.json` - Posts mentioning male terms
- `data/raw/all_gender_posts.json` / `.csv` - Combined dataset
- `data/raw/comments.json` / `.csv` - Comments from top engagement posts

### Custom Lexicons

| Category | Keywords |
|----------|----------|
| Gender - Women (Target) | woman, female, girl, she, her, hers |
| Gender - Men (Baseline) | man, male, guy, he, his, him |
| Hard AI / Core Tech | LLM, Transformer, PyTorch, Tensor, GPU, Architecture, Optimization, Algorithm |
| Soft AI / Ethics & UX | Ethics, Bias, Fairness, Policy, Regulation, Prompt Engineering, Design, Art, Moral |

### LIWC Categories
- Power & Achievement
- Cognitive Processes
- Negative Emotions
- Certainty
- Affiliation / Social
- Perceptual Processes

## Methodology

### 1. Data Collection & Labeling
- Automated collection from selected subreddits
- Implicit labeling via search queries and lexicons
- First-person filtering (I, my) for self-talk analysis

### 2. Statistical Analysis
- **Chi-square test**: Compare Hard/Soft AI word frequencies between gender groups

### 3. Sentiment Analysis
- **VADER**: Measure sentiment in responses to posts
- Compare negative sentiment in responses to women vs men in Hard AI contexts

### 4. Psychological Profiling
- **LIWC Analysis**: Compare power, achievement, certainty, and cognitive process scores
- Focus on self-talk posts across genders

### 5. Semantic Analysis
- **Word2Vec**: Train embeddings on full dataset
- **Cosine similarity**: Measure distance between gender terms and AI category terms
- **t-SNE visualization**: Display semantic space relationships

## Project Structure

```
stereotypes_in_ai/
├── README.md
├── data/
│   ├── raw/                 # Raw Reddit data
│   ├── processed/           # Cleaned and labeled data
│   └── lexicons/            # Custom word lists
├── src/
│   ├── collection/          # Reddit data collection scripts
│   ├── preprocessing/       # Data cleaning and labeling
│   ├── analysis/
│   │   ├── statistical/     # Chi-square tests
│   │   ├── sentiment/       # VADER analysis
│   │   ├── liwc/            # LIWC processing
│   │   └── embeddings/      # Word2Vec and similarity
│   └── visualization/       # t-SNE and charts
├── notebooks/               # Jupyter notebooks for exploration
├── results/                 # Output figures and tables
└── reports/                 # Final research report
```

## Deliverables

1. **Labeled Dataset**: CSV/JSON with implicit and algorithmic tags (LIWC scores, sentiment)
2. **Python Code**: Complete, documented scripts for all analysis stages
3. **Research Report**: Comprehensive report with statistical comparisons, LIWC findings, and visualizations

## Requirements

```
praw                 # Reddit API
pandas
numpy
scipy                # Statistical tests
vaderSentiment       # Sentiment analysis
gensim               # Word2Vec
scikit-learn         # t-SNE, preprocessing
matplotlib
seaborn
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Collect data
python src/collection/reddit_collector.py

# Run analysis pipeline
python src/analysis/run_analysis.py

# Generate visualizations
python src/visualization/generate_plots.py
```

## License

[Add license information]

## Authors

[Add author information]

## Acknowledgments

[Add acknowledgments]
