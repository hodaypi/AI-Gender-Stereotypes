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
| Technical Core | r/MachineLearning, r/OpenAI, r/deeplearning, r/LanguageTechnology, r/learnmachinelearning, r/MLQuestions, r/datascience, r/LocalLLaMA, r/Oobabooga, r/LLMDevs |
| Creative/Usage | r/Midjourney, r/StableDiffusion, r/AIArt, r/comfyui, r/dalle, r/Leonardo_AI |
| Career | r/CSCareerQuestions, r/ExperiencedDevs, r/cscareerquestionsEU, r/MLjobs, r/datasciencecareer |
| Ethics & Soft AI | r/artificial, r/singularity, r/ChatGPT, r/AIethics, r/ControlProblem, r/AItechnology|
| AI Tools | r/ClaudeAI, r/Bard, r/bing, r/Perplexity, r/NotionAI, r/GPT3 |
| General | r/ArtificialInteligence, r/technology, r/FutureTechnology|
## Dataset Summary

### Posts by Subreddit

|## Posts per Subreddit by Gender Context (Categorized)

| Category              | Subreddit              | Female | Male  | Total |
|-----------------------|------------------------|--------|-------|-------|
| Technical Core        | MachineLearning        | 228    | 575   | 803   |
| Technical Core        | OpenAI                 | 310    | 724   | 1034  |
| Technical Core        | deeplearning           | 65     | 568   | 633   |
| Technical Core        | LanguageTechnology     | 56     | 457   | 513   |
| Technical Core        | learnmachinelearning   | 214    | 623   | 837   |
| Technical Core        | MLQuestions            | 52     | 553   | 605   |
| Technical Core        | datascience            | 231    | 567   | 798   |
| Technical Core        | LocalLLaMA             | 206    | 580   | 786   |
| Technical Core        | Oobabooga              | 21     | 120   | 141   |
| Technical Core        | LLMDevs                | 24     | 343   | 367   |
| Creative/Usage        | Midjourney             | 1228   | 1136  | 2364  |
| Creative/Usage        | StableDiffusion        | 1146   | 954   | 2100  |
| Creative/Usage        | AIArt                  | 2160   | 1997  | 4157  |
| Creative/Usage        | comfyui                | 422    | 657   | 1079  |
| Creative/Usage        | dalle                  | 96     | 287   | 383   |
| Career                | CSCareerQuestions      | 1357   | 2336  | 3693  |
| Career                | ExperiencedDevs        | 298    | 853   | 1151  |
| Career                | cscareerquestionsEU    | 466    | 837   | 1303  |
| Career                | MLjobs                 | 15     | 88    | 103   |
| Ethics & Soft AI      | artificial             | 420    | 922   | 1342  |
| Ethics & Soft AI      | singularity            | 360    | 1077  | 1437  |
| Ethics & Soft AI      | ChatGPT                | 1768   | 2229  | 3997  |
| Ethics & Soft AI      | AIethics               | 2      | 11    | 13    |
| Ethics & Soft AI      | ControlProblem         | 41     | 217   | 258   |
| Ethics & Soft AI      | AItechnology           | 3      | 12    | 15    |
| AI Tools              | ClaudeAI               | 287    | 940   | 1227  |
| AI Tools              | Bard                   | 244    | 704   | 948   |
| AI Tools              | bing                   | 257    | 430   | 687   |
| AI Tools              | Perplexity             | 0      | 9     | 9     |
| AI Tools              | GPT3                   | 102    | 467   | 569   |
| General               | ArtificialInteligence  | 589    | 1294  | 1883  |
| General               | technology             | 1089   | 1742  | 2831  |
| General               | FutureTechnology       | 0      | 5     | 5     |
| **TOTAL**             | —                      | **13757** | **24314** | **38071** |


### Collection Totals
| Data Type | Count |
|-----------|-------|
| Total Posts | 57,392 |
| Female-related Posts | 13,757 |
| Male-related Posts | 24,314 |
| Comments | 942,108 |

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
| Hard AI / Core Tech | LLM, Transformer, PyTorch, Tensor, GPU, Architecture, Optimization, Algorithm and more |
| Soft AI / Ethics & UX | Ethics, Bias, Fairness, Policy, Regulation, Prompt Engineering, Design, Art, Moral and more  |

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
