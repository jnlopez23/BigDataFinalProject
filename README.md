# Big Data Final Project

Analysis of associations between **CPS voter registration** (survey-based) and **social media** volume/sentiment at the state level. **Turnout is not modeled** in the main pipeline.

---

## Environment setup

From the project root, use a virtual environment and install:

```bash
pip install pandas matplotlib seaborn scikit-learn kagglehub vaderSentiment deep-translator langdetect
```

Optional: **`statsmodels`** for `multiple_linreg.ipynb`.  
Optional (Kaggle translation notebook / co-occurrence inputs): **`torch`**, **`transformers`**, **`tqdm`**, **`scipy`**.

---

## Data setup

### Voter registration (IPUMS CPS)

1. Place `abrv_voter_reg.dat` at `data/raw/voter_registration/abrv_voter_reg.dat`
2. Run:

```bash
python "scripts/voter-reg.py"
```

**Output:** `data/processed/voter_reg/voter_reg_individual_response_subset.csv`  
(Registration indicator and demographics; **no turnout column**.)

### Social media (Kaggle pipeline)

```bash
python "scripts/download_data.py"
python "scripts/clean_data.py"
python "scripts/preprocessing.py"
```

Preprocessed tweet CSVs: `data/processed/social_media/`.

---

## Combined analysis script (main quantitative pipeline)

```bash
python "scripts/combined_analysis.py"
```

## 1. Merged analysis dataset

The analysis begins by aggregating two separate data sources at the state-year level.

The voter dataset is grouped by state × year, producing:
- voter registration rate (`registration_rate`)
- number of voter records (`n_voter_records`)

The social media dataset is also aggregated by state × year, producing:
- tweet volume (`tweet_volume`)
- mean sentiment (`mean_sentiment`, based on VADER compound scores)
- sentiment variability (`sentiment_std`)

These two datasets are merged using an inner join on (state, year), resulting in:
`data/processed/analysis/merged_state_year.csv`

Because tweet coverage does not perfectly align with all CPS survey years, the final merged dataset is typically limited to overlapping years (often primarily 2020 depending on tweet availability).

Supporting intermediate outputs:
- `voter_state_year.csv`
- `social_state_year.csv`

---

## 2. Descriptive figures (`reports/figures/combined_analysis/`)

- `registration_rate_by_year.png`: average voter registration across CPS survey years (full voter dataset)
- `mean_sentiment_by_year.png`: average tweet sentiment over time
- `mean_sentiment_vs_registration_rate.png`: relationship between sentiment and registration at the state-year level

There is no turnout visualization since turnout variables were removed.

---

## 3. Correlation analysis

Correlation matrices:
- `correlation_pearson.csv`
- `correlation_spearman.csv`

Variables included:
- registration rate
- tweet volume
- mean sentiment
- sentiment standard deviation

---

## 4. K-means clustering

Features used:
- tweet volume
- mean sentiment

K-means clustering is run with k = 3 (or fewer if insufficient data). Each row is assigned:
- `social_cluster`

Outputs:
- `merged_state_year_with_clusters.csv`

Figures:
- `cluster_sizes.png`
- `registration_rate_by_cluster.png`

---

## 5. Predictive modeling (baseline regression)

Model:
- registration_rate ~ tweet_volume + mean_sentiment

Output:
- `linear_model_registration_results.csv`

Metrics include:
- R²
- RMSE
- coefficients

---

## Notebook summaries

Some notebooks reference older folder names (e.g. translated_data vs data/processed). All paths should be updated to match the final project structure.

---

### cooccurence_network.ipynb

This notebook builds keyword co-occurrence networks from tweet text.

It uses translated Biden and Trump tweet datasets and filters tweets by US states. Tweets are scanned for political keywords (e.g. vote, biden, trump, climate), and co-occurring keywords are counted.

A NetworkX graph is constructed where:
- nodes = keywords
- edges = co-occurrence frequency
- edge weight = frequency
- edge color = average sentiment

Separate networks are created for Biden, Trump, and combined data to compare thematic structure and sentiment across political groups.

---

### Kaggle_translation.ipynb

This notebook is designed for Kaggle execution.

It uses a transformer model (cardiffnlp/twitter-xlm-roberta-base-sentiment) to compute sentiment scores for tweets using GPU acceleration when available.

It:
- loads datasets from Kaggle inputs
- computes sentiment per tweet
- saves outputs with a `sentiment_score` column
- writes results to `/kaggle/working/`

For local use, this is replicated using the scripts pipeline or VADER-based approach.

---

### multiple_linreg.ipynb

This notebook performs state-level regression analysis.

It restricts CPS data to 2020 and merges:
- voter registration rates (CPS)
- Biden/Trump tweet volume
- sentiment measures

Model:
registration_rate ~ biden_vol + biden_sent + trump_vol + trump_sent

Includes:
- predicted vs actual plots
- coefficient plots
- seaborn regression plots
- heatmaps

Note: some plots still reference “turnout,” but the dependent variable is registration rate.

---

## Datasets used

- US Election 2024 Social Media Sentiment Dataset  
https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv

- US Election 2020 Tweets  
https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets

- Trump-related tweets (Election Day 2020)  
https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020