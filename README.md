# Big Data Final Project

Analysis of associations between **CPS voter registration** (survey-based) and **social media** volume/sentiment at the state level. **Turnout is not modeled** in the main pipeline.

---

## How the pieces fit together (no duplicated work)

| What you need | Where it lives | What it is *not* |
|---------------|----------------|------------------|
| Reproducible **state–year** merge, VADER sentiment, correlations, clustering, sklearn regression | **`scripts/combined_analysis.py`** | Not the same grain as the exploratory notebook (see below). |
| **State-only**, **2020** slice, **split** Biden vs Trump volume & **absolute** translated sentiment, **statsmodels OLS**, extra plots (actual vs predicted, coef bars, heatmaps) | **`notebooks/multiple_linreg.ipynb`** | Different features, merge key (state only), and estimator than the script. |

Running the script does **not** replace re-running the notebook for *that* workflow, and vice versa. For a class report, cite **one canonical table** per research question or clearly label “state–year VADER panel” vs “state-only 2020 OLS panel.”

---

## Environment setup

From the project root, use a virtual environment and install:

```bash
pip install pandas matplotlib seaborn scikit-learn kagglehub vaderSentiment deep-translator langdetect
```

Optional: `statsmodels` if you extend the notebook (`pip install statsmodels`).

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

This single run completes the **registration-only** task list that is *not* duplicated in the notebook:

### 1) Merged analysis table

- **Voter:** `state` × `year` → `registration_rate`, `n_voter_records`
- **Social:** `state` × `year` → `tweet_volume`, `mean_sentiment`, `sentiment_std` (VADER compound on tweet text)
- **Merge:** `inner` on `(state, year)` → `data/processed/analysis/merged_state_year.csv`  
  Intersection years depend on tweet timestamps (often **2020-only** rows in the merge even when CPS has multiple years).

Supporting files: `voter_state_year.csv`, `social_state_year.csv`.

### 2) Descriptive figures (`reports/figures/combined_analysis/`)

| File | Description |
|------|-------------|
| `registration_rate_by_year.png` | Mean registration over **all CPS years** in the voter file (full voter panel, not limited by tweet years). |
| `mean_sentiment_by_year.png` | Mean VADER sentiment by **calendar year** in the combined tweet files. |
| `mean_sentiment_vs_registration_rate.png` | Scatter + trend on **merged** state–year rows. |

There is **no** turnout bar chart (turnout not in the CPS subset used here).

### 3) Correlations

- `correlation_pearson.csv`, `correlation_spearman.csv`  
- Variables: `registration_rate`, `tweet_volume`, `mean_sentiment`, `sentiment_std`

### 4) K-means clustering (fills gap vs. older task lists)

- Features (standardized): **`tweet_volume`**, **`mean_sentiment`** on merged rows with non-null features.
- Default **k = 3**; automatically reduced if fewer than 3 valid rows exist.
- **`social_cluster`** column appended → `merged_state_year_with_clusters.csv`
- Figures: `cluster_sizes.png`, `registration_rate_by_cluster.png` (mean **registration** by cluster, not turnout)

### 5) Predictive model (simple baseline)

- `linear_model_registration_results.csv`: **sklearn** `LinearRegression` for  
  `registration_rate ~ tweet_volume + mean_sentiment` on merged rows (R², RMSE, coefficients).

This is **distinct** from **`multiple_linreg.ipynb`**, which uses **statsmodels OLS** with **four** predictors (Biden/Trump volume and sentiment variants) on a **state-only, 2020** table.

---

## Notebooks (exploration & alternate specification)

- **`notebooks/multiple_linreg.ipynb`** — state-level merge, 2020 CPS filter, candidate-specific social features, OLS, diagnostic plots (e.g. actual vs predicted, sentiment coefficient bars). **Titles in some cells may still say “turnout”; axes use registration where noted.**
- **`notebooks/cooccurence_network.ipynb`**, **`notebooks/Kaggle_translation.ipynb`** — network / translation workflows; separate from the merged state–year script.

---

## Datasets used

- [US Election 2024 Social Media Sentiment Dataset](https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv)
- [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
- [Trump-related tweets (US Election Day 2020)](https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020)
