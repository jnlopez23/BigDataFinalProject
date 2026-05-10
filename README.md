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

## Notebook summaries

Paths inside notebooks may say `translated_data/` or `data/processed/voter_reg_individual_response_subset.csv` — if you run locally from the **project root**, point them at your actual outputs (e.g. `data/translated/social_media/`, `data/processed/voter_reg/voter_reg_individual_response_subset.csv`).

### `notebooks/cooccurence_network.ipynb` (keyword co-occurrence networks)

- Loads **translated** Biden and Trump tweet CSVs, filters to a fixed **US state** list, concatenates for a combined view.
- Scans tweet text for a **political keyword list** (`vote`, `biden`, `trump`, `climate`, etc.) and builds **co-occurrence pairs** of keywords that appear in the same tweet.
- Keeps edges where a pair appears more than a **frequency threshold** (default: more than 5 co-occurrences); edge **weight** ≈ count, edge **color** encodes **average `sentiment_score`** on those tweets.
- Uses **NetworkX** (`spring_layout`, degree-based node sizes) and **Matplotlib** to draw **three** networks: Biden-only, Trump-only, and **combined**.
- Prints top edges (keyword pair, frequency, sentiment label). Suited for a **complex-networks** framing of “which themes co-occur together and with what tone,” separate from the CPS merge in `combined_analysis.py`.

### `notebooks/Kaggle_translation.ipynb` (multilingual sentiment scoring on Kaggle)

- Intended to run on **Kaggle** (`kaggle_secrets` for a Hugging Face token; GPU if available).
- Loads **`cardiffnlp/twitter-xlm-roberta-base-sentiment`** (`transformers` + `torch`), runs **forward passes** on tweet text, converts logits to a **single score** (positive softmax minus negative softmax, scaled roughly between −1 and +1).
- Reads preprocessed Trump/Biden CSV paths from **Kaggle input**, writes **`trump_translated.csv`** and **`biden_translated.csv`** to **`/kaggle/working/`** with a new **`sentiment_score`** column.
- Offline / local workflows can instead use **`scripts/`** preprocessing and **`combined_analysis.py`** (VADER) or replicate this notebook with adjusted paths.

### `notebooks/multiple_linreg.ipynb` (state-level multiple regression)

- Imports translated Biden/Trump data and CPS voter subset; restricts voters to **`year == 2020`** for alignment with typical tweet windows.
- Builds **`registration_pct_by_demo`**: CPS **registration indicator** (0/1) aggregated to **registration rate %** by `state`, `race_label`, `gender` (registration only—not turnout).
- Aggregates tweets to **state** level: Biden volume, Trump volume, and **mean absolute** `sentiment_score` per side (`biden_sent`, `trump_sent`).
- Merges social + CPS on **`state`** only (`inner`) → **`final_regression_df`** with `actual_reg_rate` (often scaled to percent).
- Fits **statsmodels OLS**: `actual_reg_rate ~ const + biden_vol + biden_sent + trump_vol + trump_sent`.
- Visualization cells: **actual vs predicted** scatter with 45° line; **horizontal bar chart** of sentiment-related coefficients ± standard errors; **Seaborn `regplot`** (e.g. Trump sentiment vs registration) with **state labels**; **heatmap** (state × trump sentiment vs registration, min–max scaled for color).
- **Note:** Some plot **titles still say “turnout”** or “youth registration” while the dependent variable is **CPS registration rate**—treat titles as legacy wording.

---

## Datasets used

- [US Election 2024 Social Media Sentiment Dataset](https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv)
- [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
- [Trump-related tweets (US Election Day 2020)](https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020)
