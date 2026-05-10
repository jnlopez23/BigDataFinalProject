# Big Data Final Project

This project analyzes potential relationships between U.S. voter behavior and social media activity.

## Environment Setup

1. Create and activate a Python environment.
2. Install required libraries:

```bash
pip install pandas kagglehub vaderSentiment deep-translator langdetect
```

## Data Setup

### Voter registration data (IPUMS CPS)

1. Download `abrv_voter_reg.dat` (the abbreviated CPS extract used in this project).
2. Place it at:
   - `data/raw/voter_registration/abrv_voter_reg.dat`
3. Run:

```bash
python "scripts/voter-reg.py"
```

This script generates the cleaned individual response subset in:
- `data/processed/voter_reg_individual_response_subset.csv`

### Social media data (Kaggle)

Run the pipeline in order:

```bash
python "scripts/download_data.py"
python "scripts/clean_data.py"
python "scripts/preprocessing.py"
```

## Combined Analysis Merge Details

To run the first-step analysis pipeline:

```bash
python "scripts/combined_analysis.py"
```

This script merges voter and social data in three stages:

1. Build voter state-year table from `data/processed/voter_reg/voter_reg_individual_response_subset.csv`
   - Group keys: `state`, `year`
   - Aggregations:
     - `registration_rate` = mean of `voter_registration`
     - `turnout_rate` = mean of `voting_turnout`
     - `n_voter_records` = count of voter rows
   - Output: `data/processed/analysis/voter_state_year.csv`

2. Build social media state-year table from:
   - `data/processed/social_media/biden_preprocessed.csv`
   - `data/processed/social_media/trump_preprocessed.csv`
   - Steps:
     - Parse `created_at` to extract `year`
     - Compute VADER sentiment compound score per tweet
     - Concatenate both social datasets
     - Group keys: `state`, `year`
   - Aggregations:
     - `tweet_volume` = count of tweets
     - `mean_sentiment` = mean compound sentiment
     - `sentiment_std` = standard deviation of compound sentiment
   - Output: `data/processed/analysis/social_state_year.csv`

3. Merge voter + social tables
   - Join type: `inner`
   - Join keys: `state`, `year`
   - Output: `data/processed/analysis/merged_state_year.csv`

Notes:
- `inner` join keeps only state-year pairs present in both tables.
- Additional outputs from this script:
  - `data/processed/analysis/correlation_pearson.csv`
  - `data/processed/analysis/correlation_spearman.csv`
  - `data/processed/analysis/linear_model_turnout_results.csv`
  - figures in `reports/figures/combined_analysis/`


## Datasets Used

- [US Election 2024 Social Media Sentiment Dataset](https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv)
- [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
- [Trump-related tweets (US Election Day 2020)](https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020)
