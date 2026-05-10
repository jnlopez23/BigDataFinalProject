# Big Data Final Project

This project analyzes potential relationships between U.S. voter behavior and social media activity.

## Project Structure

```text
BigDataFinalProject/
├── data/
│   ├── raw/
│   │   └── voter_registration/
│   │       └── abrv_voter_reg.dat
│   ├── processed/
│   │   ├── voter_reg_individual_response_subset.csv
│   │   ├── voter_registration_clean.csv
│   │   └── social_media/
│   │       ├── biden_preprocessed.csv
│   │       └── trump_preprocessed.csv
│   └── translated/
│       └── social_media/
│           ├── biden_translated.csv
│           └── trump_translated.csv
├── scripts/
│   ├── voter-reg.py
│   ├── download_data.py
│   ├── clean_data.py
│   └── preprocessing.py
├── notebooks/
│   ├── Kaggle_translation.ipynb
│   ├── cooccurence_network.ipynb
│   └── multiple_linreg.ipynb
└── reports/
    └── figures/
        ├── actual_vs_predicted.png
        ├── coefficient_impact.png
        ├── scatter.png
        ├── sentiment_impact_refined.png
        └── state_heatmap.png
```

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

## What Has Been Completed

- Project folder structure reorganized into `scripts`, `notebooks`, `reports/figures`, and `data` stages.
- Voter registration raw file is placed under `data/raw/voter_registration/`.
- Processed voter outputs are stored under `data/processed/`.
- Preprocessed social media outputs are stored under `data/processed/social_media/`.
- Translated social media outputs are stored under `data/translated/social_media/`.
- Existing analysis notebooks and exported figures have been organized.

## Datasets Used

- [US Election 2024 Social Media Sentiment Dataset](https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv)
- [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
- [Trump-related tweets (US Election Day 2020)](https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020)
