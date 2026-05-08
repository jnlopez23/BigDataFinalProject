
1. Run python download_data.py to download raw kaggle csv data. Includes data from 2020 & 2024.
2. Run python clean_data.py to clean the data into 3 files, one for biden-specific tweets, one for trump-specific tweets, and one for combined.
3. Run python preprocessing.py to remove whitespace, URLs, and empty rows.

4. Run sentiment_analysis.py


Required Libraries:
- pandas
- kagglehub
- vaderSentiment
- deep-translator
- langdetect

Datasets used: 
- US Election 2024 Social Media Sentiment Dataset (https://www.kaggle.com/datasets/imaadmahmood/us-election-2024-social-media-sentiment-dataset?select=election_tweets.csv)

- US Election 2020 Tweets (https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)

- Trump-related tweets (US Election Day 2020) (https://www.kaggle.com/datasets/wyewlee/trumprelated-tweets-us-election-day-2020)
