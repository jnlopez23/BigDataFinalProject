import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# load cleaned tweets
df = pd.read_csv('./cleaned_data/combined_cleaned_data.csv')

analyzer = SentimentIntensityAnalyzer()

# 'compound' score (-1 to 1)
def get_sentiment(text):
    if pd.isna(text): return 0
    return analyzer.polarity_scores(text)['compound']

# 4. Apply to your tweets
df['sentiment_score'] = df['tweet'].apply(get_sentiment)

# 5. Categorize the sentiment
df['sentiment_type'] = pd.cut(df['sentiment_score'], 
                              bins=[-1, -0.05, 0.05, 1], 
                              labels=['Negative', 'Neutral', 'Positive'])

# 6. Group by state to see who is the "happiest" or "angriest"
state_mood = df.groupby('state')['sentiment_score'].mean()
print(state_mood)