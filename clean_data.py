import pandas as pd
import os
                                                   

# 1. Load the dataset
manchunhui_trump_tweets_df = pd.read_csv("./raw_data/manchunhui_hashtag_donaldtrump.csv", lineterminator='\n')

manchunhui_biden_tweets_df = pd.read_csv("./raw_data/manchunhui_hashtag_joebiden.csv", lineterminator='\n')

# 2. Keep only the requested columns
trump_df_cleaned = manchunhui_trump_tweets_df[['created_at', 'tweet', 'state']]
biden_df_cleaned = manchunhui_biden_tweets_df[['created_at', 'tweet', 'state']]

# 3. Optional: Remove rows where 'state' is empty (NaN)
trump_df_cleaned = trump_df_cleaned.dropna(subset=['state'])
biden_df_cleaned = biden_df_cleaned.dropna(subset=['state'])

target_folder = "./cleaned_data"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# save to cleaned_data folder
trump_df_cleaned.to_csv(os.path.join(target_folder, 'trump_cleaned_data.csv'), index=False)
biden_df_cleaned.to_csv(os.path.join(target_folder, 'biden_cleaned_data.csv'), index=False)

# combine into a new csv file
combined_df = pd.concat([trump_df_cleaned, biden_df_cleaned], ignore_index=True)
combined_df.to_csv(os.path.join(target_folder, 'combined_cleaned_data.csv'), index=False)

print("\n cleaned data saved to cleaned_data folder")
