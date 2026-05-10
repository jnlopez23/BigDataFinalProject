import pandas as pd
import re
import os


def clean_tweet(text):
    ''' 
    Basic tweet cleaning function:
    - Remove URLs
    - Strip extra whitespace
    '''
    text = str(text) # Force to string

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.replace('@', '')
    text = " ".join(text.split())
    text = re.sub(r'[^\w\s!\?\.]', '', text)

    # remove extra whitespace
    text = " ".join(text.split())

    return text.strip()



def process_file(file_path, output_name):
    '''
    Processes a single CSV file: cleans tweets and translates if needed, then saves the result.
    '''

    target_folder = "./processed_data"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


    print(f"preprocessing for {file_path}...")

    df = pd.read_csv(file_path, lineterminator='\n')
        
    df['tweet'] = df['tweet'].apply(clean_tweet)
    
    df.to_csv(os.path.join(target_folder, output_name), index=False)
    print(f"saved to {output_name}")


# Run for both files
process_file('./cleaned_data/trump_cleaned_data.csv', 'trump_preprocessed.csv')
process_file('./cleaned_data/biden_cleaned_data.csv', 'biden_preprocessed.csv')
