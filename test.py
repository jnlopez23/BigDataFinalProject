from langdetect import detect
import pandas as pd
# Take a tiny sample and check languages

df = pd.read_csv('./cleaned_data/trump_cleaned_data.csv', lineterminator='\n')

def safe_detect(text):
    try:
        # We only try to detect if there is actual text
        if pd.isna(text) or str(text).strip() == "":
            return "unknown"
        return detect(str(text))
    except:
        # This catches "No features in text" errors
        return "unknown"

sample_langs = df['tweet'].apply(safe_detect)
print(sample_langs.value_counts())