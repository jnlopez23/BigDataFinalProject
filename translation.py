from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import os
import pandas as pd

# seed for language detection 
DetectorFactory.seed = 0

def translate_if_needed(text):
    '''
    Detects language and translates to English if not already in English.
    If detection fails (e.g. only emojis), returns original text.
    '''
    if not text or len(text) < 3:
        return text
    try:
        lang = detect(text)
        if lang != 'en':
            # translate to English
            return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        pass
    return text


target_folder = "./processed_data"

def process_file(file_path, output_name):
    df = pd.read_csv(file_path, lineterminator='\n')
        
    df['tweet'] = df['tweet'].apply(translate_if_needed)
    
    df.to_csv(os.path.join(target_folder, output_name), index=False)
    print(f"saved to {output_name}")

process_file('./processed_data/trump_preprocessed.csv', 'trump_translated.csv')
process_file('./processed_data/biden_preprocessed.csv', 'biden_translated.csv')

