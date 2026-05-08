# 1. Install necessary libraries in Colab
!pip install transformers scipy torch

import os

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import torch

# 2. Setup Model and GPU
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_multilingual_sentiment(text):
    """Returns a score between -1 (Negative) and 1 (Positive)"""
    if not text or len(str(text)) < 5:
        return 0
    
    try:
        # Preprocess and tokenize
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        
        # Get model output
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Convert output to probabilities
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        
        # XLM-T output is: [Negative, Neutral, Positive]
        # Calculate a weighted compound score
        ranking = scores[2] - scores[0] # (Positive - Negative)
        return ranking
    except:
        return 0




def process_file(file_path, output_name):
    target_folder = "./processed_data"
    
    df = pd.read_csv(file_path, lineterminator='\n')
        
    df['tweet'] = df['tweet'].apply(get_multilingual_sentiment)
    
    df.to_csv(os.path.join(target_folder, output_name), index=False)
    print(f"saved to {output_name}")

process_file('./processed_data/trump_preprocessed.csv', 'trump_translated.csv')
process_file('./processed_data/biden_preprocessed.csv', 'biden_translated.csv')

