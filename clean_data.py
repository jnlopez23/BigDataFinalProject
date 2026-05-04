import pandas as pd
import os

def clean_data(file_path):
    print(f" for {file_path}")
    
    df = pd.read_csv(file_path)
    
    df.dropna(inplace=True) 
    df.drop_duplicates(inplace=True) 
    
    cleaned_file_path = file_path.replace("raw_data", "cleaned_data")
    os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
    df.to_csv(cleaned_file_path, index=False)
    
    print(f" saved {cleaned_file_path}")

if __name__ == "__main__":
    raw_folder = "./raw_data"
    for file in os.listdir(raw_folder):
        if file.endswith(".csv"):
            clean_data(os.path.join(raw_folder, file))
