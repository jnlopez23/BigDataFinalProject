import kagglehub
import pandas as pd
import os
import shutil

# kaggle datasets
datasets = [
    "sohumgokhale/multi-platform-social-sentiment-evolution",
    "imaadmahmood/us-election-2024-social-media-sentiment-dataset",
    "programmer3/political-tweets-and-social-reactions"
]

target_folder = "./raw_data"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"Created folder: {target_folder}")

for dataset in datasets:
    print(f"\n--- Fetching: {dataset} ---")
    
    path = kagglehub.dataset_download(dataset)
    
    # Find all CSV files and move them
    files = os.listdir(path)
    for file in files:
        if file.endswith(".csv"):
            source_path = os.path.join(path, file)
            # We add the dataset creator's name to the filename 
            # so they don't overwrite each other if they have the same name!
            new_name = f"{dataset.split('/')[0]}_{file}"
            shutil.copy(source_path, os.path.join(target_folder, new_name))
            print(f"✅ Saved to {target_folder}/{new_name}")

print("\nDone! All datasets are in your 'raw_data' folder.")