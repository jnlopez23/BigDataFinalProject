import kagglehub
import pandas as pd
import os
import shutil

# kaggle datasets
datasets = [
    # 2024 (Current)
    "imaadmahmood/us-election-2024-social-media-sentiment-dataset",

    # 2021-2023 (The Gap Years)
    "thedevastator/analyzing-the-political-discourse-of-reddit-s-su",
    # 2020 (Historical Baseline)
    "manchunhui/us-election-2020-tweets",
    "noorsaeed/usa-election-sentiment-analysis-dataset",
    "yewleewong/trump-related-tweets-us-election-day-2020"
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