import kagglehub
import pandas as pd
import os
import shutil

# 1. Download the latest version of the dataset
# This downloads it to a "cache" folder on your Mac
path = kagglehub.dataset_download("imaadmahmood/us-election-2024-social-media-sentiment-dataset")
print("Downloaded to cache at:", path)

# 2. Look at the files inside that folder
files = os.listdir(path)
print("Files found in download:", files)

# 3. Move the CSV from the cache to your current Project folder
# We'll look for any file ending in .csv and copy it here
for file in files:
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), f"./{file}")
        print(f"Successfully moved {file} to your project folder!")

# 4. Optional: Load it to make sure it works
# df = pd.read_csv("sentiment_dataset.csv") # Use the actual filename here
# print(df.head())