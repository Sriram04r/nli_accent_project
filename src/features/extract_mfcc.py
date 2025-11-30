# src/features/extract_mfcc.py
import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Function to compute MFCC features
def compute_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mean + std pooling
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat

# Process each split
splits = ["train", "val", "test"]

for split in splits:
    df = pd.read_csv(f"splits/{split}.csv")

    X, y = [], []

    print(f"Extracting MFCC for: {split} ...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        feat = compute_mfcc(row["filename"])
        X.append(feat)
        y.append(row["language"])

    X = np.array(X)
    y = np.array(y)

    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/{split}_mfcc_X.npy", X)
    np.save(f"embeddings/{split}_mfcc_y.npy", y)

    print(f"Saved MFCC features for {split}")
