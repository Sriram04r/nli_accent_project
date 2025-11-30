# src/features/extract_hubert.py
import torch
import numpy as np
import pandas as pd
import os
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm

# choose model; you can use "facebook/hubert-base-ls960" or "facebook/hubert-large-ls960-ft"
MODEL_NAME = "facebook/hubert-base-ls960"

fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

def extract_mean(path):
    y, sr = sf.read(path)
    if sr != 16000:
        import librosa
        y = librosa.resample(y.astype(float), sr, 16000)
    inputs = fe(y, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        out = model(input_values, output_hidden_states=True)
    # choose last hidden state and mean-pool over time
    emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    return emb

os.makedirs("embeddings", exist_ok=True)
splits = ["train", "val", "test"]
for split in splits:
    df = pd.read_csv(f"splits/{split}.csv")
    X, y = [], []
    print("Extracting HuBERT for", split)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        emb = extract_mean(row["filename"])
        X.append(emb)
        y.append(row["language"])
    X = np.stack(X)
    y = np.array(y)
    np.save(f"embeddings/{split}_hubert_X.npy", X)
    np.save(f"embeddings/{split}_hubert_y.npy", y)
    print("Saved embeddings for", split)
