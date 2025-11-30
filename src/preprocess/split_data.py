# src/preprocess/split_data.py
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

# Use the cleaned index
IN_CSV = "index_clean.csv"

df = pd.read_csv(IN_CSV)
print("Total cleaned files:", len(df))

# Make output folder for splits
os.makedirs("splits", exist_ok=True)

# First split: train (70%) + temp (30%)
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df["speaker_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)

# Second split: temp → val (15%) + test (15%)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["speaker_id"]))

val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print("Train files:", len(train_df))
print("Val files:", len(val_df))
print("Test files:", len(test_df))

train_df.to_csv("splits/train.csv", index=False)
val_df.to_csv("splits/val.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)

print("Saved: splits/train.csv, splits/val.csv, splits/test.csv")
