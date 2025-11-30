# src/preprocess/audio_cleaner.py
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Input index CSV (created earlier)
INDEX_CSV = "index.csv"

# Output folder
OUT_ROOT = "data_clean"
SR = 16000      # target sample rate
TOP_DB = 20     # silence trimming threshold

os.makedirs(OUT_ROOT, exist_ok=True)

df = pd.read_csv(INDEX_CSV)

# TEST_LIMIT = 20  # for testing on 20 files only
TEST_LIMIT = None

clean_rows = []

iterable = list(df.iterrows()) if TEST_LIMIT is None else list(df.iterrows())[:TEST_LIMIT]

for i, row in tqdm(iterable, total=len(iterable)):
    infile = row["filename"]
    try:
        # load audio and resample to 16kHz
        y, sr = librosa.load(infile, sr=SR)

        # trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

        # create output path under data_clean
        rel_path = os.path.relpath(infile, "data_raw")
        out_path = os.path.join(OUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # save cleaned file
        sf.write(out_path, y_trimmed, SR)

        clean_rows.append([
            out_path,
            row["speaker_id"],
            row["language"],
            row.get("age", ""),
            row.get("unit_type", "")
        ])

    except Exception as e:
        print("FAILED:", infile, e)

# save cleaned CSV
clean_df = pd.DataFrame(clean_rows, columns=["filename", "speaker_id", "language", "age", "unit_type"])
clean_df.to_csv("index_clean.csv", index=False)

print("Saved index_clean.csv with", len(clean_df), "rows")
