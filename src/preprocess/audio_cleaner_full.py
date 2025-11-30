# src/preprocess/audio_cleaner_full.py
"""
Robust, resumable audio cleaner.

- Reads `index.csv` (created earlier).
- For each row, writes cleaned audio to data_clean/<same-relpath>.
- Skips files already cleaned (idempotent).
- Writes/updates index_clean.csv progressively so you can resume.
- Logs failures to clean_failures.txt.
"""
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

INDEX_CSV = "index.csv"          # input (raw file list)
OUT_ROOT = "data_clean"          # cleaned audio output root
CLEAN_INDEX = "index_clean.csv"  # output progressive index
FAIL_LOG = "clean_failures.txt"
SR = 16000
TOP_DB = 20

os.makedirs(OUT_ROOT, exist_ok=True)

# Load master index
df = pd.read_csv(INDEX_CSV)
total = len(df)
print(f"Master index rows: {total}")

# Load or create cleaned index (so we can resume)
if os.path.exists(CLEAN_INDEX):
    cleaned_df = pd.read_csv(CLEAN_INDEX)
    cleaned_files = set(cleaned_df['filename'].astype(str).tolist())
    print(f"Found existing {CLEAN_INDEX} with {len(cleaned_df)} rows -> resuming.")
else:
    cleaned_df = pd.DataFrame(columns=["filename","speaker_id","language","age","unit_type"])
    cleaned_files = set()
    print(f"No existing {CLEAN_INDEX}. Starting fresh.")

fail_log_f = open(FAIL_LOG, "a", encoding="utf-8")

# iterate over master index
for i, row in tqdm(df.iterrows(), total=total):
    raw_path = str(row['filename'])
    # compute output path preserving subfolders under data_raw
    try:
        rel = os.path.relpath(raw_path, "data_raw")
    except Exception:
        rel = os.path.basename(raw_path)
    out_path = os.path.join(OUT_ROOT, rel)

    # If already cleaned (exists in index or file exists), skip
    if out_path in cleaned_files or os.path.exists(out_path):
        # ensure it's recorded in cleaned_df (might exist from previous run)
        if out_path not in cleaned_files:
            cleaned_df = pd.concat([cleaned_df, pd.DataFrame([[out_path, row.get('speaker_id', ''), row.get('language',''), row.get('age',''), row.get('unit_type','')]], columns=cleaned_df.columns)], ignore_index=True)
            cleaned_files.add(out_path)
        continue

    # attempt to load, trim, resample and save
    try:
        y, sr = librosa.load(raw_path, sr=SR)
        y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)
        # ensure out dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y_trim, SR)
        # append to cleaned_df and flush to disk
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([[out_path, row.get('speaker_id', ''), row.get('language',''), row.get('age',''), row.get('unit_type','')]], columns=cleaned_df.columns)], ignore_index=True)
        cleaned_files.add(out_path)
        # write updated CSV every 100 files to avoid too many writes
        if len(cleaned_files) % 100 == 0:
            cleaned_df.to_csv(CLEAN_INDEX, index=False)
    except Exception as e:
        msg = f"FAILED: {raw_path} -> {repr(e)}\n"
        print(msg.strip())
        fail_log_f.write(msg)
        fail_log_f.flush()

# final save
cleaned_df.to_csv(CLEAN_INDEX, index=False)
fail_log_f.close()
print(f"Done. Cleaned files recorded in {CLEAN_INDEX}. Failures (if any) in {FAIL_LOG}")
