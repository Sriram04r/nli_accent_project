# src/preprocess/build_index.py
import os
import csv

# Root where you extracted the dataset
DATA_RAW = "data_raw"

# Output CSV path
OUT_CSV = "index.csv"

# Helper: find all audio files under DATA_RAW
audio_exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

rows = []
for root, dirs, files in os.walk(DATA_RAW):
    for f in files:
        if f.lower().endswith(audio_exts):
            full = os.path.join(root, f)
            # try to infer language and speaker from path
            # typical path: data_raw/<language>/<speaker>/<file.wav>
            parts = os.path.normpath(full).split(os.sep)
            # find index of data_raw in parts
            try:
                idx = parts.index(os.path.basename(DATA_RAW))
            except ValueError:
                # fallback: assume DATA_RAW is the first entry
                idx = 0
            # language = next part after data_raw if present
            language = parts[idx + 1] if len(parts) > idx + 1 else "unknown"
            speaker = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
            rows.append([full, speaker, language, "", ""])  # age and unit_type left blank

# write CSV (header: filename,speaker_id,language,age,unit_type)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as wf:
    w = csv.writer(wf)
    w.writerow(["filename", "speaker_id", "language", "age", "unit_type"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
print("Example rows:")
for r in rows[:5]:
    print(r)
