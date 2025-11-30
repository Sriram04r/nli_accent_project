from datasets import load_dataset
import os
import shutil

# Download dataset
print("Downloading IndicAccentDb dataset...")
ds = load_dataset("DarshanaS/IndicAccentDb")

print("Download complete.")

# Output folder
out_root = "data_raw"
os.makedirs(out_root, exist_ok=True)

print("Saving audio files into data_raw/...")

for split in ds:
    for i, item in enumerate(ds[split]):
        audio = item["audio"]
        lang = item.get("accent", "unknown")
        speaker = item.get("speaker_id", f"speaker_{i}")

        # Folder structure: data_raw/<language>/<speaker>/
        folder = os.path.join(out_root, lang, speaker)
        os.makedirs(folder, exist_ok=True)

        out_path = os.path.join(folder, f"{split}_{i}.wav")
        shutil.copy(audio["path"], out_path)

print("All audio files saved to data_raw/")
