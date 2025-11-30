# src/demo_app/app.py
import os
import tempfile
import joblib
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch
import datetime
import shutil
import csv

# --- Config ---
MODEL_PATH = "saved_models/hubert_rf.pkl"   # ensure this file exists
HUBERT_NAME = "facebook/hubert-base-ls960"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload

# --- Ensure folders exist ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load classifier model (RandomForest) with a friendly error if missing ---
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# --- Load HuBERT feature extractor and model ---
try:
    fe = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
    hubert = HubertModel.from_pretrained(HUBERT_NAME).to(DEVICE)
    hubert.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load HuBERT model {HUBERT_NAME}: {e}")

# create app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# mapping to friendly display names and cuisines
LABEL_MAP = {
    "andhra_pradesh": "Telugu - English",
    "telugu": "Telugu - English",
    "tamilnadu": "Tamil - English",
    "tamil": "Tamil - English",
    "hindi": "Hindi",
    "kannada": "Kannada - English",
    "malayalam": "Malayalam - English",
    "kerala": "Malayalam - English",
    "kerala_state": "Malayalam - English",
    "gujarati": "Gujarati - English",
    "american": "American English",
    "australian": "Australian English"
}

CUISINE_MAP = {
    "Telugu - English": ["Biryani", "Gongura", "Pulihora"],
    "Tamil - English": ["Idli", "Dosa", "Pongal"],
    "Hindi": ["Chole Bhature", "Paratha", "Rajma"],
    "Kannada - English": ["Bisi Bele Bath", "Maddur vada", "Ragi mudde"],
    "Malayalam - English": ["Puttu", "Appam", "Kerala fish curry"],
    "Gujarati - English": ["Dhokla", "Khandvi", "Undhiyu"],
    "American English": ["Burger", "BBQ", "Pancakes"],
    "Australian English": ["Meat Pie", "Pavlova", "Lamington"]
}

# Hardcoded model performance numbers to show exactly like your screenshot
MODEL_PERF = {
    "mfcc": "78.45%",
    "hubert": "97.84%",
    "age_generalization": "80.12%",
    "sentence_level": "85.66%",
    "word_level": "88.21%"
}

PREDICTION_LOG = "predictions_log.csv"

# ensure CSV header exists
if not os.path.exists(PREDICTION_LOG):
    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["timestamp_utc", "input_filename", "pred_code", "display_name", "confidence_pct"])

def normalize_display_name_from_code(code: str):
    """Return display name using LABEL_MAP with a few fallbacks."""
    if not code:
        return None
    code_norm = str(code).strip().lower()
    # direct mapping
    if code_norm in LABEL_MAP:
        return LABEL_MAP[code_norm]
    # try remove spaces/underscores
    alt = code_norm.replace(" ", "_")
    if alt in LABEL_MAP:
        return LABEL_MAP[alt]
    # try title-casing (e.g., 'malayalam' -> 'Malayalam - English' via LABEL_MAP keys)
    for k, v in LABEL_MAP.items():
        if k.lower() == code_norm:
            return v
    # as final fallback, try to find a display_name that *contains* the token
    for k, v in LABEL_MAP.items():
        if code_norm in k.lower():
            return v
    return None

def lookup_cuisines(display_name: str, pred_code: str):
    """Robust lookup for cuisines: try display_name, normalized forms, and label map fallback."""
    if not display_name:
        # try to derive from pred_code
        display_name = normalize_display_name_from_code(pred_code)
    # try exact
    if display_name in CUISINE_MAP:
        return CUISINE_MAP[display_name]
    # try title-cased form
    display_title = display_name.title() if display_name else None
    if display_title and display_title in CUISINE_MAP:
        return CUISINE_MAP[display_title]
    # try mapping via label_map using pred_code
    alt = normalize_display_name_from_code(pred_code)
    if alt and alt in CUISINE_MAP:
        return CUISINE_MAP[alt]
    # fallback
    return ["Local cuisine suggestions not available"]

def extract_hubert_embedding(path):
    """Return mean-pooled last hidden-state embedding for audio at path (16kHz)."""
    y, sr = sf.read(path)
    # convert multichannel to mono if needed
    if y is None:
        raise ValueError("Empty audio file or unreadable format.")
    if hasattr(y, "ndim") and y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(float), sr, 16000)
    inputs = fe(y, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)
    with torch.no_grad():
        out = hubert(input_values, output_hidden_states=True)
    emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
    return emb

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "time": datetime.datetime.utcnow().isoformat() + "Z"})

@app.route("/predict", methods=["POST"])
def predict():
    # Basic checks
    if "audio" not in request.files:
        return jsonify({"error": "no audio file provided"}), 400

    f = request.files["audio"]
    filename = f.filename or "recording.wav"

    # Save a copy to uploads for debugging / audit
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        f.save(tmp_path)
        saved_path = os.path.join("uploads", f"{int(datetime.datetime.utcnow().timestamp())}_{filename}")
        shutil.copy(tmp_path, saved_path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": f"failed to store uploaded file: {e}"}), 500

    try:
        # Extract embedding
        emb = extract_hubert_embedding(tmp_path)

        # Prediction with classifier
        try:
            probs = clf.predict_proba([emb])[0]
            classes = list(clf.classes_)
            # zip and sort descending by probability
            pairs = list(zip(classes, probs))
            pairs_sorted = sorted(pairs, key=lambda x: float(x[1]), reverse=True)
            top = [{"language": str(p[0]), "prob": float(p[1])} for p in pairs_sorted[:5]]
            pred_code = str(pairs_sorted[0][0])
            confidence = float(pairs_sorted[0][1])
        except Exception:
            # fallback
            pred_code = str(clf.predict([emb])[0])
            confidence = None
            top = [{"language": pred_code, "prob": None}]

        # determine display name robustly
        display_name = normalize_display_name_from_code(pred_code)
        if display_name is None:
            # fallback to label_map direct or raw pred_code
            display_name = LABEL_MAP.get(pred_code, pred_code)

        cuisines = lookup_cuisines(display_name, pred_code)

        response = {
            "input_filename": filename,
            "prediction": pred_code,
            "display_name": display_name,
            # confidence as percentage rounded to 1 decimal (or None)
            "confidence": round(confidence * 100, 1) if confidence is not None else None,
            "top": top,
            "cuisines": cuisines,
            "perf": MODEL_PERF
        }

        # Log an easy-to-read server line for debugging
        print(f"[{datetime.datetime.utcnow().isoformat()}] PREDICT input={filename} -> {response['display_name']} ({response['confidence']}%)")

        # Append to CSV log
        try:
            with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as wf:
                writer = csv.writer(wf)
                writer.writerow([
                    datetime.datetime.utcnow().isoformat(),
                    filename,
                    pred_code,
                    response["display_name"],
                    response["confidence"]
                ])
        except Exception as e:
            print("Warning: failed to write prediction log:", e)

        return jsonify(response)

    except Exception as e:
        print(f"[{datetime.datetime.utcnow().isoformat()}] ERROR during predict:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        # remove the temp file we used for processing (keep uploads copy)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    print("Starting demo on http://127.0.0.1:5000 — device:", DEVICE)
    app.run(debug=True, host="127.0.0.1", port=5000)
