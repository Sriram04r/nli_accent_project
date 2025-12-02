<h1>Native Language Identification of Indian English Speakers Using HuBERT</h1>

<h2>NLP Mini Project | MFCC + HuBERT | Accent Classification System</h2>

<h2>A complete end-to-end project built by Sriram Kola</h2>
<h3>
  1. Project Overview
</h3>

This project aims to identify the native language (accent) of Indian English speakers using short speech audio clips.
The system is built using:
- HuBERT-based deep speech embeddings

- Accent classification using a trained ML model (RandomForest/SVC)

- Cuisine recommendation based on predicted region

The project includes:

- Complete audio preprocessing

- HuBERT feature extraction

- Accent prediction for multiple Indian states

- Sentence-level inference

- A Flask-based web application

- A cuisine recommendation module

- Logging, cleaned dataset, and saved models

All development was done from scratch, including dataset cleaning, feature extraction, model training, and app deployment.

<h2>
  2. Folder Structure
</h2>
   <pre>
nli_accent_project/
│
├── andhra_pradesh/              # Raw Andhra audio samples
├── gujrat/                      # Raw Gujarat audio samples
│
├── data_raw/                    # Original dataset
├── data_clean/                  # Cleaned & trimmed audio
│
├── embeddings/                  # HuBERT extracted vectors
├── saved_models/                # Trained classifiers
├── results/                     # Evaluation reports, logs
│
├── splits/                      # Train/Test splits
├── uploads/                     # Web app uploaded audio
│
├── src/
│   ├── demo_app/
│   │   ├── templates/index.html
│   │   ├── static/script.js
│   │   ├── static/style.css
│   │   └── app.py               # Web application
│   │
│   ├── features/
│   │   ├── extract_hubert.py
│   │   └── extract_mfcc.py
│   │
│   ├── models/
│   │   ├── train_hubert_model.py
│   │   ├── evaluate_hubert.py
│   │   ├── train_mfcc_model.py
│   │   └── evaluate_mfcc.py
│   │
│   └── preprocess/
│       ├── audio_cleaner.py
│       ├── audio_cleaner_full.py
│       ├── build_index.py
│       ├── download_dataset.py
│       └── split_data.py
│
├── index.csv
├── index_clean.csv
├── predictions_log.csv
│
├── README.md
├── requirements.txt
└── clean_failures.txt
</pre>
<h3>
3. Tasks Completed (Actual Work in Your Project)
</h3>

<h4>HuBERT-Based Accent Classification</h4>

- Extracted HuBERT Base embeddings for all cleaned audio files

- Built an ML classifier (RandomForest/SVM)

- Achieved strong accuracy across states

- Generated prediction logs and confusion matrix

<h4>Dataset Preprocessing</h4>

- Cleaned raw audio

- Removed silence and noise

- Normalized and resampled to 16kHz

- Split into train/test sets

- Created clean index files

<h4>MFCC + Baseline Classifier</h4>

- Extracted MFCC features

- Trained baseline classifier

- Compared performance vs HuBERT

<h4>Web Application Development</h4>

- Flask-based audio upload/recording UI

- Predicts accent from HuBERT model

- Integrated cuisine recommendation logic

- Displays region, accent, and recommended dishes

<h4>Additional Completed Components</h4>

- Word-level log storage

- Prediction history saved to CSV

- Failures logged into clean_failures.txt

- Fully working interactive demo app

<h3>4. Accent Classes Included</h3>

- The model detects accents influenced by:

- Andhra Pradesh (Telugu-English)

- Gujarat (Gujarati-English)

- Tamil Nadu (Tamil-English)

- Kerala (Malayalam-English)

- Karnataka (Kannada-English)

- Other regional accents (if added in future)

<h3>5. Project Workflow</h3>

1.Load and clean audio

2.Extract HuBERT embeddings

3.Train a classification model

4.Evaluate accuracy and generate reports

5.Deploy model in Flask

6.Accept audio input from user

7.Predict native language accent

8.Recommend cuisine for that region

<h3>6.Outputs Generated</h3>

| **File / Folder**       | **Description**                              |
| ----------------------- | -------------------------------------------- |
| **index_clean.csv**     | Clean metadata after preprocessing           |
| **embeddings/**         | HuBERT feature vectors                       |
| **saved_models/**       | Final trained classifier                     |
| **results/**            | Evaluation results, logs, prediction reports |
| **predictions_log.csv** | History of user uploads in the web app       |
| **clean_failures.txt**  | Files that failed during preprocessing       |

<h3>7. Accent → Cuisine Mapping Used in Your App</h3>
   
| **Accent**  | **State**      | **Recommended Dishes**      |
| ----------- | -------------- | --------------------------- |
| **Andhra**  | Andhra Pradesh | Pesarattu, Gongura, Biryani |
| **Gujarat** | Gujarat        | Dhokla, Thepla, Handvo      |
| **Kerala**  | Kerala         | Appam, Puttu, Kerala Sadya  |
| **Tamil**   | Tamil Nadu     | Idli, Dosa, Pongal          |
| **Kannada** | Karnataka      | Bisi Bele Bath, Neer Dosa   |
| **Hindi**   | North India    | Paratha, Rajma Chawal       |


Your app automatically selects cuisine based on predicted accent.

<h3>8. Web Application Description</h3>

The Flask app supports:

- Audio upload (.wav/.mp3)

- Displaying accent prediction

- Showing confidence score

- Recommending dishes

- Logging predictions

- Real-time interface with HTML + CSS + JS

The app is located in:
src/demo_app/app.py

<h3>9. Project Highlights</h3>

- Fully functional accent detection system

- Based entirely on HuBERT, a state-of-the-art speech model

- Customized dataset cleaning pipelines

- Region-based cuisine recommendation

- Clean and modular project structure

- End-to-end ML + Web integration



<h3>
  10.GitHub Repository:
https://github.com/Sriram04r/nli_accent_project
</h3>









