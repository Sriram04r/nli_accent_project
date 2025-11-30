# src/models/train_mfcc_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load MFCC features
X_train = np.load("embeddings/train_mfcc_X.npy")
y_train = np.load("embeddings/train_mfcc_y.npy")

X_val = np.load("embeddings/val_mfcc_X.npy")
y_val = np.load("embeddings/val_mfcc_y.npy")

print("Shapes:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:", X_val.shape, " y_val:", y_val.shape)

# Train a simple RandomForest classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate on validation set
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print("\nValidation Accuracy:", val_acc)
print("\nClassification report:")
print(classification_report(y_val, val_pred))

# Save the trained model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/mfcc_model.pkl")

print("\nModel saved to saved_models/mfcc_model.pkl")
