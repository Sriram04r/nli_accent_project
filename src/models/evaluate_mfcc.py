# src/models/evaluate_mfcc.py
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load model and data
model = joblib.load("saved_models/mfcc_model.pkl")
X_test = np.load("embeddings/test_mfcc_X.npy")
y_test = np.load("embeddings/test_mfcc_y.npy")

# predict
y_pred = model.predict(X_test)

# metrics
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))

# confusion matrix
labels = sorted(list(set(list(y_test) + list(y_pred))))
cm = confusion_matrix(y_test, y_pred, labels=labels)

os.makedirs("results/mfcc", exist_ok=True)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("MFCC model - Confusion matrix")
plt.tight_layout()
plt.savefig("results/mfcc/confusion_matrix.png")
print("\nSaved confusion matrix to results/mfcc/confusion_matrix.png")
