# src/models/evaluate_hubert.py
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Make results directory
os.makedirs("results/hubert", exist_ok=True)

# Load model + test embeddings
clf = joblib.load("saved_models/hubert_rf.pkl")
X_test = np.load("embeddings/test_hubert_X.npy")
y_test = np.load("embeddings/test_hubert_y.npy")

# Predict
y_pred = clf.predict(X_test)

# Print accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
labels = sorted(list(set(list(y_test) + list(y_pred))))
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("HuBERT Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/hubert/confusion_matrix.png")

print("\nSaved: results/hubert/confusion_matrix.png")
