# src/models/train_hubert_model.py
import numpy as np, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = np.load("embeddings/train_hubert_X.npy")
y_train = np.load("embeddings/train_hubert_y.npy")
X_val = np.load("embeddings/val_hubert_X.npy")
y_val = np.load("embeddings/val_hubert_y.npy")

print(X_train.shape, X_val.shape)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
val_pred = clf.predict(X_val)
print("Val acc:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

os.makedirs("saved_models", exist_ok=True)
joblib.dump(clf, "saved_models/hubert_rf.pkl")
print("Saved:", "saved_models/hubert_rf.pkl")
