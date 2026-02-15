import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report

rf = pickle.load(open("models/rf.pkl", "rb"))
lr = pickle.load(open("models/lr.pkl", "rb"))
meta = pickle.load(open("models/meta.pkl", "rb"))

X = pd.read_csv("data/month2/F_test.csv")
y = pd.read_csv("data/month2/y_test_mc.csv").values.ravel()

P_rf = rf.predict_proba(X)
P_lr = lr.predict_proba(X)
X_meta = np.hstack([P_rf, P_lr])
P = meta.predict_proba(X_meta)

STRONG_CLASSES = [3, 4, 5, 6, 7]
RARE_CLASSES = [0, 1, 2, 8, 9]

pred = []

for probs in P:
    cls = np.argmax(probs)
    conf = probs[cls]

    # 🔥 FINAL STRICT RULE (accuracy boost)
    if cls in RARE_CLASSES and conf < 0.55:
        cls = max(STRONG_CLASSES, key=lambda c: probs[c])

    pred.append(cls)

print(classification_report(y, pred, digits=4))
