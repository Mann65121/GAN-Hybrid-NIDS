import pandas as pd, pickle, numpy as np
from sklearn.metrics import classification_report

# Load data
X = pd.read_csv("data/month2/F_test.csv")
y_true = pd.read_csv("data/month2/y_test_mc.csv").values.ravel()

# Load models
bin_clf = pickle.load(open("models/bin_rf.pkl","rb"))
atk_clf = pickle.load(open("models/attack_rf.pkl","rb"))

# 🔥 LOAD YOUR ORIGINAL FLAT META MODEL
meta = pickle.load(open("models/meta.pkl","rb"))
rf = pickle.load(open("models/rf.pkl","rb"))
lr = pickle.load(open("models/lr.pkl","rb"))

# Probabilities
P_bin = bin_clf.predict_proba(X)

# Flat model probs
P_rf = rf.predict_proba(X)
P_lr = lr.predict_proba(X)
X_meta = np.hstack([P_rf, P_lr])
P_flat = meta.predict_proba(X_meta)

# Thresholds
LOW = 0.40
HIGH = 0.70

y_pred = []

for i, probs in enumerate(P_bin):
    p_attack = probs[1]

    if p_attack > HIGH:
        # confident attack → hierarchical
        y_pred.append(atk_clf.predict(X.iloc[[i]])[0])

    elif p_attack < LOW:
        # confident normal
        y_pred.append(6)

    else:
        # 🔥 uncertain → fallback to flat model
        y_pred.append(np.argmax(P_flat[i]))

print("Hybrid Hierarchical + Flat IDS")
print(classification_report(y_true, y_pred, digits=4))
