import pandas as pd, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

X = pd.read_csv("data/month2/F_train.csv")
y = pd.read_csv("data/month2/y_train_balanced_mc.csv").values.ravel()

# ---- class weights (important) ----
counts = Counter(y)
total = sum(counts.values())
class_weights = {cls: total / cnt for cls, cnt in counts.items()}

print("Class weights:", class_weights)

# Cost‑sensitive Random Forest
rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=14,
    class_weight=class_weights,
    n_jobs=-1,          # 🔥 USE ALL CORES
    random_state=42
)


# Logistic Regression with balanced weights
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=2
)

rf.fit(X, y)
lr.fit(X, y)

pickle.dump(rf, open("models/rf.pkl", "wb"))
pickle.dump(lr, open("models/lr.pkl", "wb"))

print("✅ Cost‑sensitive ML models trained")
