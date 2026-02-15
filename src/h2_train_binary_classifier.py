import pandas as pd, pickle
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("data/month2/F_train.csv")
y = pd.read_csv("data/month2/y_train_bin.csv").values.ravel()

clf = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    class_weight="balanced",
    n_jobs=2,
    random_state=42
)
clf.fit(X, y)

pickle.dump(clf, open("models/bin_rf.pkl","wb"))
print("✅ Level‑1 Binary IDS trained")
