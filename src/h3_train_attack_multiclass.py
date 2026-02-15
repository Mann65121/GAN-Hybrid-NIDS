import pandas as pd, pickle
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("data/month2/F_train.csv")
y = pd.read_csv("data/month2/y_train_balanced_mc.csv").values.ravel()

# keep only attacks (exclude Normal = 6)
mask = y != 6
X_a = X[mask]
y_a = y[mask]

clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    class_weight="balanced",
    n_jobs=2,
    random_state=42
)
clf.fit(X_a, y_a)

pickle.dump(clf, open("models/attack_rf.pkl","wb"))
print("✅ Level‑2 Attack classifier trained")
