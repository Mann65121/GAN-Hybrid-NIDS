import pandas as pd, pickle, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

rf = pickle.load(open("models/rf.pkl","rb"))
lr = pickle.load(open("models/lr.pkl","rb"))

X = pd.read_csv("data/month2/F_train.csv")
y = pd.read_csv("data/month2/y_train_balanced_mc.csv").values.ravel()

P_rf = rf.predict_proba(X)
P_lr = lr.predict_proba(X)

X_meta = np.hstack([P_rf, P_lr])

meta = HistGradientBoostingClassifier(
    max_iter=200,      # 🔥 thoda aur trees
    learning_rate=0.08,
    max_depth=6
)


meta.fit(X_meta, y)
pickle.dump(meta, open("models/meta.pkl","wb"))

print("✅ FAST meta‑learner trained")
