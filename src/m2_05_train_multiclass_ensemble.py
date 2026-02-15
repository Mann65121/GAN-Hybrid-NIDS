import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

print("\n===== STEP‑5: MULTICLASS ENSEMBLE TRAINING (FAST + STABLE) =====\n")

# Load balanced multiclass data
X = pd.read_csv("data/month2/X_train_balanced_mc.csv")
y = pd.read_csv("data/month2/y_train_balanced_mc.csv").values.ravel()

print("Training samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(set(y)))

# -------------------------------
# FAST & STABLE MODELS
# -------------------------------

print("\n[1/3] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=80,        # 🔥 reduced
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=2,               # VirtualBox safe
    random_state=42
)
rf.fit(X, y)
print("✅ Random Forest training complete")

print("\n[2/3] Training Gradient Boosting (FAST MODE)...")
gb = GradientBoostingClassifier(
    n_estimators=40,        # 🔥 reduced a lot
    learning_rate=0.1,     # 🔥 faster convergence
    max_depth=3,
    subsample=0.8,         # 🔥 speed + regularization
    random_state=42
)
gb.fit(X, y)
print("✅ Gradient Boosting training complete")

print("\n[3/3] Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=500,
    n_jobs=2
)
lr.fit(X, y)
print("✅ Logistic Regression training complete")

# -------------------------------
# ENSEMBLE
# -------------------------------
print("\nCombining models into ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("gb", gb),
        ("lr", lr)
    ],
    voting="soft"
)

ensemble.fit(X, y)
print("✅ Ensemble training complete")

# Save model
os.makedirs("models", exist_ok=True)
with open("models/month2_multiclass_ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

print("\n===== STEP‑5 DONE: MODEL SAVED SUCCESSFULLY =====\n")
