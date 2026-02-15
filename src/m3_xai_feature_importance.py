import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
rf = pickle.load(open("models/rf.pkl", "rb"))

# Load features
X = pd.read_csv("data/month2/F_train.csv")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10,6))
plt.title("Top 20 Important Features (Random Forest)")
plt.bar(range(20), importances[indices])
plt.xticks(range(20), indices, rotation=90)
plt.tight_layout()
plt.savefig("results_xai_rf_features.png")
plt.show()

print("✅ XAI feature importance saved")

