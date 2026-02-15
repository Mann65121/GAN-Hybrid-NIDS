import pandas as pd

y_tr = pd.read_csv("data/month2/y_train_balanced_mc.csv").values.ravel()
y_te = pd.read_csv("data/month2/y_test_mc.csv").values.ravel()

# 0 = Normal (class 6), 1 = Attack
y_tr_bin = (y_tr != 6).astype(int)
y_te_bin = (y_te != 6).astype(int)

pd.Series(y_tr_bin).to_csv("data/month2/y_train_bin.csv", index=False)
pd.Series(y_te_bin).to_csv("data/month2/y_test_bin.csv", index=False)

print("✅ Binary labels created (Normal vs Attack)")
