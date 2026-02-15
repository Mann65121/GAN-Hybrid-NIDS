import pickle, pandas as pd, numpy as np

rf = pickle.load(open("models/rf.pkl","rb"))
lr = pickle.load(open("models/lr.pkl","rb"))

X = pd.read_csv("data/month2/F_test.csv")
P = (rf.predict_proba(X) + lr.predict_proba(X)) / 2

np.save("models/final_probs.npy", P)
