import pandas as pd

df = pd.read_csv("data/raw/UNSW_NB15.csv")

print("Multiclass distribution:")
print(df['attack_cat'].value_counts())

print("\nPercentage distribution:")
total = len(df)
for k, v in df['attack_cat'].value_counts().items():
    print(f"{k}: {v/total:.2%}")
