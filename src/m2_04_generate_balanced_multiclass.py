import torch, torch.nn as nn
import pandas as pd, numpy as np, os

X = pd.read_csv("data/month2/X_train_mc.csv").values
y = pd.read_csv("data/month2/y_train_mc.csv").values.ravel()

latent_dim = 16
feature_dim = X.shape[1]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
    def forward(self, z): return self.net(z)

counts = pd.Series(y).value_counts()
max_count = counts.max()

Xb, yb = list(X), list(y)

for cls, cnt in counts.items():
    if cnt < 0.1 * max_count:
        need = int(2.5 * (max_count - cnt))
    elif cnt < 0.3 * max_count:
        need = int(1.5 * (max_count - cnt))
    else:
        need = max_count - cnt

    if need <= 0: continue

    path = f"models/month2_gan/gan_class_{cls}.pth"
    if not os.path.exists(path): continue

    G = Generator()
    G.load_state_dict(torch.load(path))
    G.eval()

    z = torch.randn(need, latent_dim)
    syn = G(z).detach().numpy()

    Xb.extend(syn)
    yb.extend([cls]*need)

pd.DataFrame(Xb).to_csv("data/month2/X_train_balanced_mc.csv", index=False)
pd.Series(yb).to_csv("data/month2/y_train_balanced_mc.csv", index=False)

print("✅ Improved balanced multiclass data ready")
