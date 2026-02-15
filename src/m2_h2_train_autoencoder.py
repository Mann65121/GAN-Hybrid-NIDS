import torch, pandas as pd, torch.nn as nn, os

X = torch.tensor(pd.read_csv("data/month2/X_train_balanced_mc.csv").values, dtype=torch.float32)

class AE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32))
        self.dec = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,d))
    def forward(self,x): return self.dec(self.enc(x))
    def encode(self,x): return self.enc(x)

ae = AE(X.shape[1])
opt = torch.optim.Adam(ae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for e in range(6):
    opt.zero_grad()
    loss = loss_fn(ae(X), X)
    loss.backward()
    opt.step()
    print(f"AE Epoch {e+1}/6 | Loss {loss.item():.4f}")

torch.save(ae.state_dict(), "models/ae.pth")
