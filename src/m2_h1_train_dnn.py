import torch, pandas as pd, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os

X = torch.tensor(pd.read_csv("data/month2/X_train_balanced_mc.csv").values, dtype=torch.float32)
y = torch.tensor(pd.read_csv("data/month2/y_train_balanced_mc.csv").values.flatten(), dtype=torch.long)

num_classes = len(torch.unique(y))
counts = Counter(y.tolist())
total = sum(counts.values())
weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, x, y):
        ce = nn.functional.cross_entropy(x, y, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

loader = DataLoader(TensorDataset(X,y), batch_size=512, shuffle=True)

class DNN(nn.Module):
    def __init__(self,d,c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,c)
        )
    def forward(self,x): return self.net(x)
    def features(self,x): return self.net[:-1](x)

model = DNN(X.shape[1], num_classes)
criterion = FocalLoss(alpha=weights, gamma=2)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for e in range(12):
    loss_sum = 0
    for xb,yb in loader:
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f"DNN Epoch {e+1}/12 | Loss {loss_sum:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/dnn.pth")
print("✅ DNN trained with rare‑class focus")
