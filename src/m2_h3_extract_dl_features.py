import torch, pandas as pd, torch.nn as nn

Xtr = torch.tensor(pd.read_csv("data/month2/X_train_balanced_mc.csv").values, dtype=torch.float32)
Xte = torch.tensor(pd.read_csv("data/month2/X_test_mc.csv").values, dtype=torch.float32)

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

num_classes = len(pd.read_csv("data/month2/y_train_balanced_mc.csv").value_counts())
dnn = DNN(Xtr.shape[1], num_classes)
dnn.load_state_dict(torch.load("models/dnn.pth"))
dnn.eval()

class AE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32))
    def encode(self,x): return self.enc(x)

ae = AE(Xtr.shape[1])
ae.load_state_dict(torch.load("models/ae.pth"), strict=False)
ae.eval()

Ftr = torch.cat([dnn.features(Xtr), ae.encode(Xtr)],1).detach().numpy()
Fte = torch.cat([dnn.features(Xte), ae.encode(Xte)],1).detach().numpy()

pd.DataFrame(Ftr).to_csv("data/month2/F_train.csv", index=False)
pd.DataFrame(Fte).to_csv("data/month2/F_test.csv", index=False)

print("✅ DL features extracted")
