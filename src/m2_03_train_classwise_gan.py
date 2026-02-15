import torch
import torch.nn as nn
import pandas as pd
import os

print("\n===== STEP‑3: CLASS‑WISE GAN TRAINING (FAST MODE) =====\n")

# Load data
X = pd.read_csv("data/month2/X_train_mc.csv").values
y = pd.read_csv("data/month2/y_train_mc.csv").values.ravel()

latent_dim = 16          # 🔥 reduced (fast)
feature_dim = X.shape[1]
epochs = 8               # 🔥 reduced (fast)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
    def forward(self, z):
        return self.net(z)

# Find minority classes
class_counts = pd.Series(y).value_counts()
max_count = class_counts.max()
minority_classes = class_counts[class_counts < 0.5 * max_count].index.tolist()

print("Detected minority classes:", minority_classes)
os.makedirs("models/month2_gan", exist_ok=True)

# Train GAN per class
for cls in minority_classes:
    print(f"\n--- Training GAN for class {cls} ---")

    X_cls = torch.tensor(X[y == cls], dtype=torch.float32)
    G = Generator()
    optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        z = torch.randn(len(X_cls), latent_dim)
        fake = G(z)

        loss = loss_fn(fake, X_cls.mean(dim=0).repeat(len(X_cls), 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Class {cls} | Epoch {epoch+1}/{epochs} | Loss = {loss.item():.6f}")

    torch.save(G.state_dict(), f"models/month2_gan/gan_class_{cls}.pth")
    print(f"✅ GAN saved for class {cls}")

print("\n===== STEP‑3 COMPLETE =====\n")
