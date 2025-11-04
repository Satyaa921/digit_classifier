import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SmallCNN(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            x = torch.zeros(1, 1, img_size, img_size)
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
            flat = x.numel()
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def make_loaders(data_dir, img_size, batch=64, seed=42):
    aug = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ])
    plain = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    full_aug = datasets.ImageFolder(data_dir, transform=aug)
    full_plain = datasets.ImageFolder(data_dir, transform=plain)
    idx = np.arange(len(full_aug))
    y = np.array(full_aug.targets)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    tmp_y = y[tmp]
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp_y, random_state=seed)
    tr_set = Subset(full_aug, tr)
    va_set = Subset(full_plain, va)
    te_set = Subset(full_plain, te)
    return (
        DataLoader(tr_set, batch_size=batch, shuffle=True),
        DataLoader(va_set, batch_size=batch),
        DataLoader(te_set, batch_size=batch),
    )

def train_model(model, loader, device, epochs=5):
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {e+1}/{epochs}, Loss = {total/len(loader):.4f}")

def evaluate(model, loader, device, name="Val"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            preds += p.cpu().numpy().tolist()
            trues += y.cpu().numpy().tolist()
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    cm = confusion_matrix(trues, preds)
    print(f"{name} Accuracy = {acc:.4f} | F1 = {f1:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}")
    return acc, f1, cm

def misclassified(model, loader, device, maxn=5):
    model.eval()
    out = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            for i in range(len(y)):
                if p[i] != y[i]:
                    out.append((int(y[i]), int(p[i])))
                    if len(out) >= maxn:
                        return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img_size", type=int, default=64)
    args = ap.parse_args()

    set_seed(42)
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_loaders(args.data, args.img_size)

    baseline = SmallCNN(args.img_size).to(device)
    print("Training baseline model...")
    train_model(baseline, train_loader, device, epochs=5)
    evaluate(baseline, val_loader, device, "Baseline Val")
    evaluate(baseline, test_loader, device, "Baseline Test")
    torch.save(baseline.state_dict(), "models/baseline.pt")

    print("\nTraining improved ResNet18...")
    net = models.resnet18(pretrained=False)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net = net.to(device)
    train_model(net, train_loader, device, epochs=args.epochs)
    evaluate(net, val_loader, device, "Improved Val")
    evaluate(net, test_loader, device, "Improved Test")
    torch.save(net.state_dict(), "models/best.pt")
    print("Misclassified examples (true, pred):", misclassified(net, test_loader, device))

if __name__ == "__main__":
    main()
