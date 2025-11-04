import argparse, os, glob, csv, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def build_model():
    m = models.resnet18(pretrained=False)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def infer(images, weights, out, img_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(images, ext))
    rows = []
    with torch.no_grad():
        for f in sorted(files):
            img = Image.open(f).convert("L")
            x = tfm(img).unsqueeze(0).to(device)
            p = model(x).argmax(1).item()
            rows.append([os.path.basename(f), p])
            print(f"{f} -> {p}")
    with open(out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["filename", "predicted"])
        writer.writerows(rows)
    print("Saved predictions to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--img_size", type=int, default=64)
    a = ap.parse_args()
    infer(a.images, a.weights, a.out, a.img_size)
