# Handwritten Digit Classifier – Data Science Intern Mini Project

This project implements a handwritten digit classifier (0–9) with:
- A baseline Small CNN
- An improved ResNet18 model

The data is expected in the following structure:

```
dataset/
  0/
  1/
  ...
  9/
```

## Setup

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy
```

## Training

Train both baseline and improved models (70/15/15 stratified split, seed=42):

```bash
python -m src.train --data ./dataset --epochs 15 --img_size 64
```

This will save:
- `models/baseline.pt`
- `models/best.pt`

## Inference

Run inference on a folder of images and save predictions to `preds.csv`:

```bash
python -m src.infer --images ./sample --weights ./models/best.pt --out preds.csv
```

The output CSV has columns:

```text
filename,predicted
```

## Report

See `report.pdf` for:
- Model choices
- Data split
- Validation and test metrics (Accuracy, Macro-F1, confusion matrix)
- 3–5 misclassified samples with one-line hypotheses
- Key takeaways
