import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch.utils.data import Dataset, DataLoader


# ==================================================
# REPRODUCIBILITY
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# CONFIG
# ==================================================

EXPERIMENT_NAME = "e1" # for change experiment you need just change digit (1, 2, 3, 4, 5)

TRAIN_CSV = f"metadata/{EXPERIMENT_NAME}_train.csv"

VAL_DIR = "dataset_real/val"

NUM_CLASSES = 8

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4


# ==================================================
# OUTPUT DIRECTORIES
# ==================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==================================================
# DEVICE
# ==================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ==================================================
# TRANSFORMS
# ==================================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==================================================
# CSV DATASET
# ==================================================

class CSVImageDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.classes = sorted(
            self.df["class_name"].unique()
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name
            in enumerate(self.classes)
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["filepath"]

        class_name = row["class_name"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================================================
# DATASETS
# ==================================================

train_dataset = CSVImageDataset(
    TRAIN_CSV,
    transform=train_transforms
)

from torchvision.datasets import ImageFolder

val_dataset = ImageFolder(
    VAL_DIR,
    transform=val_transforms
)

print(f"Classes: {train_dataset.classes}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images: {len(val_dataset)}")


# ==================================================
# DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==================================================
# MODEL
# ==================================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    1280,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ==================================================
# LOSS & OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAIN
# ==================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    return (
        running_loss / len(train_loader),
        100 * correct / total
    )


# ==================================================
# VALIDATE
# ==================================================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return (
        running_loss / len(val_loader),
        100 * correct / total
    )


# ==================================================
# TRAINING LOOP
# ==================================================

history = []

best_val_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            f"models/{EXPERIMENT_NAME}_best_model.pth"
        )

        print("Best model saved.")


# ==================================================
# SAVE RESULTS
# ==================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    f"results/{EXPERIMENT_NAME}_history.csv",
    index=False
)

with open(
    f"results/{EXPERIMENT_NAME}_summary.txt",
    "w"
) as f:

    f.write(
        f"Experiment: {EXPERIMENT_NAME}\n"
    )

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%\n"
    )

print("\nTraining completed.")
print(f"Best Epoch: {best_epoch}")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)

