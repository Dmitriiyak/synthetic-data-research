import os
import random

import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch.utils.data import DataLoader


# ==================================================
# REPRODUCIBILITY
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# CONFIG
# ==================================================

TRAIN_DIR = "dataset_real/train"
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
# DATASETS
# ==================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
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

model = efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    in_features=1280,
    out_features=NUM_CLASSES
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
# TRAIN FUNCTION
# ==================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ==================================================
# VALIDATION FUNCTION
# ==================================================

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ==================================================
# TRAINING LOOP
# ==================================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
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
            "models/best_model.pth"
        )

        print("Best model saved.")

with open("results/training_history.txt", "w") as f:

    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")

    for i in range(NUM_EPOCHS):

        f.write(
            f"Epoch {i+1}: "
            f"Train Loss={history['train_loss'][i]:.4f}, "
            f"Train Acc={history['train_acc'][i]:.2f}, "
            f"Val Loss={history['val_loss'][i]:.4f}, "
            f"Val Acc={history['val_acc'][i]:.2f}\n"
        )

print("\nTraining completed.")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
