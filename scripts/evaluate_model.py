import os
import pandas as pd

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = "e1" # for change experiment you need just change digit (1, 2, 3, 4, 5)

TEST_DATASET = "real" # for change experiment you need just change name ("real" or "synthetic")

if TEST_DATASET == "real":
    TEST_DIR = "dataset_real/test"
else:
    TEST_DIR = "dataset_synthetic/test"

MODEL_PATH = f"models/{MODEL_NAME}_best_model.pth"

NUM_CLASSES = 8
BATCH_SIZE = 32

# ==================================================
# DEVICE
# ==================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# ==================================================
# DATA
# ==================================================

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test images:", len(test_dataset))

# ==================================================
# MODEL
# ==================================================

model = efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    1280,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

# ==================================================
# EVALUATION
# ==================================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = outputs.max(1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==================================================
# SAVE RESULTS
# ==================================================

results_df = pd.DataFrame([{
    "model": MODEL_NAME,
    "test_dataset": TEST_DATASET,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

results_df.to_csv(
    f"results/final_results_{MODEL_NAME}_{TEST_DATASET}.csv",
    index=False
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=test_dataset.classes,
    columns=test_dataset.classes
)

cm_df.to_csv(
    f"results/confusion_matrix_{MODEL_NAME}_{TEST_DATASET}.csv"
)

print("\nResults saved.")