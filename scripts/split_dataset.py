import random
import shutil
from pathlib import Path

# ==========================
# CONFIG
# ==========================

RANDOM_SEED = 42

SOURCE_DIR = Path("Dataset_real")
OUTPUT_DIR = Path("Dataset_real_split")

TEST_SIZE = 30
VAL_SIZE = 30

random.seed(RANDOM_SEED)

# ==========================
# CREATE OUTPUT STRUCTURE
# ==========================

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

# ==========================
# SPLIT EACH CLASS
# ==========================

for class_dir in SOURCE_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    images = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        images.extend(class_dir.glob(ext))

    images = list(images)

    random.shuffle(images)

    if len(images) < TEST_SIZE + VAL_SIZE:
        raise ValueError(
            f"Class '{class_name}' contains only {len(images)} images"
        )

    test_images = images[:TEST_SIZE]
    val_images = images[TEST_SIZE:TEST_SIZE + VAL_SIZE]
    train_images = images[TEST_SIZE + VAL_SIZE:]

    split_mapping = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split_name, split_images in split_mapping.items():

        target_dir = OUTPUT_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(
                image_path,
                target_dir / image_path.name
            )

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}, "
        f"test={len(test_images)}"
    )

print("\nDataset split completed successfully.")
print(f"Output directory: {OUTPUT_DIR}")