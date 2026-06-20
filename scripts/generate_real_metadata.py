from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("dataset_real")

rows = []
image_id = 1

for split in ["train", "val", "test"]:

    split_dir = DATASET_ROOT / split

    if not split_dir.exists():
        continue

    for class_dir in split_dir.iterdir():

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_file in sorted(class_dir.iterdir()):

            if not image_file.is_file():
                continue

            rows.append({
                "image_id": image_id,
                "file_name": image_file.name,
                "class_name": class_name,
                "split": split,
                "source": "real"
            })

            image_id += 1

df = pd.DataFrame(rows)

output_path = Path("metadata") / "real_images.csv"
output_path.parent.mkdir(exist_ok=True)

df.to_csv(output_path, index=False)

print(f"Saved {len(df)} records to {output_path}")