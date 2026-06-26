import random
from pathlib import Path

import pandas as pd

# ==================================================
# CONFIG
# ==================================================

SEED = 42

REAL_DIR = Path("dataset_real/train")
SYNTH_DIR = Path("dataset_synthetic/train")

OUTPUT_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)

# ==================================================
# EXPERIMENTS
# ==================================================

EXPERIMENTS = {
    "e1": (100, 0),
    "e2": (75, 25),
    "e3": (50, 50),
    "e4": (25, 75),
    "e5": (0, 100),
}

# ==================================================
# CLASSES
# ==================================================

classes = sorted(
    [
        p.name
        for p in REAL_DIR.iterdir()
        if p.is_dir()
    ]
)

print("Classes found:")
for c in classes:
    print(f"  - {c}")

# ==================================================
# CREATE EXPERIMENTS
# ==================================================

for experiment_name, (real_pct, synth_pct) in EXPERIMENTS.items():

    rows = []

    print(f"\nCreating {experiment_name}...")

    for class_name in classes:

        real_files = [
            str(p)
            for p in (REAL_DIR / class_name).glob("*")
            if p.is_file()
        ]

        synth_files = [
            str(p)
            for p in (SYNTH_DIR / class_name).glob("*")
            if p.is_file()
        ]

        random.shuffle(real_files)
        random.shuffle(synth_files)

        effective_count = min(
            len(real_files),
            len(synth_files)
        )

        real_files = real_files[:effective_count]
        synth_files = synth_files[:effective_count]

        n_real = round(
            effective_count * real_pct / 100
        )

        n_synth = effective_count - n_real

        selected_real = real_files[:n_real]
        selected_synth = synth_files[:n_synth]

        for filepath in selected_real:

            rows.append({
                "filepath": filepath,
                "class_name": class_name,
                "source": "real"
            })

        for filepath in selected_synth:

            rows.append({
                "filepath": filepath,
                "class_name": class_name,
                "source": "synthetic"
            })

        print(
            f"{class_name:<25} "
            f"total={effective_count:<4} "
            f"real={n_real:<4} "
            f"synthetic={n_synth:<4}"
        )

    random.shuffle(rows)

    df = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR /
        f"{experiment_name}_train.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n{experiment_name}: "
        f"{len(df)} images saved"
    )

print("\nDone.")
