import os
import pandas as pd

UTK_DIR = "dataset_utk"
LIP_DIR = "dataset_lips"
OUT_DIR = "dataset_combined"

os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:

    utk_path = os.path.join(UTK_DIR, f"{split}.csv")
    lip_path = os.path.join(LIP_DIR, f"{split}.csv")

    utk = pd.read_csv(utk_path)
    lip = pd.read_csv(lip_path)

    combined = pd.concat(
        [utk, lip],
        ignore_index=True
    )

    # Shuffle while preserving the existing split
    combined = combined.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    output_path = os.path.join(
        OUT_DIR,
        f"{split}.csv"
    )

    combined.to_csv(
        output_path,
        index=False
    )

    print(
        f"{split}: "
        f"UTKFace={len(utk)}, "
        f"Collected={len(lip)}, "
        f"Combined={len(combined)}"
    )


# Combine full.csv as well
utk_full = pd.read_csv(
    os.path.join(UTK_DIR, "full.csv")
)

lip_full = pd.read_csv(
    os.path.join(LIP_DIR, "full.csv")
)

combined_full = pd.concat(
    [utk_full, lip_full],
    ignore_index=True
)

combined_full = combined_full.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

combined_full.to_csv(
    os.path.join(OUT_DIR, "full.csv"),
    index=False
)

print()
print("==============================================")
print(" Combined dataset created successfully")
print("==============================================")
print(f"Total samples: {len(combined_full)}")
print(f"Output folder: {OUT_DIR}")