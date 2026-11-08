"""
Upgraded Dataset Preparation for Cheiloscopy
=============================================

Supports:
    1. UTKFace-derived groove dataset
    2. Collected lip_samples dataset

The script converts both datasets into a common CSV format
that can be used by:

    - CNN
    - MobileNetV2
    - SVM

Age groups:
    Young : 0  - 25
    Adult : 26 - 50
    Old   : 51+

Gender:
    0 = Male
    1 = Female

Output:
    dataset/
        train.csv
        val.csv
        test.csv
        full.csv
        stats.txt

Common CSV columns:
    filename
    groove_path
    age
    age_group
    age_idx
    gender_label
    gender_idx
    dataset
    subject_id

Examples
--------

UTKFace:

python prepare_dataset.py ^
    --grooves "ext/enhanced/lip_groove" ^
    --output "dataset_utk" ^
    --source utk

Collected lip samples:

python prepare_dataset.py ^
    --grooves "quality ext/lip_groove" ^
    --output "dataset_lip" ^
    --source lip_samples
"""

import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================================
# 1. AGE GROUP DEFINITION
# ============================================================================

def encode_age_group(age):
    """
    Convert numerical age into the project's three age groups.

    Young : 0  - 25 -> 0
    Adult : 26 - 50 -> 1
    Old   : 51+     -> 2
    """

    if age <= 25:
        return "Young", 0

    elif age <= 50:
        return "Adult", 1

    else:
        return "Old", 2


# ============================================================================
# 2. GENDER ENCODING
# ============================================================================

def encode_gender(gender_int):
    """
    Convert gender integer into readable label.

    0 -> Male
    1 -> Female
    """

    return {
        0: "Male",
        1: "Female"
    }.get(gender_int, None)


# ============================================================================
# 3. UTKFACE FILENAME PARSER
# ============================================================================

def parse_utk(filename):
    """
    Parse UTKFace filename.

    Expected format:

        age_gender_race_date.jpg

    Example:

        25_0_0_201701.jpg

    Returns:
        age
        gender_int
        subject_id

    Gender:
        0 = Male
        1 = Female
    """

    try:

        name = os.path.splitext(filename)[0]

        parts = name.split("_")

        if len(parts) < 2:
            return None, None, None

        age = int(parts[0])

        gender_int = int(parts[1])

        # Validate age
        if age < 0 or age > 116:
            return None, None, None

        # Validate gender
        if gender_int not in (0, 1):
            return None, None, None

        # UTKFace does not reliably provide a unique person ID
        # in the filename. Therefore filename is used as identity.
        subject_id = name

        return age, gender_int, subject_id

    except Exception:
        return None, None, None


# ============================================================================
# 4. COLLECTED LIP-SAMPLE FILENAME PARSER
# ============================================================================

def parse_lip_sample(filename):
    """
    Parse collected lip-sample filename.

    Expected format:

        AGE.GENDER.ID.jpg

    Examples:

        13y.f.241.jpg
        18y.m.155.jpg
        48y.f.268.jpg
        55y.m.216.jpg

    Interpreted as:

        AGE     -> numerical age
        GENDER  -> m / f
        ID      -> subject/sample identifier

    Returns:
        age
        gender_int
        subject_id

    Gender:
        m -> 0 -> Male
        f -> 1 -> Female
    """

    try:

        name = os.path.splitext(filename)[0]

        parts = name.split(".")

        if len(parts) < 3:
            return None, None, None

        # ------------------------------------------------------------
        # AGE
        # ------------------------------------------------------------

        age_text = parts[0].strip().lower()

        # Remove optional 'y'
        if age_text.endswith("y"):
            age_text = age_text[:-1]

        age = int(age_text)

        # ------------------------------------------------------------
        # GENDER
        # ------------------------------------------------------------

        gender_text = parts[1].strip().lower()

        if gender_text == "m":
            gender_int = 0

        elif gender_text == "f":
            gender_int = 1

        else:
            return None, None, None

        # ------------------------------------------------------------
        # AGE VALIDATION
        # ------------------------------------------------------------

        if age < 0 or age > 116:
            return None, None, None

        # ------------------------------------------------------------
        # SUBJECT / SAMPLE ID
        # ------------------------------------------------------------

        subject_id = parts[2].strip()

        if not subject_id:
            return None, None, None

        return age, gender_int, subject_id

    except Exception:
        return None, None, None


# ============================================================================
# 5. SELECT THE CORRECT PARSER
# ============================================================================

def parse_filename(filename, source):
    """
    Select the appropriate filename parser.

    source:
        utk
        lip_samples
    """

    if source == "utk":

        return parse_utk(filename)

    elif source == "lip_samples":

        return parse_lip_sample(filename)

    else:

        return None, None, None


# ============================================================================
# 6. BUILD MASTER DATAFRAME
# ============================================================================

def build_dataframe(groove_dir, source):
    """
    Build a common dataframe for either dataset.

    The resulting dataframe contains:

        filename
        groove_path
        age
        age_group
        age_idx
        gender_label
        gender_idx
        dataset
        subject_id
    """

    records = []

    skipped = 0

    # ------------------------------------------------------------------------
    # Check directory
    # ------------------------------------------------------------------------

    if not os.path.isdir(groove_dir):

        print(f"\nERROR: Groove directory does not exist:")
        print(f"       {groove_dir}")

        return pd.DataFrame()

    # ------------------------------------------------------------------------
    # Find image files
    # ------------------------------------------------------------------------

    files = [
        f
        for f in os.listdir(groove_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    print(f"  Found {len(files)} groove images")
    print(f"  Source : {source}")
    print(f"  Folder : {groove_dir}")

    # ------------------------------------------------------------------------
    # Parse each file
    # ------------------------------------------------------------------------

    for fname in sorted(files):

        age, gender_int, subject_id = parse_filename(
            fname,
            source
        )

        # Invalid filename
        if age is None or gender_int is None:

            skipped += 1

            continue

        # --------------------------------------------------------------------
        # Encode age
        # --------------------------------------------------------------------

        age_label, age_idx = encode_age_group(age)

        # --------------------------------------------------------------------
        # Encode gender
        # --------------------------------------------------------------------

        gender_label = encode_gender(gender_int)

        if gender_label is None:

            skipped += 1

            continue

        # --------------------------------------------------------------------
        # Full groove path
        # --------------------------------------------------------------------

        groove_path = os.path.abspath(
            os.path.join(
                groove_dir,
                fname
            )
        )

        # --------------------------------------------------------------------
        # Create record
        # --------------------------------------------------------------------

        records.append({

            "filename": fname,

            "groove_path": groove_path,

            "age": age,

            "age_group": age_label,

            "age_idx": age_idx,

            "gender_label": gender_label,

            "gender_idx": gender_int,

            "dataset": source,

            "subject_id": subject_id,

        })

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print(f"  Parsed  : {len(records)}")
    print(f"  Skipped : {skipped}")

    return pd.DataFrame(records)


# ============================================================================
# 7. STRATIFIED SPLIT
# ============================================================================

def split_dataset(
    df,
    train=0.80,
    val=0.10,
    test=0.10,
    seed=42
):
    """
    Split dataframe into train / validation / test.

    Stratification is based on:

        gender + age_group

    This preserves the class distribution across splits.
    """

    # ------------------------------------------------------------------------
    # Validate split percentages
    # ------------------------------------------------------------------------

    assert abs(
        train + val + test - 1.0
    ) < 1e-6, (
        "Train + validation + test must equal 1.0"
    )

    # ------------------------------------------------------------------------
    # Create combined stratification label
    # ------------------------------------------------------------------------

    df = df.copy()

    df["_strat"] = (
        df["gender_label"]
        + "_"
        + df["age_group"]
    )

    # ------------------------------------------------------------------------
    # Check class counts
    # ------------------------------------------------------------------------

    strat_counts = df["_strat"].value_counts()

    print("\n  Stratification classes:")

    for label, count in strat_counts.items():

        print(
            f"    {label:<20} {count:>5}"
        )

    # ------------------------------------------------------------------------
    # First split
    #
    # Train vs Validation + Test
    # ------------------------------------------------------------------------

    df_train, df_temp = train_test_split(

        df,

        test_size=(val + test),

        stratify=df["_strat"],

        random_state=seed,
    )

    # ------------------------------------------------------------------------
    # Second split
    #
    # Validation vs Test
    # ------------------------------------------------------------------------

    relative_test = test / (val + test)

    df_val, df_test = train_test_split(

        df_temp,

        test_size=relative_test,

        stratify=df_temp["_strat"],

        random_state=seed,
    )

    # ------------------------------------------------------------------------
    # Remove helper column
    # ------------------------------------------------------------------------

    for d in [df_train, df_val, df_test]:

        d.drop(
            columns=["_strat"],
            inplace=True,
            errors="ignore"
        )

    df.drop(
        columns=["_strat"],
        inplace=True,
        errors="ignore"
    )

    # ------------------------------------------------------------------------
    # Reset indexes
    # ------------------------------------------------------------------------

    df_train = df_train.reset_index(drop=True)

    df_val = df_val.reset_index(drop=True)

    df_test = df_test.reset_index(drop=True)

    df = df.reset_index(drop=True)

    return (
        df_train,
        df_val,
        df_test
    )


# ============================================================================
# 8. DATASET STATISTICS
# ============================================================================

def print_stats(
    df_full,
    df_train,
    df_val,
    df_test
):
    """
    Generate a detailed dataset statistics report.
    """

    lines = []

    lines.append("=" * 65)
    lines.append("             CHEILOSCOPY DATASET STATISTICS")
    lines.append("=" * 65)

    # ------------------------------------------------------------------------
    # Overall counts
    # ------------------------------------------------------------------------

    lines.append("")

    lines.append(
        f"Total       : {len(df_full)}"
    )

    lines.append(
        f"Train       : {len(df_train)} "
        f"({100 * len(df_train) / len(df_full):.1f}%)"
    )

    lines.append(
        f"Validation  : {len(df_val)} "
        f"({100 * len(df_val) / len(df_full):.1f}%)"
    )

    lines.append(
        f"Test        : {len(df_test)} "
        f"({100 * len(df_test) / len(df_full):.1f}%)"
    )

    # ------------------------------------------------------------------------
    # Dataset source
    # ------------------------------------------------------------------------

    lines.append("")
    lines.append("Dataset source:")

    for label, count in (
        df_full["dataset"]
        .value_counts()
        .items()
    ):

        lines.append(
            f"    {label:<15} {count:>5} "
            f"({100 * count / len(df_full):.1f}%)"
        )

    # ------------------------------------------------------------------------
    # Gender distribution
    # ------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "Gender distribution:"
    )

    for label, count in (
        df_full["gender_label"]
        .value_counts()
        .items()
    ):

        lines.append(
            f"    {label:<15} {count:>5} "
            f"({100 * count / len(df_full):.1f}%)"
        )

    # ------------------------------------------------------------------------
    # Age group distribution
    # ------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "Age group distribution:"
    )

    for label in [
        "Young",
        "Adult",
        "Old"
    ]:

        count = len(
            df_full[
                df_full["age_group"] == label
            ]
        )

        lines.append(
            f"    {label:<15} {count:>5} "
            f"({100 * count / len(df_full):.1f}%)"
        )

    # ------------------------------------------------------------------------
    # Gender × Age
    # ------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "Gender × Age group:"
    )

    cross = (
        df_full
        .groupby(
            ["gender_label", "age_group"]
        )
        .size()
    )

    for (gender, age_group), count in cross.items():

        lines.append(
            f"    {gender:<10} / "
            f"{age_group:<10} {count:>5}"
        )

    # ------------------------------------------------------------------------
    # Subject count
    # ------------------------------------------------------------------------

    if "subject_id" in df_full.columns:

        lines.append("")
        lines.append(
            "Unique subject/sample IDs:"
        )

        unique_subjects = (
            df_full["subject_id"]
            .nunique()
        )

        lines.append(
            f"    {unique_subjects}"
        )

    # ------------------------------------------------------------------------
    # Age statistics
    # ------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "Age statistics:"
    )

    lines.append(
        f"    Minimum : {df_full['age'].min()}"
    )

    lines.append(
        f"    Maximum : {df_full['age'].max()}"
    )

    lines.append(
        f"    Mean    : {df_full['age'].mean():.2f}"
    )

    lines.append(
        f"    Median  : {df_full['age'].median():.2f}"
    )

    lines.append("")

    lines.append("=" * 65)

    return "\n".join(lines)


# ============================================================================
# 9. SAVE DATASET
# ============================================================================

def save_dataset(
    df,
    df_train,
    df_val,
    df_test,
    output_dir
):
    """
    Save all CSV files.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # Full dataset
    # ------------------------------------------------------------------------

    df.to_csv(
        os.path.join(
            output_dir,
            "full.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------

    df_train.to_csv(
        os.path.join(
            output_dir,
            "train.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    df_val.to_csv(
        os.path.join(
            output_dir,
            "val.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------------

    df_test.to_csv(
        os.path.join(
            output_dir,
            "test.csv"
        ),
        index=False
    )


# ============================================================================
# 10. MAIN
# ============================================================================

def main(args):

    print("\n")
    print("=" * 65)
    print("             CHEILOSCOPY DATASET PREPARATION")
    print("=" * 65)

    print(f"  Source      : {args.source}")
    print(f"  Grooves     : {args.grooves}")
    print(f"  Output      : {args.output}")
    print(
        f"  Split       : "
        f"{args.train}/{args.val}/{args.test}"
    )
    print(f"  Seed        : {args.seed}")

    print("=" * 65)
    print()

    # ------------------------------------------------------------------------
    # Build dataframe
    # ------------------------------------------------------------------------

    df = build_dataframe(
        args.grooves,
        args.source
    )

    # ------------------------------------------------------------------------
    # Check dataset
    # ------------------------------------------------------------------------

    if len(df) == 0:

        print(
            "\nERROR:"
            "\nNo valid records were found."
        )

        print(
            "\nCheck:"
            "\n1. Groove directory path"
            "\n2. Filename format"
            "\n3. Dataset source"
        )

        return

    # ------------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------------

    try:

        (
            df_train,
            df_val,
            df_test
        ) = split_dataset(

            df,

            train=args.train,

            val=args.val,

            test=args.test,

            seed=args.seed,
        )

    except ValueError as e:

        print(
            "\nERROR during stratified split:"
        )

        print(e)

        print(
            "\nThis usually means one or more "
            "gender × age classes contain too "
            "few samples."
        )

        print(
            "\nTry:"
            "\n1. Checking the dataset statistics"
            "\n2. Changing age groups"
            "\n3. Increasing the number of samples"
        )

        return

    # ------------------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------------------

    save_dataset(

        df,

        df_train,

        df_val,

        df_test,

        args.output
    )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    stats = print_stats(

        df,

        df_train,

        df_val,

        df_test
    )

    print(stats)

    # ------------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------------

    stats_path = os.path.join(
        args.output,
        "stats.txt"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(stats)

    # ------------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("                    SAVED FILES")
    print("=" * 65)

    print(
        f"  {args.output}/full.csv"
        f"   — {len(df)} records"
    )

    print(
        f"  {args.output}/train.csv"
        f"  — {len(df_train)} records"
    )

    print(
        f"  {args.output}/val.csv"
        f"    — {len(df_val)} records"
    )

    print(
        f"  {args.output}/test.csv"
        f"   — {len(df_test)} records"
    )

    print(
        f"  {args.output}/stats.txt"
        f"  — statistics"
    )

    print("\n")
    print("CSV columns:")
    print("  filename       → image filename")
    print("  groove_path    → model input image")
    print("  age            → original numerical age")
    print("  age_group      → Young / Adult / Old")
    print("  age_idx        → 0 / 1 / 2")
    print("  gender_label   → Male / Female")
    print("  gender_idx     → 0 / 1")
    print("  dataset        → utk / lip_samples")
    print("  subject_id     → subject/sample identifier")

    print("=" * 65)
    print()


# ============================================================================
# 11. COMMAND-LINE ARGUMENTS
# ============================================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser(

        description=(
            "Prepare UTKFace or collected "
            "lip_samples groove dataset "
            "for cheiloscopy model training."
        )
    )

    # ------------------------------------------------------------------------
    # Dataset source
    # ------------------------------------------------------------------------

    ap.add_argument(

        "--source",

        choices=[
            "utk",
            "lip_samples"
        ],

        default="utk",

        help=(
            "Dataset source: "
            "'utk' for UTKFace or "
            "'lip_samples' for collected data."
        )
    )

    # ------------------------------------------------------------------------
    # Groove directory
    # ------------------------------------------------------------------------

    ap.add_argument(

        "--grooves",

        required=True,

        help=(
            "Path to folder containing "
            "lip groove images."
        )
    )

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    ap.add_argument(

        "--output",

        required=True,

        help=(
            "Folder where train/val/test "
            "CSV files will be saved."
        )
    )

    # ------------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------------

    ap.add_argument(

        "--train",

        type=float,

        default=0.80,

        help="Training split."
    )

    ap.add_argument(

        "--val",

        type=float,

        default=0.10,

        help="Validation split."
    )

    ap.add_argument(

        "--test",

        type=float,

        default=0.10,

        help="Testing split."
    )

    # ------------------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------------------

    ap.add_argument(

        "--seed",

        type=int,

        default=42,

        help="Random seed."
    )

    # ------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------

    main(
        ap.parse_args()
    )