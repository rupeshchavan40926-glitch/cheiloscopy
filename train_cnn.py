"""
CNN Trainer — Smart Cheiloscopy

Experiment 3 / revised training pipeline.

Input CSV must contain:
    groove_path
    gender_idx
    age_idx

The CSV points to ALREADY GENERATED GROOVE MAPS.

Training input:
    groove image
        ↓
    grayscale
        ↓
    aspect-ratio-preserving resize
        ↓
    centered padding to 256 x 256
        ↓
    normalize to [0, 1]
        ↓
    CNN

The model does NOT regenerate groove maps during training.

Tasks:
    gender
    age
    both

Example:
    python train_cnn.py ^
        --dataset "dataset_combined_final" ^
        --output "models(E3)/cnn_256" ^
        --task both
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256

INPUT_SHAPE = (
    IMG_HEIGHT,
    IMG_WIDTH,
    1
)

BATCH_SIZE = 32
EPOCHS = 15

LEARNING_RATE = 0.0005

RANDOM_SEED = 42


# ============================================================================
# RANDOM SEED
# ============================================================================

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================================
# ASPECT-RATIO-PRESERVING IMAGE PREPARATION
# ============================================================================

def letterbox_grayscale(
    image,
    target_size=(256, 256),
    pad_value=0
):
    """
    Resize a grayscale image while preserving its original aspect ratio.

    NO stretching is performed.

    Example:

        992 x 810
             ↓
        proportional resize
             ↓
        256 x 209
             ↓
        centered padding
             ↓
        256 x 256

    Returns:
        uint8 grayscale image of target_size.
    """

    target_w, target_h = target_size

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    src_h, src_w = image.shape[:2]

    if src_h <= 0 or src_w <= 0:
        raise ValueError(
            "Invalid input image dimensions."
        )

    # ------------------------------------------------------------
    # Compute scale without changing aspect ratio
    # ------------------------------------------------------------

    scale = min(
        target_w / src_w,
        target_h / src_h
    )

    new_w = max(
        1,
        int(round(src_w * scale))
    )

    new_h = max(
        1,
        int(round(src_h * scale))
    )

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------------
    # Center padding
    # ------------------------------------------------------------

    canvas = np.full(
        (
            target_h,
            target_w
        ),
        pad_value,
        dtype=np.uint8
    )

    x_offset = (
        target_w -
        new_w
    ) // 2

    y_offset = (
        target_h -
        new_h
    ) // 2

    canvas[
        y_offset:
        y_offset + new_h,
        x_offset:
        x_offset + new_w
    ] = resized

    return canvas


# ============================================================================
# LOAD IMAGES
# ============================================================================

def load_images(
    csv_path,
    label_col,
    n_classes,
    desc="Loading"
):
    """
    Load already-generated groove maps from groove_path.

    Every image is:

        grayscale
        ->
        aspect-ratio-preserving resize/padding
        ->
        256 x 256
        ->
        normalize
        ->
        add channel dimension
    """

    df = pd.read_csv(
        csv_path
    )

    if "groove_path" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain "
            f"'groove_path' column."
        )

    if label_col not in df.columns:
        raise ValueError(
            f"{csv_path} must contain "
            f"'{label_col}' column."
        )

    X = []
    y = []

    skipped = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=desc
    ):

        groove_path = str(
            row["groove_path"]
        )

        # --------------------------------------------------------
        # Read already-generated groove map
        # --------------------------------------------------------

        groove = cv2.imread(
            groove_path,
            cv2.IMREAD_GRAYSCALE
        )

        if groove is None:

            skipped += 1
            continue

        try:

            # ----------------------------------------------------
            # Preserve aspect ratio
            # ----------------------------------------------------

            groove = letterbox_grayscale(
                groove,
                target_size=(
                    IMG_WIDTH,
                    IMG_HEIGHT
                ),
                pad_value=0
            )

        except Exception as exc:

            print(
                f"\nWarning: failed to prepare "
                f"{groove_path}: {exc}"
            )

            skipped += 1
            continue

        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------

        groove = (
            groove.astype(
                np.float32
            )
            / 255.0
        )

        # --------------------------------------------------------
        # Add grayscale channel
        #
        # (256,256) -> (256,256,1)
        # --------------------------------------------------------

        groove = np.expand_dims(
            groove,
            axis=-1
        )

        X.append(
            groove
        )

        y.append(
            int(row[label_col])
        )

    print(
        f"{desc}: "
        f"Loaded {len(X)}, "
        f"skipped {skipped}"
    )

    if len(X) == 0:

        raise ValueError(
            f"No valid groove images "
            f"could be loaded from "
            f"{csv_path}"
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    y = to_categorical(
        y,
        num_classes=n_classes
    )

    print(
        f"{desc} shape: "
        f"X={X.shape}, "
        f"y={y.shape}"
    )

    return X, y


# ============================================================================
# CNN MODEL
# ============================================================================

def build_cnn(n_classes):
    """
    Lightweight CNN for 256 x 256 grayscale lip-groove maps.

    Designed for CPU training while retaining enough capacity
    to learn groove structure.
    """

    inputs = layers.Input(
        shape=INPUT_SHAPE
    )

    # ============================================================
    # BLOCK 1
    # ============================================================

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.20
    )(x)

    # ============================================================
    # BLOCK 2
    # ============================================================

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.25
    )(x)

    # ============================================================
    # BLOCK 3
    # ============================================================

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    # ============================================================
    # GLOBAL FEATURE REPRESENTATION
    # ============================================================

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.40
    )(x)

    # ============================================================
    # OUTPUT
    # ============================================================

    outputs = layers.Dense(
        n_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    # ============================================================
    # COMPILE
    # ============================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

    # ========================================================================
    # COMPILE
    # ========================================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(
    model_path
):
    """
    Training callbacks.
    """

    return [

        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    model,
    X_test,
    y_test_cat,
    class_names
):
    """
    Evaluate test set.
    """

    y_pred_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    y_true = np.argmax(
        y_test_cat,
        axis=1
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: "
        f"{acc * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    return acc


# ============================================================================
# PLOT
# ============================================================================

def save_plot(
    history,
    output_dir,
    task
):
    """
    Save training/validation accuracy and loss plots.
    """

    plt.figure(
        figsize=(12, 4)
    )

    # ------------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------------

    plt.subplot(
        1,
        2,
        1
    )

    plt.plot(
        history.history["accuracy"],
        label="Train"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Val"
    )

    plt.title(
        f"{task} Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    plt.subplot(
        1,
        2,
        2
    )

    plt.plot(
        history.history["loss"],
        label="Train"
    )

    plt.plot(
        history.history["val_loss"],
        label="Val"
    )

    plt.title(
        f"{task} Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"{task.lower()}_training_plot.png"
    )

    plt.savefig(
        path,
        dpi=120
    )

    plt.close()

    print(
        f"Plot saved: {path}"
    )


# ============================================================================
# TRAIN ONE TASK
# ============================================================================

def run_task(
    task,
    label_col,
    n_classes,
    class_names,
    args
):
    """
    Train one CNN task.
    """

    print(
        "\n" + "=" * 65
    )

    print(
        f"CNN Training — {task}"
    )

    print(
        "=" * 65
    )

    train_csv = os.path.join(
        args.dataset,
        "train.csv"
    )

    val_csv = os.path.join(
        args.dataset,
        "val.csv"
    )

    test_csv = os.path.join(
        args.dataset,
        "test.csv"
    )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    print(
        "\nLoading already-generated groove maps..."
    )

    X_train, y_train = load_images(
        train_csv,
        label_col,
        n_classes,
        desc="Train"
    )

    X_val, y_val = load_images(
        val_csv,
        label_col,
        n_classes,
        desc="Val"
    )

    X_test, y_test = load_images(
        test_csv,
        label_col,
        n_classes,
        desc="Test"
    )

    print(
        "\nData shapes:"
    )

    print(
        f"Train: "
        f"{X_train.shape}"
    )

    print(
        f"Val  : "
        f"{X_val.shape}"
    )

    print(
        f"Test : "
        f"{X_test.shape}"
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = build_cnn(
        n_classes
    )

    print(
        "\nModel summary:"
    )

    model.summary()

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    os.makedirs(
        args.output,
        exist_ok=True
    )

    model_path = os.path.join(
        args.output,
        f"{task.lower()}_cnn_256.keras"
    )

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=get_callbacks(
            model_path
        ),

        verbose=1
    )

    # ------------------------------------------------------------------------
    # Reload best model
    # ------------------------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path
    )

    # ------------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------------

    print(
        "\nTest Results:"
    )

    acc = evaluate(
        model,
        X_test,
        y_test,
        class_names
    )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    save_plot(
        history,
        args.output,
        task
    )

    return acc


# ============================================================================
# MAIN
# ============================================================================

def main(
    args
):
    """
    Main training entry point.
    """

    print(
        "\n" + "=" * 65
    )

    print(
        "Smart Cheiloscopy CNN Training — 256 x 256"
    )

    print(
        "=" * 65
    )

    print(
        "Dataset:",
        args.dataset
    )

    print(
        "Output :",
        args.output
    )

    print(
        "Input  :",
        INPUT_SHAPE
    )

    print(
        "Epochs :",
        EPOCHS
    )

    print(
        "Batch  :",
        BATCH_SIZE
    )

    print(
        "LR     :",
        LEARNING_RATE
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        "GPUs available:",
        len(gpus)
    )

    if len(gpus) == 0:

        print(
            "Training will run on CPU."
        )

    results = {}

    # ========================================================================
    # GENDER
    # ========================================================================

    if args.task in (
        "gender",
        "both"
    ):

        acc = run_task(
            task="Gender",
            label_col="gender_idx",
            n_classes=2,
            class_names=[
                "Male",
                "Female"
            ],
            args=args
        )

        results[
            "Gender CNN"
        ] = acc

    # ========================================================================
    # AGE
    # ========================================================================

    if args.task in (
        "age",
        "both"
    ):

        acc = run_task(
            task="Age",
            label_col="age_idx",
            n_classes=3,
            class_names=[
                "Young",
                "Adult",
                "Old"
            ],
            args=args
        )

        results[
            "Age CNN"
        ] = acc

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 65
    )

    for name, acc in results.items():

        print(
            f"{name:<25}: "
            f"{acc * 100:.2f}%"
        )

    print(
        "=" * 65
    )


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train CNN using pre-generated "
            "lip groove maps."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Folder containing "
            "train.csv, val.csv, test.csv"
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Folder to save trained models"
        )
    )

    parser.add_argument(
        "--task",
        default="both",
        choices=[
            "gender",
            "age",
            "both"
        ]
    )

    args = parser.parse_args()

    main(
        args
    )