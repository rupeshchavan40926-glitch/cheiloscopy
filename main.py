"""
main.py
=======

Smart input router and preprocessing tester for the current Phase-1 system.

Supported input types:

    1. Full-face image
       -> groove_extract.py --mode face

    2. Lip-only / collected lip-print image
       -> groove_extract.py --mode lip

    3. Auto detection
       -> automatically choose face or lip mode

The final CNN representation is always:

    256 x 256 x 1

IMPORTANT
---------
This file does NOT perform CNN classification.

Its responsibilities are:

    input image
        ↓
    input-type routing
        ↓
    groove_extract.py
        ↓
    groove representation
        ↓
    256 x 256 model image
        ↓
    save / inspect outputs


Usage
-----

AUTO:
    python main.py ^
        --image "sample.jpg" ^
        --output "output" ^
        --mode auto

FULL FACE:
    python main.py ^
        --image "face.jpg" ^
        --output "output" ^
        --mode face

LIP PRINT:
    python main.py ^
        --image "lip.jpg" ^
        --output "output" ^
        --mode lip
"""

import os
import cv2
import argparse
import numpy as np

from groove_extract import process_image


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SIZE = (256, 256)  # width, height


# ============================================================================
# INPUT TYPE DETECTION
# ============================================================================

def detect_real_lipprint(image_path):
    """
    Heuristic detection of a collected/scanned lip print.

    Returns:
        True  -> likely lip print
        False -> likely full-face / real lip photograph

    This is ONLY a routing heuristic.
    It is not a machine-learning classifier.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    total_pixels = float(
        height * width
    )

    # ------------------------------------------------------------------------
    # Red / pink ink
    # ------------------------------------------------------------------------

    red1 = cv2.inRange(
        hsv,
        (0, 50, 50),
        (25, 255, 255)
    )

    red2 = cv2.inRange(
        hsv,
        (135, 40, 50),
        (180, 255, 255)
    )

    ink = cv2.bitwise_or(
        red1,
        red2
    )

    ink_ratio = (
        np.count_nonzero(ink)
        /
        total_pixels
    )

    # ------------------------------------------------------------------------
    # White background
    # ------------------------------------------------------------------------

    white_bg = cv2.inRange(
        hsv,
        (0, 0, 210),
        (180, 45, 255)
    )

    white_ratio = (
        np.count_nonzero(white_bg)
        /
        total_pixels
    )

    # ------------------------------------------------------------------------
    # Skin-like pixels
    # ------------------------------------------------------------------------

    skin1 = cv2.inRange(
        hsv,
        (0, 20, 60),
        (30, 170, 255)
    )

    skin2 = cv2.inRange(
        hsv,
        (160, 20, 60),
        (180, 150, 255)
    )

    skin = cv2.bitwise_or(
        skin1,
        skin2
    )

    skin_ratio = (
        np.count_nonzero(skin)
        /
        total_pixels
    )

    # ------------------------------------------------------------------------
    # Background smoothness
    # ------------------------------------------------------------------------

    white_pixels = gray[
        white_bg > 0
    ]

    if white_pixels.size > 50:

        white_std = float(
            np.std(white_pixels)
        )

    else:

        white_std = 999.0

    # ------------------------------------------------------------------------
    # Print debug
    # ------------------------------------------------------------------------

    print()
    print("Input Detection")
    print("----------------")
    print(
        f"Image size  : {width} x {height}"
    )

    print(
        f"Ink ratio   : {ink_ratio:.3f}"
    )

    print(
        f"White ratio : {white_ratio:.3f}"
    )

    print(
        f"Skin ratio  : {skin_ratio:.3f}"
    )

    print(
        f"White std   : {white_std:.3f}"
    )

    # ------------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------------

    is_lipprint = (
        ink_ratio > 0.08
        and
        white_ratio > 0.35
        and
        skin_ratio < 0.35
        and
        white_std < 25
    )

    print(
        "Detected    : "
        + (
            "LIP PRINT"
            if is_lipprint
            else "FACE / REAL LIP PHOTO"
        )
    )

    return is_lipprint


# ============================================================================
# ASPECT-RATIO-PRESERVING MODEL CONVERSION
# ============================================================================

def letterbox_grayscale(
    image,
    target_size=MODEL_SIZE,
    pad_value=0
):
    """
    Convert an arbitrary grayscale image to 256x256 without stretching.

    Same basic strategy as train_cnn.py:

        source
          ↓
        proportional resize
          ↓
        centered padding
          ↓
        256 x 256

    This is especially important for the UTKFace branch because its
    legacy groove map is still approximately 128 x 64.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    target_w, target_h = target_size

    src_h, src_w = image.shape[:2]

    if src_h <= 0 or src_w <= 0:

        raise ValueError(
            "Invalid image dimensions."
        )

    scale = min(
        target_w / src_w,
        target_h / src_h
    )

    new_w = max(
        1,
        int(
            round(
                src_w * scale
            )
        )
    )

    new_h = max(
        1,
        int(
            round(
                src_h * scale
            )
        )
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

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
# PREPARE CNN INPUT
# ============================================================================

def prepare_model_image(
    result,
    mode
):
    """
    Obtain the final 256x256 grayscale model representation.

    lip mode:
        groove_extract.py already creates lip_groove_model.

    face mode:
        groove_extract.py currently retains the old UTKFace groove
        representation, so we convert it here using aspect-preserving
        letterboxing.
    """

    # ------------------------------------------------------------------------
    # Collected lip-print path
    # ------------------------------------------------------------------------

    if mode == "lip":

        model_image = result.get(
            "lip_groove_model"
        )

        if model_image is None:

            raise ValueError(
                "lip_groove_model was not returned "
                "by groove_extract.py"
            )

        model_image = cv2.cvtColor(
            model_image,
            cv2.COLOR_BGR2GRAY
        ) if len(
            model_image.shape
        ) == 3 else model_image

        if model_image.shape != (
            MODEL_SIZE[1],
            MODEL_SIZE[0]
        ):

            model_image = letterbox_grayscale(
                model_image,
                MODEL_SIZE
            )

        return model_image

    # ------------------------------------------------------------------------
    # Full-face / UTKFace path
    # ------------------------------------------------------------------------

    groove = result.get(
        "lip_groove"
    )

    if groove is None:

        raise ValueError(
            "lip_groove was not returned "
            "by groove_extract.py"
        )

    return letterbox_grayscale(
        groove,
        MODEL_SIZE
    )


# ============================================================================
# OVERLAY PREPARATION
# ============================================================================

def make_overlay(
    result
):
    """
    Return the native-resolution overlay generated by groove_extract.py.
    """

    overlay = result.get(
        "lip_overlay"
    )

    return overlay


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_result(
    result,
    model_image,
    output_dir,
    input_type
):
    """
    Save preprocessing results for inspection.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # Native outputs
    # ------------------------------------------------------------------------

    output_map = {
        "lip_crop": "lip_crop.png",
        "lip_mask": "lip_mask.png",
        "lip_gray": "lip_gray.png",
        "lip_clahe": "lip_clahe.png",
        "lip_groove": "lip_groove.png",
        "lip_groove_response": "lip_groove_response.png",
        "lip_groove_skeleton": "lip_groove_skeleton.png",
        "lip_overlay": "lip_overlay.png",
    }

    for key, filename in output_map.items():

        image = result.get(
            key
        )

        if image is not None:

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    filename
                ),
                image
            )

    # ------------------------------------------------------------------------
    # Final CNN representation
    # ------------------------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            output_dir,
            "lip_groove_model.png"
        ),
        model_image
    )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "input_info.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Input type: {input_type}\n"
        )

        f.write(
            "Model input size: 256 x 256 x 1\n"
        )

        f.write(
            "Aspect ratio preserved: yes\n"
        )


# ============================================================================
# PROCESS ONE IMAGE
# ============================================================================

def process_input(
    image_path,
    mode="auto"
):
    """
    Main reusable processing function.

    mode:
        auto
        face
        lip

    Returns:
        result
        model_image
        input_type
        detected_mode
    """

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ------------------------------------------------------------------------
    # Auto routing
    # ------------------------------------------------------------------------

    if mode == "auto":

        is_lipprint = (
            detect_real_lipprint(
                image_path
            )
        )

        detected_mode = (
            "lip"
            if is_lipprint
            else "face"
        )

    else:

        detected_mode = mode

    # ------------------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Could not read image."
        )

    # ------------------------------------------------------------------------
    # New groove_extract pipeline
    # ------------------------------------------------------------------------

    result = process_image(
        image,
        mode=detected_mode
    )

    if result is None:

        raise RuntimeError(
            "groove_extract.py could not "
            "produce a valid lip/groove result."
        )

    # ------------------------------------------------------------------------
    # Prepare model image
    # ------------------------------------------------------------------------

    model_image = prepare_model_image(
        result,
        detected_mode
    )

    input_type = (
        "LIP PRINT"
        if detected_mode == "lip"
        else "FACE / REAL LIP PHOTO"
    )

    return (
        result,
        model_image,
        input_type,
        detected_mode
    )


# ============================================================================
# CLI
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Smart Cheiloscopy router and "
            "groove preprocessing tester."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image"
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output folder"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "auto",
            "face",
            "lip"
        ],
        default="auto",
        help=(
            "auto = automatic detection, "
            "face = full-face input, "
            "lip = lip-print/lip-only input"
        )
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("                 CHEILOSCOPY SMART ROUTER")
    print("=" * 70)

    try:

        (
            result,
            model_image,
            input_type,
            detected_mode
        ) = process_input(
            args.image,
            args.mode
        )

        print()
        print(
            f"Input type : {input_type}"
        )

        print(
            f"Mode       : {detected_mode}"
        )

        print(
            f"Model size : {model_image.shape}"
        )

        save_result(
            result,
            model_image,
            args.output,
            input_type
        )

        print()
        print(
            f"Saved to   : {args.output}"
        )

    except Exception as exc:

        print()
        print(
            "ERROR:"
        )

        print(
            str(exc)
        )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()