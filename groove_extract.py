"""
groove_extract.py
=================

Two modes:

1. --mode face
   Full-face / UTKFace images.
   Uses MediaPipe Face Landmarker to locate lips.

2. --mode lip
   Already-collected / already-cropped lip-print images.

Strict collected-lip pipeline:

    Lip print
       ↓
    Strict print mask
       ↓
    Grayscale
       ↓
    CLAHE
       ↓
    Bright groove response (Top-hat)
       +
    Dark groove response (Black-hat)
       +
    Hessian ridge response
       ↓
    Combined response
       ↓
    Strict threshold
       ↓
    Directional continuity filtering
       ↓
    Small-gap bridging
       ↓
    Length filtering
       ↓
    Elongation filtering
       ↓
    Strict final groove map
       ↓
    Overlay

IMPORTANT:

    lip_groove/
        Native-resolution strict groove map

    lip_groove_response/
        Continuous groove response

    lip_groove_skeleton/
        Optional one-pixel skeleton

    lip_groove_model/
        256 x 256 aspect-ratio-preserving model input


Example:

    python groove_extract.py ^
        --input "D:\quality ext" ^
        --output "D:\quality ext\lip_groove_strict" ^
        --mode lip


UTKFace:

    python groove_extract.py ^
        --input "ext/filtered" ^
        --output "ext/grooves" ^
        --mode face
"""

import os
import cv2
import argparse
import urllib.request
import numpy as np

from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# New CNN model input
# ---------------------------------------------------------------------------

MODEL_SIZE = (256, 256)     # width, height

# ---------------------------------------------------------------------------
# Existing UTKFace output
# Keep unchanged for compatibility with old pipeline
# ---------------------------------------------------------------------------

FACE_OUTPUT_SIZE = (128, 64)

# ---------------------------------------------------------------------------
# Reference native resolution for collected samples
# ---------------------------------------------------------------------------

REFERENCE_SIZE = (1200, 900)


# ============================================================================
# STRICT GROOVE PARAMETERS
# ============================================================================

# Higher percentile = stricter groove selection
GROOVE_PERCENTILE = 81

# Minimum connected component size
MIN_COMPONENT_AREA = 10

# Minimum longest dimension
MIN_COMPONENT_LENGTH = 18

# Minimum elongation
MIN_ELONGATION = 2.0

# Only small gaps are bridged
GAP_BRIDGE_SIZE = 3

# Minimum line length from directional structure
DIRECTIONAL_MIN_LENGTH = 11


# ============================================================================
# MEDIAPIPE
# ============================================================================

_landmarker = None


def download_model():
    """Download MediaPipe model when required."""

    if not os.path.exists(MODEL_PATH):

        print(
            "Downloading face_landmarker.task (~30 MB) ..."
        )

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print(
            f"Saved -> {MODEL_PATH}"
        )


def get_landmarker():
    """Create MediaPipe FaceLandmarker once."""

    global _landmarker

    if _landmarker is None:

        download_model()

        base_options = mp_tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        _landmarker = (
            mp_vision.FaceLandmarker
            .create_from_options(options)
        )

    return _landmarker


# ============================================================================
# LIP LANDMARKS
# ============================================================================

OUTER_LIP = [
    61, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 291,
    375, 321, 405, 314, 17,
    84, 181, 91, 146
]

INNER_LIP = [
    78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308,
    324, 318, 402, 317, 14,
    87, 178, 88, 95
]

ALL_LIP_IDX = list(
    set(OUTER_LIP + INNER_LIP)
)


# ============================================================================
# FACE MODE
# ============================================================================

def extract_lip_region(image_bgr):
    """
    Existing UTKFace/full-face lip extraction.

    Returns:
        lip_crop
        lip_mask
        success
    """

    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = get_landmarker().detect(
        mp_image
    )

    if not result.face_landmarks:
        return None, None, False

    landmarks = result.face_landmarks[0]

    lip_pts = np.array(
        [
            (
                int(landmarks[i].x * w),
                int(landmarks[i].y * h)
            )
            for i in ALL_LIP_IDX
        ],
        dtype=np.int32
    )

    x_min = max(
        lip_pts[:, 0].min() - 12,
        0
    )

    x_max = min(
        lip_pts[:, 0].max() + 12,
        w
    )

    y_min = max(
        lip_pts[:, 1].min() - 10,
        0
    )

    y_max = min(
        lip_pts[:, 1].max() + 10,
        h
    )

    lip_w = x_max - x_min
    lip_h = y_max - y_min

    if lip_w < 30 or lip_h < 10:
        return None, None, False

    lip_crop = image_bgr[
        y_min:y_max,
        x_min:x_max
    ].copy()

    shifted_pts = (
        lip_pts -
        np.array([x_min, y_min])
    )

    mask = np.zeros(
        (lip_h, lip_w),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [shifted_pts],
        255
    )

    return lip_crop, mask, True


# ============================================================================
# LIP-ONLY MODE
# ============================================================================

def use_lip_image_directly(image_bgr):
    """
    Collected images are already lip-print images.

    No MediaPipe is used.
    Native resolution is preserved.
    """

    if image_bgr is None:
        return None, False

    h, w = image_bgr.shape[:2]

    if h < 100 or w < 100:
        return None, False

    return image_bgr.copy(), True


# ============================================================================
# STRICT PRINT MASK
# ============================================================================

def build_strict_print_mask(image_bgr):
    """
    Build a strict mask for the actual lip-print region.

    Primary evidence:
        red/pink dominance
        saturation

    Secondary evidence:
        local darkness

    Strong spatial filtering is used to suppress:
        handwritten labels
        isolated pen marks
        borders
        random background noise
    """

    h, w = image_bgr.shape[:2]

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    b, g, r = cv2.split(
        image_bgr.astype(np.int16)
    )

    # ------------------------------------------------------------------------
    # Red dominance
    # ------------------------------------------------------------------------

    redness = r - g

    redness = np.clip(
        redness,
        0,
        255
    ).astype(np.uint8)

    positive_red = redness[
        redness > 0
    ]

    if positive_red.size == 0:
        return None, False

    red_threshold = max(
        12.0,
        float(
            np.percentile(
                positive_red,
                65
            )
        )
    )

    red_mask = (
        redness >= red_threshold
    ).astype(np.uint8) * 255

    # ------------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------------

    saturation = hsv[:, :, 1]

    positive_sat = saturation[
        saturation > 0
    ]

    if positive_sat.size == 0:
        return None, False

    sat_threshold = max(
        10.0,
        float(
            np.percentile(
                positive_sat,
                68
            )
        )
    )

    sat_mask = (
        saturation >= sat_threshold
    ).astype(np.uint8) * 255

    # ------------------------------------------------------------------------
    # Primary mask
    #
    # Use AND rather than OR to be stricter.
    # The region should have BOTH color evidence and red/pink evidence.
    # ------------------------------------------------------------------------

    mask = cv2.bitwise_and(
        red_mask,
        sat_mask
    )

    # ------------------------------------------------------------------------
    # Mild closing
    # ------------------------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # ------------------------------------------------------------------------
    # Remove tiny connected components BEFORE large-scale operations
    # ------------------------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return None, False

    clean = np.zeros_like(mask)

    image_area = float(h * w)

    candidates = []

    for label in range(
        1,
        num_labels
    ):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        cw = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        ch = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        cx, cy = centroids[label]

        area_ratio = (
            area /
            image_area
        )

        # --------------------------------------------------------------------
        # Centrality
        # --------------------------------------------------------------------

        dx = abs(
            cx - w / 2
        ) / max(
            w / 2,
            1
        )

        dy = abs(
            cy - h / 2
        ) / max(
            h / 2,
            1
        )

        centrality = max(
            0.0,
            1.0 - (
                0.7 * dx +
                0.3 * dy
            )
        )

        # --------------------------------------------------------------------
        # Minimum dimensionality
        # --------------------------------------------------------------------

        if cw < 0.20 * w:
            continue

        if ch < 0.08 * h:
            continue

        if area_ratio < 0.003:
            continue

        score = (
            area_ratio *
            centrality
        )

        candidates.append(
            (
                label,
                score,
                area_ratio,
                cx,
                cy,
                cw,
                ch
            )
        )

    if not candidates:
        return None, False

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Keep the strongest 1–2 meaningful print regions
    selected = 0

    for (
        label,
        score,
        area_ratio,
        cx,
        cy,
        cw,
        ch
    ) in candidates:

        if abs(
            cx - w / 2
        ) > 0.40 * w:
            continue

        if abs(
            cy - h / 2
        ) > 0.38 * h:
            continue

        clean[
            labels == label
        ] = 255

        selected += 1

        if selected >= 2:
            break

    if not np.any(clean):
        return None, False

    # ------------------------------------------------------------------------
    # Fill the print region gently
    # ------------------------------------------------------------------------

    fill_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        fill_kernel,
        iterations=1
    )

    # ------------------------------------------------------------------------
    # Keep large contours only
    # ------------------------------------------------------------------------

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        largest = cv2.contourArea(
            contours[0]
        )

        contour_mask = np.zeros_like(
            clean
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area >= (
                0.15 *
                largest
            ):

                cv2.drawContours(
                    contour_mask,
                    [contour],
                    -1,
                    255,
                    thickness=cv2.FILLED
                )

        clean = contour_mask

    # ------------------------------------------------------------------------
    # Erode boundary slightly.
    # ------------------------------------------------------------------------

    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    clean = cv2.erode(
        clean,
        erode_kernel,
        iterations=1
    )

    ratio = np.mean(
        clean > 0
    )

    if ratio < 0.02:
        return None, False

    if ratio > 0.65:
        return None, False

    return clean, True


# ============================================================================
# BASIC PREPROCESSING
# ============================================================================

def to_gray(image_bgr):

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )


def apply_clahe(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(12, 12)
    )

    return clahe.apply(
        gray
    )


# ============================================================================
# BRIGHT RESPONSE
# ============================================================================

def compute_bright_response(
    image,
    scale
):
    """
    Bright thin line detector using top-hat.
    """

    response = np.zeros_like(
        image
    )

    sizes = [
        max(
            7,
            int(
                round(
                    7 * scale
                )
            )
        ),
        max(
            11,
            int(
                round(
                    11 * scale
                )
            )
        ),
        max(
            15,
            int(
                round(
                    15 * scale
                )
            )
        )
    ]

    for size in sizes:

        if size % 2 == 0:
            size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )

        top_hat = cv2.morphologyEx(
            image,
            cv2.MORPH_TOPHAT,
            kernel
        )

        response = cv2.max(
            response,
            top_hat
        )

    return cv2.normalize(
        response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


# ============================================================================
# DARK RESPONSE
# ============================================================================

def compute_dark_response(
    image,
    scale
):
    """
    Dark thin line detector using black-hat.
    """

    response = np.zeros_like(
        image
    )

    sizes = [
        max(
            7,
            int(
                round(
                    7 * scale
                )
            )
        ),
        max(
            11,
            int(
                round(
                    11 * scale
                )
            )
        ),
        max(
            15,
            int(
                round(
                    15 * scale
                )
            )
        )
    ]

    for size in sizes:

        if size % 2 == 0:
            size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )

        black_hat = cv2.morphologyEx(
            image,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        response = cv2.max(
            response,
            black_hat
        )

    return cv2.normalize(
        response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


# ============================================================================
# HESSIAN RESPONSE
# ============================================================================

def compute_hessian_response(
    image,
    lip_mask,
    scale
):
    """
    Multi-scale Hessian line response.
    """

    img = image.astype(
        np.float32
    )

    response = np.zeros_like(
        img
    )

    valid = (
        lip_mask > 0
    )

    if not np.any(valid):
        return np.zeros_like(
            image
        )

    sigmas = [
        1.5 * scale,
        2.5 * scale,
        3.5 * scale
    ]

    for sigma in sigmas:

        blurred = cv2.GaussianBlur(
            img,
            (0, 0),
            sigma
        )

        Ixx = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            2,
            0,
            ksize=3
        )

        Iyy = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            0,
            2,
            ksize=3
        )

        Ixy = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            1,
            1,
            ksize=3
        )

        discriminant = np.sqrt(
            (Ixx - Iyy) ** 2 +
            4.0 * (Ixy ** 2)
        )

        l1 = 0.5 * (
            Ixx +
            Iyy +
            discriminant
        )

        l2 = 0.5 * (
            Ixx +
            Iyy -
            discriminant
        )

        abs_l1 = np.abs(
            l1
        )

        abs_l2 = np.abs(
            l2
        )

        small = np.minimum(
            abs_l1,
            abs_l2
        )

        large = np.maximum(
            abs_l1,
            abs_l2
        )

        ratio = (
            small /
            (
                large +
                1e-6
            )
        )

        strength = np.sqrt(
            small ** 2 +
            large ** 2
        )

        c = np.percentile(
            strength[valid],
            90
        )

        c = max(
            c,
            1e-6
        )

        beta = 0.35

        ridge = (
            np.exp(
                -(ratio ** 2)
                /
                (
                    2.0 *
                    beta ** 2
                )
            )
            *
            (
                1.0 -
                np.exp(
                    -(strength ** 2)
                    /
                    (
                        2.0 *
                        c ** 2
                    )
                )
            )
        )

        ridge *= (
            sigma ** 2
        )

        response = np.maximum(
            response,
            ridge
        )

    response = cv2.normalize(
        response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return response.astype(
        np.uint8
    )


# ============================================================================
# DIRECTIONAL CONTINUITY
# ============================================================================

def directional_continuity(response):
    """
    Keep structures that have elongated local support.

    Four directions are tested:

        horizontal
        vertical
        diagonal /
        diagonal \

    This is used as a strict structural filter.

    The purpose is to suppress:
        random dots
        speckle
        small blobs
        isolated texture
    """

    # ------------------------------------------------------------------------
    # Directional kernels
    # ------------------------------------------------------------------------

    horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (DIRECTIONAL_MIN_LENGTH, 3)
    )

    vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, DIRECTIONAL_MIN_LENGTH)
    )

    # Approximate diagonal line kernels
    diag1 = np.zeros(
        (
            9,
            9
        ),
        dtype=np.uint8
    )

    diag2 = np.zeros_like(
        diag1
    )

    for i in range(9):

        diag1[
            i,
            i
        ] = 1

        diag2[
            i,
            8 - i
        ] = 1

    # Slightly thicken diagonal support
    diag1 = cv2.dilate(
        diag1,
        np.ones(
            (3, 3),
            dtype=np.uint8
        ),
        iterations=1
    )

    diag2 = cv2.dilate(
        diag2,
        np.ones(
            (3, 3),
            dtype=np.uint8
        ),
        iterations=1
    )

    # ------------------------------------------------------------------------
    # Morphological opening gives local line support.
    # ------------------------------------------------------------------------

    h_support = cv2.morphologyEx(
        response,
        cv2.MORPH_OPEN,
        horizontal
    )

    v_support = cv2.morphologyEx(
        response,
        cv2.MORPH_OPEN,
        vertical
    )

    d1_support = cv2.morphologyEx(
        response,
        cv2.MORPH_OPEN,
        diag1
    )

    d2_support = cv2.morphologyEx(
        response,
        cv2.MORPH_OPEN,
        diag2
    )

    continuity = cv2.max(
        h_support,
        v_support
    )

    continuity = cv2.max(
        continuity,
        d1_support
    )

    continuity = cv2.max(
        continuity,
        d2_support
    )

    return continuity


# ============================================================================
# STRICT GROOVE EXTRACTION
# ============================================================================

def extract_strict_grooves(
    gray,
    lip_mask
):
    """
    Strict groove extraction.

    Main principle:

        strength
            +
        elongation
            +
        continuity
            =
        valid groove

    Returns:

        groove
        continuous_response
        skeleton
    """

    h, w = gray.shape[:2]

    # ------------------------------------------------------------------------
    # Resolution factor
    # ------------------------------------------------------------------------

    scale = max(
        w,
        h
    ) / 1200.0

    scale = max(
        0.6,
        min(
            scale,
            2.0
        )
    )

    # ------------------------------------------------------------------------
    # Mild smoothing
    # ------------------------------------------------------------------------

    smooth = cv2.bilateralFilter(
        gray,
        d=7,
        sigmaColor=35,
        sigmaSpace=7
    )

    # ------------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(12, 12)
    )

    enhanced = clahe.apply(
        smooth
    )

    # ------------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------------

    bright = compute_bright_response(
        enhanced,
        scale
    )

    dark = compute_dark_response(
        enhanced,
        scale
    )

    ridge = compute_hessian_response(
        enhanced,
        lip_mask,
        scale
    )

    # ------------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------------

    raw_response = (
        0.30 *
        bright.astype(np.float32)
        +
        0.30 *
        dark.astype(np.float32)
        +
        0.40 *
        ridge.astype(np.float32)
    )

    raw_response = np.clip(
        raw_response,
        0,
        255
    ).astype(
        np.uint8
    )

    raw_response[
        lip_mask == 0
    ] = 0

    # ------------------------------------------------------------------------
    # Directional continuity
    # ------------------------------------------------------------------------

    continuity = directional_continuity(
        raw_response
    )

    # Normalize continuity
    if np.max(
        continuity
    ) > 0:

        continuity = cv2.normalize(
            continuity,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    # ------------------------------------------------------------------------
    # Strength + continuity
    #
    # A true groove should have strong response AND structure.
    # ------------------------------------------------------------------------

    combined = (
        0.55 *
        raw_response.astype(
            np.float32
        )
        +
        0.45 *
        continuity.astype(
            np.float32
        )
    )

    combined = np.clip(
        combined,
        0,
        255
    ).astype(
        np.uint8
    )

    combined[
        lip_mask == 0
    ] = 0

    # ------------------------------------------------------------------------
    # Remove slow illumination variation
    # ------------------------------------------------------------------------

    baseline = cv2.GaussianBlur(
        combined,
        (0, 0),
        sigmaX=max(
            10,
            int(
                14 * scale
            )
        )
    )

    corrected = (
        combined.astype(
            np.float32
        )
        -
        0.35 *
        baseline.astype(
            np.float32
        )
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(
        np.uint8
    )

    corrected[
        lip_mask == 0
    ] = 0

    # ------------------------------------------------------------------------
    # Mild smoothing
    # ------------------------------------------------------------------------

    corrected = cv2.GaussianBlur(
        corrected,
        (3, 3),
        0
    )

    # ------------------------------------------------------------------------
    # STRICT threshold
    # ------------------------------------------------------------------------

    values = corrected[
        lip_mask > 0
    ]

    values = values[
        values > 0
    ]

    if values.size == 0:

        return (
            np.zeros_like(gray),
            corrected,
            np.zeros_like(gray)
        )

    threshold = np.percentile(
        values,
        GROOVE_PERCENTILE
    )

    threshold = max(
        threshold,
        12
    )

    _, groove = cv2.threshold(
        corrected,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    groove[
        lip_mask == 0
    ] = 0

    # ------------------------------------------------------------------------
    # Small-gap bridging only
    # ------------------------------------------------------------------------

    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            GAP_BRIDGE_SIZE,
            GAP_BRIDGE_SIZE
        )
    )

    groove = cv2.morphologyEx(
        groove,
        cv2.MORPH_CLOSE,
        gap_kernel,
        iterations=1
    )

    # ------------------------------------------------------------------------
    # NO aggressive opening here
    #
    # We already applied a strict response threshold.
    # Opening at this stage can break real grooves.
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Connected-component filtering
    # ------------------------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            groove,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        groove
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        long_side = max(
            width,
            height
        )

        short_side = max(
            min(
                width,
                height
            ),
            1
        )

        elongation = (
            long_side /
            short_side
        )

        # ----------------------------------------------------
        # Strict criteria
        # ----------------------------------------------------

        valid_area = (
            area >=
            MIN_COMPONENT_AREA
        )

        valid_length = (
            long_side >=
            MIN_COMPONENT_LENGTH
        )

        valid_shape = (
            elongation >=
            MIN_ELONGATION
        )

        # Allow a slightly larger but less elongated structure
        large_structure = (
            area >=
            MIN_COMPONENT_AREA * 6
            and
            long_side >=
            MIN_COMPONENT_LENGTH * 0.75
        )

        if (
            valid_area
            and
            valid_length
            and
            valid_shape
        ) or large_structure:

            cleaned[
                labels == label
            ] = 255

    groove = cleaned

    groove[
        lip_mask == 0
    ] = 0

    # ------------------------------------------------------------------------
    # Separate skeleton
    #
    # IMPORTANT:
    # This does NOT replace the primary groove map.
    # ------------------------------------------------------------------------

    skeleton = groove.copy()

    if hasattr(
        cv2,
        "ximgproc"
    ):

        skeleton = cv2.ximgproc.thinning(
            skeleton,
            thinningType=(
                cv2.ximgproc.THINNING_ZHANGSUEN
            )
        )

    return (
        groove,
        corrected,
        skeleton
    )


# ============================================================================
# ASPECT-RATIO-PRESERVING MODEL CONVERSION
# ============================================================================

def groove_map_to_model(
    groove,
    target_size=(256, 256)
):
    """
    Convert the native binary groove map to the CNN model size.

    The groove geometry is preserved by:
        1. small line reinforcement
        2. aspect-ratio-preserving resize
        3. nearest-neighbor interpolation
        4. light cleanup

    No interpolation of gray groove-response values is used.
    """

    target_w, target_h = target_size

    # ---------------------------------------------------------
    # Ensure binary groove map
    # ---------------------------------------------------------

    binary = np.where(
        groove > 0,
        255,
        0
    ).astype(np.uint8)

    src_h, src_w = binary.shape[:2]

    if src_h == 0 or src_w == 0:
        raise ValueError(
            "Invalid groove image dimensions."
        )

    # ---------------------------------------------------------
    # Slightly reinforce very thin groove segments
    #
    # This is intentionally small.
    # ---------------------------------------------------------

    binary = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2)
        ),
        iterations=1
    )

    # ---------------------------------------------------------
    # Preserve aspect ratio
    # ---------------------------------------------------------

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

    resized = cv2.resize(
        binary,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------------------------------------------------------
    # Center on black canvas
    # ---------------------------------------------------------

    canvas = np.zeros(
        (target_h, target_w),
        dtype=np.uint8
    )

    x_offset = (
        target_w - new_w
    ) // 2

    y_offset = (
        target_h - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # ---------------------------------------------------------
    # Small cleanup
    #
    # Remove isolated single pixels but do not erode
    # connected groove lines.
    # ---------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            canvas,
            connectivity=8
        )
    )

    clean = np.zeros_like(
        canvas
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        long_side = max(
            width,
            height
        )

        # Keep even relatively short but non-trivial lines
        if (
            area >= 3
            or
            long_side >= 5
        ):

            clean[
                labels == label
            ] = 255

    return clean


# ============================================================================
# OVERLAY
# ============================================================================

def overlay_grooves(
    base_gray,
    groove
):
    """
    Show strict grooves in red.
    """

    vis = cv2.cvtColor(
        base_gray,
        cv2.COLOR_GRAY2BGR
    )

    vis = cv2.convertScaleAbs(
        vis,
        alpha=1.10,
        beta=-5
    )

    vis[
        groove > 0
    ] = [
        0,
        0,
        255
    ]

    return vis


# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

def make_output_dirs(
    base
):

    folders = [

        "lip_crop",

        "lip_mask",

        "lip_gray",

        "lip_clahe",

        "lip_groove",

        "lip_groove_response",

        "lip_groove_skeleton",

        "lip_overlay",

        "lip_groove_model"
    ]

    paths = {}

    for folder in folders:

        path = os.path.join(
            base,
            folder
        )

        os.makedirs(
            path,
            exist_ok=True
        )

        paths[folder] = path

    return paths


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    dirs,
    stem,
    outputs
):

    for key, image in outputs.items():

        path = os.path.join(
            dirs[key],
            stem + ".png"
        )

        if not cv2.imwrite(
            path,
            image
        ):

            print(
                f"\nWarning: failed to save {path}"
            )


# ============================================================================
# FACE IMAGE PROCESSING
# ============================================================================

def process_face_image(
    image_bgr
):

    lip_crop, lip_mask, ok = (
        extract_lip_region(
            image_bgr
        )
    )

    if not ok:
        return None

    gray = to_gray(
        lip_crop
    )

    clahe = apply_clahe(
        gray
    )

    edges = cv2.Canny(
        clahe,
        25,
        75
    )

    groove = np.zeros_like(
        edges
    )

    groove[
        lip_mask > 0
    ] = edges[
        lip_mask > 0
    ]

    overlay = overlay_grooves(
        gray,
        groove
    )

    crop_out = cv2.resize(
        lip_crop,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    mask_out = cv2.resize(
        lip_mask,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    gray_out = cv2.resize(
        gray,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    clahe_out = cv2.resize(
        clahe,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    groove_out = cv2.resize(
        groove,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    overlay_out = cv2.resize(
        overlay,
        FACE_OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return {

        "lip_crop":
            crop_out,

        "lip_mask":
            mask_out,

        "lip_gray":
            gray_out,

        "lip_clahe":
            clahe_out,

        "lip_groove":
            groove_out,

        "lip_groove_response":
            groove_out,

        "lip_groove_skeleton":
            groove_out,

        "lip_overlay":
            overlay_out,

        "lip_groove_model":
            groove_out
    }


# ============================================================================
# COLLECTED LIP PROCESSING
# ============================================================================

def process_lip_image(
    image_bgr
):

    lip_crop, ok = (
        use_lip_image_directly(
            image_bgr
        )
    )

    if not ok:
        return None

    # ------------------------------------------------------------------------
    # STRICT PRINT MASK
    # ------------------------------------------------------------------------

    lip_mask, mask_ok = (
        build_strict_print_mask(
            lip_crop
        )
    )

    if not mask_ok:
        return None

    # ------------------------------------------------------------------------
    # GRAY
    # ------------------------------------------------------------------------

    gray = to_gray(
        lip_crop
    )

    # ------------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------------

    clahe = apply_clahe(
        gray
    )

    # ------------------------------------------------------------------------
    # STRICT GROOVE EXTRACTION
    # ------------------------------------------------------------------------

    groove, response, skeleton = (
        extract_strict_grooves(
            gray,
            lip_mask
        )
    )

    # ------------------------------------------------------------------------
    # OVERLAY
    # ------------------------------------------------------------------------

    overlay = overlay_grooves(
        clahe,
        groove
    )

    # ------------------------------------------------------------------------
    # 256 x 256 MODEL MAP
    #
    # Uses continuous response, NOT skeleton.
    # ------------------------------------------------------------------------

    groove_model = groove_map_to_model(
    groove,
    MODEL_SIZE
    )

    return {

        # Native
        "lip_crop":
            lip_crop,

        "lip_mask":
            lip_mask,

        "lip_gray":
            gray,

        "lip_clahe":
            clahe,

        "lip_groove":
            groove,

        "lip_groove_response":
            response,

        "lip_groove_skeleton":
            skeleton,

        "lip_overlay":
            overlay,

        # Model
        "lip_groove_model":
            groove_model
    }


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

def process_image(
    image_bgr,
    mode="face"
):

    if mode == "face":

        return process_face_image(
            image_bgr
        )

    return process_lip_image(
        image_bgr
    )


# ============================================================================
# MAIN
# ============================================================================

def main(
    args
):

    os.makedirs(
        args.output,
        exist_ok=True
    )

    dirs = make_output_dirs(
        args.output
    )

    files = [
        f
        for f in os.listdir(
            args.input
        )
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff"
            )
        )
    ]

    print()
    print("=" * 76)
    print("                    LIP GROOVE EXTRACTOR")
    print("=" * 76)

    print(
        f"Mode              : {args.mode}"
    )

    print(
        f"Input             : {args.input}"
    )

    print(
        f"Output            : {args.output}"
    )

    print(
        f"Images            : {len(files)}"
    )

    print(
        f"Model size        : "
        f"{MODEL_SIZE[0]} x {MODEL_SIZE[1]}"
    )

    print(
        f"Groove percentile : "
        f"{GROOVE_PERCENTILE}"
    )

    print(
        f"Min area          : "
        f"{MIN_COMPONENT_AREA}"
    )

    print(
        f"Min length        : "
        f"{MIN_COMPONENT_LENGTH}"
    )

    print(
        f"Min elongation    : "
        f"{MIN_ELONGATION}"
    )

    print(
        f"Directional length: "
        f"{DIRECTIONAL_MIN_LENGTH}"
    )

    print(
        f"ximgproc          : "
        f"{'AVAILABLE' if hasattr(cv2, 'ximgproc') else 'NOT AVAILABLE'}"
    )

    print("=" * 76)

    if args.mode == "lip":

        print()
        print("STRICT COLLECTED-LIP MODE")
        print("-------------------------")
        print("  Strict print segmentation")
        print("  Bright groove response")
        print("  Dark groove response")
        print("  Hessian ridge response")
        print("  Directional continuity")
        print("  88th-percentile threshold")
        print("  Length filtering")
        print("  Elongation filtering")
        print("  Small-gap bridging")
        print("  No thinning of primary groove map")
        print("  Separate skeleton output")
        print("  256 x 256 aspect-preserving model output")

    else:

        print()
        print("UTKFACE MODE")
        print("------------")
        print("  Existing MediaPipe lip extraction")
        print("  Existing CLAHE + Canny groove path")
        print(
            f"  Existing output: "
            f"{FACE_OUTPUT_SIZE[0]} x "
            f"{FACE_OUTPUT_SIZE[1]}"
        )

    print("=" * 76)
    print()

    success = 0
    failed = 0

    for fname in tqdm(
        files,
        desc="Extracting grooves",
        unit="img"
    ):

        path = os.path.join(
            args.input,
            fname
        )

        image = cv2.imread(
            path
        )

        if image is None:

            failed += 1
            continue

        if args.mode == "lip":

            h, w = image.shape[:2]

            if (
                abs(
                    w -
                    REFERENCE_SIZE[0]
                ) > 500
                or
                abs(
                    h -
                    REFERENCE_SIZE[1]
                ) > 500
            ):

                print(
                    f"\nWarning: "
                    f"{fname} has size "
                    f"{w}x{h}"
                )

        stem = os.path.splitext(
            fname
        )[0]

        outputs = process_image(
            image,
            mode=args.mode
        )

        if outputs is None:

            failed += 1
            continue

        save_outputs(
            dirs,
            stem,
            outputs
        )

        success += 1

    print()
    print("=" * 76)
    print("                              DONE")
    print("=" * 76)

    print(
        f"Successfully processed : "
        f"{success}"
    )

    print(
        f"Failed                 : "
        f"{failed}"
    )

    print(
        f"Results saved in       : "
        f"{args.output}"
    )

    if args.mode == "lip":

        print()
        print(
            "Primary native groove map:"
        )

        print(
            os.path.join(
                args.output,
                "lip_groove"
            )
        )

        print()
        print(
            "Continuous groove response:"
        )

        print(
            os.path.join(
                args.output,
                "lip_groove_response"
            )
        )

        print()
        print(
            "Skeleton:"
        )

        print(
            os.path.join(
                args.output,
                "lip_groove_skeleton"
            )
        )

        print()
        print(
            "256 x 256 model map:"
        )

        print(
            os.path.join(
                args.output,
                "lip_groove_model"
            )
        )

    print("=" * 76)
    print()


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract lip groove maps from "
            "full-face or collected lip-print images."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image folder"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "face",
            "lip"
        ],
        default="face",
        help=(
            "face = full-face images, "
            "lip = collected lip prints"
        )
    )

    args = parser.parse_args()

    main(
        args
    )