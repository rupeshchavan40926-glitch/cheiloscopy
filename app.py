"""
app.py
======

Streamlit application for the current Phase-1 CNN system.

Supported inputs:

    - Full-face image
    - Cropped lip image
    - Collected / scanned lip-print image

Preprocessing:

    input
      ↓
    face or lip routing
      ↓
    groove_extract.py
      ↓
    256 x 256 model representation
      ↓
    Gender CNN
    Age CNN
"""

import os
import cv2
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

from main import (
    detect_real_lipprint,
    prepare_model_image,
    process_input
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="AI Cheiloscopy CNN System",
    page_icon="🧬",
    layout="wide"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F7EEDC;
        color: #1E1E1E;
    }

    html, body, [class*="css"] {
        color: #1E1E1E;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #2E1F1A !important;
    }

    p, span, div, label {
        color: #1E1E1E !important;
    }

    .main-title {
        text-align: center;
        color: #3B2A24 !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #4E342E !important;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        background-color: #FFF8E8;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        color: #1E1E1E;
    }

    .result-box {
        background-color: #FAE6C8;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #8B5E3C;
        font-size: 20px;
        font-weight: 700;
        color: #2B1B17 !important;
    }

    .result-box * {
        color: #2B1B17 !important;
    }

    .small-text {
        color: #3E2723 !important;
        font-size: 15px;
    }

    .stButton > button {
        background-color: #8B5E3C;
        color: white !important;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: 700;
        border: none;
    }

    .stButton > button:hover {
        background-color: #6D4C41;
        color: white !important;
    }

    section[data-testid="stFileUploader"] {
        color: #1E1E1E !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "models(E3)/cnn/cnn_combined"

GENDER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "gender_cnn_256.keras"
)

AGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "age_cnn_256.keras"
)

MODEL_SIZE = (256, 256)

GENDER_CLASSES = [
    "Male",
    "Female"
]

AGE_CLASSES = [
    "Young",
    "Adult",
    "Old"
]


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """
    Load the Phase-1 combined-dataset CNN models.
    """

    if not os.path.exists(
        GENDER_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Gender model not found:\n"
            f"{GENDER_MODEL_PATH}"
        )

    if not os.path.exists(
        AGE_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Age model not found:\n"
            f"{AGE_MODEL_PATH}"
        )

    gender_model = tf.keras.models.load_model(
        GENDER_MODEL_PATH
    )

    age_model = tf.keras.models.load_model(
        AGE_MODEL_PATH
    )

    return (
        gender_model,
        age_model
    )


# ============================================================================
# MODEL INPUT PREPARATION
# ============================================================================

def prepare_cnn_tensor(
    model_image
):
    """
    Convert 256x256 grayscale model image into:

        (1, 256, 256, 1)
    """

    if model_image is None:

        raise ValueError(
            "Model image is None."
        )

    if len(
        model_image.shape
    ) == 3:

        model_image = cv2.cvtColor(
            model_image,
            cv2.COLOR_BGR2GRAY
        )

    if model_image.shape != (
        MODEL_SIZE[1],
        MODEL_SIZE[0]
    ):

        raise ValueError(
            f"Unexpected model image shape: "
            f"{model_image.shape}. "
            f"Expected (256, 256)."
        )

    x = (
        model_image.astype(
            np.float32
        )
        /
        255.0
    )

    x = np.expand_dims(
        x,
        axis=-1
    )

    x = np.expand_dims(
        x,
        axis=0
    )

    return x


# ============================================================================
# PREDICTION
# ============================================================================

def predict(
    image_path,
    input_mode
):
    """
    Run preprocessing + CNN prediction.
    """

    gender_model, age_model = (
        load_models()
    )

    (
        result,
        model_image,
        input_type,
        detected_mode
    ) = process_input(
        image_path,
        input_mode
    )

    x = prepare_cnn_tensor(
        model_image
    )

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------

    gender_pred = (
        gender_model.predict(
            x,
            verbose=0
        )[0]
    )

    age_pred = (
        age_model.predict(
            x,
            verbose=0
        )[0]
    )

    gender_idx = int(
        np.argmax(
            gender_pred
        )
    )

    age_idx = int(
        np.argmax(
            age_pred
        )
    )

    return {

        "gender":
            GENDER_CLASSES[
                gender_idx
            ],

        "gender_conf":
            float(
                gender_pred[
                    gender_idx
                ]
                * 100.0
            ),

        "gender_scores":
            gender_pred,

        "age":
            AGE_CLASSES[
                age_idx
            ],

        "age_conf":
            float(
                age_pred[
                    age_idx
                ]
                * 100.0
            ),

        "age_scores":
            age_pred,

        "lip_crop":
            result.get(
                "lip_crop"
            ),

        "lip_mask":
            result.get(
                "lip_mask"
            ),

        "lip_gray":
            result.get(
                "lip_gray"
            ),

        "lip_clahe":
            result.get(
                "lip_clahe"
            ),

        "groove":
            result.get(
                "lip_groove"
            ),

        "groove_response":
            result.get(
                "lip_groove_response"
            ),

        "skeleton":
            result.get(
                "lip_groove_skeleton"
            ),

        "overlay":
            result.get(
                "lip_overlay"
            ),

        "model_image":
            model_image,

        "pipeline":
            input_type,

        "mode":
            detected_mode
    }


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown(
    '<div class="main-title">'
    'AI-Based Cheiloscopy System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'CNN-based Gender and Age Prediction '
    'using Lip Groove Patterns'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================================
# TOP SECTION
# ============================================================================

left, right = st.columns(
    [1, 1]
)


# ----------------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------------

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "Upload Lip Image / Lip Print"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    st.markdown(
        """
        <p class="small-text">
        Supported inputs:
        full-face image, cropped lip image,
        and collected/scanned lip-print image.
        </p>
        """,
        unsafe_allow_html=True
    )

    input_mode_label = st.selectbox(
        "Input type",
        options=[
            "Auto Detect",
            "Full Face",
            "Lip Print / Lip-only"
        ],
        help=(
            "Auto Detect uses the project's heuristic router. "
            "Manual selection can be used when the image type "
            "is known."
        )
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------
# SYSTEM INFORMATION
# ----------------------------------------------------------------------------

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "System Information"
    )

    st.write(
        "Model: CNN"
    )

    st.write(
        "Tasks: Gender Prediction + Age Group Prediction"
    )

    st.write(
        "Input Size: 256 × 256 × 1"
    )

    st.write(
        "Preprocessing: groove_extract.py"
    )

    st.write(
        "Representation: Aspect-ratio-preserving groove map"
    )

    st.write(
        f"Model Directory: {MODEL_DIR}"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================================
# PROCESS UPLOADED IMAGE
# ============================================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    # ------------------------------------------------------------------------
    # Save temporary image
    # ------------------------------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_path = temp_file.name

        image.save(
            temp_path,
            format="JPEG"
        )

        # --------------------------------------------------------------------
        # Preview
        # --------------------------------------------------------------------

        st.markdown(
            "## Preview"
        )

        preview_left, preview_right = (
            st.columns(2)
        )

        with preview_left:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # --------------------------------------------------------------------
        # Translate UI mode
        # --------------------------------------------------------------------

        if input_mode_label == "Auto Detect":

            actual_mode = "auto"

        elif input_mode_label == "Full Face":

            actual_mode = "face"

        else:

            actual_mode = "lip"

        # --------------------------------------------------------------------
        # Analyze button
        # --------------------------------------------------------------------

        analyze_clicked = st.button(
            "Analyze Lip Pattern"
        )

        if analyze_clicked:

            with st.spinner(
                "Processing image and running CNN prediction..."
            ):

                try:

                    result = predict(
                        temp_path,
                        actual_mode
                    )

                except Exception as exc:

                    st.error(
                        f"Processing failed: {exc}"
                    )

                    result = None

            if result is not None:

                # ============================================================
                # DETECTED LIP
                # ============================================================

                with preview_right:

                    if result["lip_crop"] is not None:

                        st.image(
                            cv2.cvtColor(
                                result["lip_crop"],
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Detected / Input Lip Region",
                            use_container_width=True
                        )

                # ============================================================
                # PIPELINE INFO
                # ============================================================

                st.markdown(
                    "## Processing Information"
                )

                p1, p2, p3 = st.columns(3)

                with p1:

                    st.markdown(
                        f"""
                        <div class="result-box">
                        Input Type<br>
                        {result["pipeline"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with p2:

                    st.markdown(
                        f"""
                        <div class="result-box">
                        Processing Mode<br>
                        {result["mode"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with p3:

                    st.markdown(
                        """
                        <div class="result-box">
                        Model Input<br>
                        256 × 256 × 1
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ============================================================
                # GROOVE VISUALIZATION
                # ============================================================

                st.markdown(
                    "## Groove Extraction Visualization"
                )

                g1, g2, g3 = st.columns(3)

                with g1:

                    if result["groove"] is not None:

                        st.image(
                            result["groove"],
                            caption=(
                                "Native Groove Map"
                            ),
                            use_container_width=True,
                            clamp=True
                        )

                with g2:

                    if result["overlay"] is not None:

                        st.image(
                            cv2.cvtColor(
                                result["overlay"],
                                cv2.COLOR_BGR2RGB
                            ),
                            caption=(
                                "Highlighted Groove Patterns"
                            ),
                            use_container_width=True
                        )

                with g3:

                    st.image(
                        result["model_image"],
                        caption=(
                            "256 × 256 CNN Model Input"
                        ),
                        use_container_width=True,
                        clamp=True
                    )

                # ============================================================
                # OPTIONAL RESPONSE
                # ============================================================

                if result["groove_response"] is not None:

                    with st.expander(
                        "Continuous Groove Response"
                    ):

                        st.image(
                            result[
                                "groove_response"
                            ],
                            caption=(
                                "Continuous groove response "
                                "before final model conversion"
                            ),
                            use_container_width=True,
                            clamp=True
                        )

                # ============================================================
                # PREDICTION RESULTS
                # ============================================================

                st.markdown(
                    "## Prediction Results"
                )

                r1, r2 = st.columns(2)

                with r1:

                    st.markdown(
                        f"""
                        <div class="result-box">
                        Gender<br>
                        {result["gender"]}<br>
                        Confidence:
                        {result["gender_conf"]:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with r2:

                    st.markdown(
                        f"""
                        <div class="result-box">
                        Age Group<br>
                        {result["age"]}<br>
                        Confidence:
                        {result["age_conf"]:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ============================================================
                # CLASS SCORES
                # ============================================================

                st.markdown(
                    "### Class Scores"
                )

                score_left, score_right = st.columns(2)

                with score_left:

                    st.write(
                        "Gender probabilities"
                    )

                    gender_scores = {
                        GENDER_CLASSES[i]:
                            float(
                                result[
                                    "gender_scores"
                                ][i]
                                * 100
                            )
                        for i in range(
                            len(
                                GENDER_CLASSES
                            )
                        )
                    }

                    st.bar_chart(
                        gender_scores
                    )

                with score_right:

                    st.write(
                        "Age-group probabilities"
                    )

                    age_scores = {
                        AGE_CLASSES[i]:
                            float(
                                result[
                                    "age_scores"
                                ][i]
                                * 100
                            )
                        for i in range(
                            len(
                                AGE_CLASSES
                            )
                        )
                    }

                    st.bar_chart(
                        age_scores
                    )

    finally:

        if (
            temp_path is not None
            and
            os.path.exists(
                temp_path
            )
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:
                pass