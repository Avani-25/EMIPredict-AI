import streamlit as st
import joblib
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "notebooks"


CLASSIFICATION_MODEL_PATH = MODEL_DIR / "best_classification_model.pkl"
REGRESSION_MODEL_PATH = MODEL_DIR / "best_regression_model.pkl"
XGB_REGRESSION_MODEL_PATH = MODEL_DIR / "best_regression_model_xgb.pkl"
REGRESSION_PREPROCESSOR_PATH = MODEL_DIR / "regression_preprocessor.pkl"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)

    regression_model = joblib.load(REGRESSION_MODEL_PATH)

    xgb_regression_model = joblib.load(XGB_REGRESSION_MODEL_PATH)

    regression_preprocessor = joblib.load(
        REGRESSION_PREPROCESSOR_PATH
    )

    target_encoder = joblib.load(
        TARGET_ENCODER_PATH
    )

    return (
        classification_model,
        regression_model,
        xgb_regression_model,
        regression_preprocessor,
        target_encoder
    )


# ============================================================
# SAFE MODEL LOADING
# ============================================================

try:

    (
        classification_model,
        regression_model,
        xgb_regression_model,
        regression_preprocessor,
        target_encoder
    ) = load_models()

except Exception as e:

    st.error("❌ Model loading failed.")

    with st.expander("Show technical details"):
        st.exception(e)

    st.stop()


