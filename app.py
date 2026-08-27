import streamlit as st
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# EMIPredict-AI/streamlit_app/app.py
#
# parent       = streamlit_app
# parent.parent = EMIPredict-AI

BASE_DIR = Path(__file__).resolve().parent

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
PAGES_DIR = Path(__file__).resolve().parent / "pages"


# ============================================================
# DATA PATHS
# ============================================================

DATA_PATH = NOTEBOOKS_DIR / "feature_engineered_dataset.csv"

MLFLOW_DB_PATH = NOTEBOOKS_DIR / "mlflow.db"


# ============================================================
# MODEL PATHS
# ============================================================

CLASSIFICATION_MODEL_PATH = (
    NOTEBOOKS_DIR / "best_classification_model.pkl"
)

REGRESSION_MODEL_PATH = (
    NOTEBOOKS_DIR / "best_regression_model.pkl"
)

XGB_REGRESSION_MODEL_PATH = (
    NOTEBOOKS_DIR / "best_regression_model_xgb.pkl"
)

REGRESSION_PREPROCESSOR_PATH = (
    NOTEBOOKS_DIR / "regression_preprocessor.pkl"
)

TARGET_ENCODER_PATH = (
    NOTEBOOKS_DIR / "target_encoder.pkl"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    xgb_regression_model = joblib.load(
        XGB_REGRESSION_MODEL_PATH
    )

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
# NAVIGATION
# ============================================================

pg = st.navigation(
    [
        st.Page(
            PAGES_DIR / "1_Home.py",
            title="Home",
            icon="🏠"
        ),

        st.Page(
            PAGES_DIR / "2_Data_Exploration.py",
            title="Data Exploration",
            icon="📊"
        ),

        st.Page(
            PAGES_DIR / "3_Classification.py",
            title="Classification",
            icon="🤖"
        ),

        st.Page(
            PAGES_DIR / "4_Regression.py",
            title="Regression",
            icon="📈"
        ),

        st.Page(
            PAGES_DIR / "5_Model_Performance.py",
            title="Model Performance",
            icon="📋"
        ),

        st.Page(
            PAGES_DIR / "6_MLflow_Dashboard.py",
            title="MLflow Dashboard",
            icon="🔬"
        ),

        st.Page(
            PAGES_DIR / "7_Admin.py",
            title="Admin",
            icon="⚙️"
        ),
    ],
    position="sidebar"
)


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "💰 EMIPredict AI • Machine Learning Application"
)


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()


