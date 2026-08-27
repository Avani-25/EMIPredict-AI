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

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation(
    [
        st.Page("1_Home.py", title="Home", icon="🏠"),
        st.Page("2_Data_Exploration.py", title="Data Exploration", icon="📊"),
        st.Page("3_Classification.py", title="Classification", icon="🤖"),
        st.Page("4_Regression.py", title="Regression", icon="📈"),
        st.Page("5_Model_Performance.py", title="Model Performance", icon="📋"),
        st.Page("6_MLflow_Dashboard.py", title="MLflow Dashboard", icon="🔬"),
        st.Page("7_Admin.py", title="Admin", icon="⚙️"),
    ],
    position="sidebar"
)

st.sidebar.markdown("---")
st.sidebar.caption("💰 EMIPredict AI • Machine Learning Application")

pg.run()


