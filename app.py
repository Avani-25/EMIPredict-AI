import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# Folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Page files
pages = [
    st.Page(
        str(BASE_DIR / "1_Home.py"),
        title="Home",
        icon="🏠"
    ),
    st.Page(
        str(BASE_DIR / "2_Data_Exploration.py"),
        title="Data Exploration",
        icon="📊"
    ),
    st.Page(
        str(BASE_DIR / "3_Classification.py"),
        title="Classification",
        icon="🤖"
    ),
    st.Page(
        str(BASE_DIR / "4_Regression.py"),
        title="Regression",
        icon="📈"
    ),
    st.Page(
        str(BASE_DIR / "5_Model_Performance.py"),
        title="Model Performance",
        icon="📋"
    ),
    st.Page(
        str(BASE_DIR / "6_MLflow_Dashboard.py"),
        title="MLflow Dashboard",
        icon="🔬"
    ),
    st.Page(
        str(BASE_DIR / "7_Admin.py"),
        title="Admin",
        icon="⚙️"
    ),
]

pg = st.navigation(pages)

pg.run()


