import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PAGE PATH
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "pages"

# --------------------------------------------------
# STREAMLIT NAVIGATION
# --------------------------------------------------
pages = [
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
    )
]

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
pg = st.navigation(
    pages,
    position="sidebar",
    expanded=True
)

# --------------------------------------------------
# SIDEBAR FOOTER
# --------------------------------------------------
st.sidebar.markdown("---")

st.sidebar.caption(
    "💰 EMIPredict AI • Machine Learning Application"
)

# --------------------------------------------------
# RUN SELECTED PAGE
# --------------------------------------------------
pg.run()

