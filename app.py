import streamlit as st
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


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
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💰 EMIPredict AI")

    st.caption(
        "AI-powered EMI prediction system"
    )

    st.markdown("---")

    st.markdown("### 📌 Modules")

    st.markdown(
        """
        🏠 Home

        📊 Data Exploration

        🤖 Classification

        📈 Regression

        📋 Model Performance

        🔬 MLflow Dashboard
        """
    )

    st.markdown("---")

    st.caption(
        "AI & ML Project"
    )


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()


