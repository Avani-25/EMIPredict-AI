import streamlit as st
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

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.2rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    /* Feature cards */
    .card {
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
        min-height: 170px;
    }

    .card h3 {
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK PAGE FILES
# ============================================================

required_pages = {
    "Home": "1_Home.py",
    "Data Exploration": "2_Data_Exploration.py",
    "Classification": "3_Classification.py",
    "Regression": "4_Regression.py",
    "Model Performance": "5_Model_Performance.py",
    "MLflow Dashboard": "6_MLflow_Dashboard.py",
}


missing_pages = []

for page_name, filename in required_pages.items():

    page_path = PAGES_DIR / filename

    if not page_path.exists():
        missing_pages.append(filename)


if missing_pages:

    st.error("❌ Some Streamlit pages are missing.")

    st.write("Missing files:")

    for filename in missing_pages:
        st.write(f"- `{filename}`")

    st.info(
        "Please make sure all page files are inside "
        "`streamlit_app/pages/`."
    )

    st.stop()


# ============================================================
# STREAMLIT NAVIGATION
# ============================================================

pages = [

    st.Page(
        str(PAGES_DIR / "1_Home.py"),
        title="Home",
        icon="🏠",
        default=True
    ),

    st.Page(
        str(PAGES_DIR / "2_Data_Exploration.py"),
        title="Data Exploration",
        icon="📊"
    ),

    st.Page(
        str(PAGES_DIR / "3_Classification.py"),
        title="Classification",
        icon="🤖"
    ),

    st.Page(
        str(PAGES_DIR / "4_Regression.py"),
        title="Regression",
        icon="📈"
    ),

    st.Page(
        str(PAGES_DIR / "5_Model_Performance.py"),
        title="Model Performance",
        icon="📋"
    ),

    st.Page(
        str(PAGES_DIR / "6_MLflow_Dashboard.py"),
        title="MLflow Dashboard",
        icon="🔬"
    ),
]


# ============================================================
# RUN NAVIGATION
# ============================================================

pg = st.navigation(pages)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💰 EMIPredict AI")

    st.caption(
        "AI-powered EMI eligibility and prediction system"
    )

    st.markdown("---")

    st.markdown("### 📌 Project Modules")

    st.markdown(
        """
        🏠 **Home**  
        📊 **Data Exploration**  
        🤖 **Classification**  
        📈 **Regression**  
        📋 **Model Performance**  
        🔬 **MLflow Dashboard**
        """
    )

    st.markdown("---")

    st.caption(
        "AI & ML Project • Streamlit"
    )


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()


