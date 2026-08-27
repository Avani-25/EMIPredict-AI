import streamlit as st

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
# NAVIGATION
# ============================================================

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


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption("💰 EMIPredict AI • Machine Learning Application")


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()


