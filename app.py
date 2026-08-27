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
# HEADER
# ============================================================

st.title("💰 EMIPredict AI")

st.subheader(
    "AI-Powered EMI Eligibility & Prediction System"
)

st.markdown("---")


# ============================================================
# WELCOME SECTION
# ============================================================

st.header("Welcome 👋")

st.write(
    """
    EMIPredict AI is an intelligent financial prediction application
    designed to assist with EMI eligibility assessment and EMI prediction.
    
    Use the navigation menu on the left to explore data, make predictions,
    evaluate models, and monitor MLflow experiments.
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 Classification")
    st.write(
        "Predict whether an applicant is eligible for EMI."
    )

with col2:
    st.info("📈 Regression")
    st.write(
        "Predict the expected EMI amount using machine learning."
    )

with col3:
    st.info("🔬 MLflow")
    st.write(
        "Monitor experiments and compare model performance."
    )


# ============================================================
# ADDITIONAL FEATURES
# ============================================================

st.markdown("---")

st.header("🚀 Application Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.write("✅ EMI Eligibility Classification")
    st.write("✅ EMI Amount Prediction")
    st.write("✅ Data Exploration")
    st.write("✅ Model Performance Analysis")

with feature_col2:
    st.write("✅ MLflow Experiment Tracking")
    st.write("✅ Admin Dashboard")
    st.write("✅ Interactive Streamlit Interface")
    st.write("✅ Production-Ready Prediction Workflow")


# ============================================================
# FOOTER / STATUS
# ============================================================

st.markdown("---")

st.success("🟢 EMIPredict AI is ready!")

st.sidebar.title("📌 Navigation")

st.sidebar.write(
    """
    Use the navigation menu above to explore:
pg = st.navigation(
    [
        st.Page("1_Home.py", title="Home", icon="🏠"),
        st.Page("2_Data_Exploration.py", title="Data Exploration", icon="📊"),
        st.Page("3_Classification.py", title="Classification", icon="🤖"),
        st.Page("4_Regression.py", title="Regression", icon="📈"),
        st.Page("5_Model_Performance.py", title="Model Performance", icon="📋"),
        st.Page("6_MLflow_Dashboard.py", title="MLflow Dashboard", icon="🔬"),
        st.Page("7_Admin.py", title="Admin", icon="⚙️"),
        """
    ],
    position="sidebar"
)
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "EMIPredict AI • Machine Learning Application"
)

