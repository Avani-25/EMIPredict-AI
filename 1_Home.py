import streamlit as st

st.set_page_config(
    page_title="Home - EMIPredict AI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Home")

st.header("Welcome to EMIPredict AI")

st.write("""
This application provides AI-powered financial prediction capabilities.
""")

st.markdown("### Available Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 EMI Eligibility")
    st.write(
        "Use the classification model to determine "
        "whether an applicant is eligible for EMI."
    )

with col2:
    st.subheader("📈 EMI Prediction")
    st.write(
        "Use the regression model to estimate "
        "the applicant's EMI amount."
    )

st.markdown("---")

st.subheader("Technology Stack")

st.write("""
- Python
- Scikit-learn
- XGBoost
- Streamlit
- MLflow
- Pandas
- NumPy
- Matplotlib
""")