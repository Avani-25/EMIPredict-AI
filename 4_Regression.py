import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_DIR / "notebooks"

MODEL_PATH = MODEL_DIR / "best_regression_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "regression_preprocessor.pkl"


# ============================================================
# LOAD TRAINED REGRESSION MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    st.sidebar.success("✅ Regression model loaded successfully")

except Exception as e:

    st.sidebar.error(
        f"❌ Regression model loading failed: {e}"
    )

    st.stop()

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EMI Prediction",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

try:
    regression_model = joblib.load(MODEL_PATH)
    regression_preprocessor = joblib.load(PREPROCESSOR_PATH)

except Exception as e:
    st.error(f"❌ Regression model loading failed: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("💰 EMI Amount Prediction")
st.write(
    "Enter the applicant details below to generate a regression prediction."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "High School",
            "Post Graduate",
            "Professional"
        ]
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Government",
            "Private",
            "Self-employed"
        ]
    )

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Large Indian",
            "MNC",
            "Mid-size",
            "Small",
            "Startup"
        ]
    )

    house_type = st.selectbox(
        "House Type",
        [
            "Family",
            "Own",
            "Rented"
        ]
    )


with col2:

    monthly_salary = st.number_input(
        "Monthly Salary",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    monthly_rent = st.number_input(
        "Monthly Rent",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    school_fees = st.number_input(
        "School Fees",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    college_fees = st.number_input(
        "College Fees",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    travel_expenses = st.number_input(
        "Travel Expenses",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    groceries_utilities = st.number_input(
        "Groceries & Utilities",
        min_value=0.0,
        value=8000.0,
        step=500.0
    )


with col3:

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    existing_loans = st.selectbox(
        "Existing Loans",
        ["No", "Yes"]
    )

    current_emi_amount = st.number_input(
        "Current EMI Amount",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
        step=1
    )

    bank_balance = st.number_input(
        "Bank Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    emergency_fund = st.number_input(
        "Emergency Fund",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Education EMI",
            "Home Appliances EMI",
            "Personal Loan EMI",
            "Vehicle EMI"
        ]
    )

    requested_amount = st.number_input(
        "Requested Amount",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )

    requested_tenure = st.number_input(
        "Requested Tenure",
        min_value=1,
        max_value=120,
        value=24,
        step=1
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "age": [age],

    "gender": [gender],

    "marital_status": [marital_status],

    "education": [education],

    "monthly_salary": [monthly_salary],

    "employment_type": [employment_type],

    "years_of_employment": [years_of_employment],

    "company_type": [company_type],

    "house_type": [house_type],

    "monthly_rent": [monthly_rent],

    "family_size": [family_size],

    "dependents": [dependents],

    "school_fees": [school_fees],

    "college_fees": [college_fees],

    "travel_expenses": [travel_expenses],

    "groceries_utilities": [groceries_utilities],

    "other_monthly_expenses": [other_monthly_expenses],

    "existing_loans": [existing_loans],

    "current_emi_amount": [current_emi_amount],

    "credit_score": [credit_score],

    "bank_balance": [bank_balance],

    "emergency_fund": [emergency_fund],

    "emi_scenario": [emi_scenario],

    "requested_amount": [requested_amount],

    "requested_tenure": [requested_tenure]
})


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict",
    type="primary",
    width="stretch"
):

    try:

        # Transform using the SAME preprocessor
        transformed_data = regression_preprocessor.transform(
            input_data
        )

        # Generate prediction
        prediction = regression_model.predict(
            transformed_data
        )

        prediction_value = float(prediction[0])

        st.success("✅ Prediction generated successfully!")

        st.metric(
            label="Predicted EMI Amount",
            value=f"₹ {prediction_value:,.2f}"
        )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


