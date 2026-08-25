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

MODEL_PATH = MODEL_DIR / "best_classification_model.pkl"
ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
target_encoder = joblib.load(ENCODER_PATH)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="EMI Eligibility",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    target_encoder = joblib.load(ENCODER_PATH)

    st.sidebar.success("✅ Model loaded successfully")

except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    st.stop()

st.title("🤖 EMI Eligibility Prediction")
st.write("Enter the applicant's financial and personal information below.")

st.markdown("---")

# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=30.0
    )

    gender = st.selectbox(
    "Gender",
    [0, 1, 2, 3, 4, 5, 6, 7]
)


    marital_status = st.selectbox(
    "Marital Status",
    [0, 1]
)

with col2:
    education = st.selectbox(
    "Education",
    [0, 1, 2, 3]
)

    employment_type = st.selectbox(
    "Employment Type",
    [0, 1, 2]
)

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0
    )

with col3:
    company_type = st.selectbox(
    "Company Type",
    [0, 1, 2, 3, 4]
)

    house_type = st.selectbox(
    "House Type",
    [0, 1, 2]
)

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=2
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.markdown("---")
st.header("💰 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_salary = st.number_input(
        "Monthly Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    monthly_rent = st.number_input(
        "Monthly Rent",
        min_value=0.0,
        value=10000.0,
        step=500.0
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

with col2:
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

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

with col3:
    current_emi_amount = st.number_input(
        "Current EMI Amount",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=700.0
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
        value=30000.0,
        step=1000.0
    )


# ============================================================
# EMI REQUEST
# ============================================================

st.markdown("---")
st.header("💳 EMI Request")

col1, col2, col3 = st.columns(3)

with col1:
    emi_scenario = st.selectbox(
    "EMI Scenario",
    [0, 1, 2, 3, 4]
)

with col2:
    requested_amount = st.number_input(
        "Requested Amount",
        min_value=0.0,
        value=500000.0,
        step=10000.0
    )

with col3:
    requested_tenure = st.number_input(
        "Requested Tenure",
        min_value=1.0,
        max_value=120.0,
        value=36.0
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

total_monthly_expenses = (
    monthly_rent
    + school_fees
    + college_fees
    + travel_expenses
    + groceries_utilities
    + other_monthly_expenses
    + current_emi_amount
)

disposable_income = (
    monthly_salary - total_monthly_expenses
)

if requested_tenure > 0:
    max_monthly_emi = requested_amount / requested_tenure
else:
    max_monthly_emi = 0

if monthly_salary > 0:
    expense_to_income_ratio = (
        total_monthly_expenses / monthly_salary
    )
else:
    expense_to_income_ratio = 0


st.markdown("---")
st.header("📊 Calculated Financial Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Monthly Expenses",
        f"₹{total_monthly_expenses:,.2f}"
    )

with col2:
    st.metric(
        "Disposable Income",
        f"₹{disposable_income:,.2f}"
    )

with col3:
    st.metric(
        "Maximum Monthly EMI",
        f"₹{max_monthly_emi:,.2f}"
    )

with col4:
    st.metric(
        "Expense / Income Ratio",
        f"{expense_to_income_ratio:.2%}"
    )

# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict EMI Eligibility",
    type="primary",
    use_container_width=True
):

    try:

        # Create input dataframe
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "max_monthly_emi": max_monthly_emi,
            "total_monthly_expenses": total_monthly_expenses,
            "disposable_income": disposable_income,
            "expense_to_income_ratio": expense_to_income_ratio
        }])

        # Exact training column order
        input_data = input_data[[
            "age",
            "gender",
            "marital_status",
            "education",
            "monthly_salary",
            "employment_type",
            "years_of_employment",
            "company_type",
            "house_type",
            "monthly_rent",
            "family_size",
            "dependents",
            "school_fees",
            "college_fees",
            "travel_expenses",
            "groceries_utilities",
            "other_monthly_expenses",
            "existing_loans",
            "current_emi_amount",
            "credit_score",
            "bank_balance",
            "emergency_fund",
            "emi_scenario",
            "requested_amount",
            "requested_tenure",
            "max_monthly_emi",
            "total_monthly_expenses",
            "disposable_income",
            "expense_to_income_ratio"
        ]]

        # Show input
        st.subheader("Input Data")
        st.dataframe(
            input_data,
            use_container_width=True
        )

        # Prediction
        prediction = model.predict(input_data)

        # Convert target number back to original label
        prediction_label = target_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        # Result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.success(
            f"EMI Eligibility: **{prediction_label}**"
        )

        # Prediction probabilities
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability_df = pd.DataFrame({
                "Class": target_encoder.classes_,
                "Probability (%)": probabilities * 100
            })

            st.subheader("📊 Prediction Probability")

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            confidence = np.max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )



    

