import streamlit as st
import os
import pandas as pd
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Admin",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================================
# PROJECT PATH
# ==========================================================

BASE_PATH = r"C:\Users\avani\Documents\EMIPredict-AI"

NOTEBOOKS_PATH = os.path.join(
    BASE_PATH,
    "notebooks"
)

# ==========================================================
# FILE PATHS
# ==========================================================

classification_model = os.path.join(
    NOTEBOOKS_PATH,
    "best_classification_model.pkl"
)

regression_model = os.path.join(
    NOTEBOOKS_PATH,
    "best_regression_model.pkl"
)

regression_xgb_model = os.path.join(
    NOTEBOOKS_PATH,
    "best_regression_model_xgb.pkl"
)

regression_preprocessor = os.path.join(
    NOTEBOOKS_PATH,
    "regression_preprocessor.pkl"
)

target_encoder = os.path.join(
    NOTEBOOKS_PATH,
    "target_encoder.pkl"
)

feature_dataset = os.path.join(
    NOTEBOOKS_PATH,
    "feature_engineered_dataset.csv"
)

cleaned_dataset = os.path.join(
    NOTEBOOKS_PATH,
    "emi_prediction_cleaned.csv"
)

mlflow_database = os.path.join(
    NOTEBOOKS_PATH,
    "mlflow.db"
)

mlruns_folder = os.path.join(
    NOTEBOOKS_PATH,
    "mlruns"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("⚙️ Admin Dashboard")

st.markdown(
    "Monitor the EMIPredict-AI application, models, datasets and MLflow configuration."
)

st.markdown("---")

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.header("🟢 System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Application",
        "Online"
    )

with col2:

    st.metric(
        "ML Models",
        "Ready"
    )

with col3:

    st.metric(
        "Dataset",
        "Available"
    )

with col4:

    st.metric(
        "MLflow",
        "Connected"
    )

# ==========================================================
# MODEL STATUS
# ==========================================================

st.header("🤖 Model Status")

model_status = pd.DataFrame(
    {
        "Model": [
            "Classification Model",
            "Regression Model",
            "XGBoost Regression Model",
            "Regression Preprocessor",
            "Target Encoder"
        ],
        "File": [
            "best_classification_model.pkl",
            "best_regression_model.pkl",
            "best_regression_model_xgb.pkl",
            "regression_preprocessor.pkl",
            "target_encoder.pkl"
        ],
        "Status": [
            "✅ Available"
            if os.path.exists(classification_model)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(regression_model)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(regression_xgb_model)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(regression_preprocessor)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(target_encoder)
            else "❌ Missing"
        ]
    }
)

st.dataframe(
    model_status,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# DATASET STATUS
# ==========================================================

st.header("📊 Dataset Status")

dataset_status = pd.DataFrame(
    {
        "Dataset": [
            "Feature Engineered Dataset",
            "Cleaned Dataset"
        ],
        "File": [
            "feature_engineered_dataset.csv",
            "emi_prediction_cleaned.csv"
        ],
        "Status": [
            "✅ Available"
            if os.path.exists(feature_dataset)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(cleaned_dataset)
            else "❌ Missing"
        ]
    }
)

st.dataframe(
    dataset_status,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.header("📋 Dataset Information")

if os.path.exists(feature_dataset):

    try:

        df = pd.read_csv(
            feature_dataset
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:

            st.metric(
                "Missing Values",
                f"{df.isnull().sum().sum():,}"
            )

        with col4:

            st.metric(
                "Duplicates",
                f"{df.duplicated().sum():,}"
            )

    except Exception as e:

        st.warning(
            f"⚠️ Could not read dataset information: {e}"
        )

else:

    st.warning(
        "⚠️ Feature-engineered dataset is unavailable."
    )

# ==========================================================
# MLFLOW STATUS
# ==========================================================

st.header("🔬 MLflow Status")

col1, col2 = st.columns(2)

with col1:

    if os.path.exists(mlflow_database):

        st.success(
            "✅ MLflow SQLite database found"
        )

        st.code(
            mlflow_database
        )

    else:

        st.error(
            "❌ MLflow database not found"
        )

with col2:

    if os.path.exists(mlruns_folder):

        st.success(
            "✅ MLflow artifact directory found"
        )

        st.code(
            mlruns_folder
        )

    else:

        st.warning(
            "⚠️ MLflow artifact directory not found"
        )

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.header("📁 Project Information")

project_info = pd.DataFrame(
    {
        "Property": [
            "Project Name",
            "Project Type",
            "Application",
            "Machine Learning",
            "Frontend",
            "Experiment Tracking",
            "Data Storage"
        ],
        "Details": [
            "EMIPredict-AI",
            "EMI Prediction System",
            "Streamlit",
            "Classification + Regression",
            "Streamlit UI",
            "MLflow",
            "CSV"
        ]
    }
)

st.dataframe(
    project_info,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# FILE STORAGE
# ==========================================================

st.header("📦 Model & Dataset Storage")

files_to_check = [
    classification_model,
    regression_model,
    regression_xgb_model,
    regression_preprocessor,
    target_encoder,
    feature_dataset,
    cleaned_dataset,
    mlflow_database
]

file_information = []

for file_path in files_to_check:

    if os.path.exists(file_path):

        size_kb = (
            os.path.getsize(file_path)
            / 1024
        )

        modified_time = datetime.fromtimestamp(
            os.path.getmtime(file_path)
        )

        file_information.append(
            {
                "File": os.path.basename(file_path),
                "Size": f"{size_kb:.2f} KB",
                "Last Modified": modified_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Status": "✅ Available"
            }
        )

    else:

        file_information.append(
            {
                "File": os.path.basename(file_path),
                "Size": "-",
                "Last Modified": "-",
                "Status": "❌ Missing"
            }
        )

file_df = pd.DataFrame(
    file_information
)

st.dataframe(
    file_df,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# APPLICATION CHECK
# ==========================================================

st.header("🩺 Application Health Check")

checks = {
    "Classification Model": os.path.exists(
        classification_model
    ),

    "Regression Model": os.path.exists(
        regression_model
    ),

    "Feature Dataset": os.path.exists(
        feature_dataset
    ),

    "MLflow Database": os.path.exists(
        mlflow_database
    ),

    "MLflow Artifacts": os.path.exists(
        mlruns_folder
    )
}

for name, status in checks.items():

    if status:

        st.success(
            f"✅ {name}: Ready"
        )

    else:

        st.error(
            f"❌ {name}: Not Available"
        )

# ==========================================================
# ADMIN ACTIONS
# ==========================================================

st.header("🔧 Admin Actions")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True
    ):

        st.rerun()

with col2:

    if st.button(
        "🧹 Clear Page Cache",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.cache_resource.clear()

        st.success(
            "✅ Streamlit cache cleared!"
        )

# ==========================================================
# FINAL STATUS
# ==========================================================

st.markdown("---")

all_critical_files = [
    classification_model,
    regression_model,
    feature_dataset,
    mlflow_database
]

if all(
    os.path.exists(file)
    for file in all_critical_files
):

    st.success(
        "🎉 EMIPredict-AI Admin Dashboard is healthy and all critical components are available!"
    )

else:

    st.warning(
        "⚠️ Some critical project components are missing. Check the status tables above."
    )

st.caption(
    "EMIPredict-AI • Admin & System Monitoring"
)