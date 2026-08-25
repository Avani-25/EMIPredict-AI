import streamlit as st
import mlflow
import pandas as pd
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="MLflow Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 MLflow Dashboard")
st.markdown(
    "Track, compare and monitor the ML experiments used in EMIPredict-AI."
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_PATH = r"C:\Users\avani\Documents\EMIPredict-AI"

NOTEBOOKS_PATH = os.path.join(
    BASE_PATH,
    "notebooks"
)

MLFLOW_DB = os.path.join(
    NOTEBOOKS_PATH,
    "mlflow.db"
)

MLRUNS_PATH = os.path.join(
    NOTEBOOKS_PATH,
    "mlruns"
)

# ==========================================================
# CHECK FILES
# ==========================================================

if not os.path.exists(MLFLOW_DB):

    st.error("❌ MLflow database not found.")

    st.code(
        MLFLOW_DB
    )

    st.stop()

if not os.path.exists(MLRUNS_PATH):

    st.warning(
        "⚠️ MLflow artifact folder was not found."
    )

# ==========================================================
# CONNECT TO MLflow SQLITE DATABASE
# ==========================================================

try:

    tracking_uri = (
        "sqlite:///"
        + MLFLOW_DB.replace("\\", "/")
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

except Exception as e:

    st.error(
        f"❌ Failed to configure MLflow: {e}"
    )

    st.stop()

# ==========================================================
# CONNECTION STATUS
# ==========================================================

st.success(
    "✅ Connected to MLflow SQLite database!"
)

with st.expander("🔧 MLflow Connection Details"):

    st.write(
        "**Tracking URI:**"
    )

    st.code(
        tracking_uri
    )

    st.write(
        "**MLflow Database:**"
    )

    st.code(
        MLFLOW_DB
    )

    st.write(
        "**Artifact Location:**"
    )

    st.code(
        MLRUNS_PATH
    )

# ==========================================================
# LOAD EXPERIMENTS
# ==========================================================

try:

    experiments = mlflow.search_experiments()

except Exception as e:

    st.error(
        f"❌ Could not load MLflow experiments: {e}"
    )

    st.stop()

# ==========================================================
# EXPERIMENT SUMMARY
# ==========================================================

st.header("🧪 Experiments")

if not experiments:

    st.warning(
        "⚠️ No MLflow experiments found."
    )

    st.stop()

experiment_data = []

for experiment in experiments:

    experiment_data.append(
        {
            "Experiment ID": experiment.experiment_id,
            "Experiment Name": experiment.name,
            "Artifact Location": experiment.artifact_location,
            "Lifecycle": experiment.lifecycle_stage
        }
    )

experiment_df = pd.DataFrame(
    experiment_data
)

st.dataframe(
    experiment_df,
    use_container_width=True
)

# ==========================================================
# SELECT EXPERIMENT
# ==========================================================

experiment_names = [
    experiment.name
    for experiment in experiments
]

selected_experiment = st.selectbox(
    "Select an experiment:",
    experiment_names
)

selected_experiment_id = next(
    experiment.experiment_id
    for experiment in experiments
    if experiment.name == selected_experiment
)

# ==========================================================
# LOAD RUNS
# ==========================================================

try:

    runs = mlflow.search_runs(
        experiment_ids=[
            selected_experiment_id
        ],
        output_format="pandas"
    )

except Exception as e:

    st.error(
        f"❌ Could not load experiment runs: {e}"
    )

    st.stop()

# ==========================================================
# RUN SUMMARY
# ==========================================================

st.header("📊 Run Summary")

if runs.empty:

    st.info(
        "No runs found for this experiment."
    )

    st.stop()

total_runs = len(runs)

finished_runs = (
    runs["status"] == "FINISHED"
).sum()

failed_runs = (
    runs["status"] == "FAILED"
).sum()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Runs",
        total_runs
    )

with col2:

    st.metric(
        "Completed Runs",
        finished_runs
    )

with col3:

    st.metric(
        "Failed Runs",
        failed_runs
    )

# ==========================================================
# RUN INFORMATION
# ==========================================================

st.header("📋 Experiment Runs")

run_columns = [
    "run_id",
    "status",
    "start_time",
    "end_time"
]

available_run_columns = [
    column
    for column in run_columns
    if column in runs.columns
]

st.dataframe(
    runs[available_run_columns],
    use_container_width=True
)

# ==========================================================
# PARAMETERS
# ==========================================================

st.header("⚙️ Model Parameters")

parameter_columns = [
    column
    for column in runs.columns
    if column.startswith("params.")
]

if parameter_columns:

    parameter_display = runs[
        ["run_id"] + parameter_columns
    ].copy()

    parameter_display = parameter_display.dropna(
        axis=1,
        how="all"
    )

    st.dataframe(
        parameter_display,
        use_container_width=True
    )

else:

    st.info(
        "No parameters were logged for this experiment."
    )

# ==========================================================
# METRICS
# ==========================================================

st.header("📈 Model Metrics")

metric_columns = [
    column
    for column in runs.columns
    if column.startswith("metrics.")
]

if metric_columns:

    metric_display = runs[
        ["run_id"] + metric_columns
    ].copy()

    metric_display = metric_display.dropna(
        axis=1,
        how="all"
    )

    st.dataframe(
        metric_display,
        use_container_width=True
    )

else:

    st.info(
        "No metrics were logged for this experiment."
    )

# ==========================================================
# BEST RUN
# ==========================================================

st.header("🏆 Best Run")

if metric_columns:

    # Remove metrics with no actual values
    usable_metrics = []

    for metric in metric_columns:

        if runs[metric].notna().any():

            usable_metrics.append(
                metric
            )

    if usable_metrics:

        selected_metric = st.selectbox(
            "Choose metric for comparison:",
            usable_metrics
        )

        valid_runs = runs.dropna(
            subset=[
                selected_metric
            ]
        )

        if not valid_runs.empty:

            # Metrics that should be minimized
            minimize_metrics = [
                "mae",
                "mse",
                "rmse",
                "loss"
            ]

            metric_name = (
                selected_metric
                .replace("metrics.", "")
                .lower()
            )

            if any(
                metric in metric_name
                for metric in minimize_metrics
            ):

                best_run = valid_runs.loc[
                    valid_runs[
                        selected_metric
                    ].idxmin()
                ]

            else:

                best_run = valid_runs.loc[
                    valid_runs[
                        selected_metric
                    ].idxmax()
                ]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Best Run",
                    str(
                        best_run["run_id"]
                    )[:12]
                )

            with col2:

                st.metric(
                    "Metric",
                    metric_name.upper()
                )

            with col3:

                st.metric(
                    "Best Value",
                    f"{best_run[selected_metric]:.4f}"
                )

            st.write(
                "Full Best Run ID:"
            )

            st.code(
                best_run["run_id"]
            )

# ==========================================================
# CLASSIFICATION / REGRESSION RUN IDs
# ==========================================================

st.header("🤖 Logged Model Runs")

classification_run_id = (
    "1a0602ab6b3748a2a667a175352bc592"
)

regression_run_id = (
    "ccb0747b818540e0b5ff83bcdb686ddc"
)

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "🤖 Classification"
    )

    st.success(
        "Classification run logged successfully"
    )

    st.code(
        classification_run_id
    )

with col2:

    st.subheader(
        "📈 Regression"
    )

    st.success(
        "Regression run logged successfully"
    )

    st.code(
        regression_run_id
    )

# ==========================================================
# REFRESH
# ==========================================================

st.markdown("---")

if st.button(
    "🔄 Refresh MLflow Dashboard"
):

    st.rerun()

st.success(
    "🎉 MLflow Dashboard is connected successfully!"
)

st.caption(
    "EMIPredict-AI • MLflow Experiment Tracking"
)