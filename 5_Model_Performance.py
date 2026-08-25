import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Model Performance")
st.markdown(
    "Evaluate and visualize the performance of the trained EMI prediction models."
)

# ==========================================================
# FILE PATHS
# ==========================================================

BASE_PATH = r"C:\Users\avani\Documents\EMIPredict-AI"

DATA_PATH = os.path.join(
    BASE_PATH,
    "notebooks",
    "feature_engineered_dataset.csv"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_PATH,
    "notebooks",
    "best_classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_PATH,
    "notebooks",
    "best_regression_model.pkl"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(f"❌ Dataset loading failed: {e}")
    st.stop()

# ==========================================================
# LOAD CLASSIFICATION MODEL
# ==========================================================

try:

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    classification_loaded = True

except Exception as e:

    classification_loaded = False

    st.warning(
        f"⚠️ Classification model could not be loaded: {e}"
    )

# ==========================================================
# LOAD REGRESSION MODEL
# ==========================================================

try:

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    regression_loaded = True

except Exception as e:

    regression_loaded = False

    st.warning(
        f"⚠️ Regression model could not be loaded: {e}"
    )

# ==========================================================
# PAGE TABS
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "🤖 Classification Performance",
        "📈 Regression Performance"
    ]
)

# ==========================================================
# CLASSIFICATION
# ==========================================================

with tab1:

    st.header("🤖 Classification Model Performance")

    if not classification_loaded:

        st.error(
            "❌ Classification model is unavailable."
        )

    else:

        # --------------------------------------------------
        # FIND TARGET
        # --------------------------------------------------

        classification_targets = [
            "emi_eligibility",
            "target",
            "Target",
            "label",
            "Label"
        ]

        target_column = None

        for col in classification_targets:

            if col in df.columns:

                target_column = col
                break

        if target_column is None:

            st.error(
                "❌ Classification target column was not found."
            )

            st.write(
                "Available columns:"
            )

            st.write(
                list(df.columns)
            )

        else:

            st.success(
                f"🎯 Classification target: `{target_column}`"
            )

            # --------------------------------------------------
            # PREPARE DATA
            # --------------------------------------------------

            X_class = df.drop(
                columns=[target_column]
            )

            y_class = df[target_column]

            # Keep numeric columns
            X_class = X_class.select_dtypes(
                include=np.number
            )

            # Remove possible NaN / infinity
            X_class = X_class.replace(
                [np.inf, -np.inf],
                np.nan
            )

            valid_rows = (
                X_class.notna().all(axis=1)
                & y_class.notna()
            )

            X_class = X_class.loc[valid_rows]
            y_class = y_class.loc[valid_rows]

            # --------------------------------------------------
            # ALIGN FEATURES WITH MODEL
            # --------------------------------------------------

            try:

                if hasattr(
                    classification_model,
                    "feature_names_in_"
                ):

                    required_features = (
                        classification_model.feature_names_in_
                    )

                    missing_features = [
                        col
                        for col in required_features
                        if col not in X_class.columns
                    ]

                    if missing_features:

                        st.error(
                            "❌ Required classification features are missing."
                        )

                        st.write(
                            missing_features
                        )

                    else:

                        X_class = X_class[
                            required_features
                        ]

                        # Prediction
                        y_pred = (
                            classification_model.predict(
                                X_class
                            )
                        )

                        # --------------------------------------------------
                        # METRICS
                        # --------------------------------------------------

                        accuracy = accuracy_score(
                            y_class,
                            y_pred
                        )

                        precision = precision_score(
                            y_class,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )

                        recall = recall_score(
                            y_class,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )

                        f1 = f1_score(
                            y_class,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )

                        # --------------------------------------------------
                        # METRIC CARDS
                        # --------------------------------------------------

                        st.subheader(
                            "📊 Classification Metrics"
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:

                            st.metric(
                                "Accuracy",
                                f"{accuracy * 100:.2f}%"
                            )

                        with col2:

                            st.metric(
                                "Precision",
                                f"{precision * 100:.2f}%"
                            )

                        with col3:

                            st.metric(
                                "Recall",
                                f"{recall * 100:.2f}%"
                            )

                        with col4:

                            st.metric(
                                "F1 Score",
                                f"{f1 * 100:.2f}%"
                            )

                        # --------------------------------------------------
                        # CONFUSION MATRIX
                        # --------------------------------------------------

                        st.subheader(
                            "🔥 Confusion Matrix"
                        )

                        cm = confusion_matrix(
                            y_class,
                            y_pred
                        )

                        fig, ax = plt.subplots(
                            figsize=(7, 5)
                        )

                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            ax=ax
                        )

                        ax.set_xlabel(
                            "Predicted"
                        )

                        ax.set_ylabel(
                            "Actual"
                        )

                        ax.set_title(
                            "Classification Confusion Matrix"
                        )

                        st.pyplot(fig)

                        plt.close(fig)

                        # --------------------------------------------------
                        # CLASSIFICATION REPORT
                        # --------------------------------------------------

                        st.subheader(
                            "📋 Classification Report"
                        )

                        report = classification_report(
                            y_class,
                            y_pred,
                            output_dict=True,
                            zero_division=0
                        )

                        report_df = pd.DataFrame(
                            report
                        ).transpose()

                        st.dataframe(
                            report_df.round(4),
                            use_container_width=True
                        )

                        st.success(
                            "✅ Classification performance calculated successfully!"
                        )

                else:

                    st.warning(
                        "⚠️ The classification model does not expose feature names."
                    )

            except Exception as e:

                st.error(
                    f"❌ Classification evaluation failed: {e}"
                )


# ==========================================================
# REGRESSION
# ==========================================================

with tab2:

    st.header("📈 Regression Model Performance")

    if not regression_loaded:

        st.error(
            "❌ Regression model is unavailable."
        )

    else:

        st.info(
            "Regression performance is calculated using the saved regression model."
        )

        # --------------------------------------------------
        # FIND POSSIBLE REGRESSION TARGET
        # --------------------------------------------------

        regression_targets = [
            "emi_amount",
            "emi",
            "monthly_emi",
            "loan_amount",
            "target",
            "Target"
        ]

        regression_target = None

        for col in regression_targets:

            if col in df.columns:

                regression_target = col
                break

        # --------------------------------------------------
        # IF TARGET FOUND
        # --------------------------------------------------

        if regression_target is None:

            st.warning(
                "⚠️ Regression target could not be automatically detected."
            )

            st.write(
                "Available columns:"
            )

            st.write(
                list(df.columns)
            )

            regression_target = st.selectbox(
                "Select regression target:",
                df.columns,
                key="regression_target"
            )

        st.success(
            f"🎯 Regression target: `{regression_target}`"
        )

        # --------------------------------------------------
        # PREPARE DATA
        # --------------------------------------------------

        X_reg = df.drop(
            columns=[regression_target]
        )

        y_reg = df[regression_target]

        X_reg = X_reg.select_dtypes(
            include=np.number
        )

        X_reg = X_reg.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid_rows = (
            X_reg.notna().all(axis=1)
            & y_reg.notna()
        )

        X_reg = X_reg.loc[valid_rows]
        y_reg = y_reg.loc[valid_rows]

        # --------------------------------------------------
        # ALIGN FEATURES
        # --------------------------------------------------

        try:

            if hasattr(
                regression_model,
                "feature_names_in_"
            ):

                required_features = (
                    regression_model.feature_names_in_
                )

                missing_features = [
                    col
                    for col in required_features
                    if col not in X_reg.columns
                ]

                if missing_features:

                    st.warning(
                        "⚠️ Some regression model features are not directly available in the feature-engineered dataset."
                    )

                    st.write(
                        "Missing features:"
                    )

                    st.write(
                        missing_features
                    )

                    st.info(
                        "The regression model uses a preprocessing pipeline. "
                        "Your existing Regression Prediction page remains the correct place for live predictions."
                    )

                else:

                    X_reg = X_reg[
                        required_features
                    ]

                    y_pred_reg = regression_model.predict(
                        X_reg
                    )

                    # --------------------------------------------------
                    # REGRESSION METRICS
                    # --------------------------------------------------

                    mae = mean_absolute_error(
                        y_reg,
                        y_pred_reg
                    )

                    mse = mean_squared_error(
                        y_reg,
                        y_pred_reg
                    )

                    rmse = np.sqrt(mse)

                    r2 = r2_score(
                        y_reg,
                        y_pred_reg
                    )

                    # --------------------------------------------------
                    # METRIC CARDS
                    # --------------------------------------------------

                    st.subheader(
                        "📊 Regression Metrics"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "MAE",
                            f"{mae:.4f}"
                        )

                    with col2:

                        st.metric(
                            "MSE",
                            f"{mse:.4f}"
                        )

                    with col3:

                        st.metric(
                            "RMSE",
                            f"{rmse:.4f}"
                        )

                    with col4:

                        st.metric(
                            "R² Score",
                            f"{r2:.4f}"
                        )

                    # --------------------------------------------------
                    # ACTUAL VS PREDICTED
                    # --------------------------------------------------

                    st.subheader(
                        "🎯 Actual vs Predicted"
                    )

                    fig, ax = plt.subplots(
                        figsize=(9, 6)
                    )

                    ax.scatter(
                        y_reg,
                        y_pred_reg,
                        alpha=0.4
                    )

                    min_value = min(
                        y_reg.min(),
                        y_pred_reg.min()
                    )

                    max_value = max(
                        y_reg.max(),
                        y_pred_reg.max()
                    )

                    ax.plot(
                        [min_value, max_value],
                        [min_value, max_value],
                        linestyle="--"
                    )

                    ax.set_xlabel(
                        "Actual Values"
                    )

                    ax.set_ylabel(
                        "Predicted Values"
                    )

                    ax.set_title(
                        "Actual vs Predicted Values"
                    )

                    st.pyplot(fig)

                    plt.close(fig)

                    st.success(
                        "✅ Regression performance calculated successfully!"
                    )

            else:

                st.warning(
                    "⚠️ Regression model does not expose feature names."
                )

        except Exception as e:

            st.warning(
                f"⚠️ Regression performance could not be calculated directly: {e}"
            )

            st.info(
                "This does not affect your Regression Prediction page. "
                "The saved regression model and preprocessing pipeline are still used there."
            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "EMIPredict-AI • Model Performance Dashboard"
)