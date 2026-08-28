import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Data Exploration",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Exploration")
st.markdown(
    "Explore the feature-engineered dataset used for the EMI prediction project."
)

# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

DATA_PATH = r"C:\Users\avani\Documents\EMIPredict-AI\notebooks\featureEngineered_sample.csv"

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)

    st.success("✅ Feature-engineered sample dataset loaded successfully!")

except Exception as e:
    st.error(f"❌ Dataset loading failed: {e}")
    st.stop()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Total Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

with col4:
    st.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.header("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------------------
# COLUMN INFORMATION
# --------------------------------------------------

st.header("🔎 Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

st.dataframe(
    column_info,
    use_container_width=True
)

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

st.header("❌ Missing Values")

missing_df = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_df.columns = [
    "Column",
    "Missing Values"
]

missing_df = missing_df[
    missing_df["Missing Values"] > 0
]

if missing_df.empty:

    st.success(
        "✅ No missing values found in the dataset."
    )

else:

    st.dataframe(
        missing_df,
        use_container_width=True
    )

# --------------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------------

st.header("🎯 Target Distribution")

target_candidates = [
    "emi_eligibility",
    "target",
    "Target",
    "label",
    "Label"
]

target_column = None

for column in target_candidates:
    if column in df.columns:
        target_column = column
        break

if target_column is None:

    st.warning(
        "⚠️ Target column could not be detected automatically."
    )

    target_column = st.selectbox(
        "Select target column:",
        df.columns
    )

else:

    st.success(
        f"🎯 Target column detected: `{target_column}`"
    )

target_counts = df[target_column].value_counts()

col1, col2 = st.columns(2)

with col1:

    st.subheader("Target Counts")

    st.dataframe(
        target_counts.rename("Count"),
        use_container_width=True
    )

with col2:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    target_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"{target_column} Distribution"
    )

    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

# --------------------------------------------------
# NUMERICAL FEATURES
# --------------------------------------------------

st.header("📈 Numerical Feature Analysis")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if numeric_columns:

    selected_feature = st.selectbox(
        "Select a numerical feature:",
        numeric_columns
    )

    # Statistics
    st.subheader(
        f"📊 Statistics — {selected_feature}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean",
            f"{df[selected_feature].mean():.2f}"
        )

    with col2:
        st.metric(
            "Median",
            f"{df[selected_feature].median():.2f}"
        )

    with col3:
        st.metric(
            "Minimum",
            f"{df[selected_feature].min():.2f}"
        )

    with col4:
        st.metric(
            "Maximum",
            f"{df[selected_feature].max():.2f}"
        )

    # Histogram
    st.subheader(
        f"📊 Distribution of {selected_feature}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        df[selected_feature].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_feature}"
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    plt.close(fig)

else:

    st.warning(
        "⚠️ No numerical columns found."
    )

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.header("🔥 Correlation Heatmap")

if len(numeric_columns) >= 2:

    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(
        figsize=(15, 10)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numerical Features"
    )

    st.pyplot(fig)

    plt.close(fig)

else:

    st.warning(
        "⚠️ Not enough numerical features for correlation analysis."
    )

# --------------------------------------------------
# OUTLIER ANALYSIS
# --------------------------------------------------

st.header("📦 Outlier Analysis")

if numeric_columns:

    outlier_feature = st.selectbox(
        "Select feature for boxplot:",
        numeric_columns,
        key="outlier_feature"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    sns.boxplot(
        x=df[outlier_feature],
        ax=ax
    )

    ax.set_title(
        f"Boxplot — {outlier_feature}"
    )

    ax.set_xlabel(
        outlier_feature
    )

    st.pyplot(fig)

    plt.close(fig)

# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

st.header("📌 Exploration Summary")

st.write(
    f"""
    **Dataset:** Feature-engineered EMI prediction dataset

    - **Rows:** {df.shape[0]:,}
    - **Columns:** {df.shape[1]}
    - **Numerical features:** {len(numeric_columns)}
    - **Missing values:** {df.isnull().sum().sum():,}
    - **Duplicate rows:** {df.duplicated().sum():,}
    - **Target column:** `{target_column}`
    """
)

st.success(
    "🎉 Data Exploration page completed successfully!"
)
