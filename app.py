import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="ML Deployment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Configuration")

dataset_option = st.sidebar.selectbox(
    "Dataset source",
    ["Telco Customer Churn", "Upload CSV"]
)

model_option = st.sidebar.selectbox(
    "Select model",
    [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ]
)

st.sidebar.markdown("---")

# ==================================================
# LOAD MODEL (PIPELINE)
# ==================================================
MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression_pipeline.pkl",
    "Random Forest": "models/random_forest_pipeline.pkl",
    "Gradient Boosting": "models/gradient_boosting_pipeline.pkl"
}

@st.cache_resource
def load_pipeline(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

pipeline = load_pipeline(MODEL_PATHS[model_option])

# ==================================================
# LOAD DATA
# ==================================================
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )
    if uploaded_file is None:
        st.info("⬅️ Upload a CSV file to continue")
        st.stop()
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(
        df.get("TotalCharges"),
        errors="coerce"
    )

# ==================================================
# MAIN TABS
# ==================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Data Overview",
    "📊 Visualizations",
    "🤖 Predictions",
    "📈 Model Performance",
    "⬇️ Export Results"
])

# ==================================================
# TAB 1 — DATA OVERVIEW
# ==================================================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.write(df.describe(include="all"))

# ==================================================
# TAB 2 — VISUALIZATIONS
# ==================================================
with tab2:
    st.subheader("Interactive Visualizations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Select numeric feature", numeric_cols)
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns available for visualization.")

    if "Churn" in df.columns:
        churn_counts = (
            df["Churn"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Churn", "Churn": "count"})
        )

        fig2 = px.bar(
            churn_counts,
            x="Churn",
            y="count",
            title="Target Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# TAB 3 — PREDICTIONS
# ==================================================
with tab3:
    st.subheader("Real-Time Predictions")

    if "Churn" not in df.columns:
        st.warning("Target column 'Churn' not found.")
        st.stop()

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    predictions = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        probabilities = np.full(len(predictions), np.nan)

    results_df = df.copy()
    results_df["Prediction"] = predictions
    results_df["Probability"] = probabilities

    st.dataframe(results_df.head(), use_container_width=True)

# ==================================================
# TAB 4 — MODEL PERFORMANCE
# ==================================================
with tab4:
    st.subheader("Model Metrics")

    y_pred = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, y_prob)
    else:
        roc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-score": f1_score(y, y_pred),
        "ROC-AUC": roc
    }

    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "N/A" if np.isnan(v) else f"{v:.3f}")

# ==================================================
# TAB 5 — EXPORT RESULTS
# ==================================================
with tab5:
    st.subheader("Export Predictions")

    st.download_button(
        "⬇️ Download CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv"
    )

    st.download_button(
        "⬇️ Download JSON",
        results_df.to_json(orient="records").encode("utf-8"),
        "predictions.json",
        "application/json"
    )

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("🚀 ML Deployment Dashboard | Streamlit + Scikit-learn")
