from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Dry Bean Classification Dashboard",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Dry Bean Classification Dashboard")
st.caption(
    "BITS Pilani | M.Tech Artificial Intelligence & Machine Learning "
    "| Machine Learning Assignment 2 - VISWESVARAN J 2026AJ05145"
)
st.write(
    "Evaluate and compare five pre-trained classifiers using the "
    "Dry Bean dataset. Models are loaded from disk and are never retrained here."
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"  # BITS assignment folder name: model/

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
TARGET = "Class"
REQUIRED_COLUMNS = FEATURES + [TARGET]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
METRIC_COLUMNS = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC",
]


# ============================================================
# 3. LOAD SAVED MODELS ONCE PER SERVER PROCESS
# ============================================================
@st.cache_resource(show_spinner="Loading the five saved classification models...")
def load_models() -> dict[str, Any]:
    """Load the five serialized models once and reuse them across reruns."""
    loaded: dict[str, Any] = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing model file: {model_path}. "
                "Check that the GitHub repository contains model/ with all five .joblib files."
            )
        loaded[model_name] = joblib.load(model_path)

    # Ensure all models were trained with the same class labels.
    reference_classes = list(loaded["Logistic Regression"].classes_)
    for model_name, model in loaded.items():
        if list(model.classes_) != reference_classes:
            raise ValueError(
                f"Class order for {model_name} differs from Logistic Regression."
            )

    return loaded


try:
    models = load_models()
except Exception as exc:
    st.error(f"Unable to load the saved models: {exc}")
    st.exception(exc)
    st.stop()

CLASS_NAMES = [str(value) for value in models["Logistic Regression"].classes_]


# ============================================================
# 4. READ DATASET
# ============================================================
st.sidebar.header("Evaluation Settings")
st.sidebar.subheader("Select Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload another labelled test CSV (optional)",
    type=["csv"],
    help=(
        "The CSV must contain the 16 Dry Bean input features and the actual "
        "Class target column. Extra columns are ignored."
    ),
)

if uploaded_file is None:
    if not TEST_DATA_PATH.is_file():
        st.error(f"Preloaded test data was not found: {TEST_DATA_PATH}")
        st.stop()
    dataset_bytes = TEST_DATA_PATH.read_bytes()
    dataset_name = "Preloaded assignment test dataset"
    st.sidebar.success("Using preloaded test_data.csv")
else:
    dataset_bytes = uploaded_file.getvalue()
    dataset_name = uploaded_file.name
    st.sidebar.info(f"Using uploaded dataset: {dataset_name}")

st.sidebar.divider()
st.sidebar.write("**Available classifiers**")
for name in MODEL_FILES:
    st.sidebar.write(f"✓ {name}")
st.sidebar.caption("Saved models are evaluated only; no fitting or retraining occurs.")

try:
    raw_df = pd.read_csv(io.BytesIO(dataset_bytes))
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The selected dataset is empty.")
    st.stop()

missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
if missing_columns:
    st.error(f"The CSV is missing required columns: {missing_columns}")
    st.info("Required columns are the 16 Dry Bean features plus the Class target.")
    st.stop()

# Select only the required columns and enforce the training feature order.
df = raw_df[REQUIRED_COLUMNS].copy()

for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

if df[FEATURES].isna().any().any():
    bad_columns = df[FEATURES].columns[df[FEATURES].isna().any()].tolist()
    st.error(f"Missing or non-numeric values were found in these features: {bad_columns}")
    st.stop()

if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
    st.error("The feature data contains positive or negative infinity values.")
    st.stop()

if df[TARGET].isna().any():
    st.error("The Class target column contains missing values.")
    st.stop()

df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
if (df[TARGET] == "").any():
    st.error("The Class target column contains blank labels.")
    st.stop()

unknown_classes = sorted(set(df[TARGET].unique()) - set(CLASS_NAMES))
if unknown_classes:
    st.error(f"Unknown target classes were found: {unknown_classes}")
    st.info(f"Expected classes: {', '.join(CLASS_NAMES)}")
    st.stop()

if len(df) < 2:
    st.error("At least two labelled observations are needed for evaluation.")
    st.stop()


# ============================================================
# 5. CACHED EVALUATION: RUN ONLY WHEN DATASET CHANGES
# ============================================================
@st.cache_data(show_spinner="Evaluating all five models on this dataset...")
def evaluate_all_models(csv_bytes: bytes):
    """Evaluate all models once per unique CSV content and cache the results."""
    eval_df = pd.read_csv(io.BytesIO(csv_bytes))
    X_eval = eval_df[FEATURES]
    y_eval = eval_df[TARGET].astype(str).str.strip().str.upper()

    model_objects = load_models()
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, list[str]] = {}
    inference_times: dict[str, float] = {}

    for model_name, model in model_objects.items():
        start = time.perf_counter()
        y_pred = model.predict(X_eval)
        predict_time = time.perf_counter() - start

        # Probability prediction is needed for the required multiclass AUC.
        probability_start = time.perf_counter()
        y_probability = model.predict_proba(X_eval)
        probability_time = time.perf_counter() - probability_start

        inference_times[model_name] = round(predict_time + probability_time, 4)
        predictions[model_name] = [str(value) for value in y_pred]

        # Weighted OvR AUC needs all model classes represented in y_true.
        if set(y_eval.unique()) == set(model.classes_):
            auc_value = roc_auc_score(
                y_eval,
                y_probability,
                labels=model.classes_,
                multi_class="ovr",
                average="weighted",
            )
        else:
            auc_value = float("nan")

        results[model_name] = {
            "Accuracy": float(accuracy_score(y_eval, y_pred)),
            "AUC": float(auc_value),
            "Precision": float(
                precision_score(y_eval, y_pred, average="weighted", zero_division=0)
            ),
            "Recall": float(recall_score(y_eval, y_pred, average="weighted", zero_division=0)),
            "F1 Score": float(f1_score(y_eval, y_pred, average="weighted", zero_division=0)),
            "MCC": float(matthews_corrcoef(y_eval, y_pred)),
        }

    return results, predictions, inference_times


# Canonical CSV bytes provide a stable cache key. Re-selecting a model or
# expanding an expander will reuse the cached evaluation result.
evaluation_bytes = df[REQUIRED_COLUMNS].to_csv(index=False).encode("utf-8")
try:
    results, predictions, inference_times = evaluate_all_models(evaluation_bytes)
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.exception(exc)
    st.stop()


y_true = df[TARGET].to_numpy()


# ============================================================
# 6. DATASET OVERVIEW
# ============================================================
st.divider()
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Input Features", len(FEATURES))
col3.metric("Bean Classes Present", df[TARGET].nunique())
col4.metric("Classification Models", len(models))
st.caption(f"Current dataset: {dataset_name}")

with st.expander("Preview dataset and class distribution"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Class distribution")
    st.bar_chart(df[TARGET].value_counts().sort_index())


# ============================================================
# 7. ALL-MODEL COMPARISON (ONE TABLE + ONE CHART)
# ============================================================
st.divider()
st.header("Overall Model Comparison")
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[METRIC_COLUMNS]

st.subheader("Evaluation Metrics")
st.dataframe(
    comparison_df.style.format("{:.4f}", na_rep="N/A"),
    use_container_width=True,
)

best_accuracy_model = comparison_df["Accuracy"].idxmax()
left, right = st.columns(2)
left.success(
    f"Highest accuracy: {best_accuracy_model} "
    f"({comparison_df.loc[best_accuracy_model, 'Accuracy']:.4f})"
)
valid_auc = comparison_df["AUC"].dropna()
if not valid_auc.empty:
    best_auc_model = valid_auc.idxmax()
    right.info(
        f"Highest AUC: {best_auc_model} "
        f"({comparison_df.loc[best_auc_model, 'AUC']:.4f})"
    )
else:
    right.warning("AUC is unavailable because not all seven classes are present in this dataset.")

st.subheader("Performance Metrics Comparison")
st.caption("Compare all six metrics across all five classifiers.")
st.bar_chart(comparison_df)


# ============================================================
# 8. INFERENCE TIME DIAGNOSTICS
# ============================================================
st.subheader("Model Evaluation Time")
st.caption(
    "Time for predict() plus predict_proba() on this dataset. "
    "These timings are measured on the first uncached evaluation and then cached."
)
time_df = pd.DataFrame.from_dict(
    inference_times,
    orient="index",
    columns=["Inference time (seconds)"],
).sort_values("Inference time (seconds)", ascending=False)
st.dataframe(time_df, use_container_width=True)


# ============================================================
# 9. ONE MODEL AT A TIME: DETAILED ANALYSIS
# ============================================================
st.divider()
st.header("Individual Model Analysis")
st.write(
    "Choose one classifier to view its confusion matrix and classification report. "
    "The five-model comparison above always remains available."
)
selected_model = st.selectbox("Select classifier", list(models.keys()))
selected_predictions = np.asarray(predictions[selected_model])
selected_metrics = results[selected_model]

metric_cols = st.columns(6)
for column, metric_name in zip(metric_cols, METRIC_COLUMNS):
    value = selected_metrics[metric_name]
    shown_value = "N/A" if np.isnan(value) else f"{value:.4f}"
    column.metric(metric_name, shown_value)

st.subheader(f"Confusion Matrix — {selected_model}")
cm = confusion_matrix(y_true, selected_predictions, labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False,
)
ax.set_title(f"Confusion Matrix — {selected_model}")
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.subheader("Classification Report")
report = classification_report(
    y_true,
    selected_predictions,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).T
st.dataframe(
    report_df.style.format("{:.4f}"),
    use_container_width=True,
)


# ============================================================
# 10. SIDE-BY-SIDE PREDICTION COMPARISON
# ============================================================
st.divider()
st.header("Model Prediction Comparison")
st.write(
    "Compare the actual class with the predictions from all five models "
    "for each observation."
)

prediction_df = pd.DataFrame({"Actual Class": y_true})
for model_name in models:
    prediction_df[f"{model_name} Prediction"] = predictions[model_name]

prediction_columns = [f"{name} Prediction" for name in models]
prediction_df["All Models Agree"] = prediction_df[prediction_columns].nunique(axis=1).eq(1)

agree_count = int(prediction_df["All Models Agree"].sum())
disagree_count = int((~prediction_df["All Models Agree"]).sum())
col_a, col_b = st.columns(2)
col_a.metric("All Models Agree", f"{agree_count:,}")
col_b.metric("Models Disagree", f"{disagree_count:,}")

with st.expander("View individual predictions"):
    st.dataframe(prediction_df, use_container_width=True, height=400)

prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download all model predictions (CSV)",
    data=prediction_csv,
    file_name="dry_bean_model_predictions.csv",
    mime="text/csv",
)


# ============================================================
# 11. FOOTER
# ============================================================
st.divider()
st.success(
    "Evaluation completed. All five saved classifiers were evaluated "
    "without retraining."
)
st.caption(
    "Dry Bean Classification | BITS Pilani M.Tech AIML | Machine Learning Assignment 2"
)
