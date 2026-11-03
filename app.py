
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Dry Bean ML Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")

st.caption(
    "BITS Pilani | M.Tech Artificial Intelligence & Machine Learning "
    "| Machine Learning Assignment 2"
)

st.write(
    "Evaluate and compare five machine learning classifiers "
    "using the Dry Bean dataset."
)


# ==================================================
# 2. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DATA_PATH = BASE_DIR / "test_data.csv"

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
    "ShapeFactor4"
]

TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# ==================================================
# 3. LOAD ALL FIVE SAVED MODELS
# ==================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = BASE_DIR / "model" / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {filename}"
            )

        loaded_models[model_name] = joblib.load(
            model_path
        )

    return loaded_models


try:

    models = load_models()

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# Use the saved model's class ordering
CLASS_NAMES = models["Logistic Regression"].classes_


# ==================================================
# 4. SIDEBAR - DATASET SELECTION
# ==================================================

st.sidebar.title("Evaluation Settings")

st.sidebar.subheader("Select Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your own test CSV (optional)",
    type=["csv"]
)

if uploaded_file is None:

    st.sidebar.success(
        "Using preloaded assignment test dataset"
    )

    if not TEST_DATA_PATH.exists():

        st.error(
            "Preloaded test_data.csv was not found "
            "in the project directory."
        )

        st.stop()

    df = pd.read_csv(TEST_DATA_PATH)

    dataset_name = "Preloaded Test Dataset"

else:

    try:

        df = pd.read_csv(uploaded_file)

        dataset_name = uploaded_file.name

        st.sidebar.info(
            f"Using uploaded dataset: {dataset_name}"
        )

    except Exception as error:

        st.error(f"Unable to read uploaded CSV: {error}")
        st.stop()


st.sidebar.markdown("---")

st.sidebar.write("**Available classifiers**")

for model_name in MODEL_FILES:
    st.sidebar.write(f"✓ {model_name}")

st.sidebar.markdown("---")

st.sidebar.caption(
    "All models are pre-trained. "
    "Uploaded datasets are used only for evaluation."
)


# ==================================================
# 5. VALIDATE THE DATASET
# ==================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


if df.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


if df[TARGET].isnull().any():

    st.error("Target column contains missing values.")
    st.stop()


# Select required features in training order
X = df[FEATURES].copy()

# Convert input features to numeric
for col in FEATURES:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# Check missing and invalid feature values
if X.isnull().any().any():

    st.error(
        "Input features contain missing or invalid values."
    )

    st.stop()


if not np.isfinite(X.to_numpy()).all():

    st.error("Input features contain infinite values.")
    st.stop()


# Normalize target label representation
y = df[TARGET].astype(str).str.strip().str.upper()


if (y == "").any():

    st.error("Target column contains empty labels.")
    st.stop()


# Validate class labels
unknown_classes = set(y.unique()) - set(CLASS_NAMES)

if unknown_classes:

    st.error(
        f"Unknown bean classes found: {unknown_classes}"
    )

    st.stop()


# ==================================================
# 6. DATASET OVERVIEW
# ==================================================

st.markdown("---")

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Observations",
    f"{len(df):,}"
)

col2.metric(
    "Input Features",
    len(FEATURES)
)

col3.metric(
    "Bean Classes",
    y.nunique()
)

col4.metric(
    "Classification Models",
    len(models)
)

st.caption(f"Current dataset: {dataset_name}")

with st.expander("Preview dataset and class distribution"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("Class distribution")

    st.bar_chart(
        y.value_counts()
    )


# ==================================================
# 7. EVALUATE ALL FIVE MODELS
# ==================================================

results = {}
predictions = {}
probabilities = {}

all_classes_present = set(y.unique()) == set(CLASS_NAMES)

for model_name, model in models.items():

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)

    predictions[model_name] = y_pred

    probabilities[model_name] = y_prob

    # AUC requires all seven target classes here
    if all_classes_present:

        auc = roc_auc_score(
            y,
            y_prob,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        )

    else:

        auc = np.nan

    results[model_name] = {

        "Accuracy": accuracy_score(
            y, y_pred
        ),

        "AUC": auc,

        "Precision": precision_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            y_pred
        )
    }


# ==================================================
# 8. OVERALL MODEL COMPARISON
# ==================================================

st.markdown("---")

st.header("Overall Model Comparison")

comparison_df = pd.DataFrame(results).T

st.subheader("Evaluation Metrics")

st.dataframe(
    comparison_df.style.format(
        "{:.4f}",
        na_rep="N/A"
    ),
    use_container_width=True
)

# Identify metric leaders
metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

best_accuracy = comparison_df["Accuracy"].idxmax()

best_auc = comparison_df["AUC"].idxmax()

col1, col2 = st.columns(2)

col1.success(
    f"Highest Accuracy: {best_accuracy} "
    f"({comparison_df.loc[best_accuracy, 'Accuracy']:.4f})"
)

if not pd.isna(comparison_df["AUC"].max()):

    col2.info(
        f"Highest AUC: {best_auc} "
        f"({comparison_df.loc[best_auc, 'AUC']:.4f})"
    )

else:

    col2.info(
        "AUC unavailable: all seven classes "
        "are not present in the dataset."
    )


# ==================================================
# 9. COMPARE ALL MODELS GRAPHICALLY
# ==================================================

st.subheader("Performance Metrics Comparison")

st.caption(
    "All six evaluation metrics are displayed together."
)

st.bar_chart(
    comparison_df[metric_columns]
)


# ==================================================
# 10. MODEL-WISE DETAILED RESULTS
# ==================================================

st.markdown("---")

st.header("Individual Model Analysis")

st.write(
    "Select a model tab to examine its performance, "
    "confusion matrix and classification report."
)

model_tabs = st.tabs(list(models.keys()))

for tab, (model_name, model) in zip(
    model_tabs,
    models.items()
):

    with tab:

        st.subheader(model_name)

        model_metrics = results[model_name]

        # Display six evaluation metrics
        metric_cols = st.columns(6)

        for col, metric in zip(
            metric_cols,
            metric_columns
        ):

            value = model_metrics[metric]

            display_value = (
                "N/A"
                if pd.isna(value)
                else f"{value:.4f}"
            )

            col.metric(
                metric,
                display_value
            )

        # Confusion matrix
        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(
            y,
            predictions[model_name],
            labels=CLASS_NAMES
        )

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_NAMES
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            xticks_rotation=45,
            colorbar=False
        )

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # Classification report
        st.markdown("#### Classification Report")

        report = classification_report(
            y,
            predictions[model_name],
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).T

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )


# ==================================================
# 11. COMPARE PREDICTIONS OF ALL FIVE MODELS
# ==================================================

st.markdown("---")

st.header("Model Prediction Comparison")

st.write(
    "Compare the predicted bean variety from all five "
    "classifiers for each uploaded observation."
)

prediction_df = pd.DataFrame({
    "Actual Class": y.to_numpy()
})

for model_name in models:

    prediction_df[
        f"{model_name} Prediction"
    ] = predictions[model_name]


prediction_df["Agreement"] = prediction_df[
    [
        f"{name} Prediction"
        for name in models
    ]
].nunique(axis=1).eq(1)


col1, col2 = st.columns(2)

col1.metric(
    "All Models Agree",
    int(prediction_df["Agreement"].sum())
)

col2.metric(
    "Models Disagree",
    int((~prediction_df["Agreement"]).sum())
)


with st.expander("View individual predictions"):

    st.dataframe(
        prediction_df,
        use_container_width=True
    )


# Download prediction comparison
prediction_csv = prediction_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download All Model Predictions",
    data=prediction_csv,
    file_name="dry_bean_model_predictions.csv",
    mime="text/csv"
)


# ==================================================
# 12. FINAL STATUS
# ==================================================

st.markdown("---")

st.success(
    "Evaluation completed successfully. "
    "All five saved classifiers were evaluated "
    "without retraining."
)

st.caption(
    "Dry Bean Classification | "
    "BITS Pilani M.Tech AIML | Machine Learning Assignment 2"
)
