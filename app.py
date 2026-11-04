from pathlib import Path
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Dry Bean Classification", page_icon="🌱", layout="wide")
st.title("🌱 Dry Bean Classification Dashboard")
st.caption("BITS Pilani | M.Tech AI & ML | Machine Learning Assignment 2")
st.write("Evaluate five saved classifiers. Models are never retrained on uploaded data.")
ROOT = Path(__file__).resolve().parent
FEATURES = ["Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRatio","Eccentricity","ConvexArea","EquivDiameter","Extent","Solidity","Roundness","Compactness","ShapeFactor1","ShapeFactor2","ShapeFactor3","ShapeFactor4"]
MODEL_FILES = {"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","KNN":"knn.joblib","Naive Bayes":"naive_bayes.joblib","SVM":"svm.joblib"}

@st.cache_resource(show_spinner="Loading saved classifiers...")
def load_models():
    loaded, failures = {}, {}
    for name, filename in MODEL_FILES.items():
        try:
            loaded[name] = joblib.load(ROOT / "model" / filename)
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {e}"
    return loaded, failures

file = st.file_uploader("Upload labelled CSV (16 features and Class)", type="csv")
if file is None:
    st.caption("Default dataset: test_data.csv")
if not st.button("Load models and evaluate", type="primary"):
    st.info("Click 'Load models and evaluate' to start. The dashboard has loaded successfully.")
    st.stop()

try:
    raw = pd.read_csv(file if file is not None else ROOT / "test_data.csv")
except Exception as e:
    st.error(f"Unable to read dataset: {e}")
    st.stop()
missing = [x for x in FEATURES+["Class"] if x not in raw.columns]
if missing:
    st.error(f"Missing CSV columns: {missing}")
    st.stop()
df = raw[FEATURES+["Class"]].copy()
for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
if df[FEATURES].isna().any().any() or not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
    st.error("Features contain missing, nonnumeric or infinite values")
    st.stop()
if df["Class"].isna().any():
    st.error("Class contains missing values")
    st.stop()
df["Class"] = df["Class"].astype(str).str.strip().str.upper()
if (df["Class"] == "").any():
    st.error("Class contains blank labels")
    st.stop()
models, failures = load_models()
if failures:
    with st.expander("Model loading errors", expanded=True):
        for name, err in failures.items():
            st.error(f"{name}: {err}")
if not models:
    st.stop()
st.success(f"Loaded {len(models)} of 5 models")
st.metric("Test observations", len(df))
X, y = df[FEATURES], df["Class"]
results, preds, errors = {}, {}, {}
for name, model in models.items():
    try:
        predicted = np.asarray(model.predict(X)).astype(str)
        labels = [str(x) for x in model.classes_]
        auc = float("nan")
        if hasattr(model, "predict_proba") and set(y.unique()) == set(labels):
            try:
                auc = roc_auc_score(y, model.predict_proba(X), labels=labels, multi_class="ovr", average="weighted")
            except ValueError:
                pass
        results[name] = {"Accuracy":accuracy_score(y,predicted),"AUC":auc,"Precision":precision_score(y,predicted,average="weighted",zero_division=0),"Recall":recall_score(y,predicted,average="weighted",zero_division=0),"F1 Score":f1_score(y,predicted,average="weighted",zero_division=0),"MCC":matthews_corrcoef(y,predicted)}
        preds[name] = predicted
    except Exception as e:
        errors[name] = f"{type(e).__name__}: {e}"
if errors:
    with st.expander("Evaluation errors", expanded=True):
        for name, err in errors.items():
            st.error(f"{name}: {err}")
if not results:
    st.stop()
st.subheader("Model comparison")
st.dataframe(pd.DataFrame.from_dict(results, orient="index").round(4))
st.subheader("Individual model analysis")
selected = st.selectbox("Choose model", list(results))
labels = [str(x) for x in models[selected].classes_]
cm = pd.DataFrame(confusion_matrix(y,preds[selected],labels=labels),index=labels,columns=labels)
st.write("Confusion matrix (actual classes in rows, predicted classes in columns)")
st.dataframe(cm)
st.write("Classification report")
report = classification_report(y,preds[selected],labels=labels,target_names=labels,output_dict=True,zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(4))
st.subheader("Predictions")
predictions = pd.DataFrame({"Actual Class":y})
for name, values in preds.items():
    predictions[name] = values
st.dataframe(predictions.head(100))
st.download_button("Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"), "dry_bean_predictions.csv", "text/csv")
