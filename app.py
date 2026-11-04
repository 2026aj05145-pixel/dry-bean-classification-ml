from __future__ import annotations
import io
from html import escape
import time
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

st.set_page_config(page_title="Dry Bean | Model Studio", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
/* Crisp, high-contrast visual system; works in Streamlit light and dark modes */
.block-container {max-width:1480px; padding-top:1.3rem; padding-bottom:3rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,rgba(20,75,89,.10),rgba(13,26,43,.025))}
[data-testid="stMetric"] {background:linear-gradient(135deg,rgba(14,165,164,.10),rgba(56,96,142,.055));border:1px solid rgba(30,119,130,.22);padding:1rem 1.1rem;border-radius:16px;min-height:116px}
[data-testid="stMetricLabel"] {font-weight:650;letter-spacing:.01em}
[data-testid="stMetricValue"] {font-size:1.75rem}
[data-testid="stTabs"] button {font-weight:650}
.hero {background:radial-gradient(circle at 85% 15%,rgba(94,234,212,.26),transparent 31%),linear-gradient(120deg,#102339,#145e69 85%);padding:30px 34px;border-radius:22px;color:white;margin:0 0 18px;box-shadow:0 12px 34px rgba(15,37,57,.14)}
.hero .eyebrow {color:#9af1df;font-weight:750;font-size:.78rem;letter-spacing:.13em;text-transform:uppercase}
.hero h1 {font-size:clamp(1.9rem,3vw,2.75rem);line-height:1.12;margin:10px 0 9px;color:#fff;font-weight:800}
.hero p {color:#d3e9ed;margin:0;font-size:1rem;max-width:850px}
.hero .pill {display:inline-block;border:1px solid rgba(185,255,239,.4);background:rgba(255,255,255,.11);color:#ecfffa;padding:7px 12px;border-radius:30px;margin:18px 8px 0 0;font-size:.78rem;font-weight:650}
.section-kicker {font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;color:#0c8c87;font-weight:800;margin:10px 0 2px}
.section-title {font-size:1.42rem;font-weight:800;margin:0 0 8px}
.note-panel {border-left:4px solid #18a69e;padding:12px 16px;background:rgba(22,163,157,.07);border-radius:0 12px 12px 0;margin:10px 0 18px}
@media(max-width:720px){.hero{padding:24px 20px}.hero h1{font-size:1.9rem}}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">BITS PILANI · M.TECH AI &amp; ML · ASSIGNMENT 2 - Viswesvaran J</div>
<h1>🌱 Dry Bean Intelligence Studio</h1>
<p>Explore your dataset. Benchmark five trained classifiers. Investigate every prediction — in one interactive workspace.</p>
<span class="pill">● Live model evaluation</span><span class="pill">5 pretrained classifiers</span><span class="pill">Zero retraining </span></div>""", unsafe_allow_html=True)
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
    st.error(f"Unable to load saved models: {exc}")
    st.stop()
CLASS_NAMES = [str(v) for v in models["Logistic Regression"].classes_]

@st.cache_data(show_spinner=False)
def parse_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def validate_dataset(raw: pd.DataFrame):
    errors = []
    if raw.empty:
        return None, ["The CSV contains no rows."]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return None, errors
    df = raw[REQUIRED_COLUMNS].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = [c for c in FEATURES if df[c].isna().any()]
    if bad:
        errors.append("Missing or non-numeric feature values in: " + ", ".join(bad))
    if not bad and not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        errors.append("Some feature values are infinite; use finite numbers only.")
    if df[TARGET].isna().any():
        errors.append("The Class column contains missing labels.")
    df[TARGET] = df[TARGET].astype("string").str.strip().str.upper()
    if df[TARGET].eq("").any():
        errors.append("The Class column contains blank labels.")
    unknown = sorted(set(df[TARGET].dropna().unique()) - set(CLASS_NAMES))
    if unknown:
        errors.append("Unknown class labels: " + ", ".join(unknown))
        errors.append("Allowed labels: " + ", ".join(CLASS_NAMES))
    if len(df) < 2:
        errors.append("Provide at least two labelled observations for evaluation.")
    return (df if not errors else None), errors

@st.cache_data(show_spinner="Evaluating saved models…")
def evaluate_all_models(canonical_csv: bytes):
    data = pd.read_csv(io.BytesIO(canonical_csv))
    X, y = data[FEATURES], data[TARGET].astype(str).str.strip().str.upper()
    results, predictions, times = {}, {}, {}
    for name, model in load_models().items():
        t0 = time.perf_counter()
        pred = model.predict(X)
        prob = model.predict_proba(X)
        times[name] = time.perf_counter() - t0
        predictions[name] = [str(v) for v in pred]
        auc = float('nan')
        if set(y.unique()) == set(str(v) for v in model.classes_):
            auc = float(roc_auc_score(y, prob, labels=model.classes_, multi_class="ovr", average="weighted"))
        results[name] = {
            "Accuracy": float(accuracy_score(y, pred)),
            "AUC": auc,
            "Precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
            "F1 Score": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "MCC": float(matthews_corrcoef(y, pred)),
        }
    return results, predictions, times

# The default dataset is the first-class experience: no button or selection required.
with st.sidebar:
    st.markdown("### 🧭 Control center")
    st.caption("Your evaluation workspace")
    st.markdown("**📂 Active dataset**")
    uploaded = st.file_uploader(
        "Replace default data with a labelled CSV",
        type=["csv"],
        help="Provide the 16 numerical features and a Class column. Clear the upload to return to the default dataset.",
    )
    if uploaded is None:
        st.success("Default test_data.csv · active")
    else:
        st.info(f"Uploaded · {uploaded.name}")
    st.caption("Clear the upload (×) to restore the default test dataset instantly.")
    st.divider()
    st.markdown("**🫘 Supported bean classes**")
    st.caption(" · ".join(CLASS_NAMES))
    st.divider()
    st.markdown("**🔒 Model integrity**")
    st.caption("Saved models are used for inference only. No uploaded observations are used to fit or retrain a model.")

if uploaded is None:
    if not TEST_DATA_PATH.is_file():
        st.error("Default test_data.csv is missing from the project folder.")
        st.stop()
    data_bytes, data_name = TEST_DATA_PATH.read_bytes(), "test_data.csv"
else:
    data_bytes, data_name = uploaded.getvalue(), uploaded.name

try:
    raw = parse_csv(data_bytes)
except Exception as exc:
    st.error(f"Cannot read CSV: {exc}")
    st.stop()

df, validation_errors = validate_dataset(raw)
if validation_errors:
    st.error("Dataset validation failed. Evaluation has not started.")
    for error in validation_errors:
        st.warning(error)
    with st.expander("Inspect uploaded columns and first rows"):
        st.write("Columns:", list(raw.columns))
        st.dataframe(raw.head(10), width="stretch")
    st.stop()

# Automatic, cached inference. Changing a tab/model never retrains or repeats work.
canonical_bytes = df[REQUIRED_COLUMNS].to_csv(index=False).encode("utf-8")
try:
    results, predictions, inference_times = evaluate_all_models(canonical_bytes)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()
comparison = pd.DataFrame(results).T[METRIC_COLUMNS]

st.markdown('<div class="section-kicker">01 / At a glance</div><div class="section-title">Your evaluation snapshot</div>', unsafe_allow_html=True)
leader = comparison["Accuracy"].idxmax()
leader_score = comparison.loc[leader, "Accuracy"]
a,b,c,d = st.columns(4)
a.metric("🏆 Accuracy leader",leader)
b.metric("🎯 Best accuracy",f"{leader_score:.2%}")
c.metric("🧪 Test observations",f"{len(df):,}")
d.metric("🫘 Classes in dataset",f"{df[TARGET].nunique()} / {len(CLASS_NAMES)}")
source_text = "Built-in test data" if uploaded is None else "Uploaded test data"
st.markdown(f'<div class="note-panel"><strong>✓ {source_text} validated and evaluated automatically.</strong> &nbsp; {escape(data_name)} · {len(FEATURES)} numerical features · {len(models)} pretrained classifiers. Choose a tab below to explore the results.</div>',unsafe_allow_html=True)

with st.expander("👀 Preview active test data", expanded=True):
    st.caption("Showing the first 8 observations. Open Dataset Explorer for the full preview and distribution.")
    st.dataframe(df.head(8), width="stretch", hide_index=True)

st.markdown('<div class="section-kicker">02 / Explore</div><div class="section-title">Interactive analysis workspace</div>', unsafe_allow_html=True)
tab_compare, tab_detail, tab_data, tab_inspect = st.tabs([
    "🏆 Model Leaderboard", "🔬 Model Deep Dive", "🗂️ Dataset Explorer", "🔎 Prediction Inspector"
])

with tab_data:
    st.subheader("Understand your evaluation dataset")
    st.caption("The default test set is loaded automatically. Uploading a CSV in the sidebar replaces it for this session.")
    a,b,c,d=st.columns(4)
    a.metric("Observations",f"{len(df):,}")
    b.metric("Input features",len(FEATURES))
    c.metric("Classes present",df[TARGET].nunique())
    d.metric("Saved models",len(models))
    st.markdown("#### Dataset preview")
    show_rows=st.slider("Rows to preview",5,min(100,len(df)),min(10,len(df))) if len(df)>=5 else len(df)
    st.dataframe(df.head(show_rows),width="stretch",hide_index=True)
    left,right=st.columns([1,1])
    with left:
        st.markdown("#### Class distribution")
        counts=df[TARGET].value_counts().reindex(CLASS_NAMES,fill_value=0)
        st.bar_chart(counts, horizontal=True)
    with right:
        st.markdown("#### Data quality")
        st.success("Required features and class labels validated")
        st.write(f"**Extra columns ignored:** {len(set(raw.columns)-set(REQUIRED_COLUMNS))}")
        st.write(f"**Missing values in required columns:** {int(df.isna().sum().sum())}")
        st.write("**Labels:** " + ", ".join(CLASS_NAMES))
        st.download_button("⬇️ Download labelled CSV template",data=df.head(0).to_csv(index=False).encode(),
            file_name="dry_bean_test_template.csv",mime="text/csv")
    with st.expander("Feature summary statistics"):
        st.dataframe(df[FEATURES].describe().T,width="stretch")

with tab_compare:
    st.subheader("Performance leaderboard")
    st.caption("Pick any models and metric. The winner and ranking update instantly — no retraining or repeated inference.")
    chosen = st.multiselect("Models to compare", list(models), default=list(models),
        help="Choose at least two models to compare side by side.")
    if len(chosen)<2:
        st.info("Select at least two models to compare.")
    else:
        metric=st.selectbox("Rank models by",METRIC_COLUMNS,index=0)
        subset=comparison.loc[chosen].sort_values(metric,ascending=False,na_position="last")
        valid=subset[metric].dropna()
        if not valid.empty:
            winner=valid.idxmax()
            x,y,z=st.columns(3)
            x.metric("🏆 Top model",winner)
            y.metric(f"Best {metric}",f"{valid.max():.4f}")
            z.metric("Models compared",len(chosen))
        else:
            st.warning(f"{metric} is unavailable on this dataset. Multiclass AUC requires all seven target classes to be present.")
        st.markdown("#### Metric leaderboard")
        st.bar_chart(subset[[metric]].dropna(),horizontal=True, color="#0c9b92")
        st.markdown("#### All evaluation metrics")
        st.dataframe(subset.style.format("{:.4f}",na_rep="N/A").highlight_max(axis=0,color="#b9e9dd"),width="stretch")
        with st.expander("Compare another metric visually"):
            other=st.multiselect("Metrics",METRIC_COLUMNS,default=["Accuracy","F1 Score"],key="other_metrics")
            if other:
                st.bar_chart(subset[other])
        st.download_button("⬇️ Download comparison CSV",subset.to_csv().encode(),"dry_bean_model_comparison.csv","text/csv")
    st.caption("Metrics use weighted multiclass averages. AUC is weighted one-vs-rest; it is N/A when the test data does not contain all trained classes.")

with tab_detail:
    st.subheader("Inside a classifier")
    st.caption("Switch between saved models to inspect six metrics, classification quality, and individual class errors.")
    selected=st.selectbox("Switch classifier",list(models),index=0)
    vals=results[selected]
    cols=st.columns(3)
    for i,key in enumerate(METRIC_COLUMNS):
        value=vals[key]
        cols[i%3].metric(key,"N/A" if np.isnan(value) else f"{value:.4f}")
    actual=df[TARGET].astype(str).to_numpy()
    pred=np.asarray(predictions[selected])
    left,right=st.columns([1.2,1])
    with left:
        st.markdown(f"#### Confusion matrix · {selected}")
        cm=confusion_matrix(actual,pred,labels=CLASS_NAMES)
        fig,ax=plt.subplots(figsize=(8,6))
        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASS_NAMES).plot(
            ax=ax,cmap="Blues",values_format="d",xticks_rotation=45,colorbar=False)
        ax.set_title(selected)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Classification report")
        report=classification_report(actual,pred,labels=CLASS_NAMES,target_names=CLASS_NAMES,
            output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"),width="stretch")
    st.caption(f"Measured inference time (predict + predict_proba): {inference_times[selected]:.4f}s on first uncached evaluation; not a stable benchmark.")

with tab_inspect:
    st.subheader("Prediction investigation lab")
    st.caption("Find disagreements and difficult observations to understand where the models make mistakes.")
    actual=df[TARGET].astype(str).to_numpy()
    pred_df=pd.DataFrame({"Row":np.arange(1,len(df)+1),"Actual Class":actual})
    for name in models:
        pred_df[name]=predictions[name]
    pred_df["All models agree"]=pred_df[list(models)].nunique(axis=1).eq(1)
    pred_df["All models correct"]=pd.concat([pred_df[name].eq(pred_df["Actual Class"]) for name in models],axis=1).all(axis=1)
    aa,bb,cc=st.columns(3)
    aa.metric("All models agree",int(pred_df["All models agree"].sum()))
    bb.metric("Models disagree",int((~pred_df["All models agree"]).sum()))
    cc.metric("All models correct",int(pred_df["All models correct"].sum()))
    filter_mode=st.selectbox("Show rows",["All predictions","Model disagreements only","At least one incorrect prediction","All models incorrect"])
    if filter_mode=="Model disagreements only":
        view=pred_df.loc[~pred_df["All models agree"]]
    elif filter_mode=="At least one incorrect prediction":
        view=pred_df.loc[~pred_df["All models correct"]]
    elif filter_mode=="All models incorrect":
        view=pred_df.loc[pd.concat([pred_df[name].ne(pred_df["Actual Class"]) for name in models],axis=1).all(axis=1)]
    else:
        view=pred_df
    st.caption(f"Showing {len(view):,} of {len(pred_df):,} rows")
    st.dataframe(view,width="stretch",hide_index=True,height=430)
    st.download_button("⬇️ Download filtered predictions",view.to_csv(index=False).encode(),
        "dry_bean_predictions.csv","text/csv")

st.divider()
st.caption("BITS Pilani · Machine Learning Assignment 2 · All evaluations use pretrained saved classifiers; uploaded datasets are never used for training.")
