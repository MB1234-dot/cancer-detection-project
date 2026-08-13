"""
Streamlit demo: Breast Mass Malignancy Risk Estimator.

Run with:  streamlit run app/app.py   (from the project root)
"""
import sys
from pathlib import Path

# allow `from src import ...` when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import load_selected_features, load_splits, TARGET_COL

st.set_page_config(page_title="Breast Mass Risk Estimator", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(config.MODEL_PATH)
    features = load_selected_features()
    with open(config.EVAL_SUMMARY_PATH) as f:
        eval_summary = json.load(f)
    stability_path = config.MODELS_DIR / "threshold_stability_report.json"
    stability = json.load(open(stability_path)) if stability_path.exists() else None
    train_df, _, test_df = load_splits()
    return model, features, eval_summary, stability, train_df, test_df


model, feature_names, eval_summary, stability, train_df, test_df = load_artifacts()
scaler = model.named_steps["scaler"]
clf = model.named_steps["clf"]

default_threshold = eval_summary["default_threshold"]
tuned_threshold = eval_summary["tuned_threshold"]
default_metrics = eval_summary["test_metrics_default_threshold"]
tuned_metrics = eval_summary["test_metrics_tuned_threshold"]

st.title("Breast Mass Malignancy Risk Estimator")
st.warning(
    "**Portfolio demo, not a medical device.** This model was trained on the "
    "public Wisconsin Diagnostic Breast Cancer dataset (569 samples) for "
    "educational purposes. It has not been clinically validated and must "
    "never be used for actual diagnosis or to inform real medical decisions.",
    icon="⚠️",
)

st.markdown(
    "Enter fine-needle-aspirate (FNA) cell nuclei measurements below, or load "
    "an example patient from the held-out test set, to see the model's "
    "prediction, its confidence, and a feature-level explanation of *why*."
)

def _ci_help(metric_name):
    """Format a bootstrap 95% CI as a metric tooltip.

    Note: built with f-strings on purpose. These strings contain a literal '%'
    (in "95% CI"), and under %-formatting that '%' is parsed as the start of a
    format spec -- "95% C" raised ValueError and took the deployed app down on
    load. See tests/test_app.py::test_no_percent_format_strings.
    """
    ci = default_metrics["bootstrap_95ci"][metric_name]
    lower = ci["ci_lower_2.5%"]
    upper = ci["ci_upper_97.5%"]
    return f"95% CI: [{lower:.3f}, {upper:.3f}]"


with st.sidebar:
    st.header(f"Model performance (held-out test set, n={eval_summary['n_test']})")
    st.metric("ROC-AUC", default_metrics["roc_auc"])
    c1, c2 = st.columns(2)
    c1.metric("Recall @ default (0.5)", default_metrics["recall"],
              help=_ci_help("recall"))
    c2.metric("Precision @ default (0.5)", default_metrics["precision"],
              help=_ci_help("precision"))
    st.caption(
        "95% CIs are bootstrap resamples of the 114-patient test set -- "
        f"with only {eval_summary['n_malignant_test']} malignant test cases, "
        "point estimates alone would overstate precision."
    )

    if stability is not None:
        with st.expander("About the 'tuned' threshold (recommended: don't use it)"):
            st.markdown(
                f"A validation-tuned threshold was computed ({tuned_threshold:.3f}) "
                f"targeting recall >= {stability['target_recall']}. A multi-seed "
                f"stability analysis (`src/stability_analysis.py`, {stability['n_seeds']} "
                f"seeds) found this threshold ranges from "
                f"**{stability['threshold_percentiles']['min']} to "
                f"{stability['threshold_percentiles']['max']}** depending on the "
                f"validation draw, and on average **{stability['recommendation']}**"
            )
    st.divider()
    st.header("Load an example patient")
    example_choice = st.selectbox(
        "Pick from the test set",
        options=["-- manual entry --"] + [f"Patient #{i}" for i in test_df.index],
    )

defaults = train_df[feature_names].mean()
mins = train_df[feature_names].min()
maxs = train_df[feature_names].max()

if example_choice != "-- manual entry --":
    idx = int(example_choice.split("#")[1])
    row = test_df.loc[idx, feature_names]
    actual_label = "malignant" if test_df.loc[idx, TARGET_COL] == 1 else "benign"
    st.info(f"Loaded {example_choice} from the test set (actual diagnosis: **{actual_label}**, not shown to the model).")
else:
    row = defaults

groups = {
    "Mean measurements": [f for f in feature_names if "worst" not in f and "error" not in f],
    "Standard error measurements": [f for f in feature_names if "error" in f],
    "'Worst' (largest) measurements": [f for f in feature_names if "worst" in f],
}

st.subheader("Input measurements")
input_values = {}
cols_per_row = 3
for group_name, feats in groups.items():
    if not feats:
        continue
    with st.expander(group_name, expanded=(group_name == "Mean measurements")):
        cols = st.columns(cols_per_row)
        for i, feat in enumerate(feats):
            with cols[i % cols_per_row]:
                input_values[feat] = st.number_input(
                    feat, min_value=float(mins[feat]) * 0.5, max_value=float(maxs[feat]) * 1.5,
                    value=float(row[feat]), format="%.4f", key=feat,
                )

X_input = pd.DataFrame([input_values])[feature_names]

if st.button("Predict", type="primary"):
    proba_malignant = model.predict_proba(X_input)[0, 1]
    default_pred = "malignant" if proba_malignant >= default_threshold else "benign"
    tuned_pred = "malignant" if proba_malignant >= tuned_threshold else "benign"

    c1, c2 = st.columns(2)
    c1.metric("Predicted probability (malignant)", f"{proba_malignant:.1%}")
    c2.metric(f"Call @ default threshold ({default_threshold}) -- recommended", default_pred)
    st.caption(
        f"For reference only, NOT recommended (see sidebar): call at the "
        f"validation-tuned threshold ({tuned_threshold:.3f}) would be **{tuned_pred}**."
    )

    if default_pred != tuned_pred:
        st.info(
            "The two thresholds disagree on this case. A multi-seed stability "
            "analysis found the tuned threshold is not reliably better than the "
            "default and is highly sensitive to which patients ended up in the "
            "validation set -- treat the default-threshold call as primary."
        )

    st.subheader("Why the model said this")
    if isinstance(clf, LogisticRegression):
        X_scaled = scaler.transform(X_input)
        background = shap.sample(train_df[feature_names], 100, random_state=config.RANDOM_STATE)
        bg_scaled = scaler.transform(background)
        explainer = shap.LinearExplainer(clf, bg_scaled)
        sv = explainer(X_scaled)
        sv.feature_names = feature_names
        sv.data = X_input.values
        fig, ax = plt.subplots()
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Note: several features here are highly correlated (see the VIF "
            "multicollinearity analysis in the README) -- read this as which "
            "*clusters* of measurements drove the prediction, not that each "
            "listed feature independently matters exactly this much."
        )
    else:
        st.caption("SHAP explanation available for the logistic regression model; "
                    "this run selected a different model type.")

st.divider()
st.caption(
    "Data source: Wisconsin Diagnostic Breast Cancer dataset (UCI ML Repository / "
    "scikit-learn). Full methodology, model comparison, and known limitations "
    "are documented in the project README."
)
