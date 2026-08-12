"""
Explainability: SHAP (SHapley Additive exPlanations).

Read this alongside models/vif_report.json: the multicollinearity analysis
found severe correlation among several feature groups (radius/perimeter/area
correlate at 0.97-1.00; concavity/concave-points correlate above 0.85) but
found that removing them costs real predictive performance, so the full
30-feature model is what's deployed. That means SHAP attributions here
should be read at the level of "this cluster of related measurements pushed
the prediction," not "this exact feature, and only this one, mattered" --
for a linear model, credit within a highly correlated cluster can shift
between near-duplicate features without changing the prediction at all.

Run: python3 -m src.shap_explain   (from project root)
"""
import json

import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src import config
from src.data import load_splits, load_selected_features, TARGET_COL

logger = config.get_logger(__name__)


def main() -> None:
    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)
    features = load_selected_features()
    train_df, _, test_df = load_splits()

    X_train, X_test = train_df[features], test_df[features]
    y_test = test_df[TARGET_COL].values

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    background = shap.sample(X_train, min(100, len(X_train)), random_state=config.RANDOM_STATE)

    if isinstance(clf, LogisticRegression):
        logger.info("Using LinearExplainer (exact, fast) for logistic regression.")
        bg_scaled = scaler.transform(background)
        X_test_scaled = scaler.transform(X_test)
        explainer = shap.LinearExplainer(clf, bg_scaled)
        shap_values = explainer(X_test_scaled)
        shap_values.feature_names = features
        shap_values.data = X_test.values
    elif isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        logger.info("Using TreeExplainer (exact, fast) for tree ensemble.")
        explainer = shap.TreeExplainer(clf)
        raw = explainer(X_test)
        if len(raw.values.shape) == 3:
            shap_values = shap.Explanation(
                values=raw.values[:, :, 1], base_values=raw.base_values[:, 1],
                data=X_test.values, feature_names=features,
            )
        else:
            shap_values = raw
            shap_values.feature_names = features
    else:
        logger.info("Using generic permutation Explainer (slower fallback).")
        import pandas as pd
        f = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))[:, 1]
        explainer = shap.Explainer(f, background, feature_names=features)
        shap_values = explainer(X_test)

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Summary: What Drives a Malignant Prediction")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Mean |SHAP value| per Feature")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    malignant_idx = np.where(y_test == 1)[0]
    if len(malignant_idx) > 0:
        i = int(malignant_idx[0])
        plt.figure()
        shap.plots.waterfall(shap_values[i], max_display=12, show=False)
        plt.title(f"Why patient #{i} (actual: malignant) was flagged")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "shap_waterfall_malignant_example.png", dpi=150, bbox_inches="tight")
        plt.close()

    benign_idx = np.where(y_test == 0)[0]
    if len(benign_idx) > 0:
        j = int(benign_idx[0])
        plt.figure()
        shap.plots.waterfall(shap_values[j], max_display=12, show=False)
        plt.title(f"Why patient #{j} (actual: benign) was cleared")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "shap_waterfall_benign_example.png", dpi=150, bbox_inches="tight")
        plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(zip(features, mean_abs), key=lambda t: -t[1])
    logger.info("Top 10 features by mean |SHAP value|:")
    for name, val in ranking[:10]:
        logger.info("  %-30s %.4f", name, val)

    with open(config.SHAP_RANKING_PATH, "w") as f:
        json.dump([{"feature": n, "mean_abs_shap": float(v)} for n, v in ranking], f, indent=2)

    logger.info("Saved SHAP figures and %s", config.SHAP_RANKING_PATH)


if __name__ == "__main__":
    main()
