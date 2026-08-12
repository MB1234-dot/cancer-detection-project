"""
Threshold selection on the VALIDATION set, final report on the TEST set --
each touched exactly once, for exactly one purpose. This is the fix for the
original leakage bug: threshold tuning used to sweep `precision_recall_curve`
directly on the test set, then report metrics on that same test set, which
made the "final" numbers optimistic (the threshold had, in effect, been
fit to the test set's specific 114 patients).

Now: the validation set (91 patients, never used for model fitting or
hyperparameter search) is used to pick an operating threshold that hits our
target recall. The test set (114 patients, used for nothing until this exact
point) is scored exactly once, at that already-chosen threshold, to produce
the numbers that actually go in the README.

We also bootstrap-resample the test set to put confidence intervals around
the final numbers. With only ~42 malignant cases in the test set, a
point-estimate like "100% recall" is fragile -- losing one case would drop
it to 97.6%. A bootstrap CI makes that fragility visible instead of implying
false precision.

Run: python3 -m src.evaluate   (from project root)
"""
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, recall_score,
    precision_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src import config
from src.data import load_splits, load_selected_features, TARGET_COL

logger = config.get_logger(__name__)


def choose_threshold(y_true, y_proba, target_recall: float) -> float:
    """Lowest-false-positive threshold that still hits >= target_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        logger.warning("No threshold hits target recall %.2f on validation set; falling back to 0.5", target_recall)
        return 0.5
    chosen_idx = valid_idx[np.argmax(thresholds[valid_idx])]
    return float(thresholds[chosen_idx])


def bootstrap_ci(y_true, y_proba, threshold: float, n_boot: int, seed: int):
    """Percentile bootstrap 95% CI for recall, precision, ROC-AUC at a fixed threshold."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    recalls, precisions, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue  # degenerate resample with only one class present
        pred = (yp >= threshold).astype(int)
        recalls.append(recall_score(yt, pred, zero_division=0))
        precisions.append(precision_score(yt, pred, zero_division=0))
        aucs.append(roc_auc_score(yt, yp))
    def ci(values):
        return {
            "point_estimate_mean_of_resamples": round(float(np.mean(values)), 4),
            "ci_lower_2.5%": round(float(np.percentile(values, 2.5)), 4),
            "ci_upper_97.5%": round(float(np.percentile(values, 97.5)), 4),
        }
    return {"recall": ci(recalls), "precision": ci(precisions), "roc_auc": ci(aucs)}


def main() -> None:
    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)
    features = load_selected_features()
    _, val_df, test_df = load_splits()

    X_val, y_val = val_df[features], val_df[TARGET_COL]
    X_test, y_test = test_df[features], test_df[TARGET_COL]

    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # --- 1. choose threshold on VALIDATION only ---
    n_pos_val = int(y_val.sum())
    achievable = sorted({round(k / n_pos_val, 4) for k in range(n_pos_val - 2, n_pos_val + 1)})
    logger.info(
        "Validation set has %d malignant cases -> achievable recall values near "
        "the target are multiples of 1/%d, e.g. %s. TARGET_RECALL=%.2f will "
        "resolve to whichever of those is the smallest value >= %.2f, which is "
        "usually NOT exactly %.2f -- see config.py's note on this.",
        n_pos_val, n_pos_val, achievable, config.TARGET_RECALL, config.TARGET_RECALL, config.TARGET_RECALL,
    )
    tuned_threshold = choose_threshold(y_val, y_proba_val, config.TARGET_RECALL)
    logger.info("Chosen operating threshold (from validation set, target recall >= %.2f): %.3f",
                config.TARGET_RECALL, tuned_threshold)

    plt.figure(figsize=(6, 5))
    p, r, _ = precision_recall_curve(y_val, y_proba_val)
    plt.plot(r, p)
    plt.axvline(config.TARGET_RECALL, color="gray", linestyle="--", label=f"target recall={config.TARGET_RECALL}")
    plt.xlabel("Recall (malignant caught)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (VALIDATION set -- used to pick threshold)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "precision_recall_curve_validation.png", dpi=150)
    plt.close()

    # --- 2. score TEST set exactly once, at both the default and tuned thresholds ---
    for label, threshold in [("default", 0.5), ("tuned", tuned_threshold)]:
        y_pred = (y_proba_test >= threshold).astype(int)
        logger.info("=== TEST SET at %s threshold (%.3f) ===", label, threshold)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["benign", "malignant"]))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["benign", "malignant"])
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues" if label == "default" else "Reds", colorbar=False)
        plt.title(f"Test Set Confusion Matrix ({label} threshold={threshold:.3f})")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"confusion_matrix_{label}.png", dpi=150)
        plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (TEST set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction malignant")
    plt.title("Calibration Curve (TEST set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close()

    # --- 3. bootstrap confidence intervals on the test set (uncertainty, not tuning) ---
    logger.info("Bootstrapping %d resamples of the test set for confidence intervals...", config.N_BOOTSTRAP)
    ci_default = bootstrap_ci(y_test, y_proba_test, 0.5, config.N_BOOTSTRAP, config.RANDOM_STATE)
    ci_tuned = bootstrap_ci(y_test, y_proba_test, tuned_threshold, config.N_BOOTSTRAP, config.RANDOM_STATE)

    y_pred_default = (y_proba_test >= 0.5).astype(int)
    y_pred_tuned = (y_proba_test >= tuned_threshold).astype(int)

    val_recall_achieved = float(recall_score(y_val, (y_proba_val >= tuned_threshold).astype(int)))
    eval_summary = {
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "n_malignant_test": int(y_test.sum()),
        "n_malignant_validation": n_pos_val,
        "target_recall_config": config.TARGET_RECALL,
        "achievable_recall_values_near_target": achievable,
        "actual_validation_recall_at_tuned_threshold": round(val_recall_achieved, 4),
        "default_threshold": 0.5,
        "tuned_threshold": round(tuned_threshold, 4),
        "test_metrics_default_threshold": {
            "recall": round(recall_score(y_test, y_pred_default), 4),
            "precision": round(precision_score(y_test, y_pred_default), 4),
            "roc_auc": round(float(roc_auc), 4),
            "bootstrap_95ci": ci_default,
        },
        "test_metrics_tuned_threshold": {
            "recall": round(recall_score(y_test, y_pred_tuned), 4),
            "precision": round(precision_score(y_test, y_pred_tuned), 4),
            "roc_auc": round(float(roc_auc), 4),
            "bootstrap_95ci": ci_tuned,
        },
    }
    with open(config.EVAL_SUMMARY_PATH, "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Saved figures to %s and %s", config.FIGURES_DIR, config.EVAL_SUMMARY_PATH)
    logger.info(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
