"""
Multicollinearity analysis via Variance Inflation Factor (VIF).

Why this matters here specifically: the correlation heatmap from eda.py shows
`mean radius`, `mean perimeter`, and `mean area` correlating at 0.99-1.00 with
each other (they're almost the same physical quantity measured three ways),
and `worst radius`/`worst perimeter` at 0.97. For a linear model like logistic
regression, severe multicollinearity means the fitted coefficients (and by
extension SHAP values, which for a linear model are a direct function of the
coefficients) become unstable -- the model can arbitrarily redistribute
"credit" between near-duplicate features. Two different runs of the same
training procedure, or even two similar rows, could get a different-looking
explanation despite reflecting essentially the same underlying signal. That
undermines the entire point of building an explainability layer.

The fix here is standard: VIF for a feature is roughly 1 / (1 - R^2) from
regressing that feature on all other features. A VIF of 10 conventionally
means "90%+ of this feature's variance is explainable by the other features"
-- i.e. it's carrying almost no independent information. We iteratively drop
the single worst offender and recompute (dropping one at a time, not all at
once, because removing a feature changes every other feature's VIF).

Critically: this entire analysis runs on the TRAINING split only. Using
validation or test data here -- even just to decide which features to keep --
would be a subtler version of the same leakage bug this project already
fixed once for threshold selection.

Run: python3 -m src.feature_analysis   (from project root)
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src import config
from src.data import load_splits, TARGET_COL

logger = config.get_logger(__name__)


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """VIF per column. X must already be numeric with no target column.

    CORRECTNESS NOTE (found by external adversarial review, confirmed by
    reproduction): `variance_inflation_factor` regresses each column on the
    others to get R^2. Without an intercept in that regression, it's forced
    through the origin, which conflates each feature's mean level with its
    collinearity -- and every WDBC feature is strictly positive, so this
    inflates VIFs by 1-2 orders of magnitude. An earlier version of this
    function omitted `add_constant`, which reproduced the exact (wrong)
    values `mean radius`=63499, `worst radius`=8872 previously reported here,
    and caused the iterative selection below to strip the entire "mean"/
    "worst" feature block (where WDBC's actual discriminative signal lives)
    before touching the weakly-correlated "error" block. Correcting this
    changes the result materially: 14 features dropped instead of 23, 16
    retained instead of 7, and `mean radius` -- the feature the buggy version
    called the single worst offender -- survives.
    """
    X_const = add_constant(X)
    vif = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
        index=X_const.columns,
    ).drop("const")
    return vif.sort_values(ascending=False)


def iterative_vif_selection(X: pd.DataFrame, threshold: float):
    """Drop the highest-VIF feature repeatedly until all VIFs < threshold.

    Returns (retained_columns, drop_log) where drop_log records, in order,
    which feature was dropped and what its VIF was at the time.
    """
    remaining = list(X.columns)
    drop_log = []
    while True:
        vif = compute_vif(X[remaining])
        worst_feature, worst_value = vif.index[0], vif.iloc[0]
        if worst_value < threshold or len(remaining) <= 2:
            break
        drop_log.append({"feature": worst_feature, "vif_at_removal": round(float(worst_value), 2)})
        remaining.remove(worst_feature)
    return remaining, drop_log


def compare_full_vs_reduced(train_df: pd.DataFrame, all_features, selected_features) -> dict:
    """Quick CV comparison: does dropping redundant features cost us accuracy?

    IMPORTANT CAVEAT (flagged by external review): this uses a fixed,
    UNTUNED logistic regression (C=1.0, the sklearn default, class_weight=
    "balanced") for both feature sets, NOT the hyperparameter-tuned model
    that train.py eventually ships. That's a real limitation, not a
    reporting choice made after the fact: feature selection has to happen
    before hyperparameter tuning in a leakage-safe pipeline (train.py reads
    the feature list this script writes), so at this point in the pipeline
    there is no "tuned C" yet to use -- tuning it here would mean tuning
    hyperparameters for a feature set that hasn't been chosen, which is
    circular. The result is a fair full-vs-reduced comparison for a generic,
    reasonably-regularized linear model, but it is NOT a guarantee that the
    specific deployed model (train.py's tuned C) responds to the reduced
    feature set the same way. Result keys are named `baseline_*` throughout
    to make that distinction visible instead of implying these numbers are
    the shipped model's numbers.
    """
    y = train_df[TARGET_COL]
    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE
    )
    results = {}
    for label, feats in [("baseline_full_30_features", all_features), ("baseline_vif_selected_features", selected_features)]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE, class_weight="balanced")),
        ])
        scores = cross_val_score(pipe, train_df[feats], y, cv=cv, scoring=config.CV_SCORING, n_jobs=-1)
        results[label] = {
            "n_features": len(feats),
            "mean_average_precision": round(float(np.mean(scores)), 4),
            "std_average_precision": round(float(np.std(scores)), 4),
            "note": "untuned baseline (C=1.0), not the tuned deployed model",
        }
    return results


def main() -> None:
    config.ensure_dirs()
    train_df, _, _ = load_splits()
    all_features = [c for c in train_df.columns if c != TARGET_COL]

    logger.info("Computing VIF on %d training rows, %d features...", len(train_df), len(all_features))
    initial_vif = compute_vif(train_df[all_features])
    logger.info("Top 5 highest-VIF features before pruning:\n%s", initial_vif.head(5).round(1).to_string())

    selected_features, drop_log = iterative_vif_selection(train_df[all_features], config.VIF_THRESHOLD)
    logger.info(
        "Dropped %d/%d features (VIF >= %.0f): %s",
        len(drop_log), len(all_features), config.VIF_THRESHOLD,
        ", ".join(d["feature"] for d in drop_log),
    )
    logger.info("Retained %d features.", len(selected_features))

    comparison = compare_full_vs_reduced(train_df, all_features, selected_features)
    logger.info("CV comparison (%s, higher is better):", config.CV_SCORING)
    for label, res in comparison.items():
        logger.info(
            "  %-22s n_features=%2d  mean=%.4f  std=%.4f",
            label, res["n_features"], res["mean_average_precision"], res["std_average_precision"],
        )

    delta = (comparison["baseline_vif_selected_features"]["mean_average_precision"]
             - comparison["baseline_full_30_features"]["mean_average_precision"])
    logger.info(
        "Delta from dropping collinear features: %+.4f average precision. "
        "%s",
        delta,
        "No meaningful accuracy cost -- proceeding with the reduced, more "
        "interpretable feature set." if delta > -0.01 else
        "Meaningful accuracy cost detected -- keeping the full feature set "
        "despite the multicollinearity (see README for reasoning)."
    )

    final_features = selected_features if delta > -0.01 else all_features

    report = {
        "vif_threshold": config.VIF_THRESHOLD,
        "all_features": all_features,
        "dropped_features": drop_log,
        "selected_features_by_vif": selected_features,
        "cv_comparison": comparison,
        "delta_average_precision": round(float(delta), 4),
        "final_decision": "reduced" if final_features is selected_features else "full",
        "final_features": final_features,
    }
    with open(config.VIF_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    with open(config.SELECTED_FEATURES_PATH, "w") as f:
        json.dump({"selected_features": final_features}, f, indent=2)

    logger.info("Saved %s and %s", config.VIF_REPORT_PATH, config.SELECTED_FEATURES_PATH)


if __name__ == "__main__":
    main()
