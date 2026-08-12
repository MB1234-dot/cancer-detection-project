"""
Post-hoc, honest measurement of what the 16-vs-30-feature decision actually
costs and buys, using the REAL shipped model -- not the untuned baseline in
feature_analysis.py.

Why this script exists: feature_analysis.py has to decide which feature set
to ship BEFORE hyperparameter tuning happens (train.py reads the feature
list it writes), so its own comparison necessarily uses an untuned baseline
logistic regression and average precision as the CV metric. That is a real,
documented, unavoidable circularity -- not a mistake -- but it means that
comparison cannot tell you how the ACTUAL tuned model (train.py's chosen C,
class_weight) behaves on the metrics people actually read off the README
(recall, precision, ROC-AUC), at the actual deployment threshold (0.5).

External adversarial review (round two) measured this directly and found
that on the real, shipped configuration, the 16-feature (VIF-reduced) model
is measurably and statistically significantly WORSE than the 30-feature
model on recall, precision, and ROC-AUC -- reproduced here. That is a real
cost, and this project should say so plainly instead of leaning on the
untuned baseline's "no meaningful cost" finding as if it settled the
question.

The same review also independently found, though didn't attempt to
quantify as we do here, that the 16-feature model's SHAP explanations are
far more stable across resampled data than the 30-feature model's -- because
severe multicollinearity in the 30-feature set lets a linear model
arbitrarily redistribute "credit" between near-duplicate features (mean
radius / mean perimeter / mean area at r=0.99-1.00) without changing a
single prediction. That instability isn't a hypothetical: below, we resample
the split 100 times, keep the model family and hyperparameters fixed, and
check how often the top-3 SHAP features by mean |value| land on the same 3
features. If they don't, "why did the model flag this patient" has a
different answer depending on which resample of the training data happened
to get used -- which is a real cost of the 30-feature model this comparison
CAN see, and the tuned-recall comparison cannot.

Net: this is a genuine trade-off, not a free lunch in either direction.
This script's whole job is to put both sides of it in one place with
numbers, instead of asserting either "no cost" or "no benefit."

Run: python3 -m src.feature_tradeoff_analysis   (after train.py)
"""
import json

import numpy as np
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_raw, make_splits, load_selected_features, TARGET_COL

logger = config.get_logger(__name__)

N_SEEDS = 100


def _fit(feats, best_params, train_df):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            C=best_params.get("clf__C", 1.0),
            class_weight=best_params.get("clf__class_weight"),
            random_state=config.RANDOM_STATE,
        )),
    ])
    pipe.fit(train_df[feats], train_df[TARGET_COL])
    return pipe


def _top3(pipe, feats, train_df, test_df):
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    background = shap.sample(train_df[feats], min(100, len(train_df)), random_state=config.RANDOM_STATE)
    explainer = shap.LinearExplainer(clf, scaler.transform(background))
    sv = explainer(scaler.transform(test_df[feats]))
    mean_abs = np.abs(sv.values).mean(axis=0)
    ranked = [feats[i] for i in np.argsort(-mean_abs)]
    return tuple(ranked[:3])


def main() -> None:
    config.ensure_dirs()
    df, all_features = load_raw()
    reduced_features = load_selected_features()

    with open(config.RESULTS_SUMMARY_PATH) as f:
        results_summary = json.load(f)
    selected_model = results_summary["selected_model"]
    best_params = results_summary["results"][selected_model]["best_params"]
    if selected_model != "logistic_regression":
        logger.warning(
            "Selected model is %s, not logistic_regression -- this analysis "
            "is hardcoded to logistic regression (for SHAP LinearExplainer "
            "and the fixed hyperparameter dict) and will not reflect the "
            "actually-deployed model. Treat results with that caveat.",
            selected_model,
        )

    logger.info(
        "Running %d splits: fit tuned logistic regression (C=%.4f, "
        "class_weight=%s from the seed-42 hyperparameter search) on BOTH "
        "the %d-feature and %d-feature sets, per split -- measuring test-set "
        "recall/precision/AUC cost and top-3 SHAP-feature consensus benefit.",
        N_SEEDS, best_params.get("clf__C", 1.0), best_params.get("clf__class_weight"),
        len(all_features), len(reduced_features),
    )

    rows = {"full": [], "reduced": []}
    top3_full, top3_reduced = [], []

    for seed in range(N_SEEDS):
        train_df, _, test_df = make_splits(df, random_state=seed)
        y_test = test_df[TARGET_COL]

        for key, feats, top3_list in [
            ("full", all_features, top3_full),
            ("reduced", reduced_features, top3_reduced),
        ]:
            pipe = _fit(feats, best_params, train_df)
            y_proba = pipe.predict_proba(test_df[feats])[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            rows[key].append({
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            })
            top3_list.append(_top3(pipe, feats, train_df, test_df))

    def consensus_fraction(top3_list):
        """Fraction of seeds whose top-3 SHAP feature SET matches the modal (most common) set."""
        sets = [frozenset(t) for t in top3_list]
        counts = {}
        for s in sets:
            counts[s] = counts.get(s, 0) + 1
        mode_count = max(counts.values())
        return mode_count / len(sets), len(counts)

    metrics_report = {}
    for metric in ("recall", "precision", "roc_auc"):
        full_vals = np.array([r[metric] for r in rows["full"]])
        reduced_vals = np.array([r[metric] for r in rows["reduced"]])
        diff = reduced_vals - full_vals
        # NOTE: seeds are independent random re-splits, not k-fold CV folds
        # sharing a training set, so the standard (not Nadeau-Bengio-corrected)
        # paired t-test applies -- but adjacent splits still overlap in which
        # rows land in train/test, so treat this as a good approximation, not
        # an exact independence guarantee. A signed-rank test is included too
        # since it doesn't assume normal differences.
        t_stat, p_ttest = stats.ttest_rel(reduced_vals, full_vals)
        w_stat, p_wilcoxon = stats.wilcoxon(reduced_vals, full_vals)
        metrics_report[metric] = {
            "mean_full_30_features": round(float(full_vals.mean()), 4),
            "mean_reduced_16_features": round(float(reduced_vals.mean()), 4),
            "mean_diff_reduced_minus_full": round(float(diff.mean()), 4),
            "paired_ttest_p": round(float(p_ttest), 6),
            "wilcoxon_p": round(float(p_wilcoxon), 6),
            "significant_at_0.05": bool(p_ttest < 0.05),
        }
        logger.info(
            "  %-10s full=%.4f reduced=%.4f diff=%+.4f (paired t-test p=%.2e, %s)",
            metric, full_vals.mean(), reduced_vals.mean(), diff.mean(), p_ttest,
            "SIGNIFICANT" if p_ttest < 0.05 else "not significant",
        )

    frac_full, n_unique_full = consensus_fraction(top3_full)
    frac_reduced, n_unique_reduced = consensus_fraction(top3_reduced)
    logger.info(
        "Top-3 SHAP feature-set consensus across %d splits: full=%d/%d splits "
        "match the modal set (%d distinct sets seen), reduced=%d/%d splits "
        "match the modal set (%d distinct sets seen).",
        N_SEEDS, int(round(frac_full * N_SEEDS)), N_SEEDS, n_unique_full,
        int(round(frac_reduced * N_SEEDS)), N_SEEDS, n_unique_reduced,
    )

    # Build the summary FROM the numbers just computed, not as a hardcoded
    # narrative -- this script exists specifically to replace an asserted
    # conclusion ("no meaningful cost") with a measured one, so its own
    # summary field has to follow the same rule instead of assuming this
    # run reproduces round-two's exact numbers (they may not, e.g. under a
    # different sklearn version -- see README's note on unpinned deps).
    significant_metrics = [m for m, r in metrics_report.items() if r["significant_at_0.05"]]
    worse_metrics = [m for m in significant_metrics if metrics_report[m]["mean_diff_reduced_minus_full"] < 0]
    better_metrics = [m for m in significant_metrics if metrics_report[m]["mean_diff_reduced_minus_full"] > 0]
    if worse_metrics:
        cost_sentence = (
            f"On THIS run, the 16-feature (VIF-reduced) model is measurably and "
            f"statistically significantly worse than the 30-feature model on: "
            f"{', '.join(worse_metrics)} (see performance_cost_reduced_vs_full). "
            f"That is a real cost, not noise."
        )
    elif better_metrics:
        cost_sentence = (
            f"On THIS run, the 16-feature model was significantly BETTER on: "
            f"{', '.join(better_metrics)} -- re-check performance_cost_reduced_vs_full "
            f"before trusting this, it does not match round-two review's finding "
            f"and may indicate an environment or methodology difference worth "
            f"investigating rather than good news to accept at face value."
        )
    else:
        cost_sentence = (
            "On THIS run, no metric showed a statistically significant "
            "difference between the 16-feature and 30-feature models -- this "
            "does not match round-two external review's finding of a "
            "significant cost, and should be treated as a discrepancy to "
            "investigate (e.g. dependency versions, see README) rather than "
            "as evidence the earlier finding was wrong."
        )

    stability_gap = frac_reduced - frac_full
    if stability_gap > 0.1:
        benefit_sentence = (
            f"It is also far more stable in which 3 features its SHAP "
            f"explanation names as most important across resampled training "
            f"data ({frac_reduced:.0%} vs {frac_full:.0%} of splits matching "
            f"the modal top-3 set) -- a real, measured benefit the untuned "
            f"feature_analysis.py comparison cannot see."
        )
    elif stability_gap < -0.1:
        benefit_sentence = (
            f"Contrary to round-two review's finding, THIS run found the "
            f"30-feature model's SHAP top-3 MORE stable ({frac_full:.0%} vs "
            f"{frac_reduced:.0%}) -- treat as a discrepancy to investigate, "
            f"not as settled."
        )
    else:
        benefit_sentence = (
            f"SHAP top-3 stability was roughly comparable between the two "
            f"feature sets on this run ({frac_reduced:.0%} reduced vs "
            f"{frac_full:.0%} full) -- no clear stability benefit observed here."
        )

    report = {
        "n_splits": N_SEEDS,
        "model": selected_model,
        "hyperparameters_used": best_params,
        "performance_cost_reduced_vs_full": metrics_report,
        "shap_top3_consensus": {
            "full_30_features": {
                "fraction_matching_modal_set": round(frac_full, 4),
                "n_distinct_sets_across_splits": n_unique_full,
            },
            "reduced_16_features": {
                "fraction_matching_modal_set": round(frac_reduced, 4),
                "n_distinct_sets_across_splits": n_unique_reduced,
            },
        },
        "summary": (
            f"{cost_sentence} {benefit_sentence} Whether a recall/precision/AUC "
            f"cost is worth an explanation-stability benefit (or vice versa) is "
            f"a judgment call, not a fact this script can settle -- the honest "
            f"thing to do is state both sides with the actual numbers from this "
            f"run, which is what this report is for."
        ),
    }
    with open(config.MODELS_DIR / "feature_tradeoff_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved %s", config.MODELS_DIR / "feature_tradeoff_report.json")


if __name__ == "__main__":
    main()
