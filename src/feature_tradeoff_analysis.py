"""
Post-hoc, honest measurement of what the 16-vs-30-feature decision actually
costs and buys, using REAL tuned models -- not the untuned baseline in
feature_analysis.py.

Why this script exists: feature_analysis.py has to decide which feature set
to ship BEFORE hyperparameter tuning happens (train.py reads the feature
list it writes), so its own comparison necessarily uses an untuned baseline
logistic regression and average precision as the CV metric. That is a real,
documented, unavoidable circularity -- not a mistake -- but it means that
comparison cannot tell you how a properly tuned model behaves on the
metrics people actually read off the README (recall, precision, ROC-AUC),
at the actual deployment threshold (0.5).

CORRECTNESS NOTE (found by external adversarial review, round three): an
earlier version of this script applied ONE hyperparameter config -- the C
tuned by train.py specifically for the 16-feature set -- to BOTH feature
sets. That's not a neutral comparison: it runs the 30-feature model at ~21x
weaker regularization than a search would pick for it, which is exactly the
regime where multicollinearity does the most damage to a linear model. That
single choice was the entire cause of a previously-reported "no significant
cost" finding that round-two review's own (correctly, independently-tuned)
measurement contradicted -- confirmed by round three, which showed the two
results are bit-identical across sklearn 1.8.0/1.9.0, ruling out an
environment explanation.

This version runs BOTH comparisons, because they answer different
questions, and reports both plainly instead of picking one:

  - "matched regularization": every arm gets the SAME hyperparameters
    (the ones train.py chose for the deployed 16-feature model). Answers
    "at these settings, does dropping features cost anything?" -- useful
    for isolating the effect of the feature set alone, but NOT the
    question "which model should ship," since nobody would deploy a
    30-feature model tuned for a 16-feature problem.
  - "independently tuned": each arm gets its OWN best C, found by a quick
    per-split CV search. Answers the actual deployment question: "if I
    tune each candidate properly and pick the best, which wins?"

The same review found, and this script also measures, that the 16-feature
model's SHAP explanations are far more stable across resampled data than
the 30-feature model's -- because severe multicollinearity in the
30-feature set lets a linear model arbitrarily redistribute "credit"
between near-duplicate features (mean radius / mean perimeter / mean area
at r=0.99-1.00) without changing a single prediction. Below, we resample
the split 100 times and check how often the top-3 SHAP features by mean
|value| land on the same 3 features -- a real cost of the 30-feature model
that the tuned-recall comparison alone cannot see, in either direction.

STATISTICAL NOTE (round three): all 100 splits resample the SAME 569 rows,
so the splits are not independent -- adjacent splits share most of their
data. This makes paired-test p-values here (and in round two's own
analysis) anti-conservative: real, but overstated in magnitude. Win/loss
counts (how many of the 100 splits favor each arm) are the more honest
summary of the same evidence and are reported alongside the p-values for
that reason -- trust the counts over the exponents.

Net: this is a genuine trade-off, not a free lunch in either direction.
This script's whole job is to put multiple honest cuts of it in one place
with numbers, instead of asserting "no cost," "definitely worse," or
picking whichever comparison tells the more flattering story.

Run: python3 -m src.feature_tradeoff_analysis   (after train.py)
"""
import json

import numpy as np
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_raw, make_splits, load_selected_features, TARGET_COL

logger = config.get_logger(__name__)

N_SEEDS = 100
C_GRID = np.logspace(-3, 2, 10)  # coarser than train.py's search (25 iters x 15 CV folds) --
                                  # this runs 200x (100 splits x 2 feature sets), so it trades
                                  # some search resolution for tractable runtime.


def _fit(feats, C, class_weight, train_df):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, C=C, class_weight=class_weight, random_state=config.RANDOM_STATE,
        )),
    ])
    pipe.fit(train_df[feats], train_df[TARGET_COL])
    return pipe


def _tune_C(feats, class_weight, train_df, seed):
    """Quick per-split C search: 5-fold CV, average precision, over C_GRID.
    Returns this feature set's own best C -- NOT the other arm's."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_c, best_score = C_GRID[0], -np.inf
    for c in C_GRID:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=c, class_weight=class_weight, random_state=config.RANDOM_STATE)),
        ])
        score = cross_val_score(
            pipe, train_df[feats], train_df[TARGET_COL], cv=cv, scoring="average_precision", n_jobs=-1,
        ).mean()
        if score > best_score:
            best_score, best_c = score, c
    return float(best_c)


def _top3(pipe, feats, train_df, test_df):
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    background = shap.sample(train_df[feats], min(100, len(train_df)), random_state=config.RANDOM_STATE)
    explainer = shap.LinearExplainer(clf, scaler.transform(background))
    sv = explainer(scaler.transform(test_df[feats]))
    mean_abs = np.abs(sv.values).mean(axis=0)
    ranked = [feats[i] for i in np.argsort(-mean_abs)]
    return tuple(ranked[:3])


def _summarize_metric(full_vals, reduced_vals):
    """One arm's comparison for one metric: means, paired tests (labeled
    anti-conservative -- see module docstring), and win/loss counts, which
    round-three review identified as the more honest summary of the same
    evidence since the underlying splits overlap."""
    diff = reduced_vals - full_vals
    t_stat, p_ttest = stats.ttest_rel(reduced_vals, full_vals)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(reduced_vals, full_vals)
    except ValueError:
        # all differences are zero -- wilcoxon is undefined, not an error condition
        w_stat, p_wilcoxon = 0.0, 1.0
    full_wins = int(np.sum(full_vals > reduced_vals))
    reduced_wins = int(np.sum(reduced_vals > full_vals))
    ties = int(np.sum(full_vals == reduced_vals))
    return {
        "mean_full_30_features": round(float(full_vals.mean()), 4),
        "mean_reduced_16_features": round(float(reduced_vals.mean()), 4),
        "mean_diff_reduced_minus_full": round(float(diff.mean()), 4),
        "full_wins": full_wins,
        "reduced_wins": reduced_wins,
        "ties": ties,
        "paired_ttest_p": round(float(p_ttest), 6),
        "wilcoxon_p": round(float(p_wilcoxon), 6),
        "significant_at_0.05": bool(p_ttest < 0.05),
        "note": "p-values here are anti-conservative (all splits resample the same 569 "
                "rows, so splits are not independent) -- trust full_wins/reduced_wins/ties "
                "over the exact p-value magnitude.",
    }


def consensus_fraction(top3_list):
    """Fraction of seeds whose top-3 SHAP feature SET matches the modal (most common) set."""
    sets = [frozenset(t) for t in top3_list]
    counts = {}
    for s in sets:
        counts[s] = counts.get(s, 0) + 1
    mode_count = max(counts.values())
    return mode_count / len(sets), len(counts)


def main() -> None:
    config.ensure_dirs()
    df, all_features = load_raw()
    reduced_features = load_selected_features()

    with open(config.RESULTS_SUMMARY_PATH) as f:
        results_summary = json.load(f)
    selected_model = results_summary["selected_model"]
    best_params = results_summary["results"][selected_model]["best_params"]
    deployed_C = best_params.get("clf__C", 1.0)
    class_weight = best_params.get("clf__class_weight")
    if selected_model != "logistic_regression":
        logger.warning(
            "Selected model is %s, not logistic_regression -- this analysis "
            "is hardcoded to logistic regression (for SHAP LinearExplainer "
            "and the fixed hyperparameter dict) and will not reflect the "
            "actually-deployed model. Treat results with that caveat.",
            selected_model,
        )

    logger.info(
        "Running %d splits x 2 arms (matched regularization at the deployed "
        "C=%.4f, class_weight=%s; and independently tuned per feature set) "
        "on the %d-feature and %d-feature sets -- measuring test-set "
        "recall/precision/AUC and top-3 SHAP-feature consensus.",
        N_SEEDS, deployed_C, class_weight, len(all_features), len(reduced_features),
    )

    matched_rows = {"full": [], "reduced": []}
    tuned_rows = {"full": [], "reduced": []}
    tuned_Cs = {"full": [], "reduced": []}
    top3_full, top3_reduced = [], []

    for seed in range(N_SEEDS):
        train_df, _, test_df = make_splits(df, random_state=seed)
        y_test = test_df[TARGET_COL]

        # Arm A: matched regularization -- both feature sets get the SAME
        # (deployed) hyperparameters. Also the source of the SHAP top-3
        # consensus numbers below, since that's what the deployed app
        # actually explains.
        for key, feats, top3_list in [
            ("full", all_features, top3_full),
            ("reduced", reduced_features, top3_reduced),
        ]:
            pipe = _fit(feats, deployed_C, class_weight, train_df)
            y_proba = pipe.predict_proba(test_df[feats])[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            matched_rows[key].append({
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            })
            top3_list.append(_top3(pipe, feats, train_df, test_df))

        # Arm B: each feature set gets its OWN best C for this split.
        for key, feats in [("full", all_features), ("reduced", reduced_features)]:
            c = _tune_C(feats, class_weight, train_df, seed)
            tuned_Cs[key].append(c)
            pipe = _fit(feats, c, class_weight, train_df)
            y_proba = pipe.predict_proba(test_df[feats])[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            tuned_rows[key].append({
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            })

        if (seed + 1) % 20 == 0:
            logger.info("  ...%d/%d splits done", seed + 1, N_SEEDS)

    matched_report, tuned_report = {}, {}
    for metric in ("recall", "precision", "roc_auc"):
        matched_report[metric] = _summarize_metric(
            np.array([r[metric] for r in matched_rows["full"]]),
            np.array([r[metric] for r in matched_rows["reduced"]]),
        )
        tuned_report[metric] = _summarize_metric(
            np.array([r[metric] for r in tuned_rows["full"]]),
            np.array([r[metric] for r in tuned_rows["reduced"]]),
        )
        logger.info(
            "  %-10s matched: full=%.4f reduced=%.4f (%d/%d/%d win/win/tie)  |  "
            "tuned: full=%.4f reduced=%.4f (%d/%d/%d win/win/tie)",
            metric,
            matched_report[metric]["mean_full_30_features"], matched_report[metric]["mean_reduced_16_features"],
            matched_report[metric]["full_wins"], matched_report[metric]["reduced_wins"], matched_report[metric]["ties"],
            tuned_report[metric]["mean_full_30_features"], tuned_report[metric]["mean_reduced_16_features"],
            tuned_report[metric]["full_wins"], tuned_report[metric]["reduced_wins"], tuned_report[metric]["ties"],
        )

    frac_full, n_unique_full = consensus_fraction(top3_full)
    frac_reduced, n_unique_reduced = consensus_fraction(top3_reduced)
    logger.info(
        "Top-3 SHAP feature-set consensus (matched-regularization arm) across %d splits: "
        "full=%d/%d splits match the modal set (%d distinct sets seen), "
        "reduced=%d/%d splits match the modal set (%d distinct sets seen).",
        N_SEEDS, int(round(frac_full * N_SEEDS)), N_SEEDS, n_unique_full,
        int(round(frac_reduced * N_SEEDS)), N_SEEDS, n_unique_reduced,
    )

    # Build the summary FROM the numbers just computed for BOTH arms, not a
    # hardcoded narrative -- and lead with the "tuned" arm since that's the
    # one that actually answers "which model should ship."
    def arm_verdict(report, arm_label):
        sig = [m for m, r in report.items() if r["significant_at_0.05"]]
        worse = [m for m in sig if report[m]["mean_diff_reduced_minus_full"] < 0]
        better = [m for m in sig if report[m]["mean_diff_reduced_minus_full"] > 0]
        if worse:
            return (f"[{arm_label}] the 16-feature model is worse on {', '.join(worse)} "
                     f"(win/loss counts: " +
                     ", ".join(f"{m} {report[m]['full_wins']}/{report[m]['reduced_wins']}/{report[m]['ties']}" for m in worse) +
                     ").")
        if better:
            return f"[{arm_label}] the 16-feature model is BETTER on {', '.join(better)}."
        return f"[{arm_label}] no metric shows a significant difference."

    tuned_verdict = arm_verdict(tuned_report, "independently tuned (which model to ship)")
    matched_verdict = arm_verdict(matched_report, "matched regularization (isolating the feature set alone)")

    stability_gap = frac_reduced - frac_full
    if stability_gap > 0.1:
        benefit_sentence = (
            f"The 16-feature model is far more stable in which 3 features its SHAP "
            f"explanation names as most important across resampled training data "
            f"({frac_reduced:.0%} vs {frac_full:.0%} of splits matching the modal "
            f"top-3 set) -- a real, measured benefit neither performance comparison "
            f"above can see."
        )
    elif stability_gap < -0.1:
        benefit_sentence = (
            f"THIS run found the 30-feature model's SHAP top-3 MORE stable "
            f"({frac_full:.0%} vs {frac_reduced:.0%}) -- contrary to prior findings, "
            f"treat as a discrepancy to investigate, not as settled."
        )
    else:
        benefit_sentence = (
            f"SHAP top-3 stability was roughly comparable between the two feature "
            f"sets on this run ({frac_reduced:.0%} reduced vs {frac_full:.0%} full)."
        )

    report = {
        "n_splits": N_SEEDS,
        "model": selected_model,
        "deployed_hyperparameters": best_params,
        "arm_a_matched_regularization": {
            "description": "Both feature sets fit with the SAME hyperparameters (the deployed C). "
                            "Answers: 'does dropping features cost anything, holding regularization fixed?' "
                            "NOT the deployment question -- see arm_b for that.",
            "performance": matched_report,
        },
        "arm_b_independently_tuned": {
            "description": "Each feature set gets its OWN best C per split (quick 5-fold CV search "
                            "over a 10-point log-spaced grid). Answers the actual deployment question: "
                            "'if each candidate is tuned properly, which wins?'",
            "mean_tuned_C_full_30_features": round(float(np.mean(tuned_Cs["full"])), 4),
            "mean_tuned_C_reduced_16_features": round(float(np.mean(tuned_Cs["reduced"])), 4),
            "performance": tuned_report,
        },
        "shap_top3_consensus": {
            "computed_on": "arm_a_matched_regularization (the deployed model's actual C)",
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
            f"{tuned_verdict} {matched_verdict} {benefit_sentence} Whether a "
            f"recall/precision/AUC cost (arm_b) is worth an explanation-stability "
            f"benefit is a judgment call this script can't settle, but arm_b is "
            f"the comparison that should drive a 'which model to ship' decision -- "
            f"arm_a is included because it isolates a different, also-useful "
            f"question (does regularization alone explain the gap?)."
        ),
    }
    with open(config.MODELS_DIR / "feature_tradeoff_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved %s", config.MODELS_DIR / "feature_tradeoff_report.json")
    logger.info("SUMMARY: %s", report["summary"])


if __name__ == "__main__":
    main()
