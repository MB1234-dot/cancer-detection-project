"""
Multi-seed stability analysis for the threshold-tuning step.

Why this exists: external adversarial review found that with only ~34
malignant cases in a single validation draw, `TARGET_RECALL` selection
collapses to (or near) a single order statistic -- the threshold is
essentially "the predicted probability of the single hardest validation
case," which is about as high-variance an estimate as you can construct.
The bootstrap CI in evaluate.py doesn't catch this because it holds the
threshold fixed and resamples the test set -- it quantifies test-set sampling
noise, not threshold-selection noise, which turned out to be the larger of
the two.

This script repeats the entire split -> fit -> threshold-select -> evaluate
loop across many random seeds (varying only the train/val/test partition;
the feature set and model hyperparameters are held fixed, both because
re-running full feature selection and hyperparameter search per seed would
be prohibitively expensive for a 364-row dataset and because the question
here is specifically about split/threshold stability, not architecture
stability) and reports the empirical spread -- which is the honest answer to
"how much should I trust the single-seed threshold and its bootstrap CI."

Run: python3 -m src.stability_analysis   (from project root; requires
train.py and evaluate.py to have been run once already, for best_params)
"""
import json

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

from src import config
from src.data import load_raw, make_splits, load_selected_features, TARGET_COL
from src.evaluate import choose_threshold

logger = config.get_logger(__name__)

N_SEEDS = 200


def main() -> None:
    config.ensure_dirs()
    df, _ = load_raw()
    features = load_selected_features()

    with open(config.RESULTS_SUMMARY_PATH) as f:
        results_summary = json.load(f)
    best_params = results_summary["results"][results_summary["selected_model"]]["best_params"]
    if results_summary["selected_model"] != "logistic_regression":
        logger.warning(
            "Selected model is %s, not logistic_regression -- this stability "
            "analysis is hardcoded to logistic regression and will not reflect "
            "the actually-deployed model. Treat results with that caveat.",
            results_summary["selected_model"],
        )

    logger.info(
        "Running %d seeds: split -> fit (fixed hyperparams from seed-42 search) "
        "-> threshold-select on validation -> evaluate on test.", N_SEEDS
    )

    thresholds, seed_results = [], []
    for seed in range(N_SEEDS):
        train_df, val_df, test_df = make_splits(df, random_state=seed)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                C=best_params.get("clf__C", 1.0),
                class_weight=best_params.get("clf__class_weight"),
                random_state=config.RANDOM_STATE,
            )),
        ])
        pipe.fit(train_df[features], train_df[TARGET_COL])

        y_val, y_proba_val = val_df[TARGET_COL], pipe.predict_proba(val_df[features])[:, 1]
        y_test, y_proba_test = test_df[TARGET_COL], pipe.predict_proba(test_df[features])[:, 1]

        threshold = choose_threshold(y_val, y_proba_val, config.TARGET_RECALL)
        thresholds.append(threshold)

        pred_default = (y_proba_test >= 0.5).astype(int)
        pred_tuned = (y_proba_test >= threshold).astype(int)
        seed_results.append({
            "seed": seed,
            "threshold": threshold,
            "recall_default": recall_score(y_test, pred_default, zero_division=0),
            "precision_default": precision_score(y_test, pred_default, zero_division=0),
            "recall_tuned": recall_score(y_test, pred_tuned, zero_division=0),
            "precision_tuned": precision_score(y_test, pred_tuned, zero_division=0),
        })

    thresholds = np.array(thresholds)
    recall_tuned = np.array([r["recall_tuned"] for r in seed_results])
    precision_tuned = np.array([r["precision_tuned"] for r in seed_results])
    recall_default = np.array([r["recall_default"] for r in seed_results])
    precision_default = np.array([r["precision_default"] for r in seed_results])

    # accounting: compare recall first (the stated goal of tuning), ties broken by precision
    improved = int(np.sum(recall_tuned > recall_default))
    unchanged = int(np.sum(recall_tuned == recall_default))
    worse_or_same_recall_lower_precision = int(np.sum((recall_tuned <= recall_default) & (precision_tuned < precision_default)))

    report = {
        "n_seeds": N_SEEDS,
        "target_recall": config.TARGET_RECALL,
        "threshold_percentiles": {
            "p05": round(float(np.percentile(thresholds, 5)), 4),
            "p25": round(float(np.percentile(thresholds, 25)), 4),
            "median": round(float(np.percentile(thresholds, 50)), 4),
            "p75": round(float(np.percentile(thresholds, 75)), 4),
            "p95": round(float(np.percentile(thresholds, 95)), 4),
            "min": round(float(thresholds.min()), 4),
            "max": round(float(thresholds.max()), 4),
        },
        "seed_42_threshold": round(float(thresholds[config.RANDOM_STATE]), 4) if config.RANDOM_STATE < N_SEEDS else None,
        "mean_recall_default_threshold": round(float(recall_default.mean()), 4),
        "mean_precision_default_threshold": round(float(precision_default.mean()), 4),
        "mean_recall_tuned_threshold": round(float(recall_tuned.mean()), 4),
        "mean_precision_tuned_threshold": round(float(precision_tuned.mean()), 4),
        "seeds_where_tuning_improved_recall": improved,
        "seeds_where_tuning_left_recall_unchanged": unchanged,
        "seeds_where_tuning_did_not_help_recall_and_cost_precision": worse_or_same_recall_lower_precision,
        "recommendation": None,
    }

    recall_gain = report["mean_recall_tuned_threshold"] - report["mean_recall_default_threshold"]
    precision_cost = report["mean_precision_default_threshold"] - report["mean_precision_tuned_threshold"]
    # Recommend tuning ONLY if it delivers a meaningful average recall gain.
    # recall_gain <= 0 means tuning is a net loss (worse or equal recall) on
    # average -- that must recommend the default regardless of precision,
    # not just when precision cost also clears some bar. (An earlier version
    # of this condition only checked precision cost and would have recommended
    # "tuned" even when tuning made recall WORSE on average -- exactly the
    # kind of one-line logic bug this whole analysis exists to catch.)
    if recall_gain > 0.01:
        report["recommendation"] = (
            f"Tuning provides a real average benefit ({recall_gain:+.1%} recall for {precision_cost:.1%} "
            f"precision) across {N_SEEDS} seeds, though the threshold value itself is still unstable "
            f"(range {report['threshold_percentiles']['min']}-{report['threshold_percentiles']['max']}); "
            f"consider using the median threshold across seeds rather than the single seed-42 value."
        )
    else:
        verdict = "made recall WORSE on average" if recall_gain < 0 else "did not meaningfully improve recall"
        report["recommendation"] = (
            f"Use the DEFAULT threshold (0.5) for deployment. Across {N_SEEDS} seeds, threshold "
            f"tuning {verdict} ({recall_gain:+.1%} average recall change) while costing "
            f"{precision_cost:.1%} average precision, and the threshold itself ranges from "
            f"{report['threshold_percentiles']['min']} to {report['threshold_percentiles']['max']} "
            f"depending on the validation draw -- not stable enough to trust as a fixed operating point."
        )

    logger.info("Threshold across %d seeds: median=%.4f, range=[%.4f, %.4f]",
                N_SEEDS, report["threshold_percentiles"]["median"],
                report["threshold_percentiles"]["min"], report["threshold_percentiles"]["max"])
    logger.info("Mean test recall/precision -- default: %.4f/%.4f, tuned: %.4f/%.4f",
                report["mean_recall_default_threshold"], report["mean_precision_default_threshold"],
                report["mean_recall_tuned_threshold"], report["mean_precision_tuned_threshold"])
    logger.info("Tuning improved recall in %d/%d seeds, unchanged in %d/%d, "
                "no-recall-benefit-but-lower-precision in %d/%d",
                improved, N_SEEDS, unchanged, N_SEEDS,
                worse_or_same_recall_lower_precision, N_SEEDS)
    logger.info("RECOMMENDATION: %s", report["recommendation"])

    with open(config.MODELS_DIR / "threshold_stability_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved %s", config.MODELS_DIR / "threshold_stability_report.json")


if __name__ == "__main__":
    main()
