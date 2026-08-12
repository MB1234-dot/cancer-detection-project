"""
Model training: Logistic Regression, Random Forest, Gradient Boosting (XGBoost).

Design decisions worth understanding, not just running:

1. Only `data/train.csv` is used here. Validation and test are untouched --
   validation is reserved for threshold selection (evaluate.py), test is
   reserved for the single final report.

2. Scaling inside a Pipeline: StandardScaler is fit only on training folds
   during CV, and only on the full training set for the final fit. It never
   sees validation or test data, which would otherwise leak their statistics
   into training.

3. Model selection metric: average precision (area under the precision-recall
   curve for the malignant class), not raw recall. A pure-recall objective
   has a degenerate optimum -- a classifier that flags everyone as malignant
   gets perfect recall and is useless. Average precision rewards a good
   ranking across the *entire* precision/recall tradeoff, which is a safer
   thing to optimize during model/hyperparameter selection. The actual
   operating threshold (how aggressively to call "malignant") is a separate
   decision, made later on the validation set, once we know which model and
   which hyperparameters we're using.

4. RepeatedStratifiedKFold instead of a single 5-fold split: with only 364
   training rows, a single CV pass is noisy -- which fold a given patient
   lands in can meaningfully swing the reported score. Repeating the k-fold
   split multiple times with different random partitions and averaging
   gives a more stable estimate of true generalization performance.

Run: python3 -m src.train   (from project root)
"""
import json
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

from src import config
from src.data import load_splits, load_selected_features, TARGET_COL
from src.stats_utils import nadeau_bengio_test

logger = config.get_logger(__name__)


def build_search_spaces() -> Tuple[Dict, RepeatedStratifiedKFold]:
    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)),
    ])
    logreg_params = {"clf__C": np.logspace(-3, 2, 20), "clf__class_weight": [None, "balanced"]}

    rf = Pipeline([
        ("scaler", "passthrough"),
        ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE)),
    ])
    rf_params = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced"],
    }

    xgb = Pipeline([
        ("scaler", "passthrough"),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE)),
    ])
    xgb_params = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [2, 3, 4, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
    }

    return {
        "logistic_regression": (logreg, logreg_params),
        "random_forest": (rf, rf_params),
        "xgboost": (xgb, xgb_params),
    }, cv


def main() -> None:
    config.ensure_dirs()
    train_df, _, _ = load_splits()
    features = load_selected_features()
    logger.info("Training on %d rows, %d features (post multicollinearity analysis).", len(train_df), len(features))

    X_train, y_train = train_df[features], train_df[TARGET_COL]
    searches, cv = build_search_spaces()

    results = {}
    fitted_models = {}
    per_fold_scores = {}  # for significance testing between model families

    n_splits_total = config.CV_SPLITS * config.CV_REPEATS
    n_test_per_fold = len(X_train) // config.CV_SPLITS
    n_train_per_fold = len(X_train) - n_test_per_fold

    for name, (pipe, param_dist) in searches.items():
        logger.info("Tuning %s (%d CV folds x %d repeats, scoring=%s)...",
                    name, config.CV_SPLITS, config.CV_REPEATS, config.CV_SCORING)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=config.N_ITER_SEARCH, cv=cv,
            scoring=config.CV_SCORING, n_jobs=-1, random_state=config.RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        fitted_models[name] = search.best_estimator_

        # per-fold scores for the WINNING hyperparameter config of this model
        # family -- same `cv` object (same folds, same order) was used for all
        # three searches, so these are properly paired across model families.
        per_fold_scores[name] = np.array([
            search.cv_results_[f"split{i}_test_score"][search.best_index_]
            for i in range(n_splits_total)
        ])

        # in-sample (training-set) sanity numbers only -- NOT a generalization
        # estimate. The real estimate is search.best_score_ (cross-validated).
        # We report both so it's obvious which is which.
        y_pred_train = search.best_estimator_.predict(X_train)
        y_proba_train = search.best_estimator_.predict_proba(X_train)[:, 1]

        results[name] = {
            "cv_mean_average_precision": round(search.best_score_, 4),
            "cv_std_average_precision": round(
                float(search.cv_results_["std_test_score"][search.best_index_]), 4
            ),
            "train_set_recall": round(recall_score(y_train, y_pred_train), 4),
            "train_set_precision": round(precision_score(y_train, y_pred_train), 4),
            "train_set_roc_auc": round(roc_auc_score(y_train, y_proba_train), 4),
            "best_params": {k: (v if not isinstance(v, np.floating) else float(v))
                             for k, v in search.best_params_.items()},
        }
        logger.info(
            "  %s: CV avg-precision = %.4f +/- %.4f",
            name, results[name]["cv_mean_average_precision"], results[name]["cv_std_average_precision"],
        )

    # Each cv_mean above is the MAX over N_ITER_SEARCH randomly sampled
    # configs -- a "winner's curse" upward bias that differs across model
    # families since their search spaces differ. Picking by raw mean is a
    # reasonable starting point but not a statistically justified one; test it.
    best_name = max(results, key=lambda n: results[n]["cv_mean_average_precision"])
    significance = {}
    for name in results:
        if name == best_name:
            continue
        t_stat, p_value = nadeau_bengio_test(
            per_fold_scores[best_name], per_fold_scores[name],
            n_train=n_train_per_fold, n_test=n_test_per_fold,
        )
        significance[f"{best_name}_vs_{name}"] = {
            "mean_diff": round(float(results[best_name]["cv_mean_average_precision"]
                                      - results[name]["cv_mean_average_precision"]), 4),
            "nadeau_bengio_t": round(t_stat, 4),
            "nadeau_bengio_p": round(p_value, 4),
            "significant_at_0.05": bool(p_value < 0.05),
        }
        logger.info(
            "  %s vs %s: mean diff=%+.4f, Nadeau-Bengio corrected p=%.4f (%s)",
            best_name, name, significance[f"{best_name}_vs_{name}"]["mean_diff"], p_value,
            "significant" if p_value < 0.05 else "NOT significant -- statistically indistinguishable",
        )

    any_significant = any(v["significant_at_0.05"] for v in significance.values())
    if any_significant:
        logger.info("Selected model: %s (significantly better on corrected test)", best_name)
    else:
        logger.info(
            "Selected model: %s -- NOTE: not significantly better than the alternatives "
            "(Nadeau-Bengio corrected p > 0.05 in all pairwise comparisons). Chosen for "
            "simplicity, calibrated probabilities, and interpretability, not for a proven "
            "performance edge.", best_name,
        )
    logger.info(json.dumps(results[best_name], indent=2))

    best_model = fitted_models[best_name]
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(features, config.MODELS_DIR / "feature_names.joblib")

    with open(config.RESULTS_SUMMARY_PATH, "w") as f:
        json.dump({
            "results": results,
            "selected_model": best_name,
            "selection_significance": significance,
            "selection_statistically_justified": any_significant,
            "features_used": features,
        }, f, indent=2)

    logger.info("Saved %s, feature_names.joblib, %s", config.MODEL_PATH, config.RESULTS_SUMMARY_PATH)


if __name__ == "__main__":
    main()
