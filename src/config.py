"""
Central configuration: every seed, split ratio, path, and search setting used
across the pipeline lives here so it's changed in exactly one place instead
of copy-pasted (and silently drifting) across scripts.
"""
import logging
from pathlib import Path

# --- paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "breast_cancer_full.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"
RESULTS_SUMMARY_PATH = MODELS_DIR / "results_summary.json"
EVAL_SUMMARY_PATH = MODELS_DIR / "eval_summary.json"
VIF_REPORT_PATH = MODELS_DIR / "vif_report.json"
SHAP_RANKING_PATH = MODELS_DIR / "shap_feature_ranking.json"

# --- reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# --- splits --------------------------------------------------------------
# test_size: fraction of the FULL dataset held out as test (touched exactly
#   once, at the very end, for the final reported numbers).
# val_size: fraction of the REMAINING (non-test) data held out as validation
#   (used for threshold selection and any other "look at the data and decide"
#   step). With 569 total rows this works out to roughly 64% train / 16%
#   validation / 20% test.
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# --- feature selection (multicollinearity) --------------------------------
# Variance Inflation Factor threshold: iteratively drop the feature with the
# highest VIF until all remaining features are below this. VIF > 10 is a
# widely used rule of thumb for "this feature's variance is meaningfully
# inflated by collinearity with other features."
VIF_THRESHOLD = 10.0

# How much average-precision cost we're willing to accept from dropping
# collinear features, before we'd rather keep the full feature set. This is
# the same magnitude the project's original (naive, pre-review) check used
# as a bare "delta > -0.01" cutoff; it's now used as a proper non-inferiority
# margin instead of an unstated one -- see feature_analysis.py.
VIF_COST_EQUIVALENCE_MARGIN = 0.01

# --- model selection -------------------------------------------------------
CV_SPLITS = 5
CV_REPEATS = 3          # RepeatedStratifiedKFold: more stable estimates on a
                         # small (569-row) dataset than a single 5-fold pass
N_ITER_SEARCH = 25       # RandomizedSearchCV iterations per model family
CV_SCORING = "average_precision"  # NOT raw recall -- see train.py docstring

# --- threshold tuning --------------------------------------------------
# CORRECTNESS NOTE (found by external adversarial review, confirmed by
# reproduction): with ~34 malignant cases in a single validation draw,
# achievable recall values are multiples of 1/34 ~= 0.0294. The previous
# value here (0.98) was NOT achievable at anything other than exactly 100%
# recall (33/34 = 0.9706 < 0.98 < 1.0 = 34/34), which silently collapsed
# threshold selection to "the predicted probability of the single hardest
# validation case" -- a min-statistic over 34 points, about the highest-
# variance estimator available. 0.95 is achievable at 33/34 as well as
# 34/34, which is less degenerate but still fragile; see
# src/stability_analysis.py, which is the actual answer to "how much should
# this be trusted" -- not this constant alone.
TARGET_RECALL = 0.95

# --- uncertainty ---------------------------------------------------------
N_BOOTSTRAP = 2000       # resamples for bootstrap confidence intervals on
                         # the held-out test set

# --- logging ---------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Call once per script/module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
