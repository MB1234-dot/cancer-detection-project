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

# --- model selection -------------------------------------------------------
CV_SPLITS = 5
CV_REPEATS = 3          # RepeatedStratifiedKFold: more stable estimates on a
                         # small (569-row) dataset than a single 5-fold pass
N_ITER_SEARCH = 25       # RandomizedSearchCV iterations per model family
CV_SCORING = "average_precision"  # NOT raw recall -- see train.py docstring

# --- threshold tuning --------------------------------------------------
TARGET_RECALL = 0.98    # minimum recall on malignant class we aim for when
                         # choosing the high-recall operating threshold

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
