# Review package: Breast Mass Malignancy Risk Estimator

> **STATUS NOTE (added after round two):** this package reflects the state
> of the project at the time of the FIRST external review ("round one") --
> the numbers and code excerpts below are pre-fix. Round one found three
> real bugs (documented in `README.md`'s "Process" and "What changed after
> external review" sections); those were fixed, then round two (given full
> repo access, not this package) found further issues, which are documented
> in `README.md`'s "Round two" section along with what was fixed in
> response. If you're doing a fresh review, `README.md` and the live repo
> are the current source of truth -- this file is kept as a historical
> record of what round one was actually shown, not as an up-to-date
> description of the project.

You are being asked to independently review this project for AI-generated
"slop" — confident-sounding output that doesn't hold up under scrutiny:
unverified claims, cosmetic rigor with no substance, internal
inconsistencies, or engineering theater. Please do not take any claim below
at face value. Check every number against the JSON evidence included. Check
every code claim against the pasted source. Where you can't verify something
from what's given, say so explicitly rather than assuming it's fine.

This package was assembled by the same AI (Claude) that built the project,
at the request of the human who owns it, specifically so a second, unrelated
AI could review it adversarially. Treat the "self-assessment" sections
below as a starting hypothesis to test, not a conclusion to accept.

---

## 1. What this actually is, no spin

A binary classifier (malignant vs. benign) on the Wisconsin Diagnostic Breast
Cancer dataset (1995, 569 rows, 30 hand-engineered features from digitized
FNA images, from `sklearn.datasets.load_breast_cancer`). This dataset is
close to linearly separable — this is not a hard prediction problem, and no
methodology applied on top changes that fact. The project's claimed value is
entirely in *process*: catching and fixing a real data-leakage bug, proper
uncertainty quantification, a documented multicollinearity analysis, tests,
and CI — not in the underlying classification task being difficult or the
result being clinically useful.

**Explicitly out of scope / not claimed:** clinical validity, regulatory
compliance, deployment readiness, novel methodology, or any benefit to an
actual patient. It is a portfolio/learning artifact.

## 2. Honest process timeline

This is the actual sequence of what happened, not a cleaned-up narrative:

1. **V1 (≈10 minutes of AI-assisted generation):** EDA, model training on a
   single 80/20 split, threshold tuning, SHAP, a Streamlit app. This version
   reused the test set for both threshold selection and final reporting —
   a real leakage bug — and reported a suspicious 100% recall as a result.
2. **Self-critique requested:** the human explicitly asked for a brutal,
   multi-expert-lens critique of V1. The AI identified: the test-set leakage,
   a recall-only CV objective with a degenerate optimum, unaddressed severe
   multicollinearity (visible in its own correlation heatmap but never acted
   on), missing confidence intervals on a small test set, and a general lack
   of engineering rigor (no tests, no CI, hardcoded paths, print-based
   logging).
3. **V2 rebuild:** the human said they wanted the fixes implemented (they
   are a Python/ML beginner and asked the AI to build rather than have them
   write it, after an initial attempt at teaching them the fix hands-on).
   The AI rebuilt the pipeline: proper train/val/test split, threshold
   selection moved to validation only, `RepeatedStratifiedKFold` +
   average-precision model selection, a VIF multicollinearity analysis, test
   suite, GitHub Actions CI, Dockerfile, and a rewritten README.
4. **This request:** the human is asking, correctly, whether any of this has
   real-world value beyond a resume line, wants hard questions that would
   actually test the work, and wants this package for independent review
   before publishing it.

**Self-check the AI did before sending this package:** cross-referenced
every number quoted in the README against the raw JSON files below. They
were internally consistent at time of writing. This is not a guarantee —
please re-verify rather than trust that claim.

## 3. Claims made in the README (verbatim excerpts)

> "The tuned version fixes [leakage] with a proper three-way split... With
> the fix in place, the same kind of threshold-tuning exercise produces
> 97.6% recall (95% CI: 92.1%–100%) at 87.2% precision."

> "Iteratively dropping the worst offender (VIF ≥ 10) down to 7 features
> costs a real −0.016 average precision (0.9934 → 0.9775 in 5×3 CV) — more
> than a 0.01 tolerance, so the analysis's own decision rule keeps the full
> 30-feature model."

> "15 passing tests... CI... runs the full pipeline and test suite on every
> push, then builds the Docker image and smoke-tests that the container
> actually serves a healthy app."

> Docker caveat, stated in the README itself: "this was written and
> reviewed carefully but could not be build-tested in the sandbox this
> project was developed in, because that environment's network blocks all
> container registries."

## 4. Raw evidence (unedited JSON outputs)

### models/results_summary.json
```json
{
  "results": {
    "logistic_regression": {
      "cv_mean_average_precision": 0.9937,
      "cv_std_average_precision": 0.0072,
      "train_set_recall": 0.9779,
      "train_set_precision": 0.9852,
      "train_set_roc_auc": 0.9975,
      "best_params": {"clf__class_weight": "balanced", "clf__C": 0.42813323987193913}
    },
    "random_forest": {
      "cv_mean_average_precision": 0.9872,
      "cv_std_average_precision": 0.0099,
      "train_set_recall": 0.9926, "train_set_precision": 1.0, "train_set_roc_auc": 1.0,
      "best_params": {"clf__n_estimators": 400, "clf__min_samples_leaf": 1, "clf__max_depth": 5, "clf__class_weight": "balanced"}
    },
    "xgboost": {
      "cv_mean_average_precision": 0.9892,
      "cv_std_average_precision": 0.01,
      "train_set_recall": 1.0, "train_set_precision": 1.0, "train_set_roc_auc": 1.0,
      "best_params": {"clf__subsample": 0.7, "clf__n_estimators": 400, "clf__max_depth": 2, "clf__learning_rate": 0.1}
    }
  },
  "selected_model": "logistic_regression"
}
```
*(note: random_forest and xgboost show train-set recall/precision of 1.0 —
perfect in-sample fit. This is disclosed as "train-set sanity numbers only,
NOT a generalization estimate" in code comments. Please assess whether that
caveat is sufficient or whether this is a red flag being explained away.)*

### models/eval_summary.json
```json
{
  "n_validation": 91, "n_test": 114, "n_malignant_test": 42,
  "default_threshold": 0.5, "tuned_threshold": 0.1741,
  "test_metrics_default_threshold": {
    "recall": 0.9762, "precision": 0.9762, "roc_auc": 0.997,
    "bootstrap_95ci": {
      "recall": {"ci_lower_2.5%": 0.921, "ci_upper_97.5%": 1.0},
      "precision": {"ci_lower_2.5%": 0.9167, "ci_upper_97.5%": 1.0},
      "roc_auc": {"ci_lower_2.5%": 0.9893, "ci_upper_97.5%": 1.0}
    }
  },
  "test_metrics_tuned_threshold": {
    "recall": 0.9762, "precision": 0.8723, "roc_auc": 0.997,
    "bootstrap_95ci": {
      "recall": {"ci_lower_2.5%": 0.921, "ci_upper_97.5%": 1.0},
      "precision": {"ci_lower_2.5%": 0.7692, "ci_upper_97.5%": 0.9608},
      "roc_auc": {"ci_lower_2.5%": 0.9893, "ci_upper_97.5%": 1.0}
    }
  }
}
```

### models/vif_report.json (key fields)
```json
{
  "vif_threshold": 10.0,
  "dropped_features_sample": [
    {"feature": "mean radius", "vif_at_removal": 63499.29},
    {"feature": "worst radius", "vif_at_removal": 8872.17},
    {"feature": "mean perimeter", "vif_at_removal": 4350.15}
  ],
  "dropped_count": 23,
  "selected_features_by_vif": ["texture error", "area error", "smoothness error",
    "concavity error", "symmetry error", "fractal dimension error", "worst concavity"],
  "cv_comparison": {
    "full_30_features": {"n_features": 30, "mean_average_precision": 0.9934, "std_average_precision": 0.0072},
    "vif_selected_features": {"n_features": 7, "mean_average_precision": 0.9775, "std_average_precision": 0.0138}
  },
  "delta_average_precision": -0.0159,
  "final_decision": "full"
}
```
*(note: the VIF-selected 7-feature set consists entirely of "error" features
and one "worst" feature — none of the "mean" features survive. Please assess
whether this specific selection makes clinical/statistical sense, or whether
it's an artifact of the greedy one-at-a-time removal order that a domain
expert would question.)*

## 5. Full source of the core pipeline (paste-verify against the claims above)

### src/config.py
```python
"""
Central configuration: every seed, split ratio, path, and search setting used
across the pipeline lives here so it's changed in exactly one place instead
of copy-pasted (and silently drifting) across scripts.
"""
import logging
from pathlib import Path

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

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
VIF_THRESHOLD = 10.0
CV_SPLITS = 5
CV_REPEATS = 3
N_ITER_SEARCH = 25
CV_SCORING = "average_precision"
TARGET_RECALL = 0.98
N_BOOTSTRAP = 2000

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

### src/data.py
```python
from typing import List, Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src import config

TARGET_COL = "diagnosis"

def load_raw() -> Tuple[pd.DataFrame, List[str]]:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET_COL] = df["target"].map({0: 1, 1: 0})  # 1 = malignant, 0 = benign
    df = df.drop(columns=["target"])
    feature_names = list(raw.feature_names)
    return df, feature_names

def make_splits(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL]
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size,
        stratify=train_val_df[TARGET_COL], random_state=random_state,
    )
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))

def load_splits():
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    return train_df, val_df, test_df

def load_selected_features() -> List[str]:
    import json
    with open(config.SELECTED_FEATURES_PATH) as f:
        return json.load(f)["selected_features"]
```

### src/evaluate.py — the threshold selection and bootstrap logic specifically
```python
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
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba); n = len(y_true)
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
        return {"ci_lower_2.5%": round(float(np.percentile(values, 2.5)), 4),
                "ci_upper_97.5%": round(float(np.percentile(values, 97.5)), 4)}
    return {"recall": ci(recalls), "precision": ci(precisions), "roc_auc": ci(aucs)}

# In main(): threshold is chosen using (y_val, y_proba_val) ONLY.
# Test set (y_test, y_proba_test) is used only afterward, for reporting.
```

### src/feature_analysis.py — the VIF pruning loop specifically
```python
def iterative_vif_selection(X: pd.DataFrame, threshold: float):
    """Drop the highest-VIF feature repeatedly until all VIFs < threshold."""
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

# Decision rule actually used:
# final_features = selected_features if delta > -0.01 else all_features
# (delta was -0.0159, so the FULL feature set was kept, not the VIF-pruned one)
```

### tests/ — full list of what's actually tested (15 tests, all passing at time of writing)
```
test_data.py:
  test_raw_data_shape                                  -- 569 rows, 30 features
  test_no_missing_values
  test_target_is_binary
  test_target_convention_matches_known_class_balance    -- catches a flipped 0/1 remap
  test_splits_do_not_overlap                            -- row-level duplicate check across train/val/test
  test_splits_cover_full_dataset
  test_splits_are_stratified                            -- class balance within 5% of overall across splits
  test_splits_are_reproducible                          -- same seed -> identical split
  test_different_seeds_give_different_splits

test_pipeline.py (skipped if model not yet trained):
  test_predict_proba_shape_and_range
  test_predictions_are_deterministic
  test_model_beats_trivial_baseline                     -- recall > 0.85 vs majority-class baseline
  TestChooseThreshold::test_hits_target_recall_on_easy_separable_data
  TestChooseThreshold::test_falls_back_gracefully_when_target_unreachable
  TestBootstrapCI::test_ci_bounds_are_ordered_and_within_unit_interval
```

## 6. Specific questions for you (the reviewing AI) to answer

Please don't just assess "does this sound rigorous" — verify specific
claims against the evidence above, and flag anything you can't verify.

1. Is the leakage fix actually correct? Trace through `evaluate.py`: is
   there any path where test-set data could still influence the threshold
   choice, even indirectly?
2. The `choose_threshold` function picks a threshold from `validation` data
   with only 91 rows (~34 malignant, since the test set has 42/114 ≈ 36.8%
   malignant and validation should be similar). Is a single validation draw
   of that size enough to make the recall-target threshold reliable? What
   would you check to know?
3. Model selection used mean CV average-precision to pick logistic
   regression over XGBoost (0.9937 vs 0.9892). Given the reported
   std (0.0072 vs 0.0100), is that difference meaningful, or within noise?
   Was any statistical test applied, or just "highest mean wins"?
4. The VIF-reduced 7-feature set is entirely "error" and "worst concavity"
   features — no "mean" features survive. Does that pattern look like a
   sound feature selection, or an artifact worth distrusting?
5. Is the bootstrap CI methodology (resample rows with replacement, refit
   nothing, just recompute metrics at a fixed threshold) valid for this use
   case? What would make it invalid?
6. Does the README's tone match the actual rigor level, or does it
   oversell anywhere? Look specifically for hedge words doing more work
   than the evidence supports.
7. Is there anything in the process timeline (Section 2) that reads as
   the AI covering for its own earlier mistake rather than genuinely fixing
   it — i.e., is the "found and fixed" narrative itself a form of slop?
8. Overall: on a scale from "templated AI output dressed up to look
   rigorous" to "genuinely sound methodology for what it claims to be,"
   where does this land, and what's the single most convincing piece of
   evidence for your verdict either way?
