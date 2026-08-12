# Breast Mass Malignancy Risk Estimator

A machine learning project built around the questions that actually matter in
a screening context — which errors are worse, how confident is the model,
whether the "held-out" test set was actually held out, whether a "fix"
actually held up under scrutiny — rather than around chasing an accuracy
number on a dataset that's been solved since the 1990s.

> **This is a portfolio/educational project, not a medical device.** The
> Wisconsin Diagnostic Breast Cancer dataset (569 samples, one institution,
> one point in time) is not a basis for real diagnosis. The model has not
> been clinically validated and must never inform actual medical decisions.

## Honest framing

Binary classifiers on this dataset are one of the most common first ML
projects there is, and the dataset is close to linearly separable — any
reasonable method gets ~95%+ accuracy. That's not a hard problem, and no
amount of tooling on top changes that. What this repo demonstrates instead
is methodology, including two full cycles of "claim rigor, get it checked,
find out some of it wasn't." Both cycles are documented below rather than
edited out, because the second one is more informative than the first.

## Process, including where it went wrong twice

1. **V1 (~10 minutes, AI-assisted):** reused the test set for both threshold
   selection and final reporting — real leakage — and reported a suspicious
   100% recall.
2. **Self-critique:** identified the leakage, plus a recall-only CV objective
   with a degenerate optimum, unaddressed multicollinearity, and missing
   uncertainty quantification.
3. **V2 rebuild:** proper train/val/test split, average-precision model
   selection, a VIF multicollinearity analysis, bootstrap confidence
   intervals, tests, CI, Docker.
4. **Independent external adversarial review**, specifically requested and
   given full source + raw metrics + no ability to just take the README's
   word for anything: reproduced every reported number exactly (the tuned
   threshold, all six test metrics, all twelve bootstrap bounds, all three
   CV means, to four decimal places), then found three things V2's own
   process had missed:
   - **A real code bug:** `compute_vif` called statsmodels'
     `variance_inflation_factor` without an intercept, inflating every VIF
     by 1-2 orders of magnitude (`mean radius` was reported at 63,499; the
     correct value is 3,939) and causing the iterative selection to strip
     the entire "mean"/"worst" feature block — where this dataset's actual
     signal lives — before touching the weakly-correlated "error" block.
   - **A design flaw reintroducing the exact bug class V1 was rebuilt to
     fix:** `TARGET_RECALL=0.98` against a ~34-case validation set is only
     achievable at exactly 100% recall (33/34 = 0.971 < 0.98), which
     silently collapsed threshold selection to "the predicted probability
     of the single hardest validation case" — a one-point minimum statistic.
     V1's bug was a degenerate CV objective; V2 fixed that and then rebuilt
     an equivalent degenerate objective one layer downstream, in the
     threshold selector, where nothing tested for it.
   - **An unjustified model-selection claim:** logistic regression was
     picked for the highest mean CV score without checking whether the gap
     to the runner-up was distinguishable from noise, and a naive
     significance test on repeated-k-fold scores is anti-conservative
     because the folds aren't independent.
5. **This version:** all three findings verified by independent reproduction
   (not just accepted), then fixed. See below for what changed and why.

If you're evaluating this repo, the honest thing to say about it is not
"rigorous ML pipeline." It's "a pipeline that made real mistakes, had them
caught by adversarial review rather than self-review, and was fixed with the
fixes checked back in." That's a different, more defensible claim, and it's
the one this README is trying to actually support rather than assert.

## What changed after external review

**VIF (`src/feature_analysis.py`):** `compute_vif` now adds an intercept
(`statsmodels.tools.tools.add_constant`) before computing VIF, per
statsmodels' own documented requirement. Corrected VIFs are far lower
(`mean radius` was reported at 63,499 by the buggy version; the corrected
worst VIF, `mean perimeter`, is 3,987 — a ~16x reduction, see table below
for the full before/after) and the iterative selection now drops 14 features
instead of 23, retaining 16 instead of 7 — critically, `mean radius`
(the single feature the buggy version called the worst offender) now
survives, because it was never actually the problem; the missing intercept
was. The corrected reduced set costs only −0.0031 average precision
(0.9934 → 0.9903, well within one CV standard deviation), so **the model
now trains on the VIF-corrected 16-feature set**, not the full 30. The
comparison estimator is explicitly labeled `baseline_*` in
`vif_report.json` and documented as an untuned reference model (feature
selection has to happen before hyperparameter tuning in a leakage-safe
order, so no "final tuned" config exists yet at this pipeline stage) —
this is now stated rather than left to look like a like-for-like comparison
with the deployed model.

**Threshold tuning (`src/evaluate.py`, `src/config.py`,
`src/stability_analysis.py`):** `TARGET_RECALL` changed from 0.98 (only
achievable at exactly 100%) to 0.95 (achievable at 33/34 as well as 34/34
— less degenerate, though still fragile with this few positives). More
importantly, a new script runs the entire split → fit → threshold-select →
evaluate loop across 200 seeds and reports the empirical spread instead of
trusting one draw:

```
Threshold across 200 seeds: median=0.568, range=[0.0005, 0.983]
Mean test recall/precision -- default: 0.952/0.948, tuned: 0.943/0.922
Tuning improved recall in 47/200 seeds, left it unchanged in 80/200,
  and failed to help while costing precision in the remaining 73/200.
```

**Threshold tuning does not help here, on average, and is not stable.**
The single seed-42 result makes this starkest: the validation-tuned
threshold (0.955) actually *dropped* test recall from 95.2% to 83.3%
(precision rose to 100%, but that's the wrong trade for a screening
context) — because the threshold that happened to catch every validation
case didn't generalize to the test set's different hard cases. **This
project's recommended, deployed operating point is therefore the plain
default threshold (0.5), not the tuned one.** The tuned-threshold analysis
is kept and shown (in the app, clearly marked "not recommended") because
the negative finding — spend effort tuning a threshold, discover it doesn't
reliably help and is highly sensitive to which patients land in a small
validation set — is a more honest and more useful result than a fabricated
positive one would have been.

**Model selection (`src/train.py`, `src/stats_utils.py`):** model choice
used to be "highest mean CV score wins," with no check on whether that gap
was real. Now a Nadeau-Bengio corrected paired t-test (the correction that
accounts for repeated-k-fold scores being non-independent, unlike a naive
paired t-test) runs between the winning model and each alternative:

```
logistic_regression vs random_forest: mean diff=+0.0020, corrected p=0.731
logistic_regression vs xgboost:       mean diff=+0.0016, corrected p=0.785
```

Neither difference is remotely significant. **Logistic regression is
selected for simplicity, calibrated probabilities, and interpretability —
not because it measurably outperforms the alternatives.** `results_summary.json`
now stores `selection_statistically_justified: false` so this is a queryable
fact, not just README prose.

**Tests:** the previous test suite included one test
(`test_falls_back_gracefully_when_target_unreachable`) that could not
actually fail — `precision_recall_curve` always returns `recall[0] == 1.0`
by construction, so the fallback branch it claimed to test is unreachable
for any `target_recall <= 1.0`, which is every value the code was ever
called with. It's been replaced with a test that exercises a genuinely
unreachable target (`> 1.0`) and a new test that documents and locks in the
min-statistic behavior described above, so it's a known, tested property
rather than a silent surprise. A new regression test
(`TestNoLeakage::test_threshold_unaffected_by_test_set_corruption`) directly
exercises `evaluate.main()` — the actual historically-buggy entry point,
not just the pure `choose_threshold()` function — by corrupting the on-disk
test set with random labels and shuffled features and asserting the chosen
threshold is unchanged. 17 tests total, all passing.

## Results (post-fix)

Three model families were tuned via `RandomizedSearchCV` with
`RepeatedStratifiedKFold` (5 folds × 3 repeats) optimizing average precision
on the VIF-corrected 16-feature training set:

| Model | CV average precision (mean ± std) | vs. logistic regression |
|---|---|---|
| **Logistic Regression (selected)** | **0.9915 ± 0.0076** | — |
| XGBoost | 0.9899 ± 0.0109 | not significant (p=0.785) |
| Random Forest | 0.9895 ± 0.0090 | not significant (p=0.731) |

### Test-set performance (default threshold, recommended)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Recall (malignant) | 95.2% | [87.5%, 100%] |
| Precision (malignant) | 90.9% | [82.0%, 97.9%] |
| ROC-AUC | 0.988 | [0.970, 1.0] |

Test set: 114 patients, 42 malignant. See `models/results_summary.json`,
`models/eval_summary.json`, `models/threshold_stability_report.json`.

### Multicollinearity (corrected)

| | Before fix (buggy, no intercept) | After fix |
|---|---|---|
| Worst VIF | `mean radius` = 63,499 | `mean perimeter` = 3,987 |
| Features dropped (VIF ≥ 10) | 23 | 14 |
| Features retained | 7 (all "error" features + 1 "worst") | 16 (spans mean/error/worst) |
| Does `mean radius` survive? | No (called the #1 offender) | **Yes** |
| Accuracy cost of pruning | −0.0159 AP (rejected: kept full 30) | −0.0031 AP (accepted: **now the deployed feature set**) |

Full detail in `models/vif_report.json`.

### What drives the model

Per SHAP (`LinearExplainer`, exact for this model, on the corrected
16-feature model): the strongest predictors are `area error`,
`worst concavity`, `mean radius`, `compactness error`, and
`concave points error`. Because several retained features are still
correlated (collinearity below the VIF≥10 threshold isn't zero
collinearity), read attributions within a related cluster as "this cluster
mattered," not as precise credit to one specific feature. See
`figures/shap_beeswarm.png` and `models/shap_feature_ranking.json`.

## Engineering practices

- **Tests** (`tests/`, 17 passing): data integrity, split non-overlap and
  stratification, split reproducibility, model output shape/range,
  determinism, threshold-selection edge cases (including the corrected
  min-statistic documentation test), and a direct regression test against
  the original leakage bug that exercises the real entry point.
- **CI** (`.github/workflows/ci.yml`): runs the full pipeline (through
  `stability_analysis.py` and `shap_explain.py`) and test suite on every
  push, then builds the Docker image and smoke-tests that the container
  actually serves a healthy app.
- **Statistical rigor beyond point estimates**
  (`src/stats_utils.py`, `src/stability_analysis.py`): Nadeau-Bengio
  corrected significance testing for model selection; a 200-seed stability
  sweep for threshold selection, in addition to the bootstrap CIs on the
  final test metrics.
- **Config centralization, single source of truth for data, containerized
  build** — as before; see `src/config.py`, `src/data.py`, `Dockerfile`.
  *Docker caveat, unchanged and still true:* written and reviewed carefully
  but not build-tested in the sandbox this project was developed in, since
  that environment blocks all container registries. CI's `docker-build` job
  is designed to build and smoke-test the image on every push — treat that
  as "designed to verify" until you've seen a green run in your own Actions
  tab, not as an assertion that it already has.

## Project structure

```
├── src/
│   ├── config.py             # central config, incl. corrected VIF/target-recall notes
│   ├── data.py                # load_raw, make_splits, load_splits
│   ├── stats_utils.py          # Nadeau-Bengio corrected significance test
│   ├── eda.py                  # exploratory analysis
│   ├── split_data.py           # train/val/test split (the original leakage fix)
│   ├── feature_analysis.py     # VIF multicollinearity analysis (intercept-corrected)
│   ├── train.py                 # RandomizedSearchCV + significance-tested model selection
│   ├── evaluate.py              # threshold on val, final report + bootstrap CI on test
│   ├── stability_analysis.py    # 200-seed threshold stability sweep
│   └── shap_explain.py          # global + per-patient SHAP explanations
├── tests/
│   ├── test_data.py             # split correctness, stratification, reproducibility
│   └── test_pipeline.py          # model contracts, threshold edge cases, leakage regression test
├── app/app.py                    # Streamlit demo (defaults to the recommended threshold)
├── .github/workflows/ci.yml
├── Dockerfile
├── AI_REVIEW_PACKAGE.md           # the package sent for external adversarial review
```

## Running it yourself

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

python3 -m src.eda
python3 -m src.split_data
python3 -m src.feature_analysis
python3 -m src.train
python3 -m src.evaluate
python3 -m src.stability_analysis
python3 -m src.shap_explain

pytest tests/ -v                  # 17 tests

streamlit run app/app.py          # run from project root
```

## Data

[Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
569 samples, loaded via `sklearn.datasets.load_breast_cancer`. No missing
values. Class balance: 62.7% benign / 37.3% malignant. sklearn's target
encoding (`0=malignant, 1=benign`) is remapped once, in `src/data.py`, to
the standard medical-ML convention; `test_target_convention_matches_known_class_balance`
guards against that remap regressing.

## Model card (abbreviated)

**Intended use:** educational demonstration of an ML methodology pipeline,
including how that pipeline holds up under independent adversarial review.
Suitable as a portfolio artifact and a base for learning, not as a
component of any real diagnostic workflow.

**Out-of-scope use:** any real clinical decision-making, patient-facing
deployment, or use as a diagnostic aid, screening tool, or second opinion.

**Training data:** single-institution, single-timepoint, 569 samples, no
demographic information for subgroup analysis, no external validation
cohort.

## Limitations

- 569 samples total is small; the ~34-case validation set is exactly why
  threshold tuning turned out to be unreliable here (see above) — this
  isn't a one-off, it's a structural consequence of the dataset size.
- No external validation set from a different institution or population.
- Cell-morphology features only, not a full pathology workup.
- Docker image unverified in a real build environment as of this commit
  (network-restricted dev sandbox); CI is the real check, once run.
- This README, and the analysis behind it, already went through one round
  of external adversarial review and correction. That doesn't mean it's
  now correct — it means it's been checked once. Treat any specific claim
  you're relying on as something to re-verify, not as settled.

## Possible next steps

- Move from tabular features to raw histopathology images (BreakHis or
  PatchCamelyon) with a CNN/transfer-learning approach.
- Add data/model versioning (DVC) and experiment tracking (MLflow/W&B).
- Deploy behind a FastAPI inference endpoint with request logging and
  drift monitoring, alongside the Streamlit demo.
- Explore an LLM/RAG layer over medical literature (e.g. PubMed abstracts)
  as a separate, distinct project.
