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
5. **Round-one fixes:** all three findings verified by independent
   reproduction (not just accepted), then fixed.
6. **Independent external adversarial review, round two** — same reviewer,
   this time given the full repo (including `train.py`/`evaluate.py`, which
   round one didn't have), plus mutation testing (deliberately
   reintroducing each fixed bug to check whether the test suite actually
   catches it, not just whether it passes). Confirmed all three round-one
   fixes were real and that the new leakage-regression test genuinely fails
   when the leak is reintroduced — but also found that **no test protected
   the VIF fix** (all 17 tests kept passing with `add_constant` removed
   again), that the threshold-stability report's seed categories didn't
   add up to 200 and silently dropped the worst-case bucket, and, most
   importantly, that **the 16-feature model shipped as a result of round
   one's fix is measurably worse than the 30-feature model on recall,
   precision, and ROC-AUC** — a real cost that the untuned pre-tuning
   comparison in `feature_analysis.py` doesn't measure and can't see.
7. **Round-two fixes:** a VIF mutation-tested regression test, corrected
   stability-report accounting, a proper significance test replacing the
   old bare `delta > -0.01` rule, and a new post-hoc script that measures —
   on the actual shipped model — both the real performance cost and a
   real, previously-unmeasured explanation-stability benefit of the
   reduced feature set. See below.
8. **Deployment found a fourth bug the pipeline never could.** Minutes
   after first deploying to Streamlit Community Cloud, the live app
   crashed on load: `ValueError` from `"95% CI: [%.3f, %.3f]" % (...)` —
   an unescaped literal `%` in a Python `%`-format string, which the `%`
   operator tried to parse as a format directive. All 18 tests at that
   point were pipeline/model tests; none of them ever executed `app.py`,
   so this shipped straight through review, mutation testing, and CI
   without anyone (or anything) running the one file that's actually the
   product. Fixed by switching those lines to f-strings and adding
   `tests/test_app.py`, which uses Streamlit's `AppTest` framework to
   actually execute the app and every button/dropdown interaction — the
   kind of test that would have caught this before it ever reached a
   user. 21 tests total now.

If you're evaluating this repo, the honest thing to say about it is not
"rigorous ML pipeline." It's "a pipeline that made real mistakes, had some
caught by adversarial review, one caught by mutation testing, and one that
got all the way to a live crash before anyone noticed the app itself had
zero test coverage — and each time, the response was a real fix plus a
test that would catch it next time, not just a patch." That's a different,
more defensible claim, and it's the one this README is trying to actually
support rather than assert.

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
was. The corrected reduced set costs −0.0031 average precision on an
**untuned baseline model** (0.9934 → 0.9903) — round two's decision rule
here was a bare `delta > -0.01` inequality with no notion of statistical
power, which round-two review correctly flagged as the most important
remaining issue. It's since been replaced with a proper Nadeau-Bengio
significance test (`nadeau_bengio_p=0.336`, not significant), but see
**"Round two"** below: that test only covers this untuned, pre-tuning
comparison, and a separate post-hoc measurement against the *actual*
shipped model found a real trade-off this comparison can't see. **The model
trains on the VIF-corrected 16-feature set**, not the full 30. The
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
  and made recall WORSE in 73/200 (mean loss 5.2pp, worst case 16.7pp).
  47 + 80 + 73 = 200 -- every seed accounted for.
```

*(Round two found the original version of this report didn't partition:
its three categories summed to 163/200 and silently dropped the worst-case
bucket — exactly the seeds where tuning hurt most. Fixed in
`stability_analysis.py`; the "made recall worse" bucket and its damage are
now first-class fields in `threshold_stability_report.json`, not an
implied remainder.)*

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
threshold is unchanged. 17 tests total, all passing at the time (round one).

## Round two: what a second, deeper review found — and what changed

Round two gave the same external reviewer the full repo (round one didn't
have `train.py`/`evaluate.py`) and asked it to **mutation-test** the fixes —
deliberately reintroduce each bug and check whether the test suite actually
fails, not just whether it currently passes. Reproduced independently here:

- **The leakage regression test genuinely works.** Reintroducing the
  original test-set leak makes `TestNoLeakage::test_threshold_unaffected_by_test_set_corruption`
  fail, as designed.
- **The VIF fix had zero test coverage.** Reintroducing the missing
  `add_constant` bug (removing the intercept) left all 17 existing tests
  passing — nothing caught it. **Fixed:** a new test,
  `TestVIFCorrectness::test_vif_uses_an_intercept`, was added and verified
  the same way — with the bug reintroduced, it fails with the exact known
  buggy value (`mean radius` VIF = 63,499.3); with the fix in place, it
  passes. 18 tests total now, all passing.
- **The threshold-stability report's categories didn't partition.**
  Described above — fixed.
- **The most important finding:** on the *actual tuned, deployed* model
  (not the untuned baseline `feature_analysis.py` uses to make its
  pre-tuning decision), the reviewer measured the 16-feature model as
  significantly worse than the 30-feature model on recall, precision, and
  ROC-AUC. That's a real, previously unmeasured cost of the VIF-selection
  decision — the untuned comparison's "no meaningful cost" finding does
  not extend to the model that's actually shipped, and the README
  shouldn't have implied that it did.

**Fixed with `src/feature_tradeoff_analysis.py`** (run after `train.py`):
a new script that fits the *actual* tuned hyperparameters (`C=8.859`,
`class_weight=balanced`) on both the 16- and 30-feature sets across 100
independent splits, and measures two things the earlier comparison
couldn't: the real performance cost, and — something the reviewer noticed
but didn't quantify — the SHAP explanation-stability *benefit* of the
smaller, less collinear feature set. Reproducing this measurement here
(`models/feature_tradeoff_report.json`):

```
recall:    full=0.9526  reduced=0.9514  diff=-0.0012  (paired t-test p=0.556, not significant)
precision: full=0.9484  reduced=0.9454  diff=-0.0030  (paired t-test p=0.361, not significant)
roc_auc:   full=0.9893  reduced=0.9893  diff=+0.0000  (paired t-test p=0.895, not significant)

SHAP top-3 feature-set consensus across 100 splits:
  30-feature model:  6/100 splits match the modal top-3 set (61 distinct sets seen)
  16-feature model:  82/100 splits match the modal top-3 set (7 distinct sets seen)
```

**This independent reproduction did NOT find a statistically significant
performance cost** in this environment (pinned to the dependency versions
in `requirements.lock`) — which does not match round two's own claim of a
significant cost (p<1e-6). That's a genuine, unresolved discrepancy, not a
result to quietly prefer: it's flagged here rather than smoothed over,
and the most likely explanation is a methodology or dependency-version
difference (see the unpinned-dependency limitation below) rather than
either side being wrong. What *did* reproduce cleanly, and by a wide
margin, is the SHAP stability finding: the 16-feature model's top-3
explanation is consistent across resampled training data roughly 14x more
often than the 30-feature model's (82% vs 6%). **The honest framing is a
trade, not a free lunch:** the reduced feature set buys a much more
trustworthy explanation, a real recall/precision/AUC cost may or may not
be part of that price depending on environment, and this project ships the
reduced set for the explainability benefit while saying exactly that
instead of "no meaningful cost."

**Also fixed:** the untuned pre-tuning decision rule in
`feature_analysis.py` no longer uses a bare `delta > -0.01` inequality —
it now runs the same Nadeau-Bengio significance test used for model-family
selection (`nadeau_bengio_p=0.336`, not significant, see above), with an
explicit code comment and log line pointing at
`feature_tradeoff_analysis.py` so nobody mistakes that untuned check for a
guarantee about the deployed model.

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
| Accuracy cost of pruning (untuned baseline) | −0.0159 AP (rejected: kept full 30) | −0.0031 AP, p=0.336 not significant (accepted: **now the deployed feature set**) |

Full detail in `models/vif_report.json`. The "accuracy cost" row above is
the untuned, pre-tuning check only — see the next section for what the
actual tuned model gives up and gains.

### Feature-set trade-off, measured on the real deployed model

`src/feature_tradeoff_analysis.py` (added after round-two review) fits the
actual tuned hyperparameters on both feature sets across 100 splits:

| | 30 features (full) | 16 features (deployed) |
|---|---|---|
| Test recall | 95.26% | 95.14% (diff not significant, p=0.556) |
| Test precision | 94.84% | 94.54% (diff not significant, p=0.361) |
| Test ROC-AUC | 0.9893 | 0.9893 (diff not significant, p=0.895) |
| SHAP top-3 consensus (100 splits) | 6% | **82%** |

In this environment, the performance difference is not statistically
significant — which does not match round-two review's own reproduction (it
found a significant cost, p<1e-6). That discrepancy is unresolved and
stated plainly rather than picked over silently; the leading suspect is a
dependency-version or methodology difference (see Limitations). What's
unambiguous is the explanation-stability benefit: the deployed model's
top-3 SHAP features land on the same 3 features in 82/100 resampled splits,
versus 6/100 for the full feature set — collinearity in the full set lets
credit shift almost arbitrarily between near-duplicate features run to run.
Full detail in `models/feature_tradeoff_report.json`.

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

- **Tests** (`tests/`, 21 passing): data integrity, split non-overlap and
  stratification, split reproducibility, model output shape/range,
  determinism, threshold-selection edge cases (including the corrected
  min-statistic documentation test), a direct regression test against the
  original leakage bug that exercises the real entry point, a
  mutation-tested regression test for the VIF intercept fix (added after
  round-two review found this was the one bug with zero test coverage;
  verified the same way round two verified it — reintroducing the bug
  makes the new test fail with the exact known buggy value), and
  (`test_app.py`) three Streamlit `AppTest` smoke tests that actually
  execute `app.py` and simulate button/dropdown interactions — added after
  a real `%`-format bug crashed the live deployed app despite every other
  test passing, because none of them ran the app itself.
- **CI** (`.github/workflows/ci.yml`): runs the full pipeline (through
  `stability_analysis.py`, `feature_tradeoff_analysis.py`, and
  `shap_explain.py`) and test suite on every push, then builds the Docker
  image and smoke-tests that the container actually serves a healthy app.
- **Statistical rigor beyond point estimates**
  (`src/stats_utils.py`, `src/stability_analysis.py`,
  `src/feature_tradeoff_analysis.py`): Nadeau-Bengio corrected significance
  testing for model selection AND for the feature-selection decision (no
  more bare inequality thresholds); a 200-seed stability sweep for
  threshold selection that now correctly partitions every seed; a 100-split
  paired-test measurement of what the deployed feature set actually costs
  and gains on the real, tuned model.
- **Config centralization, single source of truth for data, containerized
  build** — as before; see `src/config.py`, `src/data.py`, `Dockerfile`.
  *Docker caveat, unchanged and still true:* written and reviewed carefully
  but not build-tested in the sandbox this project was developed in, since
  that environment blocks all container registries. CI's `docker-build` job
  is designed to build and smoke-test the image on every push — treat that
  as "designed to verify" until you've seen a green run in your own Actions
  tab, not as an assertion that it already has.
- **Pinned dependency snapshot** (`requirements.lock`, generated via
  `pip freeze`): round two noted that `requirements.txt`'s `>=`-only
  constraints make some quoted statistics environment-dependent (a
  Nadeau-Bengio p-value that differs between sklearn versions, for
  example). `requirements.lock` records the exact versions this README's
  numbers were produced with (`pip install -r requirements.lock` for an
  exact reproduction); `requirements.txt` is left loose for normal
  development/CI use.

## Project structure

```
├── src/
│   ├── config.py                    # central config, incl. corrected VIF/target-recall notes
│   ├── data.py                       # load_raw, make_splits, load_splits
│   ├── stats_utils.py                 # Nadeau-Bengio corrected significance test
│   ├── eda.py                         # exploratory analysis
│   ├── split_data.py                  # train/val/test split (the original leakage fix)
│   ├── feature_analysis.py            # VIF multicollinearity analysis (intercept-corrected, significance-tested decision)
│   ├── train.py                        # RandomizedSearchCV + significance-tested model selection
│   ├── evaluate.py                     # threshold on val, final report + bootstrap CI on test
│   ├── stability_analysis.py           # 200-seed threshold stability sweep (partition-corrected)
│   ├── feature_tradeoff_analysis.py    # post-hoc: real cost + SHAP-stability benefit of the deployed feature set
│   └── shap_explain.py                 # global + per-patient SHAP explanations
├── tests/
│   ├── test_data.py             # split correctness, stratification, reproducibility
│   ├── test_pipeline.py          # model contracts, threshold edge cases, VIF + leakage regression tests
│   └── test_app.py                # AppTest smoke tests -- actually runs app.py, not just src/
├── app/app.py                    # Streamlit demo (defaults to the recommended threshold)
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.lock              # exact pinned versions this README's numbers were produced with
├── AI_REVIEW_PACKAGE.md           # the package sent for external adversarial review
```

## Running it yourself

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.lock for an exact reproduction

python3 -m src.eda
python3 -m src.split_data
python3 -m src.feature_analysis
python3 -m src.train
python3 -m src.evaluate
python3 -m src.stability_analysis
python3 -m src.feature_tradeoff_analysis
python3 -m src.shap_explain

pytest tests/ -v                  # 21 tests

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
- `TARGET_RECALL=0.95` still doesn't operationally mean 0.95: with 34
  validation positives, the nearest achievable levels are 32/34≈0.941,
  33/34≈0.971, and 34/34=1.0 — `evaluate.py` now logs this explicitly and
  records the actually-achieved validation recall in `eval_summary.json`,
  but the underlying degeneracy (a small positive count makes threshold
  selection close to an order statistic) is reduced, not eliminated. See
  `stability_analysis.py`'s 200-seed sweep for the practical consequence.
- The 200-seed stability sweep and the 100-split feature trade-off analysis
  both hold the feature set and hyperparameters fixed at whatever the
  single seed-42 run of `feature_analysis.py`/`train.py` happened to
  choose (re-running full feature selection and hyperparameter search per
  seed was judged too expensive for a 364-row training set to repeat
  hundreds of times). Both are therefore a **lower bound** on true
  end-to-end instability, not the complete picture.
- The independent reproduction of round two's feature-tradeoff finding
  (see "Round two" above) did NOT find a statistically significant
  performance cost in this pinned environment, contradicting round two's
  own claim of p<1e-6. This discrepancy is unresolved — most likely a
  dependency-version or methodology difference — and is stated here
  rather than silently resolved in either direction.
- This README, and the analysis behind it, has now gone through two rounds
  of external adversarial review and correction, including mutation
  testing in round two. That doesn't mean it's now correct — it means it's
  been checked twice, and the second check found real problems the first
  one missed (an untested bug, a broken report, an unmeasured trade-off).
  Treat any specific claim you're relying on as something to re-verify,
  not as settled, and assume a third review would likely find more.

## Possible next steps

- Move from tabular features to raw histopathology images (BreakHis or
  PatchCamelyon) with a CNN/transfer-learning approach.
- Add data/model versioning (DVC) and experiment tracking (MLflow/W&B).
- Deploy behind a FastAPI inference endpoint with request logging and
  drift monitoring, alongside the Streamlit demo.
- Explore an LLM/RAG layer over medical literature (e.g. PubMed abstracts)
  as a separate, distinct project.
- Resolve the unreproduced feature-tradeoff significance discrepancy
  (this repo's environment vs. round two's) by pinning both sides to the
  same `requirements.lock` and re-running the comparison.
- Re-derive `TARGET_RECALL` from the validation positive count at runtime
  (or drop the target-recall abstraction entirely in favor of directly
  reasoning about achievable order statistics) instead of a hardcoded
  value that doesn't mean what it says.
