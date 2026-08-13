# Breast Mass Malignancy Risk Estimator

[![CI](https://github.com/MB1234-dot/cancer-detection-project/actions/workflows/ci.yml/badge.svg)](https://github.com/MB1234-dot/cancer-detection-project/actions/workflows/ci.yml)

**Live demo:** https://cancer-detection-project.streamlit.app/

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
9. **Independent external adversarial review, round three** — same
   process, given the actual live public repo this time (not a package
   description), and it found something more important than any single
   code bug: **the crash fix from step 8, despite being correct and
   verified locally, had never actually been pushed to GitHub.** The
   reviewer cloned the public repo and reproduced the exact live crash,
   because the commit fixing it existed only on this machine. Root cause:
   the GitHub credential used to push was cleared for security after the
   first successful push, and the *next* push attempt failed outright
   (`could not read Username for 'https://github.com'`) with no one
   watching for that failure — a clear illustration of why "the agent
   said it pushed" is not evidence something is actually live; only
   `git ls-remote` against the real remote (or checking GitHub yourself)
   is. The review also found three real, separate issues, all now fixed:
   a methodology bug in `feature_tradeoff_analysis.py` (below), inverted
   significance-test logic in `feature_analysis.py` (below), and a
   dependency lockfile (`requirements.lock`) that nothing in CI, Docker,
   or Streamlit Cloud actually installed from — it existed but was never
   consumed. See **"Round three"** below for full detail on each.

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
remaining issue at the time. That was first replaced with a plain
significance test against zero, which round-three review then correctly
flagged as backwards: "not significantly different from zero" is not
evidence of equivalence, especially with a conservative correction making
genuine costs harder to detect. **It's now a proper one-sided
non-inferiority test** (`nadeau_bengio_noninferiority_test` in
`src/stats_utils.py`, the one-sided form of TOST equivalence testing): it
requires *positive* evidence that the cost is no worse than a declared
1-point-of-average-precision tolerance before shipping the reduced set,
rather than treating an inconclusive result as a pass. On the current data
that positive evidence exists (one-sided Nadeau-Bengio p=0.023 for
"cost ≤ 1pp", see `models/vif_report.json`), so the reduced set still
ships — but now because non-inferiority was demonstrated, not because a
cost merely failed to reach significance. See **"Round two"** and
**"Round three"** below: this test only covers this untuned, pre-tuning
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

**Update (round three): the discrepancy above is now resolved — it was a
methodology bug, not a dependency-version difference.** `feature_tradeoff_analysis.py`
was applying ONE shared hyperparameter (`C=8.859`, tuned specifically for
the 16-feature set) to BOTH feature sets in the comparison. That ran the
30-feature model at roughly 21x weaker regularization than it would choose
for itself (`C=0.428` when tuned independently) — precisely the regime
where multicollinearity damages a linear model most — which biased this
"matched regularization" comparison against detecting a real difference.
Confirmed bit-identical results under both sklearn 1.8.0 and 1.9.0, ruling
out the dependency-version theory. **Fixed:** the script now reports
*two* arms explicitly instead of silently picking one — see the updated
table below. Arm B (each feature set independently tuned, the actual
deployment-relevant comparison) reproduces round two's originally-claimed
cost; Arm A (both sets held at the same, deployed C) is kept because it
answers a genuinely different question — "does regularization strength
alone explain the gap?" — not because it's the more relevant one. What
did reproduce cleanly the whole time, in both arms, is the SHAP stability
finding: the 16-feature model's top-3 explanation is consistent across
resampled training data roughly 14x more often than the 30-feature
model's (82% vs 6%). **The honest framing is a trade, not a free lunch:**
the reduced feature set buys a much more trustworthy explanation at a
real, now-confirmed cost to recall/precision/ROC-AUC, and this project
ships the reduced set for the explainability benefit while saying exactly
that.

**Also fixed:** the untuned pre-tuning decision rule in
`feature_analysis.py` no longer uses a bare `delta > -0.01` inequality,
and — after round-three review flagged the first replacement as
backwards (see the VIF section above) — no longer uses a plain
significance-against-zero test either. It now runs a proper one-sided
non-inferiority test with an explicit code comment and log line pointing
at `feature_tradeoff_analysis.py` so nobody mistakes that untuned check
for a guarantee about the deployed model.

## Round three: what a third review found — and what changed

Round three was given the live public GitHub repo directly (not a
description of it) and asked to independently verify everything claimed
above. What it found, in order of importance:

1. **The round-two crash fix never reached GitHub.** Confirmed and fixed —
   see step 9 in "Process" above. This is the most important finding of
   the three rounds so far, precisely because it isn't a code bug at all:
   every fix described in this README was correct and locally verified,
   and none of that mattered until it was actually confirmed live on the
   real remote.
2. **The `feature_tradeoff_analysis.py` methodology bug** described above —
   real, root-caused, and fixed with an explicit two-arm comparison.
3. **The inverted significance-test logic** in `feature_analysis.py` —
   real, and fixed with a proper non-inferiority test, described above.
4. **`requirements.lock` was decorative** — it existed and was accurate,
   but nothing in `.github/workflows/ci.yml`, `Dockerfile`, or Streamlit
   Community Cloud (which only ever reads `requirements.txt`) actually
   installed from it, so it provided no real reproducibility guarantee.
   **Fixed:** `requirements.txt` is now pinned to exact versions (so
   Streamlit Cloud gets them too), and CI/Docker now install from
   `requirements.lock` directly for the full transitive-dependency
   pin.
5. **`tests/test_app.py` had weak coverage** — it only asserted "no
   exception," and its example-patient test used a patient whose
   prediction never exercised the default/tuned-threshold-disagreement
   code path (9 of 114 test patients hit that branch; the test used one
   of the other 105). **Fixed:** the three `AppTest` tests now also
   assert on rendered content (sidebar metrics, prediction metrics, info
   banners), and the example-patient test uses a patient chosen
   specifically to land in the disagreement band.
6. **A hyperparameter-search non-determinism I initially flagged myself
   was refuted.** Two pipeline runs had shown the same random-forest CV
   score with different winning hyperparameters, which I raised as a
   possible reproducibility concern. Round three re-ran the search and
   found it's exactly deterministic within a fixed environment
   (bit-identical to 10 decimal places); the original observation was
   most likely an artifact of comparing rounded values across different
   dependency versions — which is really the same root cause as finding
   4 above, not a new one.

All fixes in this round were verified by actually re-running the affected
scripts and the full test suite, not just read for plausibility — see
each subsection above for the specific before/after numbers.

### In detail: the tests all passed and the live app was down anyway

Findings 2-5 above concern methodology. Finding 1 doesn't, and it is
arguably the most instructive of the lot, so it's worth spelling out.

`app/app.py` built several strings with `%`-formatting that contained a
literal percent sign — `"95% CI: [%.3f, %.3f]"`. Python does not read that
`%` as a percent sign; it reads `% C` as the start of a format spec and
raises:

```
ValueError: unsupported format character 'C' (0x43) at index 4
```

That code runs at module import, so the deployed Streamlit app failed on
load. Not on a particular input, not intermittently — the demo linked at
the top of this README was a stack trace for anyone who clicked it.

Why nothing caught it:

- The full test suite passed, because the failing line is module-level
  Streamlit code, not an importable function. Nothing ever executed it.
- Local development didn't surface it either, since the workflow was
  "change code, run tests," not "change code, open the app."
- The bug string is *displayed correctly* in the source. `"95% CI"` looks
  right. It only misbehaves at the moment `%` is applied.

The fix converts the affected strings to f-strings, which removes the
failure mode rather than escaping around it (`%%` would also work, but
leaves the next person one keystroke from reintroducing it).

`tests/test_app.py` now covers this in three layers:

1. **A static AST check** that fails if `%`-formatting on a string literal
   reappears anywhere in `app.py` (integer modulo, as in `i % cols_per_row`,
   is correctly not flagged). This catches the bug class without needing to
   render anything.
2. **`AppTest` rendering tests with content assertions** — the disclaimer,
   the sidebar metrics, the CI tooltip text, the inputs, and a `Predict`
   click yielding a probability in range. The original version of this file
   only asserted `not at.exception`, which is close to worthless: an app
   that renders nothing at all passes it.
3. **A threshold-disagreement test** using Patient #3, whose predicted
   probability (~0.64) lands in the band where the default and tuned
   thresholds disagree — 9 of 114 test patients do. The earlier test used
   Patient #0 (probability ~0.0001), which never reaches that branch.

Both directions were checked. The new tests fail against the pre-fix
`app.py`, reproducing the exact production `ValueError`, and pass against
the fix. The deployed app was then confirmed rendering in a browser — not
inferred from a green test run.

**The transferable lesson:** a passing test suite is evidence about the
code paths the tests execute, and nothing more. "Tests pass" and "the thing
works" are different claims, and the gap between them is exactly where
deployment bugs live. Verify on the real system.

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
| Accuracy cost of pruning (untuned baseline) | −0.0159 AP (rejected: kept full 30) | −0.0031 AP, non-inferiority demonstrated within a 1pp tolerance (p=0.023) (accepted: **now the deployed feature set**) |

Full detail in `models/vif_report.json`. The "accuracy cost" row above is
the untuned, pre-tuning check only — see the next section for what the
actual tuned model gives up and gains.

### Feature-set trade-off, measured on the real deployed model

`src/feature_tradeoff_analysis.py` fits both feature sets across 100
splits in two ways: **Arm A** (both sets forced to the same, deployed
hyperparameters — isolates "does the feature set alone matter, holding
regularization fixed?") and **Arm B** (each feature set gets its own
independently-tuned hyperparameters — the actual deployment-relevant
question, "if each candidate is tuned properly, which wins?"). Round
three found Arm A alone was biased against detecting a real difference
(see "Round three" above); both are now reported. Win/loss counts across
the 100 splits are the primary evidence here, not the p-values, which are
anti-conservative because every split resamples the same 569 rows (see
`models/feature_tradeoff_report.json`'s `"note"` fields):

| | Arm A: matched regularization | Arm B: independently tuned (ship decision) |
|---|---|---|
| Test recall | full=95.26%, reduced=95.14% (23W/21L/56T, not significant) | full=96.07%, reduced=95.45% (32W/11L/57T, **significant, full wins**) |
| Test precision | full=94.84%, reduced=94.54% (46W/36L/18T, not significant) | full=96.94%, reduced=95.37% (59W/16L/25T, **significant, full wins**) |
| Test ROC-AUC | full=0.9893, reduced=0.9893 (53W/40L/7T, not significant) | full=0.9942, reduced=0.9908 (71W/26L/3T, **significant, full wins**) |
| SHAP top-3 consensus (100 splits, Arm A) | 6% (full) | **82%** (reduced) |

Arm B is the comparison that should drive a "which model to ship"
decision, and it shows a real, consistent cost to the reduced feature
set — the full model wins more resampled splits than it loses on every
metric. This now reproduces round two's originally-claimed finding
(previously thought to be an unresolved environment discrepancy; see
"Round three" above for the actual root cause). What's unambiguous
either way is the explanation-stability benefit: the deployed model's
top-3 SHAP features land on the same 3 features in 82/100 resampled
splits, versus 6/100 for the full feature set — collinearity in the full
set lets credit shift almost arbitrarily between near-duplicate features
run to run. **This project ships the 16-feature model anyway**, treating
the measured explanation-stability benefit as worth the measured
performance cost for an educational/portfolio demo where interpretability
is a stated goal — not because the cost turned out to be zero. Full
detail in `models/feature_tradeoff_report.json`.

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
- **Pinned dependencies, actually enforced** (`requirements.lock`,
  `requirements.txt`): round two noted that loose `>=`-only constraints
  make some quoted statistics environment-dependent (a Nadeau-Bengio
  p-value that differs between sklearn versions, for example) and added
  `requirements.lock` (exact versions via `pip freeze`) — but round three
  found that lockfile was decorative: nothing in CI, Docker, or Streamlit
  Community Cloud (which only reads `requirements.txt`) actually installed
  from it. **Fixed:** `requirements.txt` itself is now pinned to exact
  versions (so Streamlit Cloud's install matches too), and both
  `.github/workflows/ci.yml` and `Dockerfile` install from
  `requirements.lock` for the full pinned transitive-dependency closure.

## Project structure

```
├── src/
│   ├── config.py                    # central config, incl. corrected VIF/target-recall notes
│   ├── data.py                       # load_raw, make_splits, load_splits
│   ├── stats_utils.py                 # Nadeau-Bengio significance + non-inferiority tests
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
├── requirements.txt                # pinned exact versions (installed by Streamlit Community Cloud)
├── requirements.lock               # full pinned transitive closure (installed by CI/Docker)
├── AI_REVIEW_PACKAGE.md            # round-one review package
├── AI_REVIEW_PACKAGE_ROUND3.md     # round-three review package
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
- The non-inferiority test in `feature_analysis.py` (see "What changed
  after external review" above) uses a declared 1-percentage-point
  average-precision tolerance. That tolerance is a judgment call, not a
  derived quantity — a different, defensible tolerance could flip the
  upstream feature-selection decision. `feature_tradeoff_analysis.py`'s
  Arm B remains the more important, post-hoc measurement of what the
  actual shipped model gives up.
- Win/loss counts across the 100 splits in `feature_tradeoff_analysis.py`
  are the primary evidence, but the splits all resample the same
  569-row dataset and aren't fully independent — the reported p-values
  are anti-conservative (more significant-looking than a fully rigorous
  test would produce). This is called out directly in the JSON output's
  `"note"` fields rather than left implicit.
- This README, and the analysis behind it, has now gone through three
  rounds of external adversarial review and correction, including
  mutation testing in round two. That doesn't mean it's now correct — it
  means it's been checked three times, and each check found real problems
  the previous ones missed (an untested bug, a broken report, an
  unmeasured trade-off, and — most importantly — a verified-but-unpushed
  fix that meant the live app was still broken after round two "fixed"
  it). Treat any specific claim you're relying on as something to
  re-verify, not as settled, and assume a fourth review would likely find
  more.

## Possible next steps

- Move from tabular features to raw histopathology images (BreakHis or
  PatchCamelyon) with a CNN/transfer-learning approach.
- Add data/model versioning (DVC) and experiment tracking (MLflow/W&B).
- Deploy behind a FastAPI inference endpoint with request logging and
  drift monitoring, alongside the Streamlit demo.
- Explore an LLM/RAG layer over medical literature (e.g. PubMed abstracts)
  as a separate, distinct project.
- Re-derive `TARGET_RECALL` from the validation positive count at runtime
  (or drop the target-recall abstraction entirely in favor of directly
  reasoning about achievable order statistics) instead of a hardcoded
  value that doesn't mean what it says.
