# Review package, round three: Breast Mass Malignancy Risk Estimator

You're being asked to independently review this project again -- specifically
for AI-generated "slop" (confident-sounding output that doesn't hold up under
scrutiny), for anything a beginner deploying their first public app might have
overlooked, and for security/safety issues, since the owner is new to
GitHub/cloud deployment and wants a second opinion before trusting this is
safe. Please don't take any claim below at face value -- check numbers against
the raw JSON, check code claims against the pasted source, and say so
explicitly if you can't verify something from what's given.

This package was assembled by the same AI (Claude) that built the project, at
the owner's request, specifically so an independent AI could review it
adversarially. Treat "self-assessment" language below as a hypothesis to test,
not a conclusion to accept.

---

## 1. What this is, and what happened before this round

A binary classifier (malignant vs. benign) on the Wisconsin Diagnostic Breast
Cancer dataset (569 rows, public, no real patient data), built as a portfolio
project, now deployed live on Streamlit Community Cloud
(`app/app.py` in the repo). Full repo:
https://github.com/MB1234-dot/cancer-detection-project

**Compressed history** (full detail in the repo's `README.md`, sections
"Process..." and "Round two"):

1. **V1** had real test-set leakage and a degenerate CV objective.
2. **V2 rebuild** fixed the architecture: proper train/val/test split,
   average-precision model selection, VIF multicollinearity analysis,
   bootstrap CIs, tests, CI, Docker.
3. **Round-one external adversarial review** (a different AI, given source +
   raw metrics) found three real bugs, independently reproduced here, then
   fixed: a VIF computation missing an intercept (`add_constant`), a
   degenerate threshold-selection target, and model selection with no
   significance test.
4. **Round-two external adversarial review** (same reviewer, full repo access
   this time, plus mutation testing -- deliberately reintroducing each fixed
   bug to check whether tests actually catch it) confirmed all round-one
   fixes were real, but found: the VIF fix had **zero test coverage** (all
   17 tests passed with the bug reintroduced), the threshold-stability
   report's seed categories didn't sum to 200 (silently dropped the
   worst-case bucket), the feature-selection decision used a bare
   `delta > -0.01` inequality with no significance test, and -- most
   importantly -- **the deployed 16-feature model measured worse than the
   30-feature model on the actual tuned model's recall/precision/ROC-AUC**,
   a cost the untuned pre-tuning comparison couldn't see.

**This package covers what happened after round two** -- section 2 below.
Round one and two's full detail (code, JSON, reviewer's own words) is in
`AI_REVIEW_PACKAGE.md` in the repo if you want it, but shouldn't be necessary
to review what's new.

## 2. What changed after round two (this is the part that needs review)

### 2a. Round-two fixes

- **VIF regression test** (`tests/test_pipeline.py::TestVIFCorrectness`):
  added and verified via the same mutation-testing method round two used --
  reintroducing the `add_constant` bug makes this specific test fail with
  the exact known buggy value (63,499.3); with the fix in place, it passes.
- **Stability report accounting fixed** (`src/stability_analysis.py`): the
  three seed-outcome categories now partition all 200 seeds exactly
  (47 improved + 80 unchanged + 73 worse = 200), and the "made recall
  worse" bucket now reports its magnitude (mean loss 5.25pp, worst case
  16.67pp) instead of being an implied, unlabeled remainder.
- **Feature-selection significance test** (`src/feature_analysis.py`): the
  bare `delta > -0.01` inequality was replaced with the same Nadeau-Bengio
  corrected paired t-test already used for model-family selection.
- **New post-hoc trade-off analysis** (`src/feature_tradeoff_analysis.py`,
  new file): fits the *actual tuned* hyperparameters (not an untuned
  baseline) on both the 16- and 30-feature sets across 100 independent
  splits, measuring both the real performance cost AND the SHAP
  explanation-stability benefit of the smaller feature set -- something
  round two noticed but didn't quantify.

**Result, reproduced independently in this environment (`models/feature_tradeoff_report.json`):**

```json
{
  "n_splits": 100,
  "performance_cost_reduced_vs_full": {
    "recall":    {"mean_full": 0.9526, "mean_reduced": 0.9514, "diff": -0.0012, "paired_ttest_p": 0.5556, "significant": false},
    "precision": {"mean_full": 0.9484, "mean_reduced": 0.9454, "diff": -0.0030, "paired_ttest_p": 0.3610, "significant": false},
    "roc_auc":   {"mean_full": 0.9893, "mean_reduced": 0.9893, "diff":  0.0000, "paired_ttest_p": 0.8947, "significant": false}
  },
  "shap_top3_consensus": {
    "full_30_features":    {"fraction_matching_modal_set": 0.06, "n_distinct_sets": 61},
    "reduced_16_features":  {"fraction_matching_modal_set": 0.82, "n_distinct_sets": 7}
  }
}
```

**This is a discrepancy worth your scrutiny.** Round two's own review claimed
a *significant* performance cost (reported as p<1e-6). This independent
reproduction, run in this repo's pinned environment
(`requirements.lock` -- sklearn 1.9.0, numpy 2.4.6, scipy 1.17.1), found NO
significant difference on any of the three metrics. The SHAP stability
finding DID reproduce cleanly (82% vs 6%). The project's own README states
this discrepancy plainly rather than picking a side -- **please assess
whether that's the right call, or whether there's a methodology flaw in
`feature_tradeoff_analysis.py` (pasted in full below) that would explain
away round two's finding incorrectly.**

### 2b. A real bug found in production, after this session already thought it was done

Minutes after first deploying to Streamlit Community Cloud, the live app
crashed on load with:

```
ValueError
File "app/app.py", line 71, in <module>
    help="95% CI: [%.3f, %.3f]" % (default_metrics["bootstrap_95ci"]["recall"]["ci_lower_2.5%"], ...)
```

**Root cause:** `"95% CI: [%.3f, %.3f]" % (...)` -- the literal `%` in `"95%"`
was parsed by Python's `%` string-format operator as the start of a format
directive, which is invalid when followed by a space, raising `ValueError`.
This is a classic gotcha (a literal `%` in a `%`-operator string must be
escaped as `%%`), and it was in `app.py` from the original V2 rebuild --
**it existed through round one review, round two review, and mutation
testing, and none of them caught it**, because none of those processes ever
executed `app.py` -- they only tested `src/` pipeline modules. It only
surfaced once a real user (the owner) loaded the actual deployed app.

**Fix:** switched the three affected strings to f-strings (verified
locally: `python3 -m py_compile`, then Streamlit's own `AppTest` framework
run against the live script, including clicking Predict and loading an
example patient -- all clean, no exceptions). **Added
`tests/test_app.py`**, three tests using `AppTest` to actually execute
`app.py` end-to-end (initial load, Predict button, example-patient
dropdown) -- specifically so this class of bug (a real code path with zero
test coverage) can't repeat.

**Please check:** (a) is the f-string fix actually correct and complete --
are there other unescaped `%` characters anywhere else in the app or in any
`%`-operator format string in `src/`? (b) do the three new `AppTest` tests
actually exercise the code paths that matter, or do they give a false sense
of coverage? (c) is there anything else in `app.py` that's untested and
could fail the same way in production?

### 2c. An observation surfaced while gathering this package (not yet
investigated, flagging honestly)

Re-running the full pipeline from scratch produced identical results for
the deployed model (logistic regression: identical `best_params`,
identical CV score, to 4 decimal places, every time) -- but the
**non-deployed** alternative model, random forest, reported the same mean
CV score (0.9895) across two different pipeline runs but with **different
winning hyperparameters** (`n_estimators=400, max_depth=5,
class_weight=balanced` in one run vs. `n_estimators=200, max_depth=None,
class_weight=None` in another), both under the same `RANDOM_STATE=42`.
This suggests either a tie in CV score being broken non-deterministically
(plausible with `n_jobs=-1` parallel search and floating-point score
comparison) or an actual reproducibility gap. **It doesn't affect the
deployed model or any reported number** (random forest was never selected
or shipped), but it's a loose thread -- please flag if you think it's worth
investigating, or if it suggests a broader reproducibility risk in
`train.py`'s `RandomizedSearchCV` setup.

## 3. Security / safety review (the owner is new to this -- please be thorough)

The owner is deploying a public GitHub repo + a public Streamlit Cloud app
for the first time and explicitly asked for a safety check. Please assess,
specifically:

1. **Secrets/credentials:** is there anything in the repo (code, config,
   committed data/model files, git history) that looks like an API key,
   token, password, or other credential that shouldn't be public? (Claude's
   own check: `grep -rniE "api[_-]?key|secret|password|token"` across the
   repo found nothing beyond the word "token" appearing in prose/comments
   about GitHub tokens used during setup, none of which contain an actual
   secret value -- please verify independently rather than trust this.)
2. **Data privacy:** the dataset is the public sklearn/UCI Wisconsin
   Diagnostic Breast Cancer dataset -- no real patient identifiers, sourced
   via `sklearn.datasets.load_breast_cancer`. Confirm there's no
   re-identification risk or PII anywhere in the repo.
3. **App-level risk surface:** `app/app.py` takes numeric inputs via
   Streamlit widgets (`st.number_input`) and runs them through a
   `scikit-learn` pipeline + SHAP explainer -- no file uploads, no raw
   text/HTML rendering of user input, no database, no auth, no
   server-side state beyond Streamlit's own caching. Is there any injection,
   XSS, or resource-exhaustion risk you'd flag in this input surface?
4. **Deployment platform:** Streamlit Community Cloud (Snowflake-owned),
   connected via GitHub OAuth, deploying from a public repo. Any
   platform-specific risk the owner (a beginner) should know about --
   e.g., app sleep/wake behavior, resource limits, what "public repo"
   actually exposes beyond the code itself?
5. **Process hygiene:** a GitHub Personal Access Token (`repo` + `workflow`
   scope) was created twice during this session to let Claude push commits
   from its sandbox, and the owner was told to revoke each one immediately
   after use. Is that an acceptable pattern for a beginner, or is there a
   safer way you'd recommend for next time (e.g., GitHub CLI device flow,
   deploy keys, fine-grained tokens scoped to just this repo)?

## 4. Current test suite (21 tests, all passing at time of writing)

```
tests/test_data.py (9):
  test_raw_data_shape, test_no_missing_values, test_target_is_binary,
  test_target_convention_matches_known_class_balance,
  test_splits_do_not_overlap, test_splits_cover_full_dataset,
  test_splits_are_stratified, test_splits_are_reproducible,
  test_different_seeds_give_different_splits

tests/test_pipeline.py (9):
  test_predict_proba_shape_and_range, test_predictions_are_deterministic,
  test_model_beats_trivial_baseline,
  TestChooseThreshold::test_hits_target_recall_on_easy_separable_data,
  TestChooseThreshold::test_falls_back_gracefully_when_target_literally_impossible,
  TestChooseThreshold::test_high_target_recall_on_small_positive_count_is_a_min_statistic,
  TestBootstrapCI::test_ci_bounds_are_ordered_and_within_unit_interval,
  TestVIFCorrectness::test_vif_uses_an_intercept,
  TestNoLeakage::test_threshold_unaffected_by_test_set_corruption

tests/test_app.py (3, NEW this round):
  test_app_loads_without_exception,
  test_predict_button_works_on_default_inputs,
  test_loading_example_patient_and_predicting_works
```

## 5. Full source of the changed/new files

### src/feature_analysis.py -- decision rule (previously a bare inequality)
```python
n_test_per_fold = len(train_df) // config.CV_SPLITS
n_train_per_fold = len(train_df) - n_test_per_fold
t_stat, p_value = nadeau_bengio_test(
    raw_scores["baseline_vif_selected_features"], raw_scores["baseline_full_30_features"],
    n_train=n_train_per_fold, n_test=n_test_per_fold,
)
significant_cost = bool(p_value < 0.05 and delta < 0)
final_features = selected_features if not significant_cost else all_features
```

### src/stability_analysis.py -- partition fix
```python
recall_delta = recall_tuned - recall_default
improved_mask = recall_delta > 0
unchanged_mask = recall_delta == 0
worse_mask = recall_delta < 0
improved = int(np.sum(improved_mask))
unchanged = int(np.sum(unchanged_mask))
worse = int(np.sum(worse_mask))
assert improved + unchanged + worse == N_SEEDS, "recall-outcome buckets must partition all seeds"

worse_damage = -recall_delta[worse_mask]
worse_case_stats = {
    "count": worse,
    "mean_recall_lost": round(float(worse_damage.mean()), 4) if worse else 0.0,
    "worst_case_recall_lost": round(float(worse_damage.max()), 4) if worse else 0.0,
}
```

### src/feature_tradeoff_analysis.py -- new file, full logic (trimmed of docstrings/logging)
```python
def _fit(feats, best_params, train_df):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, C=best_params.get("clf__C", 1.0),
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

# main(): for seed in range(100): make_splits(df, random_state=seed);
# fit both feature sets with the SAME tuned hyperparameters; predict at
# threshold=0.5; record recall/precision/roc_auc; record _top3(...).
# Then: paired t-test (scipy.stats.ttest_rel) + wilcoxon per metric;
# consensus_fraction = (count of splits matching the modal top-3 SET) / n_splits.
# The "summary" field is built FROM these computed values (branching on
# significant_metrics / stability_gap), not a hardcoded narrative -- an
# earlier draft of this script hardcoded "the reduced model is
# significantly worse," which Claude caught and rewrote as data-driven
# before this ever shipped. Worth verifying this claim against the actual
# file in the repo rather than trusting this summary.
```

### app/app.py -- the fix (before/after)
```python
# BEFORE (crashed in production):
c1.metric("Recall @ default (0.5)", default_metrics["recall"],
           help="95% CI: [%.3f, %.3f]" % (default_metrics["bootstrap_95ci"]["recall"]["ci_lower_2.5%"],
                                            default_metrics["bootstrap_95ci"]["recall"]["ci_upper_97.5%"]))

# AFTER (fixed, f-strings):
recall_ci = default_metrics["bootstrap_95ci"]["recall"]
precision_ci = default_metrics["bootstrap_95ci"]["precision"]
c1.metric(
    "Recall @ default (0.5)", default_metrics["recall"],
    help=f"95% CI: [{recall_ci['ci_lower_2.5%']:.3f}, {recall_ci['ci_upper_97.5%']:.3f}]",
)
c2.metric(
    "Precision @ default (0.5)", default_metrics["precision"],
    help=f"95% CI: [{precision_ci['ci_lower_2.5%']:.3f}, {precision_ci['ci_upper_97.5%']:.3f}]",
)
st.caption(
    f"95% CIs are bootstrap resamples of the {eval_summary['n_test']}-patient test set -- "
    f"with only {eval_summary['n_malignant_test']} malignant test cases, point estimates "
    f"alone would overstate precision."
)
```

### tests/test_app.py -- new file, in full
```python
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src import config

APP_PATH = str(config.ROOT_DIR / "app" / "app.py")

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Model not trained yet -- run the full pipeline first (see README).",
)

def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception, f"App raised on initial load: {at.exception}"

def test_predict_button_works_on_default_inputs():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.button[0].click().run(timeout=60)
    assert not at.exception, f"App raised after clicking Predict: {at.exception}"

def test_loading_example_patient_and_predicting_works():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    selectbox = at.sidebar.selectbox[0]
    assert len(selectbox.options) > 1
    selectbox.set_value(selectbox.options[1]).run(timeout=60)
    assert not at.exception, f"App raised after selecting an example patient: {at.exception}"
    at.button[0].click().run(timeout=60)
    assert not at.exception, f"App raised after Predict on a loaded example patient: {at.exception}"
```

## 6. Specific questions for you

1. Is the app-level bug fix (section 2b) actually complete, or are there
   other latent issues in `app.py` the new tests wouldn't catch?
2. On the unresolved significance discrepancy (section 2a) -- does
   `feature_tradeoff_analysis.py`'s methodology look sound to you? Is there
   a plausible reason this run and round two's claimed p<1e-6 disagree that
   isn't "one of us made a mistake"?
3. Section 2c's random-forest hyperparameter non-determinism -- worth
   investigating, or a non-issue since it doesn't affect the deployed model?
4. Security/safety (section 3) -- anything you'd flag before this stays
   live and public, especially for someone who's new to GitHub/cloud
   deployment?
5. Overall: has the pattern of "external review finds real things,
   they get fixed and mutation/AppTest-verified" actually converged toward
   a trustworthy project, or does finding a fourth class of bug (this time
   in the one file with zero review coverage) suggest there's a systemic
   gap in how this project verifies itself that a fifth round would also
   find?
