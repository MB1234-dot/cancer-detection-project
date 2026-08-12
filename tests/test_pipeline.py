"""
Tests for the trained model artifact and the threshold/CI logic in
evaluate.py. These require the pipeline to have been run at least once
(train.py + split_data.py) so models/best_model.joblib exists -- they're
integration tests against real artifacts, not pure unit tests against mocks,
which is deliberate: the thing we most want to catch (a shape mismatch, a
probability outside [0,1], a threshold function that doesn't actually hit its
target) only shows up against the real fitted pipeline.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_splits, load_selected_features, TARGET_COL
from src.evaluate import choose_threshold, bootstrap_ci

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Model not trained yet -- run `python3 -m src.split_data && python3 -m src.train` first.",
)


@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    features = load_selected_features()
    _, _, test_df = load_splits()
    return test_df[features], test_df[TARGET_COL]


def test_predict_proba_shape_and_range(model, test_data):
    X_test, _ = test_data
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predictions_are_deterministic(model, test_data):
    X_test, _ = test_data
    p1 = model.predict_proba(X_test)
    p2 = model.predict_proba(X_test)
    np.testing.assert_array_equal(p1, p2)


def test_model_beats_trivial_baseline(model, test_data):
    """A model that just predicts the majority class (benign) would get
    ~63% accuracy on the full dataset while catching zero cancers. Our model
    needs to substantially beat that on recall for the malignant class."""
    from sklearn.metrics import recall_score
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    assert recall_score(y_test, y_pred) > 0.85


class TestChooseThreshold:
    def test_hits_target_recall_on_easy_separable_data(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 50 + [1] * 50)
        # scores that clearly separate the classes
        y_proba = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
        threshold = choose_threshold(y_true, y_proba, target_recall=0.95)
        y_pred = (y_proba >= threshold).astype(int)
        from sklearn.metrics import recall_score
        assert recall_score(y_true, y_pred) >= 0.95

    def test_falls_back_gracefully_when_target_literally_impossible(self):
        """CORRECTNESS NOTE (external adversarial review): the original
        version of this test used target_recall=0.999, which does NOT
        exercise the fallback branch -- sklearn's precision_recall_curve
        always returns recall[0] == 1.0 (the lowest-threshold point catches
        every positive by construction), so `recalls[:-1] >= target_recall`
        is satisfied at index 0 for ANY target_recall <= 1.0. The fallback
        (`len(valid_idx) == 0`) is only reachable for a target_recall that
        exceeds 1.0, i.e. a genuinely invalid input -- which is what this
        test now actually uses. The previous version passed regardless of
        whether the fallback code was correct, which is worse than not
        having the test at all: it looked like coverage without being any.
        """
        y_true = np.array([0] * 10 + [1] * 10)
        y_proba = np.full(20, 0.5)
        threshold = choose_threshold(y_true, y_proba, target_recall=1.5)
        assert threshold == 0.5  # documented fallback value, not just "didn't crash"

    def test_high_target_recall_on_small_positive_count_is_a_min_statistic(self):
        """CORRECTNESS NOTE (external adversarial review): with a small
        number of positives, a target_recall close to 1.0 doesn't select
        "a robust threshold hitting ~X% recall" -- it collapses to exactly
        the minimum predicted probability among the positives, because that
        is the only threshold at or below which every positive is caught.
        This isn't a bug to fix (it's a mathematical fact about the
        precision-recall curve), but it WAS being treated like a normal,
        stable statistic in this project until external review pointed out
        that this makes the "tuned" threshold a single order statistic with
        very high variance. This test documents and locks in that behavior
        so it's a known, tested property rather than a silent surprise --
        see src/stability_analysis.py for the actual variance quantification.
        """
        rng = np.random.default_rng(7)
        y_true = np.array([0] * 20 + [1] * 6)
        y_proba = np.concatenate([rng.uniform(0, 0.6, 20), rng.uniform(0.1, 0.9, 6)])
        min_positive_proba = y_proba[y_true == 1].min()
        threshold = choose_threshold(y_true, y_proba, target_recall=0.99)  # only achievable at 6/6=1.0
        assert threshold == pytest.approx(min_positive_proba)


class TestBootstrapCI:
    def test_ci_bounds_are_ordered_and_within_unit_interval(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 100)
        y_proba = rng.uniform(0, 1, 100)
        result = bootstrap_ci(y_true, y_proba, threshold=0.5, n_boot=200, seed=1)
        for metric in ("recall", "precision", "roc_auc"):
            lower = result[metric]["ci_lower_2.5%"]
            upper = result[metric]["ci_upper_97.5%"]
            assert 0.0 <= lower <= upper <= 1.0


class TestNoLeakage:
    """Regression test for the ORIGINAL bug this whole project was rebuilt
    to fix: evaluate.py used to select its threshold by sweeping the test
    set, then report metrics on that same test set. This test exercises the
    actual entry point that had the bug (evaluate.main(), not just the pure
    choose_threshold() function) and asserts the chosen threshold is
    unaffected by completely corrupting the on-disk test set -- because
    threshold selection must depend only on the validation set. If someone
    reintroduces the leak (e.g. accidentally passes test data into
    choose_threshold during a refactor), this test will fail.
    """

    def test_threshold_unaffected_by_test_set_corruption(self, tmp_path):
        from src import evaluate as evaluate_module

        evaluate_module.main()
        with open(config.EVAL_SUMMARY_PATH) as f:
            real_threshold = json.load(f)["tuned_threshold"]

        test_df = pd.read_csv(config.TEST_PATH)
        rng = np.random.default_rng(999)
        corrupted = test_df.copy()
        corrupted[TARGET_COL] = rng.integers(0, 2, len(corrupted))  # random labels
        for col in corrupted.columns:
            if col != TARGET_COL:
                corrupted[col] = rng.permutation(corrupted[col].values)  # shuffle features
        corrupted_path = tmp_path / "test_corrupted.csv"
        corrupted.to_csv(corrupted_path, index=False)

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(config, "TEST_PATH", corrupted_path)
                evaluate_module.main()
                with open(config.EVAL_SUMMARY_PATH) as f:
                    corrupted_threshold = json.load(f)["tuned_threshold"]
            assert real_threshold == corrupted_threshold, (
                "Threshold changed when the test set was corrupted -- this means "
                "threshold selection is reading from the test set, which is the "
                "original leakage bug."
            )
        finally:
            # config.TEST_PATH is restored automatically at the end of the
            # `with` block above; regenerate eval_summary.json against the
            # REAL test set so this test doesn't leave stale/corrupted
            # artifacts on disk for other scripts (e.g. the Streamlit app)
            # to pick up.
            evaluate_module.main()
