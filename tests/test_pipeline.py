"""
Tests for the trained model artifact and the threshold/CI logic in
evaluate.py. These require the pipeline to have been run at least once
(train.py + split_data.py) so models/best_model.joblib exists -- they're
integration tests against real artifacts, not pure unit tests against mocks,
which is deliberate: the thing we most want to catch (a shape mismatch, a
probability outside [0,1], a threshold function that doesn't actually hit its
target) only shows up against the real fitted pipeline.
"""
import numpy as np
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

    def test_falls_back_gracefully_when_target_unreachable(self):
        y_true = np.array([0] * 10 + [1] * 10)
        y_proba = np.full(20, 0.5)  # no separation at all
        threshold = choose_threshold(y_true, y_proba, target_recall=0.999)
        assert 0.0 <= threshold <= 1.0  # doesn't crash, returns something sane


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
