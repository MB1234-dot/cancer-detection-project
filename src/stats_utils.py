"""
Statistical helpers that go beyond "compare the means."

The Nadeau-Bengio correction exists because k-fold CV, repeated or not,
produces folds that share training data with each other -- a single row
appears in multiple folds' training sets across repeats, so per-fold scores
are correlated, not independent. A naive paired t-test on per-fold scores
assumes independence and is well known to be anti-conservative (it reports
significance far too often) for exactly this reason. Nadeau & Bengio (2003,
"Inference for the Generalization Error") derive a corrected variance
estimate for this case; this is a standard, if under-used, fix.
"""
from typing import Tuple

import numpy as np
from scipy import stats


def nadeau_bengio_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_train: int, n_test: int
) -> Tuple[float, float]:
    """Corrected paired t-test for repeated k-fold CV scores.

    scores_a, scores_b: paired per-fold scores (same folds, same order) for
    two models being compared -- e.g. split0_test_score..splitN_test_score
    from two RandomizedSearchCV runs that shared the same `cv` object.
    n_train, n_test: number of rows in a single fold's train/test partition
    (used for the correction factor n_test/n_train).

    Returns (t_statistic, two_sided_p_value).
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(diff)
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    if var_diff == 0:
        return (np.inf if mean_diff != 0 else 0.0), (0.0 if mean_diff != 0 else 1.0)
    corrected_var = var_diff * (1.0 / n + n_test / n_train)
    t_stat = mean_diff / np.sqrt(corrected_var)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return float(t_stat), float(p_value)
