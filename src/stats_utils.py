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


def nadeau_bengio_noninferiority_test(
    scores_reduced: np.ndarray, scores_full: np.ndarray, n_train: int, n_test: int, margin: float
) -> Tuple[float, float]:
    """One-sided non-inferiority test, using the same Nadeau-Bengio corrected
    variance as `nadeau_bengio_test` above, but against a declared margin
    instead of zero.

    WHY THIS EXISTS (found by external adversarial review): a plain
    two-sided significance test against zero, combined with "ship the
    reduced set unless a cost is proven significant," is an
    absence-of-evidence-as-evidence-of-absence error -- failing to detect a
    difference is not the same as proving there isn't a meaningful one,
    especially with a conservative correction like Nadeau-Bengio making
    genuine costs *harder* to detect. The statistically correct question
    is not "is the cost significantly different from zero?" but "can we
    positively demonstrate the cost is no worse than a tolerance we're
    willing to accept?" -- a non-inferiority test (the one-sided special
    case of TOST equivalence testing, since we only care about the reduced
    set being *worse*, not about it being suspiciously *better*).

    H0 (null, the thing we need to positively reject): the reduced set is
    worse than `full - margin`, i.e. mean(reduced - full) <= -margin.
    H1 (what "non-inferior" means here): mean(reduced - full) > -margin.

    We reject H0 -- i.e. conclude the reduced set is non-inferior and safe
    to ship -- only when there's positive statistical evidence for it, not
    merely an absence of evidence against it. If the test is underpowered
    (small n, noisy scores), it correctly defaults to "not proven
    non-inferior" rather than silently assuming equivalence.

    Returns (t_statistic, one_sided_p_value) for H0 above. p < 0.05 means
    non-inferiority is positively demonstrated at the 5% level.
    """
    diff = np.asarray(scores_reduced) - np.asarray(scores_full)
    n = len(diff)
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    if var_diff == 0:
        # No variance across folds: non-inferiority holds iff the (identical)
        # diff is already above the margin.
        return (np.inf if mean_diff > -margin else -np.inf), (0.0 if mean_diff > -margin else 1.0)
    corrected_var = var_diff * (1.0 / n + n_test / n_train)
    se = np.sqrt(corrected_var)
    t_stat = (mean_diff - (-margin)) / se
    p_value = 1 - stats.t.cdf(t_stat, df=n - 1)
    return float(t_stat), float(p_value)
