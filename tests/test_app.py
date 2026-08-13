"""
Smoke tests for the Streamlit app (app/app.py) using Streamlit's own
AppTest framework, which actually executes the script and every callback --
not just imports it.

Why this file exists: a real bug (an unescaped literal "%" inside a
Python %-format string -- "95% CI: ..." -- which the % operator tries to
parse as a format directive and raises ValueError) shipped to production
and crashed the deployed app on first load. It slipped past all 18 tests
in test_data.py/test_pipeline.py because none of them ever executed
app.py -- they only test the src/ pipeline modules. AppTest closes that
gap by actually running the app's script (and simulating widget
interactions) the same way Streamlit Cloud would.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP_PATH = str(config.ROOT_DIR / "app" / "app.py")

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Model not trained yet -- run the full pipeline first (see README).",
)


# Patient #3 in the held-out test set (predicted probability of malignancy
# ~=0.64) sits in the band where the default (0.5) and validation-tuned
# thresholds disagree -- 9 of the 114 test patients fall in this band. The
# earlier version of this test used Patient #0 (probability ~0.0001), which
# never exercises the threshold-disagreement branch in app.py at all.
DISAGREEMENT_PATIENT = "Patient #3"


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception, f"App raised on initial load: {at.exception}"
    # Content assertions, not just absence-of-exception: the sidebar
    # performance metrics should actually be populated from eval_summary.json.
    metric_labels = [m.label for m in at.sidebar.metric]
    assert "ROC-AUC" in metric_labels
    assert any("Recall" in label for label in metric_labels)
    assert any("Precision" in label for label in metric_labels)


def test_predict_button_works_on_default_inputs():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.button[0].click().run(timeout=60)
    assert not at.exception, f"App raised after clicking Predict: {at.exception}"
    # A prediction should actually render: a probability metric and a
    # default-threshold call, both non-empty.
    metric_labels = [m.label for m in at.metric]
    assert any("Predicted probability" in label for label in metric_labels)
    assert any("Call @ default threshold" in label for label in metric_labels)


def test_loading_example_patient_and_predicting_works():
    """Exercises the sidebar 'load an example patient' path, then Predict,
    using a patient whose probability falls in the band where the default
    and validation-tuned thresholds disagree -- a different and previously
    untested code path than the default-inputs test above."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    selectbox = at.sidebar.selectbox[0]
    assert DISAGREEMENT_PATIENT in selectbox.options, (
        f"Expected {DISAGREEMENT_PATIENT} in the example-patient dropdown -- "
        "if the train/test split changed, pick a new patient id whose "
        "predicted probability falls between the default and tuned "
        "thresholds (see eval_summary.json) and update DISAGREEMENT_PATIENT."
    )
    selectbox.set_value(DISAGREEMENT_PATIENT).run(timeout=60)
    assert not at.exception, f"App raised after selecting an example patient: {at.exception}"
    # The "actual diagnosis" info banner should reflect the real test-set label.
    assert any("actual diagnosis" in info.value for info in at.info), (
        "Expected the loaded-patient info banner to appear"
    )

    at.button[0].click().run(timeout=60)
    assert not at.exception, f"App raised after Predict on a loaded example patient: {at.exception}"
    # This patient is specifically chosen to disagree between thresholds,
    # so the app's disagreement warning must render.
    assert any("thresholds disagree" in info.value for info in at.info), (
        "Expected the default/tuned threshold disagreement banner to render "
        f"for {DISAGREEMENT_PATIENT}, which was chosen because it sits in "
        "that band -- if this fails, either the model/split changed or the "
        "disagreement-banner logic in app.py regressed."
    )
