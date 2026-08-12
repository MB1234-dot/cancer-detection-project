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
    """Exercises the sidebar 'load an example patient' path, then Predict --
    a different code path (real test-set row + real diagnosis label) than
    the default-inputs test above."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    selectbox = at.sidebar.selectbox[0]
    # options[0] is "-- manual entry --"; options[1] is the first real patient
    assert len(selectbox.options) > 1, "Expected at least one example patient in the dropdown"
    selectbox.set_value(selectbox.options[1]).run(timeout=60)
    assert not at.exception, f"App raised after selecting an example patient: {at.exception}"
    at.button[0].click().run(timeout=60)
    assert not at.exception, f"App raised after Predict on a loaded example patient: {at.exception}"
