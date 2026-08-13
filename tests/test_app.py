"""Tests for the Streamlit app (app/app.py).

Why this file exists
--------------------
The deployed app crashed on load with a ValueError, and no test caught it,
because the bug was not in any importable function -- it was in module-level
Streamlit code that only runs when the app is actually rendered.

The specific bug: strings containing a literal '%' (as in "95% CI") were being
built with %-formatting. Python parses "95% C" as a format spec, not as a
literal percent sign, and raises:

    ValueError: unsupported format character 'C' (0x43) at index 4

Two layers of defence here:

1. `test_no_percent_format_on_string_literals` -- a static AST check that fails
   if anyone reintroduces %-formatting on a string literal in app.py. This is
   fast and needs no model artifacts.
2. The AppTest tests -- actually render the app and assert on its *content*,
   not merely that it did not raise. An earlier version of this file only
   asserted `not at.exception`, which is a weak guarantee: a Streamlit app that
   silently renders nothing also passes that check.
"""
import ast
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "app.py"


def test_app_file_exists():
    assert APP_PATH.is_file(), f"expected the Streamlit app at {APP_PATH}"


def test_no_percent_format_on_string_literals():
    """Fail if app.py uses %-formatting on a string literal.

    `i % cols_per_row` (integer modulo) is fine and is not flagged -- we only
    look at BinOp nodes whose left operand is a string constant.
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)):
            continue
        left = node.left
        # unwrap implicit concatenation of adjacent string literals
        while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Add):
            left = left.left
        if isinstance(left, ast.Constant) and isinstance(left.value, str):
            offenders.append((node.lineno, left.value[:60]))

    assert not offenders, (
        "app.py uses %-formatting on string literals at "
        + ", ".join(f"line {line} ({text!r})" for line, text in offenders)
        + ". Use an f-string instead -- a literal '%' in these strings (e.g. "
        "'95% CI') is parsed as a format spec and raises ValueError at import, "
        "which takes the deployed app down on load."
    )


def test_literal_percent_strings_are_present_and_safe():
    """The '95% CI' text should still be in the app, just not %-formatted.

    Guards against 'fixing' the crash by deleting the confidence intervals.
    """
    source = APP_PATH.read_text(encoding="utf-8")
    assert "95% CI" in source
    assert "95% CIs are bootstrap resamples" in source


# --- rendering tests -------------------------------------------------------
# These need the trained model artifacts, so they are skipped if the pipeline
# has not been run.

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed"
)
AppTest = streamlit_testing.AppTest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_artifacts_present = (MODELS_DIR / "best_model.joblib").exists() and (
    MODELS_DIR / "eval_summary.json"
).exists()

requires_artifacts = pytest.mark.skipif(
    not _artifacts_present,
    reason="model artifacts missing -- run the pipeline first",
)


def _run_app(timeout=120):
    at = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    at.run()
    return at


@requires_artifacts
def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception, f"app raised on load: {at.exception}"


@requires_artifacts
def test_app_renders_expected_content():
    """Assert on actual rendered content, not just absence of an exception."""
    at = _run_app()
    assert not at.exception

    # the disclaimer must survive any refactor -- this is a medical demo
    elements = (
        list(at.markdown)
        + list(at.caption)
        + list(at.info)
        + list(at.warning)
        + list(at.subheader)
    )
    all_text = " ".join(
        str(getattr(el, "value", "")) + " " + str(getattr(el, "body", ""))
        for el in elements
    )
    assert "clinically validated" in all_text or "never be used" in all_text, (
        "the not-for-clinical-use disclaimer is missing from the rendered app"
    )

    # sidebar performance metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert any("ROC-AUC" in label for label in metric_labels), metric_labels
    assert any("Recall" in label for label in metric_labels), metric_labels

    # the CI tooltip rendered as a literal percent, not a mangled format spec
    ci_helps = [m.help for m in at.metric if m.help]
    assert any(h.startswith("95% CI: [") for h in ci_helps), ci_helps

    # input controls exist
    assert len(at.number_input) > 0, "no measurement inputs rendered"
    assert len(at.button) > 0, "no Predict button rendered"


@requires_artifacts
def test_predict_button_produces_a_probability():
    at = _run_app()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception, f"app raised on Predict: {at.exception}"

    metric_labels = [m.label for m in at.metric]
    assert any("Predicted probability" in label for label in metric_labels), metric_labels

    proba = next(
        m.value for m in at.metric if "Predicted probability" in m.label
    )
    pct = float(str(proba).rstrip("%"))
    assert 0.0 <= pct <= 100.0, proba
