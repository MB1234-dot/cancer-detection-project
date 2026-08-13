# Project status

Living state file. Update it whenever the state changes, so a chat handoff or
a new session never has to reconstruct where things stand.

**Last updated:** 2026-08-13

---

## Current state

| | |
|---|---|
| Local `HEAD` | `a13db57` |
| `origin/main` | `a13db57` |
| In sync? | Yes — verified by fresh clone, not by push output |
| Live app | https://cancer-detection-project.streamlit.app/ |
| Live app verified? | **Yes** — loaded cleanly after the `requirements.txt` pin took effect on this deploy. See "What is verified, and how" below. |
| CI on `a13db57` | **Passed** — first run installing from `requirements.lock`, full pipeline + all 25 tests. See below. |
| Test suite | 25 passed, verified both in CI and against a fresh clone of the public repo |

## What is verified, and how

Verification means checked on the real system, not inferred from a local
result.

- **`origin/main` matches local** — `git ls-remote`, plus a fresh
  `git clone` into a separate directory, running the test suite against that
  clone. "The push command succeeded" is not accepted as evidence.
- **The crash fix actually fixes the crash** — the app tests were run against
  the pre-fix `app.py` and fail there with the exact production error
  (`ValueError: unsupported format character 'C' (0x43) at index 4`), and
  pass with the fix. A test that only ever passes proves nothing.
- **The feature-tradeoff numbers are environment-stable** —
  `src/feature_tradeoff_analysis.py` was re-run from scratch in a fresh
  container and reproduced the committed
  `models/feature_tradeoff_report.json` **byte for byte**, including every
  win/loss/tie count quoted in the README. This settles the "maybe it's a
  dependency-version discrepancy" question raised during round three: it
  isn't.
- **The non-inferiority decision reproduces** — `src/feature_analysis.py`
  re-run gives Nadeau-Bengio one-sided p=0.0232, matching the documented
  0.023, and regenerates identical artifacts.
- **The live app loads after the `requirements.txt` pin** — this was the one
  change in the `fb30e50`/`a13db57` push that could have broken the
  deployment, since Streamlit Community Cloud only reads `requirements.txt`
  and this push pinned it to exact versions for the first time. Checked
  directly in a browser: the app renders (title, disclaimer banner, input
  form) with no stack trace.
- **CI on `a13db57` passed, under the pinned lockfile** — checked on
  GitHub, not inferred from the push succeeding. Run
  [#5](https://github.com/MB1234-dot/cancer-detection-project/actions/runs/31680362983):
  **Success**, 10m 36s total. Both jobs green — `test` (5m 54s: installs from
  `requirements.lock`, regenerates the pipeline, then runs the full suite
  including `tests/test_app.py`) and `docker-build` (4m 29s: builds the image
  and smoke-tests the container). This is the first CI run under the pinned
  lockfile, so a green result demonstrates the reproducibility claim rather
  than just asserting it. Two informational warnings only (Node.js 20 →  24
  runner deprecation notice on `actions/checkout`/`actions/setup-python`),
  not failures.

## Open watch item

None currently. The `requirements.txt` pin (the one change in the last push
that could have broken the deployment) has been confirmed live, and CI has
confirmed green under the pinned lockfile — see "What is verified, and how"
above.

## History note: how the round-three work was nearly lost

Commits `59696fc`, `2f59baf`, `5ad7b54` were made in a sandbox session, fully
tested, and never pushed. That sandbox was reclaimed and the commits went with
it. They were recovered only because the session was still alive long enough
to emit `git format-patch` files, which were downloaded and re-uploaded to a
later session.

The crash fix had already been independently rebuilt as `9456fc4` by then, so
the recovered patch conflicted with it. Resolved as the union of both — see
merge commit `7d2057b` for detail. The net result is stronger than either side
alone: the AST guard from the rebuild, plus the threshold-disagreement test
from the original.

**The lesson, for the third time in this project: if it isn't pushed, it
doesn't exist.**

## Next steps

Nothing is blocking. Both post-deploy checks from the last round are done and
green (live app, CI on `a13db57`).

1. **Optional, low value:** a fourth review round has diminishing returns.
   Round three's most important finding was a process failure (unpushed
   commits), not a code bug, and that gap is now closed by this file plus the
   verification habit. Don't start another round without a specific reason.

## Operating notes

- Push requires a GitHub token; the sandbox git proxy must be bypassed for the
  push command only (`env -u https_proxy -u HTTPS_PROXY ... git push`).
- Use fine-grained tokens scoped to this repo with Contents: Read and write.
  The "Contents" permission only appears **after** selecting "Only select
  repositories" — that ordering is not obvious. Delete the token once done; a
  classic `ghp_` token grants access to every repo on the account.
- Each chat session gets its own sandbox. Nothing on a session's filesystem
  survives it. Push early; a finished-but-unpushed commit is indistinguishable
  from no commit at all.
