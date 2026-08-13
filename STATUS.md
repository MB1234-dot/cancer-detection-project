# Project status

Living state file. Update it whenever the state changes, so a chat handoff or
a new session never has to reconstruct where things stand.

**Last updated:** 2026-08-13

---

## Current state

| | |
|---|---|
| Local `HEAD` | `7d2057b` (merge) |
| `origin/main` | `7d2057b` |
| In sync? | Yes — verified by fresh clone, not by push output |
| Live app | https://cancer-detection-project.streamlit.app/ |
| Live app verified? | See "Open watch item" below |
| Test suite | 25 passed |

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

## Open watch item

**`requirements.txt` is now pinned to exact versions, and that is the file
Streamlit Community Cloud reads.** The live app will reinstall dependencies
on this deploy. Confirm it still loads after this push — this is the one
change in the batch that can break the deployment. If it fails, the app logs
are under "Manage app" in the Streamlit Cloud dashboard, and the most likely
cause is a pinned version unavailable for Streamlit Cloud's Python runtime.

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

Nothing is blocking. In rough priority order:

1. **Confirm the live app after this deploy** (see Open watch item).
2. **Check the CI run** on this push — `.github/workflows/ci.yml` now installs
   from `requirements.lock` and runs the full pipeline plus all 25 tests,
   including `tests/test_app.py`. This is the first run under the pinned
   lockfile; if it goes green, the reproducibility claim is demonstrated
   rather than asserted.
3. **Optional, low value:** a fourth review round has diminishing returns.
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
