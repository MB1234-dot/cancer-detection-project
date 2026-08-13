# Project status

Living state file. Update it whenever the state changes, so a chat handoff or
a new session never has to reconstruct where things stand.

**Last updated:** 2026-08-13

---

## Current state

| | |
|---|---|
| Local `HEAD` | `9456fc4` |
| `origin/main` | `9456fc4` |
| In sync? | Yes — verified by fresh clone, not by push output |
| Live app | https://cancer-detection-project.streamlit.app/ |
| Live app verified? | Yes — loads and renders in a browser, 2026-08-13 |
| Test suite | 24 passed |

## What is verified, and how

Verification means checked on the real system, not inferred from a local
result.

- **`origin/main` is at `9456fc4`** — `git ls-remote`, plus a fresh
  `git clone` into a separate directory, confirming the fix is present in the
  public code and that `pytest tests/test_app.py` passes against that clone
  (6/6). "The push command succeeded" was not accepted as evidence.
- **The live app loads** — opened in a browser and confirmed rendering the
  title, disclaimer, sidebar metrics, inputs, and Predict button. No red
  error box.
- **The crash fix actually fixes the crash** — the new tests were run against
  the pre-fix `app.py` and fail there with the exact production error
  (`ValueError: unsupported format character 'C' (0x43) at index 4`), and
  pass with the fix. A test that only ever passes proves nothing.

## Not done / known gaps

1. **Round-three methodology work is not in this history.** A previous
   session produced commits (`59696fc`, `2f59baf`, `5ad7b54`) containing:
   - a two-arm rewrite of `src/feature_tradeoff_analysis.py` (Arm A: matched
     regularization; Arm B: each feature set independently tuned), fixing a
     bug where one shared `C` tuned for the 16-feature set was applied to the
     30-feature arm at roughly 21x weaker regularization than it would choose
     for itself
   - `nadeau_bengio_noninferiority_test` in `src/stats_utils.py`, replacing
     an inverted significance argument in `src/feature_analysis.py` that
     treated "not significantly different from zero" as proof of equivalence
   - exact version pins in `requirements.txt` (Streamlit Cloud only reads
     that file) with CI and Dockerfile installing from `requirements.lock`
   - `AI_REVIEW_PACKAGE_ROUND3.md`

   Those commits were never pushed and exist only in that session's sandbox,
   which is not reachable from later sessions. The crash fix was rebuilt
   independently as `9456fc4`; the methodology items above are **not** in the
   repo and the README does not claim they are.

   To recover: have that session emit `git format-patch` or `git bundle`,
   download the file, upload it to a new session, and rebase onto `9456fc4`.
   If the sandbox is gone, the work must be redone from the description above.

2. **`requirements.txt` is still unpinned** (`>=` ranges). Deliberately not
   changed in the same push that un-broke the deploy — one variable at a
   time. `requirements.lock` exists but Streamlit Cloud does not read it.

3. **No CI check that the app actually renders.** The tests exist; CI should
   run `tests/test_app.py` so a future app-level crash fails the build rather
   than reaching production.

## Next steps

Roughly in priority order:

1. Recover or redo the round-three methodology work (see gap 1).
2. Pin `requirements.txt`, then confirm the live app still deploys.
3. Wire `tests/test_app.py` into CI.

## Operating notes

- Push requires a GitHub token; the sandbox git proxy must be bypassed for
  the push command only (`env -u https_proxy -u HTTPS_PROXY ... git push`).
- Prefer fine-grained tokens scoped to this repo with Contents: Read and
  write. Delete the token once the push is done — a classic `ghp_` token
  grants access to every repo on the account.
- Each chat session gets its own sandbox. Nothing on a session's filesystem
  survives it. **If it isn't pushed, it doesn't exist.**
