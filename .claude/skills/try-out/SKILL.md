---
name: try-out
description: Use when the user wants to test a new approach, training recipe, architecture, preprocessing change, or proposal card on Kaggle — "let's try X", "run P-nn", "send this to Kaggle", "try 16 slices per slot" — i.e. any change to the pipeline that needs a kernel run to measure.
---

# Try out an approach on Kaggle

Turn an idea into a falsifiable card, one edit to `src/`, and a **smoke** kernel run — then
stop and report.

## The hard stop

```
This skill pushes with FORCE_SMOKE = True and stops when the smoke run is green.
It never pushes a real run. It never submits.
```

Both of those need Tian's explicit go-ahead in a fresh message. A green smoke run is a
report, not permission. See the rationalization table at the bottom before you argue with
this.

## 1. Resolve the input into a card

**Given `P-nn`:** read that card in `docs/proposals.md`. Implement exactly its `Measure`
and `Noise floor` — not a variant you think is better. If the card's `Depends on` is unmet,
say so and stop.

**Given a free-text idea:** write the card *first*, in `docs/proposals.md`, using that
file's template — Status, Hypothesis, Origin, Evidence, Measure, Noise floor, Cost, If it
works, If it fails, Depends on — and add its row to the ranked index.

> If the idea cannot be written as a falsifiable card, it is not ready to run. Say what is
> missing (usually: what number would prove it wrong) and stop. Do not push a kernel to
> "see what happens" — a Kaggle session is hours, and an unfalsifiable result is unusable.

## 2. Pre-flight — three greps that save a session

```powershell
$env:PYTHONUTF8 = "1"
Select-String -Path docs/experiments.md -Pattern "<keyword>"   # already settled? ❌ DEAD END?
Select-String -Path docs/traps.md       -Pattern "<keyword>"   # known failure mode?
git status --short                                             # start from a clean tree
```

A ❌ DEAD END is only retried with a stated new reason, written into the card.

## 3. Edit `src/`, never the `.ipynb`

- Training / inference changes → `src/kaggle_pipeline.py`
- Preprocessing-cache changes → `src/cache_pipeline.py`
- Nearly every experiment is a field in the `Config` dataclass in the config cell. Prefer a
  config field over scattered edits, so the change is one line and is saved next to the
  checkpoints.
- **Bump `cfg.version`** (`v03` → `v04`) whenever the change alters targets, inputs, or the
  model. Checkpoints are named `{version}_fold{k}_best.pt`; reusing a version silently mixes
  two experiments' weights.
- Change **one thing**. Two changes in one run buy one number and no attribution.
- If the change alters what the cache stores (`cache_px`, `cache_n_slices`, `crop_mm`,
  `lat_dead_zone_mm`), the mounted cache no longer matches and the cache kernels must be
  rebuilt first — that is a separate, larger job. Flag it rather than shipping a mismatch.

## 4. Set the run switches

In the config cell of `src/kaggle_pipeline.py`:

| Field | Value for this push | Why |
|---|---|---|
| `FORCE_SMOKE` | **`True`** | Non-negotiable on the first push after any edit. A crash in the inference cell after six hours costs a whole session. |
| `MODE` | `"train"` or `"auto"` | `"infer"` is only for the submission kernel. |
| `cfg.folds` | leave; smoke forces `(0,)` | |

Then check `kernel-metadata.json` for the kernel you are pushing:

- `"machine_shape": "NvidiaTeslaT4"` — **never P100**; Kaggle's PyTorch ships no Pascal
  kernels and the session dies at the first convolution. Re-check on every new or forked
  kernel.
- `enable_gpu: true` for train/infer, `false` for the cache kernels.
- `enable_internet: false`.
- `kernel_sources` / `model_sources` / `dataset_sources` still list what the code expects.

## 5. Optional local smoke (CPU, free, catches syntax and shape errors)

```powershell
$env:PYTHONUTF8 = "1"; $env:PYTHONPATH = "src"
python src/kaggle_pipeline.py
```

Run from the repo root. Worth the two minutes before spending a Kaggle push.

## 6. Generate the notebook and push

```powershell
$env:PYTHONUTF8 = "1"
python src/nbgen.py src/kaggle_pipeline.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
kaggle kernels push -p kaggle/rsna-knee-train
```

Cache kernels — shard 1 is `sed`'d from the same source, never edited by hand:

```powershell
python src/nbgen.py src/cache_pipeline.py kaggle/rsna-knee-cache-a/rsna-knee-cache-a.ipynb
sed 's/^SHARD = 0 /SHARD = 1 /' src/cache_pipeline.py > artifacts/cache_b.py   # sed ships with Git for Windows
python src/nbgen.py artifacts/cache_b.py kaggle/rsna-knee-cache-b/rsna-knee-cache-b.ipynb
kaggle kernels push -p kaggle/rsna-knee-cache-a
kaggle kernels push -p kaggle/rsna-knee-cache-b
```

## 7. Poll until it settles

Poll in the background rather than blocking; a smoke run is typically 5–15 minutes.

```powershell
$env:PYTHONUTF8 = "1"
kaggle kernels status tiankljucanin/rsna-knee-train
```

**Green (`COMPLETE`):** pull the log and read it — completing is not the same as working.

```powershell
kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/<ver>
```

Confirm in the log that the change actually took effect (the config print at the top), that
`use_cache=True` did not fall back (`! use_cache=True but no cache is mounted`), and that a
`submission.csv` was written and validated.

**Red (`ERROR`):** fetch the log, find the root cause, fix `src/`, re-push the smoke. Do not
"fix" by switching off the assertion that caught it. A new silent failure mode discovered
here goes into `docs/traps.md` via `/update`.

## 8. Update the card, then stop

Set the card's `Status:` and its ranked-index row to `🔧 implemented, effect pending`
(smoke green, nothing measured yet). Commit and push the source change:

```powershell
git add -A
git commit -m "P-nn: <what changed> (smoke green, kernel v<N>)"
git push origin main
```

Then **report and stop**. The report says: what changed and where, the kernel version and
smoke result, what the card predicts, the measurement that will settle it, and the exact
command to launch the real run — for Tian to approve.

```powershell
# after approval only: set FORCE_SMOKE = False, regenerate, push
```

## Red flags — stop and re-read the hard stop

- "The smoke run was green, so the real run is safe to launch"
- "It's a one-line change, smoke is overkill"
- "The OOF beats the baseline, so I'll just submit it"
- "P100 is what was available"
- "I'll edit the `.ipynb` directly, it's faster"
- "I'll bundle these two changes to save a session"

| Rationalization | Reality |
|---|---|
| "Green smoke means the real run is approved" | Smoke proves the code runs. It proves nothing about whether hours of GPU should go to this idea — that is Tian's call. |
| "It's too small a change to smoke-test" | Every silent failure in `traps.md` came from a small change. Smoke is minutes; a dead session is hours. |
| "The result is good, submitting is the obvious next step" | Submissions are limited and traceable. Ask. |
| "The card is bureaucracy for an obvious idea" | An idea with no falsifying number produces a result nobody can act on. |
| "I'll write the card after the run" | Then the measure gets chosen to fit the number. Card first. |
