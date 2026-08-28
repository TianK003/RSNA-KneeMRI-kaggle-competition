# Traps, bugs and silent failure modes

Things that go wrong **without raising an error**. That is the organising principle: a
crash tells you about itself, so it needs no document. Everything here either fails
quietly, reports success while doing nothing, or looks like a modelling result when it is
actually a bug.

Ordered by how much damage it does before you notice.

---

## Tier 1 — corrupts results while looking fine

### 1. Sorting DICOM slices by filename

The filename is the **SOP Instance UID**, assigned to be *unique*, not *ordered*. Sorting
by it produces a slice sequence uncorrelated with anatomy, and nothing complains.

**Measured on 12 sample series:** Spearman ρ between filename order and true spatial
position is **mean −0.012**, range −0.31 … +0.25, with `|ρ|>0.99` in **0 of 12** series.

This destroys the entire premise of a 2.5D triplet — the three channels are supposed to be
*adjacent* slices, and instead they are three random slices from the volume. The model still
trains. The loss still falls. The result is just quietly worse.

**Do:** sort by projecting `ImagePositionPatient` onto the slice normal (cross product of
the two `ImageOrientationPatient` vectors). Fall back to `InstanceNumber`, then filename.
Implemented in `ordered_slice_paths()`.

### 2. Ungrouped folds

Two independent leak channels, both of which *raise your validation score* — the most
dangerous direction for a bug to fail in.

- **Shared report text.** Measured: 4,273 distinct reports across 4,407 studies; **49 texts
  are shared by more than one study, covering 183 studies, largest group 37**. Studies
  sharing a report share a target vector. Split that group across folds and the model
  memorises the answer in training and is scored on it in validation.
- **Scanner/site fingerprints in the pixels.** Community reports random K-fold inflates AUC
  ~0.05; one team measured a **+0.136** grouped-vs-random gap.

Report-text grouping is implemented and asserted (`max folds touched by any group == 1`).
**Site grouping is NOT implemented** — see [brainstorm.md](brainstorm.md); it is the largest
known correctness gap in our validation.

### 3. Reading gold labels as a metric after copying them into the targets

Our own bug. `build_targets` overwrites the weak targets with the 58 gold labels, then an
early version scored the result against gold — grading gold against itself and printing
**"teacher gold macro-AUC 1.0000"**. Plausible-looking, completely meaningless.

**Do:** compute teacher quality *before* the gold override. Correct value: **0.8948** (blend;
0.8934 for the diagnostic rank blend).

### 4. Treating noise as a result

With 58 gold studies the Hanley–McNeil SE of an AUC near 0.8 is **≈0.09** — a 95% interval
of roughly **±0.17**. The top 10 public LB teams span **0.006** total.

> A gold-AUC gap under ~0.05, or a public-LB gap under ~0.005, is **not evidence**.

This is how the top public notebook overfit the leaderboard — its author says so in the
notebook, after a fork-and-republish race chasing movements of 0.001–0.003. Judge label
changes on **coverage** (does the rule fire at all, per language) and on **OOF across all
4,407 studies**, not on the 58 gold alone.

### 5. Trusting `Fluid_Sensitive` / `Fat_Suppression` as shipped

They are **degenerate**. Verified across all 24,371 training series: only `(1,1)` (14,010
rows) and `(0,0)` (10,361) ever occur — **never a mixed pair**. Two physically independent
properties (contrast weighting, set by TR/TE, vs. a fat-suppression preparation applied on
top of any weighting) arrive collapsed into one bit.

Header recovery produced **3 of 4** combinations on just 12 sample series, including
`(fluid=True, fat_sat=False)`, which the CSV cannot express at all.

**Do:** recover both from `ScanningSequence`, `SeriesDescription`, `SequenceName`,
`ScanOptions`, and TR/TE. Note gradient echo has a short TR *by design*, so it breaks the
TR/TE rule and must be caught from `ScanningSequence` first.

**But:** `Anatomical_Plane` **is** trustworthy — 100% agreement with the plane derived from
`ImageOrientationPatient`. Only the two flags are bad. Don't waste effort recomputing plane.

### 6. Global intensity windowing

Max intensity spans **690 … 8,736** across sample series — a **12.7×** range. A global
window silently crushes some series to near-black and saturates others.

**Do:** clip each triplet **jointly** at its own 1st/99th percentile, so its three channels
stay mutually comparable, then scale to [0,1]. Also handle `MONOCHROME1` (inverted — high
value means black) and `RescaleSlope`/`RescaleIntercept`. Our sample happened to be all
`MONOCHROME2` with trivial rescale, but the hidden test spans 16–19 sites.

### 6b. Building a BCE target in rank space

Our own bug, found by the research critic and verified on the data. `rank(pct=True)` gives
tied values their **average** rank, so on a label where most reports say exactly 0 every
confident negative received a target of **0.28–0.39** while the 58 gold rows sat at a hard
0/1 with 8× weight. **No study on any label (0%)** had a target below 0.1 before the gold
override (under the probability blend it is 2–72% per label: Synovitis 2%, Baker's 26%,
MCL 72%). AUC is rank-invariant, which is exactly why this looked principled — but BCE fits
the *value*, so the network was being taught "definitely absent" = 0.31.

**Do:** rank space for *scoring* and for *ensembling predictions*; probability space (mean
of sources) for *targets*. `build_targets.py` prints the target quantiles so a regression is
visible. Measured on 2026-08-28; student effect pending (P-00).

### 6c. Two "independent" label sources that decide identically

`hans_v4` and `sol56` make **identical decisions at the 0.5 cut** — agreement 99.45% over all
4,407 studies, error φ = 1.000 on gold for every label. Raw values differ, so this is
consistent with — not proof of — the v4 blend including the sol56 table. Either way the three
sources are ~1.5 effective votes: averaging them double-counts one reader and inflates the
`agreement` term of `confidence_weights`. Check pairwise error correlation before trusting a
blend of public label sets; `src/label_audit.py` does it.

### 6d. One flag meaning both "which preprocessing" and "read the cache"

`cfg.use_cache` selected the *preprocessing* (130 mm crop, per-series 1/99 normalisation,
laterality) **and** whether to `np.load` a cached array. Only the second is unavailable at
inference — no test study is ever in the cache. So when `rsna-knee-infer` mounted no cache, the
loader flipped `use_cache = False` and fed a **v03 model v02 pixels**: uncropped, unmirrored.
Nothing raised; the submission would just have scored lower for no visible reason.

**Do:** when a flag controls two things, ask whether both are actually unavailable in the branch
that turns it off. Here the fallback is correct for training and wrong for inference, so it is
now gated on `mode == "train"`. **Verify by equality, not by absence of errors**: after the fix
the infer kernel reproduced the training kernel's predictions byte-for-byte on the same studies.

### 6e. Augmentation that never augments

PyTorch's DataLoader seeds only **torch's** RNG per worker. `numpy` and `random` are inherited
from the parent by fork, and workers are recreated every epoch from a parent whose state has not
moved — so `np.random`-driven slice jitter and `random`-driven noise produce **byte-identical
augmentation in every epoch**. An ablation of such an augmentation measures nothing and reports
"no effect."

**Do:** pass a `worker_init_fn` that reseeds `numpy` and `random` from `torch.initial_seed()`.
Note this cannot be reproduced on Windows, which spawns workers instead of forking — a local
green run is not evidence the bug is absent on Kaggle.

### 7. Base-rate collapse

The known failure mode of this setup: the model outputs nearly the same score for every
study. **The loss looks fine.** AUC is 0.5.

**Do:** watch `pred_std` (mean per-label prediction standard deviation). Near zero is the
alarm. Healthy on our smoke run: **0.127**. It is a diagnostic, **never an optimisation
target** — you can trivially inflate spread without improving ranking.

---

## Tier 2 — breaks the run, wastes a session

### 8. A fold finishing with no `best.pt`

Our own bug, and the nastiest. When AUC is undefined for an epoch (a fold with no positives
for any label), `score` is `nan`, and **`nan > best` is always `False`** in Python. So no
best checkpoint was ever written — and at inference the fold was **silently skipped**,
quietly shrinking the ensemble.

**Fixed:** fall back to negative loss when AUC is undefined, and always write a checkpoint
if none exists yet. (Since v02 the question is moot: `best.pt` is the EMA weights after the
last completed epoch — fixed-epoch selection — and the per-epoch score is logged only.)

### 8b. Resume that never resumed

Our own bug, fixed in kernel v4. The resume logic looked for `{version}_fold{k}_last.pt` in
`WORK` (`/kaggle/working`), but a mounted previous-run output lives **read-only under
`/kaggle/input/...`**. The lookup missed, no error was raised, and every "resumed" run
silently restarted at epoch 0 — burning the whole session it was meant to save, while the
log looked like a normal training run.

**Fixed:** resume now copies the mounted `_last.pt` / `_best.pt` into `WORK` before looking
them up; EMA resume falls back to the raw weights if the checkpoint has no EMA state.
`MODE=auto` picks infer only if **every** configured fold has a mounted `best.pt`.

### 9. Caching an empty scan

Our own bug. The header scan wrote its cache unconditionally. A scan that found 0 series
(wrong image root) cached an empty CSV — and every **resumed** session then hit that cache
and trained on nothing.

**Fixed:** never write a cache for an empty result.

### 10. Hard-coding `/kaggle/input` paths

Kernel v1 died instantly with
`FileNotFoundError: /kaggle/input/rsna-knee-abnormality-detection/train.csv` **even though
the competition was correctly attached.**

All three of our inputs resolved to the *non-obvious* path:

| Input | Actual path on Kaggle |
|---|---|
| Competition | `/kaggle/input/competitions/rsna-knee-abnormality-detection` |
| Backbone | `/kaggle/input/models/metaresearch/dinov2/pytorch/small/1` |
| Label datasets | `/kaggle/input/datasets/<owner>/<slug>/…` |

The dataset layout was found **only** by the filename-glob fallback. Both layouts exist
depending on how the kernel was created, which is why every strong public notebook probes.

**Do:** keep `resolve_dir()` and the glob fallback. Do not "simplify" them. It prints the
real directory listing on failure.

### 11. The P100 accelerator

Kaggle's current PyTorch ships **no Pascal CUDA kernels**, so a P100 session dies at the
first convolution — after you have already paid the setup time.

**Do:** `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` (the CLI accepts it).
Re-check on every new or forked kernel; the default is not guaranteed.

> Terminology collision: a **Kaggle kernel** is a notebook; a **CUDA kernel** is a GPU
> function. "No Pascal kernels" means the second one.

### 12. Pushing a long run without a smoke run first

Five folds do not fit in one 9 h session. A crash in the *inference* cell after six hours
of training costs the entire session.

**Do:** `FORCE_SMOKE = True` on every new or edited notebook, always. Only then flip it.

### 12b. A submission that scores exactly 0.500

Submission #1 (smoke, kernel v2) **completed** and scored 0.500 — to three decimals. A
near-random model on ~1,000 hidden studies scores 0.47–0.53; an exact 0.500 is a **constant**
submission. Our own fallbacks produced it: an empty test manifest → `fillna(0.5)`, or every
slot directory missing → all-masked inputs → constant head output. Either way the hidden
test tree did not match the assumed `test_series/<study>/<series>/*.dcm` layout, and the
rerun log is invisible in a code competition, so nothing told us.

**Do:** probe the test image root by globbing for a real series UID; accept non-`.dcm`
filenames; write **no placeholder** (a crash → missing submission → visible scoring error, by
design — a 0.5-filled placeholder is exactly what produced the 0.500); then **refuse to
submit** (raise) when fewer than 90% of test studies are imaged *and* have ≥ 1 slot, or more
than half the labels are constant. A scoring error is diagnosable; a 0.500 looks like a bad
model. Also: submit smoke runs
*before* real ones — this cost a submission slot, not a session.

### 12c. Re-training inside the submitted notebook

A code competition **re-runs the notebook** on the hidden test. If the notebook trains and
infers in one pass, the rerun trains a *new* model (different data order, possibly killed by
the time limit) and scores that, not the one you validated. `MODE="infer"` loads
`{version}_fold*_best.pt` from a mounted kernel output and only predicts. Submit the
inference-mode version, never the training one.

### 12d. A real-mode default that smoke mode can never reveal

`Config.__post_init__` forces `folds=(0,)` in smoke. So a smoke run cannot show you that in
**real** mode every arm inherits `folds=(0,1,2,3,4)` — five arms would have been 25 folds, ≈18 h
against an 8.3 h guard. The smoke was green four times over.

**Do:** for anything smoke mode overrides (`folds`, `epochs`, `slices_per_slot`,
`runtime_limit_hours`, `num_workers`), read the real-mode value from the config rather than
inferring it from a smoke log — and print the resolved value in the run banner so the real run
says it out loud in its first seconds. Related: 12 (always smoke first) tells you the code runs;
it does not tell you what the code will do at production settings.

### 13. Horizontal-flip augmentation

`Medial OA` and `Lateral OA` are **different labels**. A horizontal flip swaps medial and
lateral, so it does not augment — it mislabels. Same reasoning makes *laterality
normalisation* (mirroring right knees so medial is always the same side) a genuine
improvement rather than a cosmetic one.

Vertical flip is also wrong: knee anatomy is not up/down symmetric.

---

## Tier 3 — tooling friction

### 14. Kaggle rate-limits file downloads, and reports success anyway

`xargs -P 10` over ~550 competition files got ~80% through then failed with **HTTP 429**.
`-P 4` also tripped it. Sequential with 20 s backoff *still* tripped it, so the quota window
is long (hours). Our sample DICOM tree stalled at **459/557**.

Two ways this bites:

- **The API returns exit code 0 on a 429.** A loop discarding stderr reports complete
  success while silently skipping hundreds of files. **Verify by file count, never exit
  status.**
- **The CLI flattens nested paths** into `-p`, so `study/series/` structure is lost unless
  you pass the nested directory yourself and rebuild the tree.

Also: `-f` takes one file at a time, there is **no folder download**, and a bare `-c` with
no `-f` would pull all **~570 GB**.

### 15. Windows `charmap` on `kaggle --csv`

Team names and reports contain non-ASCII; Windows' default codec throws
`'charmap' codec can't encode characters`. **Do:** `export PYTHONUTF8=1` before any Kaggle
CLI call. It is in every command block in our docs for this reason.

### 15b. `charmap` bites any Python that prints non-ASCII, not just the Kaggle CLI

Hit it again today printing research findings containing `≈`. On Windows **every** Python
process that prints or writes text without an explicit encoding needs `PYTHONUTF8=1`. Set it
once in the shell profile rather than per command.

### 16. Line-splitting `train.csv`

Reports are multi-line free text. `train.csv` is **58,556 physical lines for 4,407
records**. Any line-based parsing silently mangles the file. Use a real CSV parser, and
raise `csv.field_size_limit` if using the stdlib module.

### 17. Editing the `.ipynb` by hand

`kaggle/rsna-knee-train/rsna-knee-train.ipynb` is **generated** from
`src/kaggle_pipeline.py`. Hand edits are destroyed on the next `nbgen.py` run. Edit the
`.py`.

### 18. Running the scripts from inside `src/`

They use repo-root-relative default paths and import each other. Run from the repo root with
`PYTHONPATH=src`. (Cost us one confusing `OSError: models/dinov2_small is not a local
folder`.)
