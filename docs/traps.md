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

### 4b. Comparing epoch N of two runs with different epoch budgets

The cosine LR schedule is built over `cfg.epochs`, so a 4-epoch run and an 8-epoch run are at
**different learning rates at every single step**. Epoch 3 of a 4-epoch run is at the end of its
decay; epoch 3 of an 8-epoch run is still near peak LR. They are not the same model at the same
point in training, and their epoch-3 scores are not comparable.

Concretely: `v04d` (4 ep) scored 0.8528 at epoch 3 and `v05b` (8 ep, same config otherwise)
scored 0.8590 at epoch 3. That +0.006 looks like a result and is an artefact of the schedule.

**Do:** across different epoch budgets, compare **final checkpoint to final checkpoint** only —
that is what the fixed-epoch policy actually ships. Within one run, the per-epoch curve is fine
for reading *shape* (does it plateau, does it decay), which is where most of the signal was in
kernel v13 anyway.

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

### 6e. ~~Augmentation that never augments~~ — **WRONG, corrected 2026-08-29**

**The original claim (kept so the reasoning error stays visible):** PyTorch's DataLoader seeds
only *torch's* RNG per worker, so `numpy` and `random` are fork-inherited from a parent whose
state has not moved between epochs — meaning `np.random`-driven slice jitter produces identical
augmentation every epoch, and any ablation of it measures nothing.

**What the direct check found.** `check_worker_rng()` now runs both arms at Kaggle startup:

```
without worker_init_fn   identical across 3 epochs = False
with seed_worker         identical across 3 epochs = False
```

Without the fix the draws **already vary**. PyTorch's `_worker_loop` seeds `random` and `numpy`
per worker itself, so the pathology does not occur on this platform. `seed_worker` is kept as an
explicit, version-proof guarantee, but it fixed nothing.

**The trap that is actually real, and the reason this entry stays:** *a plausible mechanism plus
a green run is not evidence.* This was written up as a fixed bug on reasoning alone, and it was
flagged unverified only because Windows spawns workers and could not reproduce it. The check that
settled it costs seconds and could have been written **before** the fix. When a platform
difference blocks local verification, write the check into the kernel instead of shipping the fix
with a caveat.

### 6f. A NEW kernel slug does not mount inputs at the same depth as an old one

`kaggle/rsna-knee-folds` declared exactly the same `kernel_sources` as `rsna-knee-train`
(cache-a + cache-b), and Kaggle mounted them. The **loader could not see them**:

| kernel | cache mount path | depth | found by `max_depth=2`? |
|---|---|---|---|
| `rsna-knee-train` (older slug) | `/kaggle/input/rsna-knee-cache-a/manifest_shard0.csv` | 2 | ✅ |
| `rsna-knee-folds` (created 2026-08-29) | type-prefixed, like datasets at `/kaggle/input/datasets/<owner>/<name>/` | 4 | ❌ |

`load_cache_manifests` was the **only** glob in the file capped at `max_depth=2`; every other
resolver uses 3 or 4 with a fallback (`first_existing`, `resolve_image_root`,
`find_mounted_checkpoints`). Exactly the failure CLAUDE.md hard constraint 4 warns about, and
the shallow one was the one that mattered.

**What it cost:** `cfg.use_cache` flipped to `False`, so the dataset took the **v02 decode
branch** — no 130 mm crop, no laterality, no per-series normalisation — at 0.99 s/study instead
of 0.18. Five folds × 4 epochs at that rate is ~19.6 h, so the run could never finish; ~9 h of
GPU was spent training a recipe we had already superseded. Nothing raised.

**Do:**
1. Search for mounted inputs at **depth ≥ 4**, never 2. A slug created today does not lay out
   `/kaggle/input` the way one created last week does.
2. **Read the cache line in every smoke log before promoting a run to real.** The smoke printed
   `! use_cache=True but no cache is mounted` in plain text and it was read past — the log was
   scanned for the *new* thing being tested and not for the *known* failure mode.
3. The fallback is now **fatal** in train mode unless `ALLOW_DECODE_FALLBACK = True`. A silent
   downgrade to a superseded preprocessing path is worth a crash (same reasoning as 12b).

**The general lesson**, and the reason this sits next to 6d: `use_cache` conflating *which
preprocessing* with *whether a file is present* has now caused two separate incidents in one
day — once at inference (6d), once in training (here). A flag that selects a science decision
should not also be a file-availability check.

**OBSERVED 2026-08-29 17:45 — the layout, and a correction to the heading.** The startup print
added after this incident (`print_input_layout()`, in every kernel log from now on) shows, on the
**old** `rsna-knee-infer` slug:

```
/kaggle/input/competitions/rsna-knee-abnormality-detection/   train.csv, test.csv, ...
/kaggle/input/datasets/<owner>/<dataset>/                      the three label tables
/kaggle/input/models/metaresearch/dinov2/pytorch/small/1
/kaggle/input/notebooks/tiankljucanin/rsna-knee-train/         the mounted kernel output
```

Kernel outputs are type-prefixed under **`notebooks/<owner>/<slug>/`** (files at depth 3 below
`/kaggle/input`), while kernel v13 at ~10:00 the same day still found the cache at
`/kaggle/input/rsna-knee-cache-a/`. So it is **not** "new slug vs old slug" — Kaggle changed the
layout for everything during 2026-08-29. The fix stands (depth ≥ 4 everywhere, fatal fallback);
the mechanism was misattributed. Read the layout block at the top of every log; it costs nothing.

### 7. Base-rate collapse

The known failure mode of this setup: the model outputs nearly the same score for every
study. **The loss looks fine.** AUC is 0.5.

**Do:** watch `pred_std` (mean per-label prediction standard deviation). Near zero is the
alarm. Healthy on our smoke run: **0.127**. It is a diagnostic, **never an optimisation
target** — you can trivially inflate spread without improving ranking.

---

### 23. A cache version string that does not encode everything the bytes depend on

`c01_p224_s16_crop130_lat20` named the cache by px, slice count, crop and dead zone — **not** by the
per-plane slice band or the normalisation percentiles. A cache rebuilt with a different band at the
same px/slices would have carried the *same* version string, and the loader (`load_cache_manifests`,
which filters shards by that string) would have mixed it with, or silently substituted it for, the
old one. Found 2026-08-30 while designing the c02 cache; no run was hurt. **Do:** the version string is
`cache_version_of(scheme, px, slot_slices, band, crop, lat)` in *both* files, identical text; c02
strings read `c02_p336_b18-12-12-14-8-8_band2-98_crop130_lat20`. Any new knob that changes stored bytes
goes into that function first, or it does not exist. Related: 6d (one flag meaning two things).

### 24. The two hand-copied preprocessing paths had drifted

`src/kaggle_pipeline.py::cache_series` is a copy of the builder's function, guarded only by a "keep in
sync" comment and a two-study manual check. On 2026-08-30 it differed in two places: it read IOP /
PixelSpacing from the **spatially first** slice (the builder reads the **filename-first** header), and it
hard-coded `[1, 99]` where the builder used `cfg.pct_lo / pct_hi`. Harmless *today* only because IOP and
PixelSpacing are constant within a series and the percentiles happened to match — a test study would have
silently been preprocessed differently from the training cache the moment either assumption broke.
**Do:** run `python src/cache_selftest.py` before any push that touches `cache_pipeline.py`, the
`cache_*` functions in `kaggle_pipeline.py`, or a cache scheme. It builds the 3 sample studies through both
modules for both schemes and exits non-zero on the first differing byte. A green selftest is the *only*
evidence the two files agree; the comment is not.

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

### 12e. Kernel output is unreachable **while that slug has a run in flight**

**CORRECTED 2026-08-29, same day.** The first version of this entry claimed old versions'
outputs were lost once a new version was pushed. That was wrong, and it was wrong because the
test was confounded: `rsna-knee-train/11` returned nothing *because v13 was running on that slug*,
which makes the output endpoint return nothing for **any** version form, not because v11's output
had been discarded.

Retested on a settled slug (`rsna-knee-infer`, versions 1–4, none running):
`kaggle kernels output tiankljucanin/rsna-knee-infer/3` returns **version 3's** files while
version 4 is the latest. **Per-version output is retrievable and outputs are not discarded.**

The real rules:

- `kaggle kernels output <slug>` gives the **latest** version's output.
- `kaggle kernels output <slug>/<version>` gives **that** version's output.
- While any version of that slug is **running**, both return nothing. Wait for it to finish.
- `--file-pattern` is a **regex, not a glob** — `"*_oof.csv"` errors with "nothing to repeat".
  Use `"(oof|manifest)"`. The log always downloads regardless, which is why `"no_match"` fetches
  the log alone.

**CORRECTED AGAIN 2026-08-29 (17:30):** the `<slug>/<version>` form is **not reliable** either.
With `rsna-knee-train` idle (v13 latest), `kaggle kernels output tiankljucanin/rsna-knee-train/11`
downloaded **v13's** files — the log's arm banners read `v05a`/`v05b`, and the csvs were
byte-identical to v13's. The `rsna-knee-infer/3` "retest" above proved less than it seemed: infer
versions 1–4 produce near-identical outputs, so "returned version 3's files" was not actually
distinguishable from "returned the latest". Treat per-version retrieval as **unavailable**; the v11
per-epoch OOF csvs are gone for practical purposes. This makes the next paragraph the rule, not a
nicety.

**Still worth doing anyway:** pull the small result files with the log in one command, rather than
relying on being able to come back for them.

```bash
kaggle kernels output <slug> -p artifacts/kaggle_out/<ver> --file-pattern "(oof|manifest)"
```

**And the trap that survives the correction:** a *smoke* run writes checkpoints with the **same
filenames** as the real run (`v04d_fold0_best.pt` from 4 studies is indistinguishable by name from
the 0.877 model). Download smoke output into a directory named as such, or it will be mistaken for
the real thing later.

### 13. Horizontal-flip augmentation

`Medial OA` and `Lateral OA` are **different labels**. A horizontal flip swaps medial and
lateral, so it does not augment — it mislabels. Same reasoning makes *laterality
normalisation* (mirroring right knees so medial is always the same side) a genuine
improvement rather than a cosmetic one.

Vertical flip is also wrong: knee anatomy is not up/down symmetric.

---

### 22. A new member family needs its weights mounted in the *infer* kernel too

`rsna-knee-infer` v9 (2026-08-30) mounted the `v06c` checkpoint (via the `rsna-knee-train` output) but
not the ConvNeXt weights dataset it is built on. Seven DINOv2 members predicted, then the eighth raised
`convnext_tiny weights not found` and the whole run was lost. The checkpoint holds the fine-tuned
weights, but `KneeNet` still calls `from_pretrained(backbone_dir)` to build the architecture, so every
family in `INFER_MEMBERS` needs its `BACKBONES` source in the infer kernel's `dataset_sources` /
`model_sources`. Fixed: the infer member scan now calls `resolve_backbone_dir()` for each member's
family before any prediction, so a missing mount fails in seconds. Checklist when adding a member: its
checkpoint mount **and** its weights mount, both in `kaggle/rsna-knee-infer/kernel-metadata.json`.

### 25. A shard manifest that is written only when something new was built

The c01 builder wrote `manifest_shard{k}.csv` inside `if len(log_df):` — i.e. only when this run built at
least one study. A **resume-only** run (kernel re-launched after a crash, everything already on disk)
wrote no manifest, and the training loader would have indexed nothing from that shard while the arrays sat
there. Never bit us because no cache kernel had to resume. **Do (c02):** every blob has a CSV sidecar, and
the shard manifest is rebuilt from *all* sidecars on every run, so the manifest reflects the disk, not
the run. If a c01 rebuild is ever needed, check the manifest exists before mounting it.

### 26. Fixed-resolution backbones vs the smoke clamps; float windows through DataLoader shared memory

Two design-time catches from the 2026-08-30 review, recorded because both would have looked like a
*model* problem: (a) `Config.__post_init__` forces `img_size = 224` in smoke — a fixed-resolution timm
hybrid (`coatnet_rmlp_2_rw_384`) then dies on a shape mismatch (`196 vs 576` tokens) *in the smoke only*,
so the smoke would have "found a bug" the real run does not have; the clamp now skips `timm:` backbones.
(b) Shipping 60 float windows per study through the DataLoader (`(60, 3, 336, 336)` fp32 = 81 MB, ×
workers × prefetch in `/dev/shm`) is the classic Kaggle "DataLoader worker killed by bus error"; the
window Dataset ships the 8 MB uint8 study + indices and the model gathers / normalises / resizes on the
GPU. Related: 12d (a real-mode value smoke mode cannot reveal).

### 28. `MODE="oof_eval"` on Kaggle: the *second* member's DataLoader worker is OOM-killed (host RAM)

Kernel `rsna-knee-eval` v2 (2026-08-30): four c01 members with 3-view TTA, `num_workers=2`. Member 1
(`v05a`) scored in 7.2 min; ~2.5 min into member 2 (`v05b`) — same cache, same geometry, same loader —
`DataLoader worker (pid 73) is killed by signal: Killed` = the Linux OOM killer on the ~30 GB host RAM, not
CUDA. Per-member memory is modest (a study is 3 × 21.7 MB through `/dev/shm`, 2 workers × prefetch 2), so
something **accumulates across members**; the `del model, st; gc.collect()` at the end of each member does
not free it. Suspects, unverified: the page cache of ~4,400 FUSE `np.load`s charged to the cgroup, the
discarded-but-referenced train loader from `make_loaders`, or the c01 per-study `.npy` reads themselves.
**Do:** treat `oof_eval` on Kaggle as one member per kernel until the cause is found, or run it off-Kaggle
(the RunPod pod has 503 GB and the same `/kaggle/input` layout — 2026-08-30 it runs mean and focal there).
The infer kernel is on a different path (decode-once to disk, one geometry group) and has not shown this.
Related: 26 (shared-memory sizing), experiments.md 2026-08-30 P-12 entry.

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

### 19. A local smoke that "resumes" and trains nothing

`artifacts/local_run/` keeps `{version}_fold0_last.pt` from every earlier local smoke. A later
1-epoch smoke of the same version found it, printed `resumed fold 0 at epoch 1`, ran **zero**
epochs, inferred from the stale checkpoint and finished green — the checkpoint-policy code it was
run to exercise never executed. Caught only because the expected log line was missing.

**Do:** `train_fold` no longer resumes when `cfg.smoke` is set (2026-08-29). If you need the old
behaviour, delete `artifacts/local_run/*.pt` first. And when a smoke is meant to exercise a
specific line, grep the log for *that line*, not for "exit 0".

### 20. The Kaggle OAuth token expires after ~12 h, and the error blames your slug

`~/.kaggle/credentials.json` (written by `kaggle auth login`) stopped working 12 h after it was
issued (09:26 → 21:31 on 2026-08-29). Every command then fails, but not with an auth message:
`kaggle kernels status <slug>` prints **"Cannot access kernel … (Permission 'kernels.get' was
denied). The most likely cause is a wrong kernel slug"** — for a slug that had worked all day.
Only `kernels list` says "Authentication required". A status-polling loop reads this as a query
failure and keeps polling nothing.

**Do:** when a Kaggle call fails with a permission/slug message on a slug you have used before,
check `ls -la ~/.kaggle/credentials.json` first; if it is ~12 h old, re-run `kaggle auth login`
(interactive, browser). There is no CLI-only refresh. Related: `kaggle kernels logs <slug>` returns
**2 bytes while the run is in flight** — the same blackout as `kernels output` (12e), so the
mid-run throughput gate is browser-only.

**Addendum 2026-08-30:** once the token has expired, `kaggle auth login` answers *"You are already
logged-in to Kaggle as [user]. Please use the --force flag to override."* — it checks that a token file
exists, not that it is valid, while every API call in the same minute says *"Authentication required"*.
The fix is `kaggle auth login --force`. Cost this morning: ~10 min and one confused exchange.

### 21. `kaggle datasets create` on Windows: two silent-looking failures

Publishing the ConvNeXt weights (2026-08-29) failed twice before it worked:

1. **Title must be 6–50 characters** — the error is printed once, then the command keeps going and
   prints a `403 ... GetDatasetStatus`, which reads like an auth problem. It is not.
2. **Run it from inside the directory with `-p .`.** With `-p models/convnext_tiny` the CLI builds
   its upload temp path from the relative path and gets
   `...\Temp\.kaggle/uploads\models/convnext_tiny_config.json.json` — `No such file or directory`,
   again followed by 403s.

A private dataset then mounts at **`/kaggle/input/<slug>/`** (depth 1, no `datasets/<owner>/`
prefix — unlike the third-party label datasets), so weight resolvers must list both layouts.
Anything a final submission relies on must be **public** (competition rule) — flip it before then.

### 27. Regenerating `rsna-knee-cache-a/-b.ipynb` now yields a c02 kernel

`src/cache_pipeline.py` defaults to `SCHEME = "c02"`, `N_SHARDS = 4` since 2026-08-30. Running the
CLAUDE.md nbgen line for `rsna-knee-cache-a` without a sed produces a notebook that would build **c02
shard 0 into the c01 kernel's output** and repoint every consumer of `rsna-knee-cache-a`. Done once by
reflex on 2026-08-30 and reverted with `git checkout` before pushing. **Do:** the c01 kernels are
committed notebooks — do not regenerate them; if a c01 rebuild is ever needed, sed `SCHEME = "c02"` →
`"c01"` and `N_SHARDS = 4` → `2` first. c02 kernels are `kaggle/rsna-knee-cache2-{a,b,c,d}` with
`SHARD = 0..3` sed'd in. Related: locally, `find_mounted_checkpoints` searches `WORK` only when `MODE`
is explicitly `"infer"` / `"oof_eval"` — with `"auto"` it would flip every local smoke into infer mode.


### 29. Shipping the repo from the Windows checkout to a Linux GPU box (RunPod, 2026-08-30)

Five things that each cost one round trip on the first RunPod run:

- **CRLF.** `git archive HEAD` from this checkout ships `\r\n` (autocrlf); `scripts/runpod_bootstrap.sh`
  then dies at line 1 with `set: pipefail: invalid option name`. **Do:** after unpacking,
  `grep -rlI $'\r' . | xargs sed -i 's/\r$//'` (49 files), or archive with `-c core.autocrlf=false`.
- **Community-cloud pods had no public IP** (two 4090 hosts in the 100.65.x pool: only proxied HTTP ports,
  `ssh.direct: null`), and the `ssh.runpod.io` proxy needs an SSH key registered in the RunPod *account*,
  which the MCP cannot do. **Do:** `cloudType: SECURE` (a 4090 in EUR-IS-2 gave `22/tcp` at once, $0.74/h vs
  $0.34/h), pass `sshPublicKey`, expose `22/tcp` + `8888/http`, set `JUPYTER_PASSWORD` as a second way in.
- **`bc` is not in `runpod/pytorch:*-ubuntu2404`**, and `verify_count` in the bootstrap uses it (`set -e`
  would abort the setup on the first shard). `pip` needs `PIP_BREAK_SYSTEM_PACKAGES=1` (PEP 668).
- **`pgrep -f <name>` inside an SSH command that mentions `<name>` matches its own shell** — the idempotent
  launcher refused to start because the probe command contained the script's filename. Use `pgrep -f
  "runpod_[b]ootstrap.sh"`, and run remote scripts via `ssh host 'bash -s' <<'EOF'` so the command line
  carries nothing to match. Also `scp a b host:/path/x` with two sources makes `x` a *directory*.
- Kaggle pulls on the pod: blobs stream at ~30 MB/s, but per-study `.npy` (c01, ~4,400 files) at ~0.7 files/s
  — pull the two c01 shards in parallel (~50 min instead of ~100).
