# Experiments log

Every experiment, measurement, and dead end, newest first within each section. The point
is to never re-run a thing we already know the answer to, and to never re-litigate a
question that was already settled — or forget that a "settled" number was actually noise.

## How to use this file

**Append, never rewrite.** A wrong result that we later corrected is more useful than a
tidy file, because it records *why* we changed course. Mark corrections inline
(`**CORRECTED 2026-09-01:** …`) rather than deleting.

Every entry gets a **verdict**, and the verdict is the thing future-you reads first:

| Verdict | Meaning |
|---|---|
| ✅ **KEEP** | Measured better. In the pipeline now. |
| ❌ **DEAD END** | Measured worse or broke. Do not retry without a new reason. |
| 🔁 **INCONCLUSIVE** | Difference smaller than the noise floor. Not evidence either way. |
| ⏳ **PENDING** | Running or not yet measured. |
| 💡 **IDEA** | Not tried. Ranked in the backlog below. |

**The noise floor is the most important number in this file.** With 58 gold studies the
Hanley–McNeil SE of an AUC near 0.8 is ≈0.09 — a 95% interval of roughly ±0.17. Public-LB
deltas under ~0.005 are also noise (the top 10 public teams span 0.006 total). So:

> A gold-AUC difference under ~0.05, or a public-LB difference under ~0.005, is
> **INCONCLUSIVE, not a win.** Record it as such. Chasing those is exactly how the top
> public notebook admits it overfit the leaderboard.

Judge label changes on **coverage** (does the rule fire at all, per language) and on
**OOF over all 4,407 studies**, not on the 58 gold alone.

---

## Scoreboard

| Date | What | Gold AUC | Public LB | Verdict |
|---|---|---|---|---|
| 2026-08-28 | LLM report labels, rank blend of 3 sources (the *teacher*) | 0.8934 | — | ✅ KEEP |
| 2026-08-28 | DINOv2 ViT-S/14 2.5D pipeline, smoke only | n/a | not submitted | ⏳ PENDING |

**External reference points** (not ours — for calibrating ambition):

| Score | What |
|---|---|
| 0.952 | Public LB #1 (2026-08-28); top 10 span 0.946–0.952 |
| ~0.809 | Public DINOv2 baseline |
| ~0.664 | Rule-weak labels + calibrated soft targets, EfficientNet-B0 |
| ~0.613 | Rule-weak labels + EfficientNet-B0 |

⚠️ The public LB leaders are one heavily-forked community ensemble whose own author warns
it is "likely overfit to the public leaderboard." Expect a private shakeup. Do not treat
0.95 as a target to reach by blending; treat it as evidence that ~0.90+ is achievable with
a pipeline you can actually validate.

---

## Label sources

### 2026-08-28 — Rank blend of public LLM report labels ✅ KEEP

Scored each public source against the 58 gold studies (Mann-Whitney AUC, macro over 12):

| Source | Gold macro-AUC |
|---|---|
| `stevenleehans/llm_labels_v4_blend.csv` | **0.8927** |
| `stevenleehans/llm_labels_v2.csv` | 0.8873 |
| `stevenleehans/llm_labels_full.csv` | 0.8780 |
| `pilkwang/report_labels_v2.csv` | 0.8700 |
| `lixin73/labels_llm_gpt56sol.csv` | 0.8352 |
| **rank blend (hans_v4 + pilkwang + sol56)** | **0.8934** |

Per-label blend AUC with Hanley–McNeil SE — note how wide the intervals are:

| Label | AUC ± SE | | Label | AUC ± SE |
|---|---|---|---|---|
| ACL | 0.989 ± 0.015 | | PF OA | 0.903 ± 0.047 |
| MCL | 0.968 ± 0.042 | | Effusion | 0.878 ± 0.045 |
| Medial Meniscus | 0.955 ± 0.030 | | Lateral Meniscus | 0.879 ± 0.050 |
| Baker's | 0.947 ± 0.046 | | Contusion | 0.862 ± 0.058 |
| Medial OA | 0.923 ± 0.049 | | Fracture | 0.825 ± 0.065 |
| | | | Lateral OA | 0.804 ± 0.084 |
| | | | **Synovitis** | **0.788 ± 0.061** |

Kept the blend. Rank space, not probability space — different LLMs' probabilities are not
on a comparable scale but their ranks are, and rank order is all AUC reads.

**Weakest labels: Synovitis (0.788), Lateral OA (0.804), Fracture (0.825).** These are
where the teacher's ceiling is lowest, so they cap the student. Improving them is worth
more than improving ACL (already 0.989).

All sources are **CC0-1.0** — no licensing question.

### 2026-08-28 — Blending more sources 🔁 INCONCLUSIVE

| Combination | Gold macro-AUC |
|---|---|
| hans_v4 alone | 0.8927 |
| hans_v4 + pilkwang | 0.8930 |
| hans_v4 + pilkwang + sol56 | 0.8934 |
| all four | 0.8928 |
| hans_v4 + hans_v2 + pilkwang | 0.8914 |

Total spread 0.002 — far inside the noise floor. **The 3-source blend is not measurably
better than hans_v4 alone.** Using it anyway on the theory that averaging independent
readers should reduce variance on the 4,349 non-gold studies where we cannot measure, but
this is a prior, not a result. Do not cite 0.8934 > 0.8927 as evidence of anything.

### 💡 IDEA — improve the weak labels ourselves
Highest-value known lever, since the teacher caps the student. Options: prompt an
open-weights multilingual model for the three weakest findings; check per-language
coverage to find where vocabulary is thin. ⚠️ Do not send report text to a hosted LLM API
(see CLAUDE.md, "Rules"). Mount or run locally instead.

---

## Folds and validation

### 2026-08-28 — Group folds by report text ✅ KEEP

Measured: **4,273 distinct report texts over 4,407 studies. 49 texts are shared by more
than one study, covering 183 studies; the largest single group is 37 studies.** Studies
sharing a report share a target vector, so splitting a group across folds leaks the
text-derived answer into validation.

Greedy largest-first assignment, gold balanced before size:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| studies | 882 | 882 | 881 | 881 | 881 |
| gold | 11 | 12 | 12 | 12 | 11 |

Max folds touched by any single report group: **1** (verified assertion). Reproduced
byte-identically on Kaggle, so folds are machine-independent.

### 💡 IDEA — also group by scanner/site
Community reports random K-fold inflates AUC ~0.05 via scanner memorisation, with one team
measuring a **+0.136** grouped-vs-random gap. Report-text grouping does **not** address
this — site identity leaks through the pixels. Derive a site proxy from DICOM headers
(`Manufacturer`, `ManufacturerModelName`, `InstitutionName`, `MagneticFieldStrength`) and
group on it too. **Not yet done, and it may be the single biggest correctness gap in our
validation.**

---

## Series selection and preprocessing

### 2026-08-28 — Sort slices spatially, never by filename ✅ KEEP

Measured on 12 sample series: Spearman ρ between filename order and the
`ImagePositionPatient` projection is **mean −0.012, range −0.31…+0.25**, with `|ρ|>0.99`
in **0 of 12** series. Filename order is anatomically meaningless and fails silently,
destroying the slice adjacency that makes a 2.5D triplet mean anything. Sort by projecting
`ImagePositionPatient` onto the normal from `ImageOrientationPatient`; fall back to
`InstanceNumber`.

### 2026-08-28 — Recover acquisition flags from DICOM headers ✅ KEEP

The shipped `Fluid_Sensitive`/`Fat_Suppression` columns are **degenerate**: across all
24,371 training series only `(1,1)` (14,010) and `(0,0)` (10,361) occur, never a mixed
pair. Two physically independent properties (contrast weighting vs. a fat-sat preparation)
arrive as one bit. Header recovery produced **3 of 4** combinations on just 12 sample
series, including `(fluid=True, fat_sat=False)` — a combination the CSV cannot express.

### 2026-08-28 — Trust `Anatomical_Plane` as shipped ✅ KEEP

100% agreement with the plane derived from `ImageOrientationPatient` on the sample series.
Only the two flags are untrustworthy, not the plane. Don't spend effort recomputing it.

### 2026-08-28 — Two-tier slot matching ✅ KEEP

Strict matching (right plane **and** fluid **and** fat-sat) left 2 of 12 sample series
unassigned and one study at 2/6 slots, because real studies routinely carry an axial fluid
series with no fat suppression. Adding a relaxed tier (right plane + fluid, ignoring
fat-sat) lifted coverage to 4/6 and 5/6.

On **real training data** (24 studies scanned on Kaggle): mean **4.96 of 6** slots.

| Slot | Fill rate |
|---|---|
| `SAG_FLUID_FS` | 100% |
| `AX_FLUID_FS` | 100% |
| `COR_FLUID_FS` | 95.8% |
| `SAG_FLUID_NOFS` | 87.5% |
| `COR_T1` | 62.5% |
| `SAG_T1` | 50% |

The two T1 slots are the weak ones — **first candidates to drop if compute needs cutting.**

### 2026-08-28 — Per-series intensity normalisation is mandatory ✅ KEEP

Max intensity spans **690 … 8,736** across sample series (12.7×). A global window would not
transfer. Clip each triplet jointly at its 1st/99th percentile so its three channels stay
mutually comparable. All sample series were `MONOCHROME2` with trivial rescale, but the
hidden test spans 16–19 sites, so `MONOCHROME1` inversion and `RescaleSlope/Intercept` must
still be handled.

---

## Model and training

### 2026-08-28 — DINOv2 ViT-S/14 + attention pooling, smoke verified ⏳ PENDING

Architecture: shared DINOv2 encoder over ≤6 slots → attention pool over slices → concat 6
slot vectors + 6-bit presence mask → linear → 12 logits. Two LRs (backbone 5e-5, head
1e-3). Confidence-weighted soft-target BCE, **no `pos_weight`**.

Ran green end to end on a Kaggle T4 in smoke mode (kernel v2, 9.5 min). **Not trained for
real, not submitted — no score exists yet.** The smoke run's `auc_soft 0.3182` is
meaningless (4 val studies, 1 epoch, near-random head). `pred_std 0.127` is healthy.

### 💡 IDEA — ensemble DINOv2 with RadImageNet ResNet-50
**Not implemented.** `models/radimagenet_r50/ResNet50.pt` is downloaded but nothing loads
it; the current "rank mean" ensembles 5 folds of the *same* architecture, not two
backbones. RadImageNet is supervised on ~1.35M radiologic images, so it has a different
inductive bias and different failure modes — the standard reason a blend helps. Every
strong public notebook rank-blends DINOv2/v3 with RadImageNet.
⚠️ **RadImageNet is CC-BY-NC-SA-4.0** (non-commercial, share-alike) — check the winner
licence clause before making it load-bearing in a final submission. DINOv2 is Apache-2.0.

### 💡 IDEA — architecture backlog, roughly by expected value
1. **Preprocessing cache** (see below) — unblocks everything else. Do first.
2. **Site-grouped folds** — correctness, not score. Do second.
3. Rank-blend RadImageNet ResNet-50 with DINOv2.
4. DINOv3 ViT-S/16 as a third ensemble member.
5. Laterality normalisation — mirror right knees. `Medial OA` and `Lateral OA` are
   *different labels*, so this is not cosmetic. Currently **not implemented**; the DICOM
   laterality tag is reportedly unreliable in this corpus.
6. Higher resolution for the small focal findings (meniscal tears) — costly.
7. Per-label specialist heads for the three weakest labels.

---

## Infrastructure

### 2026-08-28 — ⚠️ Throughput is the open risk, and it blocks the real run

Fold 0 took **36 s for 8 study-passes at 2 slices/slot with `num_workers=0`** (~4.5 s per
pass). The real config triples the slices; naive extrapolation to 4,407 passes × 4 epochs
lands near **6–8 h per fold**, i.e. five folds ≈ five sessions and one fold barely fits in
one. **This is an extrapolation from 8 studies, not a measurement.**

Diagnosis: the bottleneck is DICOM decode, not the ViT — at 6 slices/slot × 3 channels ×
~5 slots that is ~90 file reads per study, **repeated every epoch**.

⏳ **Next action: benchmark properly, then build a preprocessing cache kernel** (CPU,
decode+resize to uint8 once; community reports ~15.9 GB and ~1 h for all 4,407 studies)
which training mounts. Turns 90 DICOM reads per study per epoch into one array read. **Do
not launch a 5-fold run before this.**

Checkpoint sizes: `best.pt` 88 MB (weights), `last.pt` 266 MB (with optimizer state).

### 2026-08-28 — Kaggle mount paths must be probed, not assumed ❌ DEAD END (hard-coding)

Kernel v1 died instantly with
`FileNotFoundError: /kaggle/input/rsna-knee-abnormality-detection/train.csv` **even though
the competition was correctly attached.** All three inputs resolved to non-obvious paths:

| Input | Actual path |
|---|---|
| Competition | `/kaggle/input/competitions/rsna-knee-abnormality-detection` |
| Backbone | `/kaggle/input/models/metaresearch/dinov2/pytorch/small/1` |
| Label datasets | `/kaggle/input/datasets/<owner>/<slug>/…` |

The dataset layout was found **only** by the filename-glob fallback. Keep `resolve_dir()`
and the glob; do not "simplify" them.

### 2026-08-28 — Kaggle rate-limits bulk file downloads ❌ DEAD END (parallel pulls)

`xargs -P 10` over ~550 competition files got ~80% through then failed with **HTTP 429**;
`-P 4` also tripped it; sequential with 20 s backoff still hit it, so the quota window is
long. **The API returns exit code 0 on a 429**, so a loop discarding stderr reports success
while silently skipping files — verify by file count, never exit status. Also, the CLI
flattens nested paths into `-p`, so the `study/series/` tree must be rebuilt manually.
Sample DICOMs stalled at **459/557** (3 series of study 3 missing); not blocking.

### 2026-08-28 — Bugs found by running the pipeline (all fixed) ✅ KEEP the fixes

1. **A fold could finish with no `best.pt`.** When AUC is undefined (no positives in a
   fold), `nan > best` is always `False`, so no best checkpoint was ever written and the
   fold would be **silently dropped from the inference ensemble**. Now falls back to
   negative loss and guarantees a checkpoint exists.
2. **Teacher AUC reported as 1.0000** — it was scored *after* gold was copied into the
   targets, i.e. grading gold against itself. Now scored pre-override → 0.8934.
3. **Empty header scans were cached**, so a resumed session would hit the empty cache and
   train on nothing. Now an empty scan is never written.

### 💡 IDEA — infrastructure backlog
- `FORCE_SMOKE = True` on every edited notebook before a long run. A crash in the inference
  cell after six hours costs a whole session.
- Never select the **P100**: Kaggle's PyTorch ships no Pascal CUDA kernels and the session
  dies at the first convolution. `"machine_shape": "NvidiaTeslaT4"`.
- Consider splitting train and inference into separate kernels once the cache exists.

---

## Submissions

None yet. Log every submission here with the exact kernel version, config diff, OOF score,
and public LB score, so a public/private divergence can be traced to a specific change.

| # | Date | Kernel ver | Config change | OOF | Public LB | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | no submissions yet |
