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

Untried ideas do **not** belong here — they go in [brainstorm.md](brainstorm.md), which
holds the ranked backlog and the open questions. This file is only for things that were
actually run and measured.

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
| 2026-08-28 | LLM report labels, rank blend of 3 sources (the *teacher*, v01) | 0.8934 | — | ❌ superseded (target-scale flaw, see P-00) |
| 2026-08-28 | LLM report labels, **mean of probabilities** (the teacher, v02) | 0.8948 | — | ✅ KEEP — for the *target scale*, not the AUC: Δ 0.0014 vs the rank blend is inside the noise floor |
| 2026-08-28 | v01 smoke model (1 fold, 1 epoch, 4 studies) — submission #1 | n/a | **0.500** | ❌ constant output at rerun (see Submissions) |
| 2026-08-28 | **v02 fold 0, real run (kernel v6)**: DINOv2-S 224, 6 slices/slot, 4 epochs, prob targets, LR 2e-5+LLRD, EMA | 0.847 (n=11, CI 0.72–0.94) · OOF-vs-teacher **0.821** (882 studies) | **0.841** | ✅ first real model; per-label table below |
| 2026-08-28 | **v03 fold 0 from the cache (kernel v8)**: v6 recipe + 130 mm crop + laterality + per-series norm | 0.906 (n=11, CI 0.81–0.97) · OOF-vs-teacher **0.843** (882 studies) | **0.871** | ✅ KEEP — LB +0.030 vs v02, 6× the 0.005 floor |

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

> Untried ideas for label improvement live in [brainstorm.md](brainstorm.md) (#4).
> Weakest teacher labels: Synovitis 0.788, Lateral OA 0.804, Fracture 0.825.

### 2026-08-28 — Rank-percentile targets put confident negatives at ~0.3 ❌ DEAD END (fixed, P-00)

Found by the research critic and verified on `artifacts/targets.csv`. `rank(pct=True)` gives
tied values their *average* rank, so on a label where most sources say exactly 0 every
confident negative landed at 0.28–0.39 while the 58 gold rows sit at a hard 0/1 (8× weight).
BCE fits the *value*, not the order — the network was being taught "definitely absent" = 0.31.

| | old rank blend | mean of probabilities (now) |
|---|---|---|
| studies with target < 0.1 | **0%** — no study on any label, before the gold override (corrected: the earlier "1%" was counted *after* gold 0/1 rows were copied in) | 2–72% per label (Synovitis 2%, Baker's 26%, MCL 72%) (corrected: earlier "30–70%") |
| MCL p25 / p50 | 0.312 / 0.43 | 0.03 / 0.04 |
| teacher gold macro-AUC | 0.8934 | **0.8948** (Δ 0.0014 — inside the noise floor; the KEEP is for target scale, not AUC) |

Rank space stays correct for *scoring* and for *ensembling predictions*; it was wrong for
*building a BCE target*. Student effect on OOF is ⏳ PENDING (v02 fold 0). Full before/after
quantiles are printed by `build_targets.py` into `artifacts/label_report.txt`.

### 2026-08-28 — Label audit (`src/label_audit.py`) — what the teacher is made of

Aggregates in `artifacts/label_audit.md` (no UIDs). Headline numbers:

- **Languages (langdetect):** en 39%, es 15%, tr 12%, hr 9%, el 7%, de 6%, bg 5%, nl 3%, fr 2%.
  Gold: 28/58 English; French has no gold.
- **hans_v4 and sol56 make identical decisions at the 0.5 cut** — agreement 99.45% over all
  4,407 studies; error-φ = 1.000 on gold for every label. Raw values differ, so this is
  consistent with — not proof of — the v4 blend including the sol56 table (corrected: an
  earlier version stated "same source / already contains" as fact). Either way the three
  sources are ~1.5 effective votes (mean pairwise error φ 0.88; literature panels ~0.39).
  Other pairs: hans_v4~pilkwang 95.4%, pilkwang~sol56 95.2%. Consequence: the `agreement`
  term in `confidence_weights` is inflated by a near-duplicate, and "blending more sources"
  (already 🔁 INCONCLUSIVE above) is now explained.
- **Silence** (pilkwang verdict `UNK`): Synovitis **84%**, Fracture 56%, Baker's 46%,
  Lateral OA 33%. pilkwang is the only source that flags silence. On UNK rows hans_v4
  averages ~0.25 (many distinct values), the blend averages ~0.18, and the confidence weight
  averages **0.69 vs 0.80–0.89 on addressed rows** — so silence is barely down-weighted and
  looks like a confident negative (the docstring claim that silent reports "pull far less"
  does not hold). (corrected: earlier text claimed each source encodes silence as a fixed
  value — hans 0.25 / pilkwang ~0.28 / sol56 hard 0 — which is not what the data show.)
- **Prevalence gap gold vs blend≥0.5:** Synovitis 47% vs 13%, Lateral Meniscus 40% vs 16%,
  Fracture 31% vs 7%, ACL 41% vs 21%. Either the 58 gold are enriched for positives or the
  reports under-call; both mean silent positives sit in the negatives.
- **Coverage by language:** Spanish is worst (Fracture UNK 80%, Lateral Meniscus 40%,
  Effusion 29%); Turkish Lateral OA UNK 68%. Source agreement (Spearman hans_v4~pilkwang,
  the near-duplicate pair excluded) is lowest for Bulgarian: bg 0.67 vs en 0.83
  (corrected: earlier 0.68 / 0.80 included the hans_v4~sol56 pair).
- **Co-occurrence φ (gold / weak; n=58, SE of gold φ ≈ 0.13):** Effusion~Synovitis
  0.40 / 0.28, Medial OA~Medial Meniscus 0.42 / 0.36, Contusion~Fracture 0.33 / 0.28,
  Medial OA~Lateral OA 0.32 / 0.49 — relevant to any per-label head (P-09). (corrected:
  earlier text swapped gold/weak for Medial OA~Medial Meniscus, quoting 0.36 as gold.)

### 2026-08-28 — Synovitis ← Effusion back-fill 🔁 INCONCLUSIVE (NOT adopted)

The public `stevenleehans` card reports +0.11 gold AUC from back-filling unaddressed
Synovitis with the Effusion label (0.678 → 0.790). On **our** probability blend the same
operation gives **0.788 → 0.729**, paired-bootstrap Δ = −0.059, 95% CI [−0.164, +0.042].
The card's gain came from a 0.678 baseline; ours is already at their post-fix level. 41 of
the 58 gold have unaddressed Synovitis and 14 of those are gold-positive, so the silence
problem is real; whether Effusion helps is not resolvable on 58 gold — the CI spans zero in
both directions (corrected: an earlier version said "leaning negative" / "Effusion is not the
answer", which over-reads the data). Weights on the back-filled rows would average 0.69
without recomputation and 0.81 with. Kept as P-07 (card only).

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

> ⚠️ **Site/scanner grouping is still missing** and is the largest correctness gap in our
> validation — report-text grouping does not address it, because site identity leaks through
> the pixels. Tracked as [brainstorm.md](brainstorm.md) #2.

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

> **Not implemented yet:** the RadImageNet ensemble, laterality normalisation, DINOv3, and
> higher resolution. All ranked in [brainstorm.md](brainstorm.md). The current rank mean
> ensembles 5 folds of the *same* DINOv2 architecture, not two backbones.

### 2026-08-28 — v02 fold 0, first real training run (kernel `rsna-knee-train` v6) ✅ KEEP as baseline

Config: DINOv2 ViT-S/14, 224 px, 6 slices/slot, gap 2, concat head, prob-mean targets,
backbone LR 2e-5 with LLRD 0.75, wd 0.02, EMA 0.998, 4 epochs, batch 1 × accum 4,
`num_workers=2`, fold 0 (3,525 train / 882 val, 11 gold). Checkpoint = epoch-3 EMA.

**Runtime:** header scan of 24,371 series 321 s; **0.99 s/study training at 6 slices/slot →
58 min/epoch + 14.5 min validation**; 5.0 h total for 4 epochs. So one fold ≈ one session
*without* the cache; the 6–8 h/fold extrapolation was pessimistic but the conclusion stands.

**Learning curve (val fold, 882 studies; gold n = 11):**

| epoch | loss | OOF macro-AUC vs teacher | gold macro-AUC [95% CI] | pred_std |
|---|---|---|---|---|
| 0 | 0.575 | 0.772 | 0.820 [0.67, 0.93] | 0.12 |
| 1 | 0.496 | 0.818 | 0.867 [0.76, 0.93] | 0.18 |
| 2 | 0.444 | **0.826** | 0.861 [0.74, 0.95] | 0.21 |
| 3 | 0.402 | 0.821 | 0.847 [0.72, 0.94] | 0.23 |

OOF vs teacher plateaus at epoch 2–3; the gold curve moves inside its own CI and is not
evidence of anything (as expected at n = 11). Recomputed on the 871 non-gold val rows:
macro 0.820; on *confidently-labelled* rows only (|target − 0.5| > 0.3) 0.848.

**Per-label OOF vs teacher at epoch 3** (weak rows; confident-only in brackets; gold n=11 for
direction only):

| label | OOF | conf-only | gold | | label | OOF | conf-only | gold |
|---|---|---|---|---|---|---|---|---|
| Fracture | 0.90 | 0.91 | 0.93 | | Medial OA | 0.84 | 0.86 | 0.93 |
| Baker's | 0.88 | 0.88 | 1.00 | | Contusion | 0.83 | 0.89 | 1.00 |
| **Synovitis** | 0.88 | **0.94** (54% confident) | **0.50** | | Medial Meniscus | 0.82 | 0.87 | 0.57 |
| Effusion | 0.85 | 0.87 | 1.00 | | PF OA | 0.79 | 0.80 | 0.73 |
| ACL | 0.81 | 0.85 | 0.93 | | Lateral OA | 0.78 | 0.80 | 0.92 |
| | | | | | **MCL** | **0.75** | 0.78 | 1.00 |
| | | | | | **Lateral Meniscus** | **0.72** | 0.74 | 0.67 |

Readings (hypotheses to test, not conclusions):
- **Weakest against the teacher: Lateral Meniscus 0.72, MCL 0.75, Lateral OA 0.78, PF OA
  0.79** — three of the four are side-specific or small focal findings → P-05 (laterality),
  P-08 (slices), P-11 (resolution) are the cards aimed at them.
- **Synovitis is the teacher-ceiling case:** the student reproduces the teacher almost
  perfectly where the teacher is confident (0.94), while gold sits at chance — the student has
  learned that "not mentioned" means negative. Only better targets can move it (P-07/P-16).
- No collapse: pred_std climbs 0.12 → 0.23 and every label is scored.
- The P-00 fix's effect on the student is **not** isolated here (no v01 real run exists); v6 is
  the baseline every later card is compared against.

### 2026-08-28 — v03 fold 0 trained from the cache (kernel `rsna-knee-train` v8) ✅ KEEP

Same recipe as v6, reading the mounted `c01_p224_s16_crop130_lat20` cache instead of decoding
DICOMs. Cache mounted clean: 4,407 studies indexed, mean slots 4.78, side resolved 97.9%, no
header scan needed.

| | v6 (v02, decode) | v8 (v03, cache) |
|---|---|---|
| s/study, training | 0.99 | **0.18** |
| per epoch | 58 min + 14.5 min val | **10.6 min + 2.9 min val** |
| header scan | 321 s | not needed |
| total, 4 epochs | 5.0 h | **0.92 h** |
| OOF-vs-teacher, ep3 | 0.821 | **0.8426** |
| gold, ep3 (n=11) | 0.847 [0.72, 0.94] | 0.9062 [0.81, 0.97] |
| pred_std | 0.23 | 0.238 |

**Speed-up is 5.4× end to end, not the ~60× the decode arithmetic suggested.** Decode was the
bottleneck; now it is not. At 0.18 s/study for ~29 ViT forwards (4.78 slots × 6 slices), the
T4 is the floor. Two consequences: further I/O work is worth nothing, and P-08's extra slices
now cost linearly in GPU time instead of riding free on saved decode.

**Learning curve (fold 0 val, 882 studies; gold n = 11):**

| epoch | loss | OOF vs teacher | gold [95% CI] | pred_std |
|---|---|---|---|---|
| 0 | 0.570 | 0.7848 | 0.837 [0.72, 0.93] | 0.128 |
| 1 | 0.487 | 0.8357 | 0.896 [0.80, 0.97] | 0.186 |
| 2 | 0.438 | **0.8457** | 0.898 [0.80, 0.97] | 0.220 |
| 3 | 0.398 | 0.8426 | 0.906 [0.81, 0.97] | 0.238 |

**Train loss falls monotonically while OOF turns over at epoch 2.** That is the overfitting
signature, and it is the argument against P-04's 8 epochs and for augmentation (the pipeline's
only augmentation was Gaussian noise at σ=0.01) — see the five-arm batch below.

**Per-label OOF vs teacher at epoch 3, against v6:**

| label | v6 | v8 | Δ | | label | v6 | v8 | Δ |
|---|---|---|---|---|---|---|---|---|
| **Lateral Meniscus** | 0.72 | **0.792** | **+0.07** | | Contusion | 0.83 | 0.855 | +0.03 |
| Medial Meniscus | 0.82 | 0.873 | +0.05 | | Medial OA | 0.84 | 0.859 | +0.02 |
| ACL | 0.81 | 0.840 | +0.03 | | PF OA | 0.79 | 0.803 | +0.01 |
| MCL | 0.75 | 0.781 | +0.03 | | Baker's | 0.88 | 0.890 | +0.01 |
| Lateral OA | 0.78 | 0.810 | +0.03 | | Fracture | 0.90 | 0.900 | 0.00 |
| | | | | | Synovitis | 0.88 | 0.873 | −0.01 |
| | | | | | Effusion | 0.85 | 0.835 | −0.02 |

**Every gain is a side-specific or small-focal finding; every global/fluid finding is flat or
marginally down.** The four weakest labels in v6 all moved up. That sorting by label semantics
is what makes the delta look mechanistic rather than like seed noise.

⚠️ **P-01's "OOF within ±0.01 = a faithful speed-up" rule was never applicable.** It assumed
v03 replayed v02's inputs. It did not: v6's config dump has no `crop_mm` and no
`lat_dead_zone_mm` keys at all, so v03 changed the pixels three ways at once — 130 mm physical
crop, laterality mirroring, per-series 1/99 normalisation. The +0.022 is real (confirmed on the
LB below) but **jointly attributed** until the `lat_undo` arm reports. Verdict is ✅ KEEP for
the combination, not for any one of the three.

Caveat: n = 1. The four epochs share one run and one seed, so they are not four samples. The
run-to-run floor was still unmeasured when this was written — that is what the batch below does.

### 2026-08-29 — Five-arm fold-0 batch (kernel `rsna-knee-train` v11) ⏳ PENDING

> **RESOLVED 2026-08-29** — all five arms completed in 4.35 h, none failed. Results and
> verdicts in the five entries below (noise floor, jitter, laterality, attention head,
> and the traps-6e correction).

Five fold-0 arms back to back in one session, ~0.9 h each (~4.6 h against the 8.3 h guard),
each writing `v04*_fold0_*`. An arm that raises is logged and skipped rather than killing the
session.

| arm | change vs `v04base` | card | what it answers |
|---|---|---|---|
| `v04base` | — (seed 42, current code) | — | the reference every other arm is compared against |
| `v04a` | `seed = 43` | P-02 | `\|v04a − v04base\|` **is** the measured OOF noise floor |
| `v04c` | `head_type = "attn"` | P-09 | per-label masked slot attention vs concat→linear |
| `v04b` | `lat_undo = True` | P-05 | splits laterality out of the v03 preprocessing gain |
| `v04d` | `cache_jitter = True` | P-08 | ±1-slice jitter as augmentation |

`v04base` exists because arms b/c/d must differ from their baseline in exactly one thing, and
kernel v8 stopped being that baseline the moment `seed_worker()` changed how augmentation is
randomised. So **v8 vs v04base measures the code revision; v04a vs v04base measures the seed.**

`lat_undo` de-canonicalises right knees at load time by re-applying the cache's own transforms
(both are involutions), which needs no cache rebuild — verified as a clean involution with no
NumPy aliasing corruption. It does not reconstruct the original bytes (the per-series
`col_to_left` sign is not in the manifest); it reproduces *chirality that varies with knee
side*, which is the thing P-05 removed. On Kaggle it fired on **2,288 of 4,407 studies
(51.9%)**. This is a cleaner test than v8-vs-v6, where the crop varied at the same time.

Read `v04c` on macro **plus** the three correlated pairs (Effusion~Synovitis, Medial OA~Medial
Meniscus, Contusion~Fracture) — research.md's stated risk is that a per-label head loses the
shared-vector benefit exactly there.

### 2026-08-29 — Run-to-run noise floor MEASURED (P-02 step 1) ✅ KEEP

Kernel v11, `v04base` (seed 42) vs `v04a` (seed 43), identical code and config, fold 0, 4 epochs
from the cache. This is the number every A/B in this file has been judged against on faith.

| epoch | `v04base` | `v04a` | \|Δ\| |
|---|---|---|---|
| 0 | 0.7851 | 0.7904 | 0.0053 |
| 1 | 0.8353 | 0.8311 | 0.0042 |
| 2 | 0.8449 | 0.8389 | 0.0060 |
| 3 | 0.8415 | 0.8339 | **0.0076** |

**Macro OOF floor = 0.008** (max over four epochs; ~0.006 typical). The asserted 0.01 was close
and slightly conservative — every verdict recorded against it stands.

**The per-label floor is much worse than assumed.** Same two runs, epoch 3, |Δ| per label:
Fracture **0.028**, Contusion 0.025, ACL 0.016, Synovitis 0.013, MCL 0.011, LatOA/PFOA 0.008,
Effusion 0.007, Baker's 0.007, MedOA/MedMen 0.005, LatMen 0.002. Mean 0.011, **max 0.028**.
proposals.md assumed 0.015–0.02 per label; **use ~0.03**. Any single-label story below that —
including several told in this file before today — is not evidence on its own. What *is* evidence
is a consistent sign across many labels, which no seed change produces.

Bonus check: `v04base` scored 0.8415 against kernel v8's 0.8426 on the same config. The
`seed_worker` code revision between them moved nothing (0.001), so v8's numbers stay comparable.

### 2026-08-29 — Slice jitter as augmentation (P-08 sub-arm, `v04d`) ✅ KEEP

`cache_jitter=True`: the K=6 slice centres move ±1 cached slice per epoch. Everything else is
`v04base`. **OOF 0.8528 vs 0.8415 = +0.0113 against a 0.008 floor** (1.4×).

The macro alone would be thin. Two things make it convincing:

**1. The overfitting turn disappears.**

| epoch | `v04base` | `v04d` |
|---|---|---|
| 0 | 0.7851 | 0.7847 |
| 1 | 0.8353 | 0.8338 |
| 2 | **0.8449** ← peak | 0.8492 |
| 3 | 0.8415 ← declining | **0.8528** ← still rising |

Train loss at epoch 3 is *higher* with jitter (0.4265 vs 0.3980) while OOF is better — less
memorisation, the textbook regularisation signature. The epoch-2 turnover that argued against
P-04's longer schedule is gone.

**2. Eleven of twelve labels improve and none regress.** MCL +0.032, ACL +0.020, Baker's +0.015,
Baker's/MedMen/LatMen/MedOA +0.010, Contusion +0.009, PFOA/Effusion +0.008, Fracture +0.007,
LatOA +0.006, Synovitis 0.000. Individually all but MCL sit inside the 0.028 per-label floor;
collectively, a seed change scatters signs and this does not.

**It had not peaked**, so 4 epochs now under-trains this config — the reason the 8-epoch arms
were launched.

### 2026-08-29 — Laterality normalisation confirmed (P-05, `v04b`) ✅ KEEP

`lat_undo=True` re-applies the cache's own transforms to the 2,288/4,407 right knees (51.9%),
restoring chirality that varies with knee side — the pre-P-05 condition — with the 130 mm crop
held constant. **OOF 0.8268 vs 0.8415 = −0.0147, ~1.9× the 0.008 floor.**

Where it costs is the mechanism:

| large drops (side-specific / focal) | | unaffected (global / fluid) |
|---|---|---|
| Baker's −0.044 (posteromedial) | | Lateral OA −0.001 |
| MCL −0.033 (medial only) | | Synovitis +0.005 |
| Medial Meniscus −0.032 | | Effusion +0.006 |
| ACL −0.018, Fracture −0.018, Lateral Meniscus −0.015 | | Contusion −0.006 |

Three of those clear even the 0.028 per-label floor on their own. Gold falls 0.8992 → 0.8259,
reported not gated (n=11).

**This resolves the v03 confound.** The v03 preprocessing change was worth +0.022 OOF and +0.030
LB; laterality accounts for **≈ +0.015** of the OOF, leaving ≈ +0.007 for the 130 mm crop and
per-series normalisation together — inside the floor, therefore **unproven**. The per-label
pattern in the v8 entry above (side-specific labels gained, fluid labels flat) predicted exactly
this.

### 2026-08-29 — Per-label attention head at 4 epochs (P-09, `v04c`) 🔁 INCONCLUSIVE — unconverged

**OOF 0.8367 vs 0.8415 = −0.0048**, inside the 0.008 floor, so formally inconclusive. But the
experiment could not have answered the question, and that is the finding:

| epoch | `v04base` (concat) | `v04c` (attn) |
|---|---|---|
| 0 | 0.7851 | 0.7528 |
| 1 | 0.8353 | 0.8114 |
| 2 | 0.8449 | 0.8323 |
| 3 | 0.8415 ← declining | 0.8367 ← **still rising** |
| train loss @ ep3 | 0.3980 | **0.4471** |

It starts further back, climbs the whole way, and has a far higher train loss at the end: a model
that has not converged. Expected after cutting the head from 27,732 to 9,300 parameters and
changing its initialisation — the 4-epoch budget was tuned for the concat head.

**The stated risk did not materialise.** On the three correlated pairs
(research.md): Effusion +0.012 / Synovitis −0.009, Medial OA +0.003 / Medial Meniscus −0.020,
Contusion +0.015 / Fracture −0.008 — mixed signs, no systematic collapse, and all inside the
0.028 per-label floor.

**Do not record this as a dead end.** The retest is 8 epochs with jitter and a matched
concat control (kernel v13, arms `v05a` / `v05b`) — it separates the head from the schedule.

### 2026-08-29 — traps 6e was WRONG: PyTorch already seeds numpy/random per worker ❌ DEAD END (the trap, not the code)

Yesterday I asserted that `numpy` and `random` are fork-inherited by DataLoader workers, so
"random" slice jitter would repeat identically every epoch, and added a `worker_init_fn` to fix
it. **That claim could not be tested locally** (Windows spawns workers) and was recorded as
unverified. It is now tested on Kaggle by `check_worker_rng()`, which runs both arms at startup:

```
worker RNG check (traps 6e):
  without worker_init_fn   identical across 3 epochs = False
  with seed_worker         identical across 3 epochs = False
```

**Without the fix the draws already vary**, because PyTorch's `_worker_loop` seeds `random` and
`numpy` per worker itself. The pathology does not exist on this platform. Consequences:

1. **The `v04d` jitter result is unconfounded** — jitter was genuinely random in v11, so the
   +0.0113 stands on its own.
2. It explains why `v04base` (0.8415) matched kernel v8 (0.8426): `seed_worker` changed nothing
   because there was nothing to change.
3. `seed_worker` is **kept** as belt-and-braces (it is free, explicit, and version-proof), but it
   is documented as a guarantee, not a fix. traps 6e is corrected in place.

The general lesson is worth more than the specific one: a plausible mechanism plus a green run is
not evidence. The cheap direct check — two DataLoader arms, seconds of runtime — is what settled
it, and it should have been written before the fix, not after.

### 2026-08-29 — 8-epoch head A/B and first 5-fold ensemble ⏳ PENDING

Two kernels launched concurrently (Kaggle allows two GPU sessions; verified by both smokes
running at once):

| kernel | arms | config | cost |
|---|---|---|---|
| `rsna-knee-train` v13 | `v05a` attn + jitter · `v05b` concat + jitter | fold 0, **8 epochs** | ~3.6 h |
| `rsna-knee-folds` v2 (new slug) | `v05f` | **5 folds** × 4 epochs, concat + jitter (the confirmed `v04d` recipe) | ~4.5 h |

`v05a` vs `v05b` is the honest P-09 retest: same schedule, same augmentation, **only the head
differs**. `v05b` alone answers P-04 (does 8 epochs beat 4 once augmentation exists?) against
`v04d`'s 0.8528.

The 5-fold run is deliberately the *current* winner rather than the eventual one — it is worth a
real ensemble and the first trustworthy LB number regardless of how the head A/B lands. 5 × 8
epochs would be ~9 h and needs the resume path instead.

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

### 2026-08-28 — Submission #1 scored exactly 0.500: constant output at rerun ❌ DEAD END (silent fallback)

Kernel v2 (v01, smoke: 1 fold, 1 epoch, 2 slices/slot) was submitted to prove the
code-competition rerun path. It completed and scored **0.500 public** — exactly. A near-random
model on ~1,000 hidden studies scores 0.47–0.53, never 0.500 to three decimals; an exact 0.500
is a *constant* submission. Two code paths produce that: an empty test manifest → the
`fillna(0.5)` fallback, or every slot directory missing → all-masked inputs → constant head
output. Both mean the hidden test tree did not match the assumed `COMP/test_series/<study>/
<series>/*.dcm` layout at rerun. The rerun log is not visible, so the exact cause is
unconfirmed. Fixes shipped in v02: probe the test image root by globbing for a real series
UID, accept non-`.dcm` filenames, **no placeholder file** (a crash → missing submission →
visible scoring error, by design; corrected: an earlier version wrote a 0.5 placeholder
first), and **refuse to submit** (SystemExit) when < 90% of test studies are imaged *and* have
≥ 1 slot, or > 6 labels are constant — a scoring error is diagnosable, a 0.500 is not. Verdict on the mechanics themselves: the submit
command works (`kaggle competitions submit -k <kernel> -v <version> -f submission.csv`).

### 2026-08-28 — Preprocessing cache built (P-01) ✅ KEEP

`rsna-knee-cache-a` v3 / `-b` v2 (CPU kernels, 4 workers), cache version
`c01_p224_s16_crop130_lat20`: **4,407/4,407 studies cached, 0 decode failures, 25 min per
shard**, 0.6 s/study wall-clock (2.45 s/study CPU). Shard A 2,115 studies / 10.19 GB, shard B
2,292 / 11.04 GB — 21.2 GB total, under the ~20 GB per-kernel cap only because of sharding.
Header pass over all 24,371 series: ~3 min.

Full-corpus facts the manifest now records (P-02 / P-05 inputs):

| | value |
|---|---|
| mean slots per study | 4.78 (24-study smoke said 4.96) |
| `Laterality` tag present | 49.6% of studies |
| side resolved from geometry (20 mm dead zone) | 96.9% |
| tag-vs-geometry agreement where both exist | **0.988** (n = 2,116) |
| conflicts (left unmirrored) | 26 studies (0.6%) |
| unresolved (no tag, centre inside dead zone) | 2.1% |
| FOV median | 160 mm; 0.0% of studies below the 130 mm crop |

These reproduce the public FINDINGS.md numbers (tag missing ~50%, geometry ~97–98%) on our own
run, so the laterality rule is no longer community-sourced.

**Training-side loader (v03, same day):** the notebook now carries the cache builder's exact
functions; cached studies are `np.load`ed, test studies are built on the fly by the same code.
Verified on the local sample: the on-the-fly array equals the cached array **bit-for-bit** for
both cached studies (one left, one right knee). Triplets are neighbouring cached slices
`[c-1, c, c+1]` (≈2–3 real slices apart); K = 6 equidistant centres, no jitter, so the v03
fold-0 run isolates *cache + crop + per-series normalisation + laterality* against v6. Expected:
OOF-vs-teacher within ±0.01 of 0.821 (P-01's sanity measure) and a large drop in s/study.

### 2026-08-28 — First measured throughput (kernel v3, T4, smoke) ⏳ PENDING at production settings

2 slices/slot, `num_workers=0`, batch 1: **2.08 s/study training**, 12 min for the whole
smoke notebook. Inference: **153 s per 100 test studies per fold** at 2 slices/slot.
Extrapolated to 6 slices/slot and ~1,300 hidden studies that is ~1.6 h **per fold model** if
each fold decodes the DICOMs again — a 5-fold ensemble would spend ~8 h on inference alone.
So decode-once inference (predict all folds per decoded study) is a *prerequisite* for any
multi-fold submission, not an optimisation (P-18). Training throughput at 6 slices and
`num_workers=2` is measured by the fold-0 real run.

### 2026-08-28 — Pipeline v02 instrumentation ✅ KEEP

Local smoke + Kaggle smoke (kernel v3, 12 min, green) of: per-label AUC/pred_std table each epoch,
OOF written **every epoch** as `{version}_fold{k}_ep{e}_oof.csv` plus `_oof.csv` for the
checkpointed epoch (corrected: earlier "at the selected epoch" — there is no selection),
2,000-rep bootstrap CI on gold macro, s/study throughput lines, layer-wise LR groups
(4.75e-7 … 2e-5 over 12 blocks), EMA 0.998 validated and saved, `MODE=infer` loading
checkpoints from a mounted kernel output (verified locally to reproduce the training-run
predictions exactly). Checkpoint selection is **fixed-epoch**: `{version}_fold{k}_best.pt`
holds the EMA weights after the last completed epoch; the per-epoch score is logged only.
Kernel v4 (post-review) also fixed resume (mounted `_last.pt`/`_best.pt` are now copied into
WORK — previously resume silently restarted at epoch 0, traps 8b), reads slices_per_slot /
triplet_gap / img_size from the checkpoint config in infer mode, and drops the placeholder
submission in favour of loud failure.

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

### 2026-08-29 — Four silent bugs found while shipping the arms (all fixed) ✅ KEEP the fixes

None of these would have raised. Each was found by reading the path a change would take,
not by a failing test.

1. **Inference silently used v02 preprocessing for a v03 model.** `rsna-knee-infer` mounts only
   `rsna-knee-train`, so no cache manifest is present; the loader then flipped
   `cfg.use_cache = False`, and `KneeStudyDataset.__getitem__` took the v02 decode branch — no
   130 mm crop, no laterality. `use_cache` conflated *which preprocessing* with *whether to read
   a .npy*, and only the second is unavailable at inference. Fixed: the flip now applies in
   train mode only. **Verified**: after the fix the infer kernel's predictions are byte-identical
   to what the v8 training kernel produced for the same 3 test studies. → traps.md 6d.
2. **Infer fold-narrowing was gated on `cfg.smoke`.** With `FORCE_SMOKE=False` and `MODE="auto"`,
   `cfg.folds` is `(0,1,2,3,4)` while only fold 0 has a checkpoint, so the auto rule decided
   `mode="train"` — the submitted notebook would have **re-trained at rerun** (traps.md 12c).
   Fixed: narrowing is unconditional, and the infer kernel is generated with `MODE="infer"`
   sed'd in, the same pattern `cache-b` already uses.
3. **A real-mode default that smoke can never reveal.** Each arm inherits `cfg.folds`, which is
   `(0,1,2,3,4)` in real mode — five arms would have been 25 folds, ≈18 h. Invisible in smoke
   because `__post_init__` forces `folds=(0,)`. Fixed with an explicit `ARM_FOLDS`. → traps.md 12d.
4. **Augmentation that never augments.** `array_to_tensor`'s jitter uses `np.random` and the
   noise augmentation uses `random`; PyTorch seeds only *torch* per worker. On Linux/fork every
   epoch's workers inherit the same parent state, so the "random" jitter would repeat identically
   in all four epochs. Fixed with a `worker_init_fn`. → traps.md 6e.
   ⚠️ **Not empirically verified**: Windows spawns workers rather than forking, so the pathology
   cannot be reproduced on the local machine. If `v04d` comes back flat, "the augmentation still
   is not random" stays a live explanation alongside "jitter does not help."

Also measured while fixing #1: `kaggle kernels output --file-pattern "<no match>"` downloads the
log alone, avoiding ~700 MB of checkpoints per log read.

### 2026-08-28 — Bugs found by running the pipeline (all fixed) ✅ KEEP the fixes

1. **A fold could finish with no `best.pt`.** When AUC is undefined (no positives in a
   fold), `nan > best` is always `False`, so no best checkpoint was ever written and the
   fold would be **silently dropped from the inference ensemble**. Now falls back to
   negative loss and guarantees a checkpoint exists.
2. **Teacher AUC reported as 1.0000** — it was scored *after* gold was copied into the
   targets, i.e. grading gold against itself. Now scored pre-override → 0.8934.
3. **Empty header scans were cached**, so a resumed session would hit the empty cache and
   train on nothing. Now an empty scan is never written.

> Operational reminders (smoke first, never P100, PYTHONUTF8) are in
> [traps.md](traps.md). Infrastructure ideas are in [brainstorm.md](brainstorm.md).

---

## Submissions

### 2026-08-29 — OOF-vs-teacher predicts the LB, with a +0.02–0.03 offset ✅ KEEP (n=2)

Two calibration points now exist, and both behave the same way:

| version | OOF vs teacher | gold (n=11) | public LB | LB − OOF |
|---|---|---|---|---|
| v02 (kernel v6) | 0.821 | 0.847 | 0.841 | +0.020 |
| v03 (kernel v8) | 0.843 | 0.906 | **0.871** | +0.028 |
| Δ | **+0.022** | +0.059 | **+0.030** | |

Three readings, in decreasing confidence:

1. **The v03 preprocessing gain is real.** +0.030 on the LB is six times the 0.005 floor, and it
   moved in the direction the OOF predicted. The +0.022 OOF was **not** a teacher-agreement
   artefact. ✅ KEEP.
2. **OOF-vs-teacher is a usable decision metric, and it under-reads.** The LB sits above the
   teacher-agreement number both times, which is what you expect when the student is graded
   against a teacher whose own gold macro-AUC is 0.8948 — the student averages out teacher noise.
   Treat +0.02–0.03 as a rough offset, not a law: n=2 fits any monotone curve.
3. **Gold overshot this time and undershot last time** (0.906 vs LB 0.871; 0.847 vs 0.841). Both
   differences sit inside the ±0.09 Hanley–McNeil interval, i.e. exactly the documented noise of
   58 studies. No change to gold's role: reported, never gated.

Position: **0.871 from one fold, one backbone, no ensemble, no TTA**, against a public top of
0.952 and ranks 2–9 spanning 0.946–0.949. The remaining gap is 0.081, and a 5-fold rank-mean of
this recipe is the cheapest standing claim on part of it.


Log every submission here with the exact kernel version, config diff, OOF score,
and public LB score, so a public/private divergence can be traced to a specific change.

| # | Date | Kernel ver | Config change | OOF | Public LB | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-08-28 | rsna-knee-train v2 | v01 smoke (1 fold, 1 epoch, 2 slices/slot, rank targets) | n/a | **0.500** | exactly 0.500 = constant output at rerun; image-root assumption failed silently. Mechanics of submitting verified. |
| 2 | 2026-08-28 | rsna-knee-infer v1 (mounts rsna-knee-train v6) | v02 fold 0: DINOv2-S 224, 6 slices/slot, 4 ep, prob targets, LR 2e-5+LLRD, EMA | 0.821 | **0.841** | First non-0.500 score — the submission path works. Above the ~0.809 public DINOv2-S baseline on one fold. |
| 3 | 2026-08-28 | rsna-knee-infer v3 (mounts rsna-knee-train v8) | v03: same recipe, trained from the cache — adds 130 mm crop + laterality + per-series norm | 0.843 | **0.871** | **+0.030 vs #2**, 6× the 0.005 LB floor. The OOF gain transferred and amplified. Also the first submission through the fixed infer path (traps 6d). |
| 4 | 2026-08-29 | rsna-knee-infer v4 (mounts rsna-knee-train v11) | v04d: v03 recipe + `cache_jitter` slice augmentation | 0.8528 | ⏳ pending | +0.0113 OOF over `v04base` against a **measured** 0.008 floor; 11/12 labels up, none down; removes the epoch-2 overfitting turn. Predicted ~0.875–0.88 from the +0.02–0.03 OOF→LB offset. |
