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
| 2026-08-29 | **v04d fold 0 (kernel v11)**: v03 recipe + `cache_jitter` slice augmentation | 0.904 (n=11, CI 0.82–0.95) · OOF-vs-teacher **0.8528** (882 studies) | **0.877** | ✅ KEEP on OOF (+0.0113 vs a 0.008 floor, 11/12 labels up); LB +0.006 is only 1.2× the 0.005 floor and is *corroboration, not proof* |
| 2026-08-29 | **P-21 blend, submission #5 (infer v5)**: rank-mean of `v05a` attn + `v05b` concat, fold 0, 8 ep | 0.927 / 0.876 (singles, n=11) · OOF-vs-teacher **0.8670** (blend, 882 studies) | **0.896** | ✅ KEEP — +0.019 LB vs #4 (3.8× floor); head diversity on one backbone is the cheapest gain found so far. Gap to the public top is now 0.056 |
| 2026-08-29 | **`v05g` 5-fold concat alone, submission #6 (infer v6)** | gold 0.8476 (all 58) · pooled OOF **0.8467** | **0.886** | ✅ five folds = +0.009 LB over one fold (1.8× floor); less than half of what the second head bought (+0.019) |
| 2026-08-29 | **by-version blend attn + concat-8ep + 5-fold concat, submission #7 (infer v8)** | fold-0 proxy 0.8680 | **0.896** | 🔁 equal to #5 — folds add nothing on top of head diversity. **Best remains 0.896** (two kernels, v5 and v8) |
| 2026-08-30 | **P-23 4-version blend + `v06c` ConvNeXt-T, submission #8 (infer v10)** | fold-0 proxy 0.8722 (v06c alone 0.8562, gold 0.905 n=11) | **0.900** | 🔁 +0.004 is under the 0.005 floor — new best on the board, not proof the family earns its place; OOF→LB offset +0.028 held (n=5). **Best is now 0.900 (infer v10)** |
| 2026-08-30 | `v07s` 16-slices-as-channels DINOv2-S, 5 folds (not submitted) | fold-0 OOF **0.7366**, gold 0.70 | — | ❌ DEAD END as built: blend −0.015 despite ρ 0.61 (experiments entry) |
| 2026-08-30 | **`v08w` fold 0 (train v17)**: DINOv2-S 224 on **c02** (2–98 % band) + window_attn, 24 random windows, 8 ep | gold 0.927 (n=11) · OOF **0.8648** | — | ✅ best single at the time; blend +0.0044 as a fifth member 🔁 |
| 2026-08-30 | **`v10c` fold 0 (RunPod 4090, 2.9 h)**: CoAtNet-2 @384 on c02 + window_attn | gold 0.913 (n=11) · OOF **0.8641** | — | ✅ member (meniscus specialist, ρ 0.73–0.81); 384 px buys nothing over 224 (v09h) |
| 2026-08-30 | **`v09h` fold 0 (RunPod 4090, 50 min)**: CoAtNet-1 @224 on c02 + window_attn | gold 0.923 (n=11) · OOF **0.8683** | — | ✅ **best single model**, cheapest strong recipe |
| 2026-08-30 | **P-23 six-version blend, submission #9 (infer v12)**: #8 set + v08w + v10c | fold-0 proxy **0.8795** | ⏳ | expected ≈ 0.905–0.907 |
| 2026-08-30 | **P-23 seven-version blend, submission #10 (infer v13)**: #9 set + v09h | fold-0 proxy **0.8820** | ⏳ | within noise of #9 expected (+0.0024 OOF) |

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

> **HALF RESOLVED 2026-08-29** — `rsna-knee-train` v13 (the head A/B) finished in 3.38 h;
> results in the two entries below. `rsna-knee-folds` v2 (5 folds) was still running at
> 14:38, ~6.6 h in against a ~4.5 h estimate — the estimate was wrong, the run is not.

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

### 2026-08-29 — 8-epoch matched head A/B (kernel `rsna-knee-train` v13): P-09 ✅ KEEP · P-04 🔁 INCONCLUSIVE

Two fold-0 arms, 8 epochs, `cache_jitter=True`, seed 42, from the cache. `v05a` uses the P-09
attention head, `v05b` the concat head. **Nothing else differs** — same schedule, same
augmentation, same seed. 3.38 h total.

| epoch | `v05a` attn | `v05b` concat |
|---|---|---|
| 0 | 0.7449 | 0.7778 |
| 1 | 0.7963 | 0.8326 |
| 2 | 0.8281 | 0.8538 |
| 3 | 0.8452 | 0.8590 |
| 4 | 0.8536 | **0.8600** ← peak |
| 5 | 0.8575 | 0.8542 |
| 6 | **0.8576** | 0.8494 |
| 7 (checkpointed) | **0.8574** | 0.8471 |
| train loss @ ep7 | 0.4129 | **0.3585** |
| gold @ ep7 (n=11) | 0.9266 | 0.8755 |

**P-09 ✅ KEEP: +0.0103 at the checkpointed epoch, 1.3× the measured 0.008 floor.**

**But the mechanism is not the one the card predicted.** P-09's hypothesis was that per-label
queries would help the *side-specific / plane-specific* findings. Per-label at epoch 7
(attn − concat), against the 0.03 per-label floor:

| attn wins | | attn loses |
|---|---|---|
| Fracture +0.040 | | **MCL −0.040** |
| Lateral OA +0.038 | | **Lateral Meniscus −0.032** |
| Contusion +0.035 | | |
| Medial OA +0.019, PF OA +0.016, Effusion/Synovitis +0.014, Baker's +0.009, Medial Meniscus +0.008, ACL +0.001 | | |

The two labels it *loses* on are the two most plane-specific in the set, and both clear the
per-label floor. **Right answer, wrong reason** — the card's rationale should not be reused as if
it were confirmed.

**What actually drives it is resistance to overfitting.** The concat head peaks at epoch 4 and
then decays for three straight epochs while its train loss falls to 0.3585; the attention head
plateaus at 0.857 and holds, with train loss only reaching 0.4129. Cutting the head from 27,732
to 9,300 parameters bought schedule robustness. This also reinterprets the v11 result: `v04c` was
not merely "unconverged", it was the same curve seen too early.

⚠️ **The verdict is policy-dependent, and that is worth stating.** At each head's *own best*
epoch it is attn 0.8576 vs concat 0.8600 — concat marginally ahead, well inside the floor. The
attention head wins **because our checkpoint policy is fixed-last-epoch** (chosen because
best-epoch selection on ~11 gold studies is a coin flip). Under best-epoch selection the two are
indistinguishable. See the new P-22 card.

**P-04 🔁 INCONCLUSIVE — 8 epochs does not beat 4.** `v05b` (concat, 8 ep) finishes at 0.8471
against `v04d` (concat, 4 ep) at 0.8528: −0.0057, inside the floor, so no gain and possibly a
small loss. Even with jitter the concat head overfits past epoch 4. The attention head is the one
that *needs* the longer schedule — it was still climbing at epoch 3 in v11.

### 2026-08-29 — Two heads rank-blend to 0.8670 (ρ = 0.773) ✅ KEEP — head-level diversity is real

Plain rank-mean of `v05a` and `v05b` epoch-7 OOF predictions over the same 882 held-out studies.
**No weights fitted**, so this is a legitimate held-out estimate, not a tuned one.

| | OOF macro |
|---|---|
| `v05b` concat | 0.8471 |
| `v05a` attn | 0.8574 |
| **rank-mean of the two** | **0.8670** |
| gain over the best single arm | **+0.0096** (1.2× the 0.008 floor) |
| gain over submitted `v04d` (0.8528) | **+0.0142** |

**Mean rank correlation between the two arms: 0.773** — strikingly low for two models sharing a
backbone, a dataset, a fold, a schedule and a seed, differing *only* in the head. Least correlated
where each is weakest: Fracture 0.664, Lateral Meniscus 0.695, MCL 0.715; most correlated on
Medial Meniscus 0.870, Effusion 0.858, Medial OA 0.853.

**Why this matters more than the +0.0103 head A/B.** P-10 and P-13 assume error diversity has to
be bought with a second architecture family (a CNN, DINOv3, RadImageNet) — which costs a session
each and, for RadImageNet, carries an unresolved licence. This says a **different head on the same
backbone** already yields ρ ≈ 0.77 and a real blend gain, at **zero extra training cost**, because
both arms were going to be run anyway as an A/B.

Caveats: fold 0 only (n=1 fold); the blend gain is 1.2× the floor, so it is real but not large;
and the +0.02–0.03 OOF→LB offset was calibrated on single models, so extrapolating this blend to
~0.891 LB is **not** supported — an ensemble need not sit on the same curve.

### 2026-08-29 — The first 5-fold run was invalid, not slow ❌ DEAD END (the run, not the idea)

`rsna-knee-folds` v2 ran ~9 h and produced nothing usable. It was **not** a slow ensemble run;
it was the **wrong recipe**. The new kernel slug mounts kernel outputs at depth 4 while
`load_cache_manifests` globbed at `max_depth=2`, so the cache was never found, `cfg.use_cache`
flipped to `False`, and the dataset took the v02 decode branch — no 130 mm crop, no laterality,
no per-series normalisation — at 0.99 s/study instead of 0.18. Full mechanism in traps 6f.

Two numbers that make the diagnosis unambiguous, and that should have been the tell hours
earlier:

| | expected (cache) | observed |
|---|---|---|
| s/study | 0.18 | ~0.99 (the v02 decode rate, measured in kernel v6) |
| 5 folds × 4 epochs | ~4.5 h | **~19.6 h** — could never fit the 8.3 h guard |

**The elapsed time was the evidence and it was misread.** At 14:38 this was recorded as "~6.6 h
in against a ~4.5 h estimate — the estimate was wrong, the run is not." That was backwards: a
1.7× overrun against a throughput figure measured three times that day was a *symptom*, and it
was explained away as estimator error instead of being investigated. The run had already been
wrong for six hours at that point.

**How it was actually caught:** the smoke log, surfaced in the browser, says
`! use_cache=True but no cache is mounted -- falling back to per-epoch DICOM decode` in plain
text at line 61. That log had already been read once — for the arm banners and the worker-RNG
check — and the cache line was skipped. The lesson is in traps 6f: read the log for the *known*
failure modes, not only for the new thing being tested.

**Unaffected:** kernel v13 (`v05a`/`v05b`) mounted the cache correctly
(`cache: 4407 studies indexed`), so P-09, P-04, the two-head blend and the noise floor all
stand. Only the 5-fold ensemble is outstanding.

Fixed in `src/kaggle_pipeline.py`: glob depth 2 → 4, and a missing cache in train mode is now a
`SystemExit` unless `ALLOW_DECODE_FALLBACK = True`.

### 2026-08-29 — P-22: checkpoint on OOF-vs-teacher instead of fixed last epoch ✅ KEEP for the concat head · 🔁 neutral for attn · policy switched

`src/oof_epoch_analysis.py` on the per-epoch OOF csvs already on disk (v13 `v05a`/`v05b`, v6
`v02`, v8 `v03`; the v11 arms are unreachable — traps 12e). Metric = the kernel's `evaluate()`
(hard = y > 0.5, macro over finite labels); the script **reproduces every logged number to 4 dp**
(`v05a` 0.8574/0.8576, `v05b` 0.8471/0.8600, `v02` 0.821) before reporting anything new.

Two estimates per arm. *In-sample*: best epoch on all 882 val studies minus the last epoch —
biased upward, it is the max of N noisy values. *Split-half*: choose the epoch on a random half
of the val studies, score on the other half, 200 splits — the honest number.

| arm | epochs | last | best (epoch) | Δ in-sample | **Δ split-half** | chosen > last | gold at best / last (n=11) |
|---|---|---|---|---|---|---|---|
| `v05b` concat + jitter | 8 | 0.8471 | 0.8600 (4) | +0.0129 | **+0.0128** (sd 0.002) | 100% | 0.873 / 0.876 |
| `v05a` attn + jitter | 8 | 0.8574 | 0.8576 (6) | +0.0002 | **−0.0002** (sd 0.0005) | 31% | 0.924 / 0.927 |
| `v03` concat | 4 | 0.8426 | 0.8457 (2) | +0.0032 | +0.0032 | 100% | 0.898 / 0.906 |
| `v02` concat, decode path | 4 | 0.8214 | 0.8257 (2) | +0.0043 | +0.0041 | 100% | 0.861 / 0.847 |

**Verdict by the card's own rule** (split-half gain > 0.008 floor **and** gold not moving against
it): ✅ **KEEP for the concat head** — +0.0128 is 1.6× the floor, chosen in 200/200 splits, and
gold at the chosen epoch is within 0.003 of gold at the last epoch (SE 0.09, direction only). For
the attention head the two policies are indistinguishable (−0.0002), so nothing is lost there.
The 4-epoch concat arms gain +0.003–0.004 — below the floor, but the same sign every time.

**Teacher-chasing check: negative.** After the OOF peak `v05b`'s OOF falls −0.013 while its gold
*rises* +0.002; `v05a` the same (−0.0002 / +0.003). Nowhere does OOF-vs-teacher rise while gold
falls, which is the pattern selecting-on-the-teacher would produce. (Gold's own peak for `v05b` is
epoch 2 at 0.9135, two epochs before the OOF peak — n=11, inside the ±0.09 interval; noted, not
acted on.)

**Verdicts that move under the new policy:**

- **P-09 becomes a tie**: attn − concat at each head's best epoch is **−0.0024** (was +0.0103 at
  the last epoch). The +0.0103 was the concat head's decay, not the attention head's gain. Both
  heads stay — the blend needs both — but "attn wins" is no longer a claim we make.
- **P-04 stays 🔁**: `v05b` at its best epoch (0.8600) vs `v04d` at its last (0.8528) is +0.0072,
  but `v04d`'s own best epoch is unknown (its csvs are unreachable), so the comparison is
  policy-mismatched.
- **P-21 blend at best-epoch checkpoints: 0.8695** (vs 0.8670 at last-epoch); still fold 0 only.
- Snapshot rank-mean of one arm's last three epochs: `v05a` 0.8579 (+0.0005), `v05b` 0.8507
  (+0.0036) — below the floor, not pursued.

**Shipped:** `Config.ckpt_policy = "best_oof"` (default) — `_best.pt` and `_oof.csv` follow the
epoch with the highest `auc_soft` so far; `_last.pt` still every epoch for resume, now carrying
`best_epoch`. `"last"` restores fixed-epoch. Gold is never the selector. Takes effect from the
`v05g` 5-fold run onward; every number above and before it was checkpointed at the last epoch.

**Caveat:** one fold, and the OOF the epoch is chosen on is the OOF later reported for that fold —
the split-half says the bias is ≤ 0.001 here, but the per-fold OOFs of a `best_oof` run are
*selected* numbers and should be read as such.

### 2026-08-29 — First valid 5-fold run (`v05g`, kernel `rsna-knee-folds` v4) ✅ KEEP as the ensemble base · fold-ensemble LB gain ⏳

Concat head + `cache_jitter`, 4 epochs, seed 42, folds 0–4 — the `v04d` recipe on every fold, from
the cache (`cache: 4407 studies indexed` at the new `/kaggle/input/notebooks/…` path, 0.17 s/study
throughout). **4.27 h** for five folds including inference; the 8.3 h guard was never near.

| fold | epoch 0 | 1 | 2 | 3 (checkpointed) | gold (n) |
|---|---|---|---|---|---|
| 0 | 0.7778 | 0.8285 | 0.8466 | **0.8508** | 0.887 (11) |
| 1 | 0.7767 | 0.8258 | 0.8393 | **0.8429** | 0.845 (12) |
| 2 | 0.7709 | 0.8290 | 0.8435 | **0.8456** | 0.824 (12) |
| 3 | 0.7805 | 0.8277 | 0.8414 | **0.8449** | 0.856 (12) |
| 4 | 0.7866 | 0.8322 | 0.8470 | **0.8503** | 0.866 (11) |

- **Mean of folds 0.8469; pooled over all 4,407 studies 0.8467** (fold-rank-normalised, same);
  gold over all 58: **0.8476**. Per-label pooled: Baker's 0.894, Synovitis 0.890, Fracture 0.877,
  Medial OA 0.874, Medial Meniscus 0.871, Effusion 0.851, Contusion 0.847, ACL 0.836, Lateral OA
  0.826, PF OA 0.812, **MCL 0.792, Lateral Meniscus 0.789** — the two side/plane-specific labels
  remain the floor of the model, as they were on fold 0 alone.
- **Fold spread 0.8429–0.8508 (range 0.008 = one noise floor).** Fold 0 is the easiest fold, not
  an outlier; every fold-0 A/B so far read a representative fold.
- **`v04d` reproduced**: fold 0 tracks `v04d`'s curve (0.8338/0.8492/0.8528) within 0.002–0.005,
  and epoch 0 equals `v05b`'s epoch 0 to 4 dp (same head, jitter and seed). The recipe is
  reproducible run to run at the floor.
- **`ckpt_policy="best_oof"` was inert here**: every fold improved monotonically, so epoch 3 was
  chosen everywhere. Consistent with P-22 — the policy only bites when a head decays, i.e. concat
  past epoch 4.
- **Folds vs heads, on fold 0's 882 studies** (the only place both heads exist): `v05a` attn 0.8574,
  `v05b` concat-8ep 0.8471, `v05g` concat-4ep 0.8508. Rank-mean a+b **0.8670**, b+g 0.8592,
  a+g 0.8650, **a+b+g 0.8680 (+0.001 over a+b — inside the floor)**. Rank correlations: a–b 0.773,
  a–g 0.835, b–g 0.842. A third member of the *same head* on the same fold adds nothing measurable;
  head diversity (ρ 0.77) beats schedule diversity (ρ 0.84). The fold-ensemble gain itself cannot be
  read from OOF (each study is held out once) — it is measured only on the LB, which is what the
  `INFER_MEMBERS=["v05g"]` infer kernel (v6) is for.
- **Vote weighting matters more than member count.** Same three fold-0 models, rank-mean with
  different weights (attn : concat-8ep : concat-4ep): flat 1:1:1 **0.8680**; **1:1:5 — what a flat
  mean over 7 checkpoints gives, since `v05g` has five folds — 0.8611**, below the two-head blend
  alone (0.8670); attn 2:1:1 0.8688 (inside the floor of 1:1:1, not adopted — no tuning on fold 0).
  The attention head is the source of the diversity and a flat mean dilutes it to 1/7. So the infer
  path now defaults to **`INFER_BLEND="by_version"`**: rank-mean the folds of each version, then
  rank-mean the versions — one vote per version, no fitted weights. The flat 7-member kernel
  (infer v7) was built and verified but is **not** to be submitted; v8 is the by-version one.

**Verdict:** ✅ the five checkpoints are the valid ensemble base (`v05g_fold{0..4}_best.pt` in
`rsna-knee-folds` v4 output; **never mount v2's `v05f`**). Whether five folds buy more than +0.005 LB
over one fold is ⏳ until the 5-fold-only submission scores. The natural final shape is five folds ×
two heads; the attention half costs ~9 h and stays a decision for the next session.

> **RESOLVED 2026-08-29 23:28 (submissions #6 and #7).** Five folds alone: **0.886**, +0.009 over
> one fold — real (1.8× floor) but half of the head blend's +0.019. Five folds *added to* the head
> blend (one vote per version): **0.896, identical to the two-head blend**. Reading: fold-averaging
> and head-blending both mostly remove the same thing — the variance of a single concat model — so
> once a second head is in the blend, folds of the first head are redundant. **The 9 h attention
> 5-fold run is therefore not launched**: its best case is the concat-side analogue (+0.009 on the
> attn member alone, ~0 in the blend). The quota is better spent on a *third source of diverse
> errors* (a different head, schedule or backbone family — P-10 with licence-clean ConvNeXt is the
> obvious candidate, fold 0 first, ~1 h) than on more folds of an existing member. P-13's
> "3 folds + a second family beats 5 folds of one" now has direct support.

---

### 2026-08-30 — 16-slices-as-channels DINOv2-S member `v07s` (P-23 candidate #3, kernel `rsna-knee-stack` v2) ❌ DEAD END (this recipe)

**Config:** `stack_mode="channels"` — each slot's 16 cached slices are the 16 input channels of ONE
image, so the encoder sees the whole stack in one pass (6 forwards per study instead of 36). DINOv2-S/14
at 224, patch-embedding conv widened 3 → 16 (init = RGB-mean kernel × 3/16, trained at `lr_stem`
2e-4; the rest of the backbone under the usual 2e-5 / LLRD 0.75), concat head, `cache_jitter` (whole
stack shifts ±1 slice), 8 epochs, `ckpt_policy=best_oof`, EMA 0.998, **five folds** in one session.
Launched 2026-08-30 00:46 after a green Kaggle smoke (v1: cache 4407 indexed on the new slug, widened
embedding, channels member through decode-once inference). Own kernel slug so the run never repoints
the `rsna-knee-train` / `rsna-knee-folds` mounts the infer kernel reads.

**Why this member:** the 0.936 notebook's second family is exactly this representation
(research.md §2.7.1); it is the most *different* input we can build from the existing cache and mounted
weights, and ~6× cheaper per epoch than a triplet member.

**Decision rule (P-23, fixed before the run):** on fold 0, own OOF ≥ 0.8574 − 0.02; mean Spearman ρ vs
the `v05a`+`v05b` rank-mean < 0.80; blend gain over 0.8670 > 0.008. `src/blend_check.py` applies it.
Accepted → `INFER_MEMBERS` gains `v07s` (all five folds, one vote by version) and one submission is
placed; rejected → logged here, no submission.

**Result (read 09:15, 4.79 h GPU, five folds complete):** every fold plateaus at **OOF 0.73–0.74**
(fold 0 0.7366 at epoch 5; folds 1–4 0.7419 / 0.7381 / 0.7324 / 0.7411), still creeping up at epoch 8
(loss 0.61 → 0.48, no overfitting signature). Gold 0.70. Throughput 0.10–0.12 s/study (vs 0.19 for a
triplet member) — the 6× fewer forwards bought ~1.7×, the rest is data loading.
`blend_check.py` on fold 0: own 0.7366 **fails** (a) by 0.10; ρ vs the `v05a`+`v05b` blend **0.609**
— the most diverse member we have ever built — but adding it moves the blend **0.8670 → 0.8524
(−0.0146)**; only Effusion (+0.001) and Synovitis (+0.004) survive, every other label drops 0.006–0.028.
**Verdict ❌ DEAD END for this recipe**: a mean-initialised 16-channel patch embed at 224 px, `lr_stem`
2e-4, 8 epochs cannot learn to separate slices through a linear 14×14 conv — the backbone effectively
sees a blurred stack average. Not evidence against the *representation* (the 0.936 notebook's version
uses a gated `DepthCompress` stem, 336 px, and presumably far more stem learning); a retry would need a
non-linear stem at ≥ 1e-3 and more epochs, and is not worth the quota this week. Checkpoints not used.


### 2026-08-30 — ConvNeXt-Tiny member `v06c` (P-10 / P-23 candidate #1, kernel `rsna-knee-train` v15) 🔁 INCONCLUSIVE by the rule · a second family at parity

**Config:** HF `facebook/convnext-tiny-224` (ImageNet-1k, Apache-2.0), concat head, `cache_jitter`,
8 epochs, `ckpt_policy=best_oof`, `lr_backbone` 1e-4 with LLRD 0.75 per stage, fold 0, from the cache.
1.86 h; 0.19 s/study (same as ViT-S/14). Launched 23:57, read 09:15 (the overnight session died).

**Own OOF curve:** 0.7795 → 0.8297 → 0.8530 → **0.8562 (epoch 3, checkpointed)** → 0.8515 → 0.8460 →
0.8425 → 0.8416. Peaks earlier and decays faster than the DINOv2 concat head (`v05b` peaked at epoch 4);
`best_oof` (P-22) kept the peak — under the old fixed-epoch policy this member would have shipped at
0.8416. Gold 0.905 (n=11).

**As a single model it is at parity with our best DINOv2 head:** 0.8562 vs `v05a` 0.8574 (attn), above
`v05b` 0.8471 and `v05g` 0.8508 (concat). A supervised ImageNet CNN at 224 matches the SSL ViT here,
which the 0.936 notebook's own gold panel (ConvNeXt-B/L ≈ 0.875 vs CoAtNet 0.9025) did not predict.

**Blend check on fold 0 (`src/blend_check.py`, `artifacts/kaggle_out/blend_verdicts.jsonl`):**

| vs base | base OOF | ρ(v06c, base blend) | ρ per member | + v06c | gain |
|---|---|---|---|---|---|
| `v05a`+`v05b` | 0.8670 | **0.831** | 0.805 (v05a), 0.767 (v05b) | 0.8729 | **+0.0059** |
| `v05a`+`v05b`+`v05g` (the submitted blend) | 0.8680 | 0.848 | 0.822 (v05g) | 0.8722 | +0.0043 |

Per label (2-version base → + v06c): **10 of 12 up** — ACL +0.016, Lateral Meniscus +0.009, Fracture
+0.008, Synovitis +0.008, Medial Meniscus +0.007, Baker's +0.007, Medial OA +0.006, Effusion +0.006,
Lateral OA +0.004, PF OA +0.004; MCL 0.000; Contusion −0.003.

**Verdict:** by the pre-registered P-23 rule (ρ < 0.80 **and** gain > 0.008) it is a **reject on both
counts, narrowly** — ρ 0.831 is in the same band as attn-vs-concat (0.773) and folds-vs-heads (0.84),
i.e. head-like diversity, not a new error profile; +0.0059 is 0.7× the 0.008 macro floor → **🔁
INCONCLUSIVE**. What argues for it is the *sign pattern*: 10/12 labels up is the same kind of evidence
that carried jitter (11/12). Tian chose to let the LB arbitrate: **submission #8 = infer v9, by-version
blend `v05a`+`v05b`+`v05g`+`v06c`** (rule set before scoring: < +0.005 over 0.896 is 🔁, not a win).
**Result: LB 0.900 (+0.004) — 🔁 by the rule, best on the board.** P-10's family bet is therefore *half* confirmed — the family is as
strong as ours, but not much more diverse than a second head at 224 px with the same slots and slices.

### 2026-08-30 — Per-label OOF of the 4-version blend: the weakest labels are the ones the discarded outer slices carry ⏳ PENDING (diagnostic → P-26)

`src/blend_check.py --base v05a v05b v05g --cand v06c` on fold 0 (882 studies), read for *where* the
0.900 blend is weak rather than for the candidate verdict:

| label | base (3 versions) | + v06c | label | base | + v06c |
|---|---|---|---|---|---|
| Fracture | 0.911 | 0.917 | Effusion | 0.857 | 0.861 |
| Baker's | 0.909 | 0.913 | **MCL** | **0.836** | **0.836** |
| Medial Meniscus | 0.896 | 0.904 | **Lateral Meniscus** | **0.827** | **0.833** |
| Synovitis | 0.884 | 0.888 | PF OA | 0.830 | 0.833 |
| ACL | 0.883 | 0.897 | Lateral OA | 0.825 | 0.828 |
| Medial OA | 0.880 | 0.883 | Contusion | 0.877 | 0.873 |

MCL and Lateral Meniscus are the two worst labels, 0.05–0.08 under the best, and are exactly the two
findings the 0.936 notebook's strongest member says it lost when the outer slices were cut (its
docstring, read 2026-08-30: 2–98 % span vs our sag 8–92 / cor 20–80). ConvNeXt-T does not move them
(MCL +0.000). This is the evidence behind P-26 (wide-band cache) and P-25 (per-label attention over
every window); the measurement that settles it is `v08w` fold 0 per label vs `v05a`. ⏳ until then.

### 2026-08-30 — P-12 slice-offset TTA, first number (kernel `rsna-knee-eval` v2, `oof_eval`, T4): v05a mean-TTA OOF 0.8621 vs 0.8574 🔁 INCONCLUSIVE · the run died OOM on member 2 → measurement moved to the RunPod pod ⏳

`MODE="oof_eval"`, `INFER_MEMBERS = [v05a, v05b, v05g, v06c]`, every member `tta_offsets=(-1, 0, 1)`,
`tta_pool="mean"`, c01 cache, fold 0 (882 held-out studies), `num_workers=2`. Only the first member
finished (7.2 min on the T4):

| v05a fold 0 | macro OOF | MCL | Lateral Meniscus | auc_gold (n=11) |
|---|---|---|---|---|
| no TTA (train-time OOF, kernel v13) | 0.8574 | 0.795 | 0.818 | — |
| TTA (-1, 0, 1) / mean | **0.8621** | 0.805 | 0.802 | 0.921 (CI 0.82–0.98) |

+0.0047 macro is **under the 0.008 OOF floor → 🔁 on its own**; MCL +0.010 and Lateral Meniscus −0.016
are both inside the ~0.03 per-label floor. The P-12 verdict is the *4-version blend* with vs without TTA
(rule: adopt per member only if the blend gains > 0.008), which needs all four `_tta_oof.csv` files.

Then, ~2.5 min into `v05b`, `RuntimeError: DataLoader worker (pid 73) is killed by signal: Killed` —
the host-RAM OOM killer, not CUDA (traps 28). Per-member RAM is small (3 views × 21.7 MB per study, 2
workers), and v05a ran clean, so something accumulates across members on Kaggle's ~30 GB box; the root
cause is **open**. Rather than spend another ~0.5 h of the ~3.6 h weekly quota on a guess, the same
`oof_eval` (mean **and** focal) runs on the RunPod pod (503 GB RAM, 4090) before its training arms —
the pod pulls the c01 shards and the three checkpoint pins for exactly this. Cost of the failed kernel:
~0.2 h GPU. ⏳ until the pod's `tta_mean/` and `tta_focal/` csvs are pulled and `blend_check.py` is run
against the untouched OOF files (base 0.8722 for the 4-version blend).

### 2026-08-30 — `v08w` fold 0 (P-25 window-attention head + P-26 wide-band c02 cache, kernel `rsna-knee-train` v17): OOF **0.8648**, the best single model · 12/12 labels up vs the blend ✅ KEEP as a member recipe · blend gain +0.0044 🔁 (REJECT as a *fifth* member by the rule) · P-26 half-confirmed (MCL +0.028, Lateral Meniscus +0.009)

DINOv2-S/14 @224 on the **c02** cache (2–98 % band, ragged 18/12/12/14/8/8 slices), `window_mode="random"`
(24 train windows, all windows at eval), `head_type="window_attn"`, 8 epochs, `best_oof`, fold 0, seed 42,
1.5 h on the T4 (train 7.2 min + val 3.7 min per epoch; **0.12 s/study on the blob loader vs 0.19 for c01**).

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| OOF (auc_soft) | 0.747 | 0.805 | 0.834 | 0.849 | 0.858 | 0.863 | 0.8646 | **0.8648** |
| gold (n=11) | 0.813 | 0.873 | 0.896 | 0.902 | 0.916 | 0.916 | 0.927 | 0.927 |

Still rising at epoch 7 (+0.0002), so 8 epochs is about right; `best_oof` picked epoch 7. `pred_std` 0.24.

**Singles:** v08w 0.8648 vs v05a 0.8574 (same backbone, same fold, c01 + fixed K=6 + attn head): **+0.0074**,
just under the 0.008 macro floor — but **all 12 labels move up** when it joins the blend (seed noise
scatters signs, so a consistent sign is evidence), and it is the first member to beat v05a. Gold 0.927 = v05a.

**Blend (`src/blend_check.py`, fold 0, 882 studies):**

| blend | OOF | note |
|---|---|---|
| v05a+v05b+v05g+v06c (LB 0.900) | 0.8722 | base |
| + v08w as fifth | 0.8766 | +0.0044 — **REJECT** by the rule (ρ vs base blend 0.866 ≥ 0.80; gain < 0.008) |
| v08w **replacing** v05a | 0.8749 | +0.0027, 🔁; the replaced v05a then adds back only +0.0017 |
| v08w + v06c | 0.8739 | two members already at the 4-member level |
| v08w + v05b | 0.8722 | = the 4-member base with two members |

Every DINOv2-based subset saturates at 0.872–0.877: **v08w is a stronger member of the same family, not new
diversity** (ρ 0.84 with v05a, 0.76 with v05b). That is the P-10/P-23 lesson again and what `v10c` (CoAtNet-2
@384, RunPod) is for.

**Per label (v08w alone vs v05a alone):** MCL **0.823 vs 0.795 (+0.028)** — the P-26 claim (+0.03) holds within
the per-label floor; Lateral Meniscus 0.827 vs 0.818 (+0.009) does not. In the 5-member blend the two labels
move +0.006 / +0.010. So the wide band buys MCL, and Lateral Meniscus needs something else (resolution / a
different family — `v10c` is the test).

Verdict: ✅ **KEEP the recipe** (c02 + random windows + window_attn is the new default member recipe; a 5-fold
`v08w` would replace `v05g` as the fold-ensemble base if folds are ever re-run), 🔁 on the blend gain — no
submission on this alone (expected LB +0.003–0.004 < the 0.005 floor); decide the blend with `v10c` in hand.

### 2026-08-30 — `v10c` fold 0 (P-23 #2b: `timm:coatnet_rmlp_2_rw_384` @384, c02, window_attn; RunPod RTX 4090, 2.9 h): OOF **0.8641** = parity with `v08w` from a second family · meniscus specialist (Lateral Meniscus 0.858, Medial Meniscus 0.922) · 6-member blend 0.8795 (+0.0073 over the LB blend) 🔁 by the single-candidate rule, ✅ KEEP as a member

The 0.936 notebook's strongest-member recipe on our cache: CoAtNet-2 (73 M, ImageNet-pretrained, offline
timm safetensors) at **384 px** over the **c02** wide-band cache, `window_mode="random"` (24 train windows,
**42 equidistant eval windows**), `head_type="window_attn"`, `lr_backbone=1e-4` (5× DINOv2's), LLRD 0.75, 8
epochs, `best_oof`, EMA 0.998, `grad_checkpoint=True`, seed 42, fold 0. **0.30 s/study → 17.7 min train + 3.2
min val per epoch, 8 GB VRAM training / 14 GB with the 42-window validation** on the 4090 (`RSNA_WORKERS=8`,
blobs on local NVMe; needs `ulimit -n` raised — traps 29). Run 1 died at epoch 0 (fd limit); run 2 is this.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| OOF (auc_soft) | 0.733 | 0.764 | 0.810 | 0.830 | 0.845 | 0.854 | 0.861 | **0.8641** |
| MCL | 0.698 | 0.707 | 0.741 | 0.751 | 0.773 | 0.784 | 0.796 | 0.807 |
| Lateral Meniscus | 0.693 | 0.702 | 0.712 | 0.732 | 0.775 | 0.816 | 0.845 | **0.858** |
| `v08w` same epoch | 0.747 | 0.805 | 0.834 | 0.849 | 0.858 | 0.863 | 0.865 | 0.8648 |

Slower start than DINOv2 (a fresh 286 k-param head over CNN-hybrid features at high LR, 10 % warmup, EMA
lag), then it closes the gap every epoch and is **still rising at epoch 7 (+0.0035)** — a 10–12-epoch rerun is
the obvious follow-up (P-23 card). Gold 0.913 (n = 11, noise).

**Per label, alone:** Lateral Meniscus **0.858** and Medial Meniscus **0.922** are the best of any member
(v08w 0.827 / 0.903; the 4-member LB blend 0.833 / 0.904); ACL 0.841 and MCL 0.807 are the weakest (v08w
0.865 / 0.823). The 384-px hybrid reads the menisci, DINOv2 reads ACL/MCL: **the two families are strong on
different labels**, which is what rank fusion pays for and what P-10/P-23 predicted.

**Blends (`src/blend_check.py`, fold 0, 882 studies):**

| blend | OOF | note |
|---|---|---|
| LB blend v05a+v05b+v05g+v06c | 0.8722 | base (LB 0.900) |
| + v10c | 0.8774 | +0.0052, ρ vs base 0.838 → 🔁 by the rule; Lateral Meniscus +0.017, Medial Meniscus +0.009 |
| + v08w + v10c (**6 members**) | **0.8795** | **+0.0073** over the LB blend; the two new members together are just under the 0.008 floor |
| v08w + v06c + v10c (three families) | 0.8785 | three members ≈ six |
| v08w + v10c | 0.8736 | **+0.0088 over v08w alone** — the first pair to clear the gain floor (ρ 0.847) |

ρ: v10c–v05b 0.731, –v06c 0.776, –v05g 0.800, –v05a 0.813 — the most different member we have (v08w–v05a
was 0.841). Verdict: ✅ **KEEP `v10c` as a member** (parity single, lowest correlation, complementary labels);
each addition alone is 🔁 under the single-candidate rule, and the six-member blend's +0.0073 predicts an LB
of ≈ 0.905–0.907 by the +0.02–0.03 offset — around the 0.005 LB floor, so a submission is an *information*
buy, not a proven gain. Infer kernel v12 (6 members) pushed 16:41 to check the mixed-geometry path end to end;
submission is Tian's call. Checkpoint: Dataset `tiankljucanin/rsna-knee-ckpt-v10c` (`ship` on the pod).

### 2026-08-30 — `v09h` fold 0 (P-23 #2a probe: `timm:coatnet_rmlp_1_rw_224` @224, c02, window_attn; RunPod 4090, 50 min): OOF **0.8683** — the best single model · 7-member blend 0.8820 (+0.0024) 🔁 as an addition · ✅ KEEP as the cheapest strong member

CoAtNet-1 (41.7 M) at **224 px** on the c02 cache, same recipe as `v10c` otherwise (24 random train windows, all
windows at eval, window_attn, `lr_backbone=1e-4`, 8 epochs, `best_oof`, EMA 0.998), fold 0, seed 42. **0.09
s/study → 5.3 min train + 0.9 min val per epoch = 50 min for the fold** (vs 2.9 h for `v10c`, 1.5 h for `v08w` on
the T4).

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| OOF | 0.759 | 0.814 | 0.836 | 0.853 | 0.864 | 0.867 | 0.868 | **0.8683** |
| Lateral Meniscus | 0.698 | 0.743 | 0.766 | 0.802 | 0.835 | 0.854 | 0.863 | **0.867** |
| MCL | 0.686 | 0.756 | 0.770 | 0.793 | 0.805 | 0.802 | 0.798 | 0.799 |

Fastest learner of the three new arms at every epoch, converged (+0.0003 over the last epoch), gold 0.923.
Singles now: **v09h 0.8683 > v08w 0.8648 > v10c 0.8641** > v05a 0.8574 — all three c02/window-attn arms beat
every c01 member. Per label alone: Lateral Meniscus **0.867** (best of all), Lateral OA 0.848 (best), ACL 0.871
(≈ v05a), MCL 0.799 (v08w 0.823 still best), Medial Meniscus 0.899 (v10c 0.922 best).

**Blends (`src/blend_check.py`, fold 0):** 6-member (#9) + v09h → **0.8820 (+0.0024, ρ 0.879 → 🔁)**;
v08w+v10c+v09h (the three new arms alone) 0.8788; + v06c 0.8816; 4 old + v09h 0.8778. ρ(v09h, v08w) 0.843,
ρ(v09h, v10c) 0.843 — a third strong member of the *same* recipe family, so it buys little diversity; the
resolution (224 vs 384) barely matters for CoAtNet here (0.8683 vs 0.8641, 3.5× the cost), which reframes the
"0.924 needs 384" reading of the 0.936 notebook: **the band + window head + hybrid backbone are the gain, not the
pixels**. Verdict: ✅ KEEP as a member (cheapest strong model — 50 min/fold makes a 5-fold `v09h` a 4 h job),
🔁 for the blend increment. Infer v13 (7 members) pushed 17:31 for submission #10 on Tian's instruction, with the
expectation that its LB is within noise of #9.

### 2026-08-30 — ⭐ What made the 0.936 notebook good, **measured**: the wide slice band + per-label window attention + a hybrid backbone — **not the pixels**. Three arms in one day (`v08w` 0.8648, `v10c` 0.8641, `v09h` 0.8683) are the three best single models we have ✅ KEEP (recipe) · the ensemble gain is small because they are one family 🔁

**The question** (research.md §2.7.1): the public 0.936 notebook's strongest member is a CoAtNet-2 @384 over 64
slices at a 2–98 % slice band with per-label attention over every window, scoring 0.924 alone; our DINOv2 recipe
was at parity with *its* DINOv2 branch (≈ 0.899 vs our 0.896 LB). Which of its ingredients carries the +0.025?
Today's three arms separate them, all on the same fold-0 split (882 studies), same teacher, same 8-epoch
`best_oof` schedule:

| arm | backbone | px | cache / band | head | fold-0 OOF | Δ vs v05a (0.8574) | wall-clock |
|---|---|---|---|---|---|---|---|
| v05a (reference) | DINOv2-S | 224 | c01: 8–92 % sag / 20–80 % cor, 16 dense | attn over 6 fixed triplets | 0.8574 | — | 1.5 h T4 |
| **v08w** | DINOv2-S | 224 | **c02: 2–98 %, ragged 18/12/12/14/8/8** | **window_attn, 24 random windows** | **0.8648** | +0.0074 | 1.5 h T4 |
| **v10c** | CoAtNet-2 (73 M) | **384** | c02 | window_attn | **0.8641** | +0.0067 | 2.9 h 4090 |
| **v09h** | CoAtNet-1 (42 M) | 224 | c02 | window_attn | **0.8683** | **+0.0109** | **0.85 h 4090** |

Readings, each backed by a row pair:

1. **Band + window head, same backbone, same pixels (v05a → v08w): +0.0074**, 12/12 labels up in the blend, MCL
   0.795 → 0.823. That is the first ingredient and it is free (0 GPU h to build the cache, same training cost).
2. **Hybrid backbone on top (v08w → v09h, both 224, both c02): +0.0035**, and a different error profile —
   Lateral Meniscus 0.867 vs 0.827, Lateral OA 0.848 vs 0.833, while MCL/ACL stay with DINOv2. Second ingredient.
3. **384 px vs 224 px, same family (v10c vs v09h): −0.0042 at 3.5× the cost.** Resolution is *not* an
   ingredient at our data size; CoAtNet-2 @384 was the notebook's choice, not its reason. (Caveat: one seed each;
   the OOF floor is 0.008, so "no gain" is the honest reading, not "worse".)
4. **The three c02 arms are one family for blending purposes** — ρ 0.84 between any two — so the 4-member LB
   blend (0.8722) goes to 0.8795 with v08w+v10c and 0.8820 with all three (+0.0098 total, three members). The
   notebook's remaining +0.01–0.02 came from *input-representation* families (16-channel ViT, RadImageNet
   frozen features) and a gold-tuned calibrator we deliberately do not copy — that is the part still open.

**Consequences.** (a) `c02 + window_mode="random" + head_type="window_attn"` is the default member recipe from
now on; every c01 member is dominated. (b) **The cheapest strong model is `v09h`: 50 min per fold on a 4090
(~$0.65)** — a 5-fold `v09h` is ~4 h / ~$3 and gives the fold-ensemble base the 5-fold `v05g` (0.8467 pooled)
gave before, but from 0.868 instead of 0.847. (c) A 12-epoch `v10c` is *not* the next arm: 384 px buys nothing
here. (d) New diversity has to come from a different input representation or pretraining (P-23 #3/#4, P-17
self-training), not from more backbones on the same windows — the notebook's own +0.001 three-backbone
counter-example, now reproduced on our side.

Submissions: **#9 (infer v12, six versions, OOF 0.8795) and #10 (infer v13, seven versions, OOF 0.8820)** were
sent 17:08 / 17:37; scores ⏳ (Scoreboard). Expected LB from the +0.02–0.03 offset: ≈ 0.905–0.908.

### 2026-08-30 — P-12 slice-offset TTA, measured on the RunPod pod (`oof_eval`, all four c01 members, (-1, 0, 1)): every member up +0.003–0.006 alone, the 4-member blend **+0.0016 mean / +0.0023 focal** (0.8722 → 0.8738 / 0.8745) 🔁 INCONCLUSIVE · not adopted

| member | OOF no TTA | mean TTA | Δ | focal TTA |
|---|---|---|---|---|
| v05a | 0.8574 | 0.8621 | +0.0047 | 0.8621 |
| v05b | 0.8471 | 0.8532 | +0.0061 | 0.8537 |
| v05g (fold 0) | 0.8508 | 0.8537 | +0.0029 | 0.8538 |
| v06c | 0.8562 | 0.8625 | +0.0063 | 0.8621 |
| **4-member blend** | **0.8722** | **0.8738** | **+0.0016** | **0.8745** (+0.0023) |
| 7-member blend (c01 members TTA'd) | 0.8820 | 0.8822 | +0.0002 | 0.8826 (+0.0006) |

The pattern is exactly what a variance-reduction technique should show: **each single model gains (consistent
sign, 4/4), the blend does not** — averaging ranks over members already removes the per-view noise that TTA
removes within a member. Cost: 3 forwards per study per c01 member at inference. Verdict: 🔁, **`INFER_OVERRIDES`
stays empty**; the mean-TTA per-member gain is real but redundant with ensembling. Files:
`artifacts/kaggle_out/eval_pod/tta_mean/`, `tta_focal/`. The Kaggle `oof_eval` kernel (traps 28) was not
needed after all — the pod ran all four members in 6 min per pool with 503 GB RAM and identical numbers for v05a
(0.8621 on both), so the pod's c01 pull and preprocessing match Kaggle.

## Infrastructure

### 2026-08-30 — Cache v2 (`c02`), window-attention path, timm hybrids and mixed-geometry inference shipped; local verification ✅ KEEP the code · Kaggle ⏳

What changed (P-25 / P-26 / P-12 / P-23 #2 / P-24; details in the cards): a second cache scheme
(`c02`: 6 slots × budgets 18/12/12/14/8/8 = 72 slices, band 2–98 % all planes, 336 px, flat
`[72, 336, 336]` per study packed into 64-study blobs with CSV sidecars; version string
`c02_p336_b18-12-12-14-8-8_band2-98_crop130_lat20`), `window_mode="random"` + `head_type="window_attn"`,
`backbone="timm:<arch>"` loaded offline, `INFER_CACHE_KEYS` / `INFER_MEMBER_KEYS` replacing the single
geometry gate (one decode-once pass per cache geometry group; member settings applied per member),
`tta_offsets` / `tta_pool` / `INFER_OVERRIDES`, `MODE="oof_eval"`, env hooks for the RunPod runner.

**Verified locally (CPU, no GPU spent), 2026-08-30 11:40–12:40:**

| check | result |
|---|---|
| c02 build on the 3 sample studies (`RSNA_N_SHARDS=1 RSNA_CACHE_SCHEME=c02`) | 1 blob (3, 72, 336, 336) = 24,385,664 B + sidecar; resume-only rerun: "1 complete, 0 to build", manifest still written |
| c01 rebuilt with the new builder vs the 2026-08-28 local files | **byte-identical** arrays and `manifest_shard0.csv` |
| `src/cache_selftest.py` (builder vs `build_study_array`, both schemes) | **bit-identical** for all 3 studies × 2 schemes; slots, side, masks, version strings, offsets, blob rows equal |
| `src/window_head_test.py` | 60 / 84 windows; sampling never picks an absent slot, no repeats, ≥ 2 per present slot; head finite in fp16 with masks and all-masked rows; `param_groups` covers every parameter exactly once for dinov2 / convnext / coatnet-1 / coatnet-2; timm offline load 0 missing / 0 unexpected |
| Local smoke train (`MODE="train"`, arms `v08w` + `v09h`) | both trained 1 epoch on c02, checkpoints + OOF csvs written, per-arm inference through the c02 decode-once group "3 studies rebuilt, identical" |
| Pre-change code (`git show HEAD:src/kaggle_pipeline.py`) vs new code, infer of `v05a`+`v05b` | **identical `submission.csv`** (max abs diff 0.0) — the c01 members are untouched |
| Mixed-geometry infer `["v05a","v08w","v09h"]` | 2 geometry groups, both decode-once passes verified, 3 members blended by version |
| TTA `INFER_OVERRIDES={"v05a": (-1,0,1)/focal}` | output differs from plain `(0,)` (max abs 0.333 on 3 studies); `(0,)` is byte-identical |
| `MODE="oof_eval"` on `v05a` (c01, TTA) and `v08w` (c02, all windows) | `{v}_fold0_tta_oof.csv` written with the `pred__/y__/w__/is_gold` schema `blend_check.py` reads |
| `build_targets.py` | teacher 0.8948 unchanged |

Also shipped: four private Datasets — `rsna-knee-ckpt-v06` (`v06c_fold0_best.pt`), `rsna-knee-ckpt-v05g`
(five folds), `timm-coatnet-rmlp-1-rw-224`, `timm-coatnet-rmlp-2-rw-384` (HF repo files, Apache-2.0) —
so `rsna-knee-train` / `-folds` can be pushed again without repointing infer v10's mounts (the infer
metadata now lists the pins and drops `rsna-knee-folds` as a kernel source). Cache kernels
`rsna-knee-cache2-a..d` launched ~11:50 and **ran concurrently** (four CPU sessions at once — verified).
**Verdict: ✅ KEEP the code (every local rung green, old members byte-identical); the first real arm is ⏳.**

> **EXTENDED 2026-08-30 ~12:15 — Kaggle smoke `rsna-knee-train` v16 ✅ green** (FORCE_SMOKE, arms
> `v08w` + `v09h`, ~2 min GPU): both caches indexed on the T4 kernel — `cache: 4407 studies indexed
> (c01_…)` **and** `(c02_…)` from the six mounted shards; `manifest from cache: 4407 studies`; `v08w`
> trained + checkpointed (`auc_soft 0.5208` on 4 val studies — smoke arithmetic, not a result); `v09h`
> loaded `timm coatnet_rmlp_1_rw_224: 445 tensors from /kaggle/input/timm-coatnet-rmlp-1-rw-224 (dropped
> head 2)`, trained + checkpointed; per-arm inference through the c02 decode-once group "3 studies
> rebuilt, identical"; `submission.csv` validated; worker-RNG check unchanged. Smoke s/study (4 studies,
> 4 windows, warm-up included) is not a throughput measurement — the real `v08w` run returns it.
>
> **EXTENDED 2026-08-30 ~12:28 — Kaggle infer smoke `rsna-knee-infer` v11 ✅ green (NOT submitted):**
> `INFER_MEMBERS = ["v05a","v05b","v05g","v06c","v08w"]` with `INFER_OVERRIDES = {"v05a": (-1,0,1)/mean}` →
> 9 checkpoints (`v05a`/`v05b` from Dataset `rsna-knee-ckpt-v05`, `v05g` ×5 from `rsna-knee-ckpt-v05g`,
> `v06c` from `rsna-knee-ckpt-v06`, smoke `v08w` from train v16) in **2 geometry groups**, each decode-once
> pass "3 studies rebuilt, identical"; v05a with 3 TTA views took 91 s/100 studies vs 21 for a single-view
> concat member (so 3-view TTA ≈ +70 s per 100 studies per member at K = 6); by-version blend over 5
> versions; `submission.csv` validated; 0.03 h. The mixed-geometry path and the pins therefore work on
> Kaggle end to end; infer v10 (the scored 0.900 blend) is unaffected.

### 2026-08-30 — Cache v2 (`c02`) built: 4,407/4,407 studies, 70 blobs, 35.8 GB, 0 decode failures, 0 GPU h ✅ KEEP

`rsna-knee-cache2-a..d` v1 (CPU, 4 workers, `SHARD = 0..3`, `N_SHARDS = 4`), all four running
concurrently, launched ~11:50 and complete by ~12:10. Version `c02_p336_b18-12-12-14-8-8_band2-98_crop130_lat20`.

| shard | studies | blobs | GB | wall (build / total) | s/study (wall / CPU) | decode failures |
|---|---|---|---|---|---|---|
| a (0) | 1,064 | 17 | 8.65 | 11.4 / 13.8 min | 0.64 / 2.46 | 0 |
| b (1) | 1,164 | 19 | 9.46 | ~12 / 14.2 min | — | 0 |
| c (2) | 1,051 | 17 | 8.54 | 11.3 / 13.8 min | 0.65 / 2.48 | 0 |
| d (3) | 1,128 | 18 | 9.17 | ~15 / 17.8 min | — | 0 |
| **total** | **4,407** | **70** (+70 sidecars) | **35.8** | ~18 min wall for all four | | **0** |

Same corpus facts as c01 (mean slots 4.78, side resolved 96.9 %, 25 conflicts, FOV median 160 mm).
Per-study wall time equals the c01 build (0.6 s/study: decode-bound, 72 slices vs 96) although each study
is 1.7× more bytes. Four CPU kernels ran at once, so the whole rebuild cost **~20 min of wall clock and no
GPU quota**. `manifest_shard{k}_c02.csv` carries `blob`, `row`, `mask`, `cached`, `decode_fails`. ✅ the
build; whether the wider band pays is P-26's measurement (`v08w` fold 0 per label), still ⏳.

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

### 2026-08-29 — Cross-version rank blend + decode-once inference shipped (P-21 / P-18, kernel `rsna-knee-infer` v5) ✅ KEEP the code · ⏳ LB pending

`INFER_MEMBERS = ["v05a", "v05b"]`: in infer mode every mounted `{version}_fold*_best.pt` of every
listed version is one member of a flat rank-mean; a listed version with no checkpoint is a
`SystemExit`; members must agree on preprocessing geometry (read from each checkpoint's saved
config) and each builds its own head from its own config. Test studies are decoded **once**
(`build_study_array`, the cache builder's function) into the system temp dir and registered in
`CACHE_INDEX`, so every member reads the same array through the training code path; the kernel
rebuilds the first three studies and asserts array + mask equality before predicting.

Verified three ways before submission #5: local CPU run in train mode (new checkpoint code
exercised, decode-once verified), local CPU run in infer mode with the real v13 checkpoints
(`v05a/fold0=attn, v05b/fold0=concat`, geometry `2 -> 6` taken from the checkpoint), and kernel v5
on Kaggle:

```
infer members (2): v05a/fold0, v05b/fold0
infer heads: v05a/fold0=attn, v05b/fold0=concat
decode-once: 3 test studies -> /tmp/rsna_test_cache/c01_p224_s16_crop130_lat20 in 0.1 min
decode-once verified: 3 studies rebuilt, identical
v05a/fold0 (attn): predicted 3 studies in 2s (61 s per 100 studies) [epoch 7, score 0.9266]
v05b/fold0 (concat): predicted 3 studies in 1s (21 s per 100 studies) [epoch 7, score 0.8755]
```

Per-member inference is now ~21 s/100 studies once decoded (the 61 s includes CUDA warm-up), so a
7-member blend on ~1,300 hidden studies is ~20 min of decode + ~5 min per member instead of
~35 min per member. Submitted as **#5** (ref 55870514); the LB reads the OOF 0.8670 blend.
Decision rule written before the score: a sub-0.005 LB move is 🔁, and the +0.02–0.03 OOF→LB
offset was calibrated on single models, so ~0.89 is **not** a prediction.

**Also observed in this log, and it changes traps 6f:** `/kaggle/input` is now laid out
type-prefixed on the *old* `rsna-knee-infer` slug too — the train kernel's output sits at
`/kaggle/input/notebooks/tiankljucanin/rsna-knee-train/` (depth 3), the cache shards will sit at
`/kaggle/input/notebooks/tiankljucanin/rsna-knee-cache-{a,b}/`. Kernel v13 at ~10:00 the same day
still saw `/kaggle/input/rsna-knee-cache-a/`. So this was a platform-wide layout change during
2026-08-29, not a property of new slugs. Every kernel now prints the layout at startup.

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

> **EXTENDED 2026-08-29 — third point, submission #4.** `v04d` (slice jitter): OOF **0.8528** →
> LB **0.877**, offset **+0.024**. Three for three inside the +0.02–0.03 band (+0.020, +0.028,
> +0.024), and the prediction made *before* submitting (0.875–0.88) contained the result. The
> offset is now good enough to plan with, though still n=3 and all from one fold of one backbone
> — an ensemble may not sit on the same curve. Best is now **0.877**; the gap to the public top
> is 0.075.
>
> Note the asymmetry that makes the OOF metric worth having: the OOF delta (+0.0102) cleared its
> floor by 1.3×, while the LB delta (+0.006) cleared its floor by only 1.2×. **Neither is decisive
> alone** — what makes jitter a ✅ KEEP is the 11-of-12 per-label sign pattern, which no
> single-number comparison would have shown.

> **EXTENDED 2026-08-29 (evening) — fourth point, submission #5, and the first blend.** The P-21
> two-head rank-mean: OOF **0.8670** → LB **0.896**, offset **+0.029**. Four for four inside
> +0.02–0.03 (+0.020, +0.028, +0.024, +0.029). The earlier caveat — "an ensemble need not sit on
> the same curve" — did not bite: the blend sits on it. So the offset can now be used for blends
> too, with the standing warning that n=4 and every point is fold 0 of one backbone. The LB delta
> (+0.019) is 3.8× its floor and the OOF delta (+0.0096) 1.2× its floor — this time the LB moved
> *more* than the OOF predicted, the opposite asymmetry from jitter. Best is now **0.896**; the gap
> to the public top (0.952) is 0.056.


Log every submission here with the exact kernel version, config diff, OOF score,
and public LB score, so a public/private divergence can be traced to a specific change.

| # | Date | Kernel ver | Config change | OOF | Public LB | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-08-28 | rsna-knee-train v2 | v01 smoke (1 fold, 1 epoch, 2 slices/slot, rank targets) | n/a | **0.500** | exactly 0.500 = constant output at rerun; image-root assumption failed silently. Mechanics of submitting verified. |
| 2 | 2026-08-28 | rsna-knee-infer v1 (mounts rsna-knee-train v6) | v02 fold 0: DINOv2-S 224, 6 slices/slot, 4 ep, prob targets, LR 2e-5+LLRD, EMA | 0.821 | **0.841** | First non-0.500 score — the submission path works. Above the ~0.809 public DINOv2-S baseline on one fold. |
| 3 | 2026-08-28 | rsna-knee-infer v3 (mounts rsna-knee-train v8) | v03: same recipe, trained from the cache — adds 130 mm crop + laterality + per-series norm | 0.843 | **0.871** | **+0.030 vs #2**, 6× the 0.005 LB floor. The OOF gain transferred and amplified. Also the first submission through the fixed infer path (traps 6d). |
| 4 | 2026-08-29 | rsna-knee-infer v4 (mounts rsna-knee-train v11) | v04d: v03 recipe + `cache_jitter` slice augmentation | 0.8528 | **0.877** | +0.0113 OOF over `v04base` against a **measured** 0.008 floor; 11/12 labels up, none down; removes the epoch-2 overfitting turn. Predicted ~0.875–0.88 from the +0.02–0.03 OOF→LB offset. |
| 5 | 2026-08-29 | rsna-knee-infer v5 (mounts rsna-knee-train v13) | P-21: rank-mean of `v05a` (attn) + `v05b` (concat), fold 0, 8 ep, jitter, last-epoch checkpoints; decode-once inference | 0.8670 (blend; singles 0.8574 / 0.8471) | **0.896** | **+0.019 vs #4, 3.8× the 0.005 LB floor** — the largest single-step LB gain since the cache (P-01). First multi-model submission. The blend's OOF→LB offset is +0.029, inside the +0.02–0.03 band that was calibrated on single models, so the offset *did* transfer (n=4 now). Rule set before scoring was < 0.005 = 🔁; this clears it by a wide margin → ✅ P-21 KEEP. |
| 6 | 2026-08-29 | rsna-knee-infer v6 (mounts rsna-knee-train v13 + rsna-knee-folds v4) | `v05g` alone: 5-fold rank-mean, concat + jitter, 4 ep, `best_oof` (= last epoch on every fold) | per-fold 0.843–0.851, pooled 0.8467 | **0.886** | **Fold-ensemble gain = +0.009 over one fold (0.877), 1.8× the LB floor → ✅ folds pay on their own.** Note the OOF→LB offset is **+0.039** here, outside the +0.02–0.03 band: pooled OOF holds each study out once and cannot see the variance reduction of averaging five models, so **the offset rule does not apply to fold ensembles**. |
| 7 | 2026-08-29 | rsna-knee-infer v8 (same mounts) | P-21 **by-version** blend: `v05a` attn + `v05b` concat-8ep + `v05g` 5-fold concat, one vote per version (`INFER_BLEND="by_version"`) | fold-0 proxy 0.8680 (a+b+g flat-by-version; the flat 7-checkpoint mean would be 0.8611) | **0.896** | **Identical to #5 → 🔁 folds add nothing on top of head diversity** (the fold-0 proxy predicted +0.001). Two things changed vs #5 at once — five concat folds were added *and* the attention head's vote fell from 1/2 to 1/3 — so "zero" may be a small gain cancelled by a small dilution; not worth a submission to disentangle (weight-tuning on the public LB is the documented trap). Infer v7 (flat 7-member) exists but was deliberately not submitted. |
| 8 | 2026-08-30 | rsna-knee-infer v10 (mounts rsna-knee-train v15 = `v06c`, rsna-knee-folds v4 = `v05g`, Dataset `rsna-knee-ckpt-v05` = `v05a`/`v05b`, Dataset `convnext-tiny-224-hf`) | **P-23**: by-version blend `v05a` attn + `v05b` concat + `v05g` 5-fold + **`v06c` ConvNeXt-T** (first non-DINOv2 member; 8 checkpoints, 4 votes). infer v9 died on the missing ConvNeXt mount (traps 22) | fold-0 proxy 0.8722 (vs 0.8680 for #7; v06c fails the P-23 rule narrowly: ρ 0.848, +0.0043 on this base, 10/12 labels up) | **0.900** | **+0.004 vs 0.896 → 🔁 INCONCLUSIVE by the pre-registered rule** (0.8× the 0.005 LB floor), although it is the best number on the board and lands exactly where fold-0 OOF predicted (0.8722 + the usual +0.028 offset = 0.900; n=5 for the offset now). Read: ConvNeXt-T is a real but *head-like* member — it adds what a third head would, not what a new family should. Default blend going forward = this 4-version one (more members, same score band, safer on private). Next member must change the input geometry (P-23 #2), not the backbone alone |
