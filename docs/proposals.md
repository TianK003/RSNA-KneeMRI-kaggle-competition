# Proposals — ranked, testable cards

*Written 2026-08-28. The forward-looking half of the lab notebook: every idea we intend to
test, written as a falsifiable card **before** it is run. Companion to
[experiments.md](experiments.md) (things measured, with verdicts) and
[research.md](research.md) (the evidence base these cards draw on).*

## How a card moves

1. An idea enters here as `💡 untested`. It needs a hypothesis, a measure, a noise floor and a
   cost — if it cannot be written in the template it is not ready to run.
2. When code ships it becomes `🔧 implemented, effect pending` (the change exists in
   `src/`, nothing has been measured yet).
3. While a run is live it is `⏳ running`.
4. When the number comes back the result goes to **experiments.md** (append-only) with a
   verdict — ✅ KEEP / ❌ DEAD END / 🔁 INCONCLUSIVE — and the card here is reduced to a
   pointer. Untried ideas never go to experiments.md; measured ones never stay here.

## Decision-metric hierarchy

| Metric | Set | Floor (a smaller delta is noise) | Used for |
|---|---|---|---|
| **OOF macro-AUC vs teacher** | all 4,407 studies (fold-out) | **macro 0.008 — MEASURED** 2026-08-29 (kernel v11, `v04base` seed 42 vs `v04a` seed 43, identical config; abs delta 0.004–0.008 over four epochs). **Per label use ~0.03**, not the 0.015–0.02 this file assumed: the same two runs move Fracture 0.028 and Contusion 0.025 on seed alone. | breakage, epoch choice, every A/B |
| Gold macro-AUC | 58 labelled studies | **0.05** macro (CLAUDE.md rule); per-label Hanley–McNeil SE ≈ 0.09 ([Andre et al.](https://arxiv.org/html/2601.17103)). A 3,000-rep study-level bootstrap of the teacher's gold macro-AUC gives SD 0.017. | direction only, never a gate |
| Public LB | hidden test (size unknown) | **0.005** (top ten span 0.006) | direction only; never tuned on |

Judge label changes on **coverage per language + OOF**, not gold alone (experiments.md
convention). Per-label OOF floors are wider than the macro floor: **~0.03, measured** 2026-08-29.

## Card template

```
### P-nn Title
Status:       💡 untested / 🔧 implemented, effect pending / ⏳ running / ✅|❌|🔁 see experiments.md
Hypothesis:   one falsifiable sentence
Origin:       peer-reviewed / competition write-up / public consensus / our hypothesis / verified in our code+data
Evidence:     numbers + [title](url); notebook-only sources marked "(notebook, not re-read)"
Measure:      exact metric + set
Noise floor:  ...
Cost:         sessions / lines
If it works:  ...
If it fails:  ...
Depends on:   ...
```

---

## Ranked index

Expected value is a judgement of *how much macro-AUC, or how much validity of every later
result*, per unit of cost. "Depends on" lists hard blockers only.

| id | title | status | expected value | cost | depends on |
|---|---|---|---|---|---|
| P-00 | Target scale: probability-space blend, not rank percentiles | 🔧 implemented, effect pending | high (rare-label targets were 0.28–0.39 for confident negatives) | done; OOF read pending | — |
| P-01 | Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised) | ✅ MEASURED — see experiments.md (5.4× end to end, OOF 0.821→0.843, **LB 0.841→0.871**) | very high — delivered speed *and* the largest scoring gain so far | done | — |
| P-02 | Seed-noise baseline, then site-grouped folds + grouped-vs-random OOF | ✅ step 1 DONE — **floor = 0.008 macro / ~0.03 per label** · step 2 💡 | delivered: every A/B in the repo now has a real floor | step 2 ~40 lines | P-01 header manifest |
| P-03 | Fine-tuning recipe: (a) LR 2e-5 + EMA shipped; (b) LLRD 0.75 vs uniform | (a) 🔧 shipped, effect pending · (b) 💡 | medium | (a) done · (b) 0.3 session | P-01 for (b) |
| P-04 | Fixed-epoch schedule, 8 epochs, chosen from fold-mean OOF curve | 🔁 8 ep does not beat 4 for concat; **epoch count is head-specific** (attn 8, concat 4) | answered | done | P-01, P-03 |
| P-05 | Laterality normalisation from DICOM geometry | ✅ CONFIRMED — **−0.0147 OOF when removed**, ~1.9× the floor | high — it is ≈ +0.015 of v03's +0.022, i.e. most of the LB gain | done | P-01 |
| P-06 | Per-label failure analysis + gold_weight {1,3,8} arm + slot-fill census | 🔧 logging shipped · arms 💡 | high (finds the cheapest lever per weak label) | 0.1–0.5 session | — |
| P-07 | Synovitis ← Effusion back-fill (measured, not adopted) | 🔁 measured on gold; OOF pending | low-medium | done (audit) | P-06 OOF |
| P-08 | Slices per slot 6 → 12–16, per-plane bands, random offsets | ✅ jitter sub-arm KEEP (**+0.0113 OOF, LB 0.877**) · K sweep **withdrawn 2026-08-30 (afternoon)**: re-presenting the same 16 band-truncated slices is superseded by the c02 cache (P-26: the *band* and per-slot budgets change) + random-window training (P-25: every window is seen across epochs) | delivered by jitter | — | P-01 |
| P-09 | Per-label masked attention head over slots | ✅ KEEP — **+0.0103** at matched 8 ep; mechanism is overfit-resistance, **not** the predicted plane-specialisation | delivered | done | P-01 |
| P-10 | Second architecture family (HF ConvNeXt-Tiny first; RadImageNet behind a flag) | 🔁 **MEASURED 2026-08-30**: `v06c` own OOF 0.8562 (parity), ρ 0.831, blend +0.0059 (10/12 labels up) — narrow reject by the rule; #8 scored **0.900 (+0.004, 🔁)**; see experiments.md · was: 🔧 implemented, effect pending — arm `v06c` (ConvNeXt-T, concat, jitter, 8 ep, fold 0); smoke pushed 2026-08-29 late; **real run `v06c` = train v15 in flight 2026-08-30** · the 0.936 notebook's own 45-gold panel ranks ConvNeXt-B/L among its *weakest* families (0.875 vs CoAtNet-384 0.9025): `v06c` is a diversity bet — judge it on ρ, not own OOF (research.md §2.7.1) | **high — first P-23 candidate**; judge on ρ < 0.77 vs both heads **and** 3-way fold-0 blend > 0.8670 + 0.008 | ~1.8 h fold-0 arm | P-01, P-04 |
| P-11 | Resolution 224 vs 336 after the 130 mm crop | 💡 untested · **raised 2026-08-30**: every branch of the 0.936 notebook runs at 336–384 px and its 0.924 single model is 384 px × 64 slices (research.md §2.7.1) | **medium** (was low-medium) — a 336/384 many-slice hybrid is P-23 candidate #2 | 1 session + sharded cache | P-01, P-08 |
| P-12 | Slice-window TTA (label-safe only) [our hypothesis] | 🔁 **measured 2026-08-30 (pod `oof_eval`)**: each c01 member +0.003–0.006 alone, the 4-member blend +0.0016, the 7-member blend +0.0002 — redundant with ensembling; `INFER_OVERRIDES` stays empty (experiments.md 2026-08-30 P-12) | low-medium — cheapest item on the list | 0.5 h GPU to measure | P-01 |
| P-13 | 3 vs 5 folds under a fixed session budget [our hypothesis] | 💡 untested, but **directly supported 2026-08-29**: 5 folds alone +0.009 LB, 5 folds on top of a second head **+0.000** — diversity beats replicates (experiments.md, "First valid 5-fold run", RESOLVED note) | **raised — the next GPU spend is a diverse member, not folds** | 1–2 sessions | P-10 |
| P-14 | DINOv2-S vs DINOv2-B (registers variant noted) | 💡 untested | low | 1 session | P-10 |
| P-15 | DINOv3-S/16 as diversity member | 💡 DINOv3 half untested · 16-channel re-scope **❌ run as `v07s` 2026-08-30, own OOF 0.74, dead as built** (experiments.md) · **retry spec from the cell-level re-read (afternoon)**: their 16-ch member runs at **336 px**, per-slice [1, 99] normalisation, `DepthCompress` (gated 1×1 depth blocks → 3 ch) or `SlotDepthMixer` (5-tap learned depth smoothing per plane) stem — *not* a linear 16-ch patch embed — with `xattn` (12 label queries over all patch tokens of all slots) readout; a retry = non-linear stem at ≥ 1e-3, 12+ epochs, on the c02 cache (16 → 18/14 slices per fluid slot) | medium as P-23 candidate #3 | ~2.5 h fold 0 (6 forwards/study) | P-26 |
| P-16 | Re-labelling with an open-weights LLM inside Kaggle (graded, native language) | 💡 untested | high but slow (raises the teacher ceiling) | 1–2 sessions | P-06 audit (done) |
| P-17 | Noise-robust loss / self-distillation | 💡 untested | low | 0.3 session each | P-01, P-04, P-06 |
| P-18 | Efficiency-track variant + decode-once inference + slot census | 🔧 infer mode + loud-failure submission shipped · **decode-once shipped 2026-08-29 (infer v5, equality-checked)** · variant 💡 | medium (separate prize) | 0.2 session + browser | P-01 |
| P-19 | Decoder wheels + TransferSyntax census | 💡 untested | insurance | 0.1 session | P-01 header pass |
| P-20 | Leave-one-slot-out ablation, T1 slot retirement | 💡 untested | low-medium | 0.1 session | first real model |
| P-21 | Blend two heads on one backbone as the default ensemble axis | ✅ **KEEP — LB 0.896 (+0.019, 3.8× floor), submission #5**; OOF 0.8670. Head diversity is now the default ensemble axis; see experiments.md | delivered (`INFER_MEMBERS`) | done | P-09 (done) |
| P-22 | Checkpoint selection on OOF-vs-teacher instead of fixed last epoch | ✅ **MEASURED, policy switched** — +0.0128 split-half for concat, ~0 for attn, gold flat; P-09 becomes a tie. See experiments.md | delivered (`ckpt_policy="best_oof"`) | done | — |
| P-23 | **Multi-family rank fusion — the ensemble is the product, not a member** | ✅ **#2b `v10c` measured 2026-08-30 (RunPod, 2.9 h): OOF 0.8641 = parity with `v08w`, lowest ρ of any member (0.73–0.81), meniscus specialist (LatMen 0.858 / MedMen 0.922); 6-member blend 0.8795 = +0.0073 over the LB blend** (experiments.md 2026-08-30 `v10c`); **#2a `v09h` (CoAtNet-1 @224, 50 min) = 0.8683, the best single**; 7-member blend 0.8820 (+0.0024, 🔁) — the three c02/window-attn arms are one family (ρ 0.84); **next: 5-fold `v09h` (~4 h, ≈ $3 — 384 px bought nothing, so no 12-epoch `v10c`)**, then a new *input representation* (#3/#4) or P-17 self-training on the multi-family OOF; the ⭐ synthesis is experiments.md 2026-08-30 "What made the 0.936 notebook good" · earlier: two candidates measured 2026-08-30: `v06c` strong-but-head-like → **#8 LB 0.900 (+0.004, 🔁)**, `v07s` 16-ch ❌ dead as built; **raised to #1 on 2026-08-30** (research.md §2.7.1) · **afternoon: candidate #2 implemented** — `v09h` (`timm:coatnet_rmlp_1_rw_224` probe, RunPod ~4 h) and `v10c` (`timm:coatnet_rmlp_2_rw_384`, 64-slice c02 cache, per-label window attention, eval 42 windows, RunPod 6–8 h fold 0) — the 0.936 notebook's 0.924 recipe; #2a (K=12) withdrawn; mixed-geometry inference (c01 + c02 members in one blend) shipped | **very high — the only axis with evidence of +0.03**; per-member acceptance rule: own OOF ≥ best − 0.02, ρ < 0.80 vs the blend, blend gain > 0.008 | 1–2 h fold-0 arm per candidate (T4); hybrids on RunPod | P-25, P-26, P-24 |
| P-24 | **Compute expansion: off-Kaggle runner (RunPod) + optional 2×T4 `DataParallel`** | 🚀 **running 2026-08-30 12:52 →** secure 4090 pod `2wend9j0lr7zf3` (EUR-IS-2, $0.74/h): `setup` → P-12 `oof_eval` mean + focal → `train v10c` → `ship` → `train v09h` → `ship` (see handoff); 🔧 runner built 2026-08-30 (`scripts/runpod_bootstrap.sh`, `requirements-gpu.txt`, `RSNA_ARM` / `RSNA_TRAIN_ONLY` / `RSNA_WORKERS` / `RSNA_RUNTIME_H`; Tian chose RunPod over Colab — paid per hour, no idle disconnects); 2×T4 half 💡 | **high** — the hybrids (P-23 #2b) run only there | (b) done · (a) 30 lines + 1 arm | P-26 (c02 blobs are what it pulls) |
| P-25 | **Window-attention head + random-window training** (12 label queries over every (slot, window) token; no label-agnostic per-slot pool) | ✅ **measured 2026-08-30 (train v17)**: `v08w` fold-0 OOF **0.8648** = best single model (v05a 0.8574; 12/12 labels up); blend +0.0044 as a fifth member 🔁 (ρ 0.866 → REJECT as extra member, KEEP as the member recipe) — experiments.md 2026-08-30 `v08w` | **high** — the 0.936 notebook's strongest member pools this way (≈ +0.005 est.); also the natural home for TTA (all windows at eval) | ~2 h fold-0 arm | P-26 |
| P-26 | **Cache v2 (`c02`): band 2–98 %, ragged budgets 18/12/12/14/8/8, 336 px, 64-study blobs** | 🔧 **built 2026-08-30 ~12:10** — 4,407/4,407 studies, 70 blobs, 35.8 GB, 0 decode failures, ~20 min wall, 0 GPU h (experiments.md "Cache v2 built"); **effect measured on `v08w` fold 0: MCL +0.028 vs v05a (claim holds), Lateral Meniscus +0.009 (does not); loader 0.12 s/study vs 0.19** — experiments.md 2026-08-30 `v08w` | **high** — MCL 0.836 / Lateral Meniscus 0.833 are our two weakest labels and the ones the discarded outer slices carry | 0 GPU h, ~1 h CPU wall | P-01 |

---

## Cards

### P-00 Target scale: probability-space blend, not rank percentiles
Status: 🔧 implemented (2026-08-28, `src/build_targets.py` + notebook cell), student effect PENDING on OOF.
Hypothesis: Feeding rank-percentile blends (average-rank ties) into BCE trains the network toward ~0.3–0.4 for clean negatives on rare labels; a mean-of-probabilities blend restores near-zero negatives and lifts per-label OOF and prediction spread on MCL, Lateral OA, Baker's, Fracture.
Origin: verified in our code+data (critic item 1).
Evidence:
- Old blend: **no study on any label (0%)** had a target < 0.1 before the gold override; confident negatives sat at **0.28–0.39** (MCL p25 = 0.312) while gold rows were hard 0/1 at 8× weight. Measured today on `artifacts/targets.csv`.
- New mean-of-probabilities blend: **2–72%** of studies < 0.1 per label (Synovitis 2%, Baker's 26%, MCL 72%). Teacher gold macro-AUC **0.8948** (rank blend 0.8934) — Δ 0.0014 is inside the noise floor; rank order of the teacher essentially unchanged, as expected.
- `confidence_weights()` already used raw probabilities, so it needed no change ([research.md §2.7](research.md)).
- Community warning that rank blending destroys 0.5 semantics: [stevenleehans label card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels).
Measure: per-label OOF AUC vs teacher and per-label `pred_std` over the fold-0 OOF csv (v02) — compared against a v01-target fold when one exists; until then, sanity = pred_std healthy and no label at chance.
Noise floor: **0.008 OOF macro, ~0.03 per label** (measured 2026-08-29; was assumed 0.01 / 0.015–0.02). There is no v01 baseline fold, so the first read is a *sanity* read, not an A/B.
Cost: done. A/B would cost one extra fold-0 run from the cache (~0.3 session).
If it works: rare-label OOF up and negatives near 0; card closes in experiments.md as ✅.
If it fails: (OOF or pred_std no better) blend was never the bottleneck; keep the probability blend anyway as the semantically correct target and move on.
Depends on: nothing.

### P-01 Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised)
Status: ✅ **MEASURED — card closed, see [experiments.md](experiments.md)** (cache build; v03 fold-0
run kernel v8; submission #3). Headline: **5.4× end to end** (not the ~60× decode arithmetic
suggested — the T4 is the bottleneck now), OOF-vs-teacher 0.821 → **0.843**, public LB 0.841 →
**0.871**. ⚠️ The card's "OOF within ±0.01 = faithful speed-up" measure was **wrong on its own
terms**: it assumed v03 replayed v02's inputs, and v02 had no crop and no laterality at all, so
the run confounds cache + crop + per-series norm + laterality. P-05's `lat_undo` arm splits it.
Historical card text below.
Status (original): ⏳ running — `src/cache_pipeline.py` shipped; shards A/B running on Kaggle. As built: **K=16 slices/slot, 224 px, 2 shards** (≈21 GB total, ~10.6 GB per shard); sagittal stacks sorted along +x (patient left) with a fixed sign and reversed for right knees; coronal/axial mirrored when (columns run to +x) == is_right; failed slices replaced by the nearest good neighbour (not zeros); presence mask written into `manifest_shard{k}.csv`; ordering drops unreadable files instead of falling back to filename order.
Hypothesis: A one-off uint8 cache of all 4,407 studies removes the per-epoch DICOM decode, cutting an epoch from hours to minutes without changing OOF-vs-teacher.
Origin: public consensus / competition write-up.
Evidence:
- 11.1 GB in-RAM uint8 cache built in ~16 min, then **103 s/epoch** over ~3,300 studies (verified log, [hida1211](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4)); ~1–1.5 h on 4 CPU procs (verified [PLATFORM.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)).
- Every RSNA winner 2019–2024 trained from pre-extracted arrays ([Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [gunesevitan](https://github.com/gunesevitan/rsna-2023-abdominal-trauma-detection)).
- Our own ~4.5 s/study decode is an **extrapolation from 8 passes**, not a measurement (experiments.md); v02 now logs s/study so the real number arrives with fold 0 (critic item 38).
- Size: as built, K=16 @ 224 uint8 = 4,407 × 6 × 16 × 224² ≈ **21 GB → two shards of ~10.6 GB** under the ~20 GB per-kernel output cap (critic item 10; the K=12 ≈ 15.9 GB figure was the planning estimate). K=12 @ 336 ≈ 36 GB — **do not cache at ≥ 336 without sharding across kernels.** `np.load(mmap_mode="r")` works on `.npy` only, silently ignored for `.npz` ([NumPy #5976](https://github.com/numpy/numpy/issues/5976)).
- Normalisation per series 1/99 → uint8, RescaleSlope/Intercept + MONOCHROME1 first ([Innolitics](https://innolitics.com/articles/medical-imaging-best-practices/), [MONAI #282](https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/282)); 130 mm centre crop inside the FOV of 99.57% of series (verified [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)).
Measure: wall-clock per epoch and per fold; OOF macro vs teacher on fold 0 from cache vs the decode-every-epoch v02 fold 0 — **with the same normalisation function on both train and val** (critic item 15), otherwise the comparison is confounded. Decode failures counted and excluded, never zero-filled (critic item 21).
Noise floor: engineering; sanity tolerance ±0.01 OOF.
Cost: 1 session (CPU kernel), ~300 lines (`src/cache_pipeline.py`, header pass, decode pass, manifest with site-proxy fields, loader). Version string encodes (K, px, crop, laterality rule). Never write an empty cache (traps #9).
If it works: P-04, P-08, P-09, P-10, P-11, P-12 become affordable; inference reuses the identical, fingerprinted normalisation.
If it fails: (cap or time) fall back to K=6 @ 224 (7.4 GB, one shard); nothing downstream proceeds.
Depends on: nothing; blocks P-02 (manifest) through P-15.

### P-02 Seed-noise baseline, then site-grouped folds and grouped-vs-random OOF
Status: **step 1 ✅ DONE — see [experiments.md](experiments.md)**: the floor is **0.008 macro,
~0.03 per label** (`v04base` vs `v04a`, identical config). The asserted 0.01 was close, so no
earlier verdict is overturned; the per-label assumption was too optimistic and every card has
been updated. Step 2 (site-grouped folds) still 💡 untested and is now the card's whole content.
Status (original): step 1 ⏳ RUNNING (kernel v11: `v04base` seed 42 vs `v04a` seed 43, identical code and
config). `|v04a − v04base|` becomes the measured OOF floor that replaces the asserted 0.01 in the
decision-metric table. Step 2 (site-grouped folds) still 💡 untested.
Hypothesis: (1) Two seeds of the same fold-0 config differ by an OOF macro that defines our real noise floor; (2) report-text-only grouping lets the model memorise scanner/site signatures, inflating OOF and favouring higher-capacity variants.
Origin: competition write-up / peer-reviewed; the seed baseline is our hypothesis about magnitude.
Evidence:
- **Step 1 (seed noise):** no card in the research measured run-to-run variance; the 0.01 floor is asserted (critic item 16). Seed effects alone move ImageNet-scale results measurably ([Picard 2021](https://arxiv.org/abs/2109.08203)).
- Grouped **0.7049** vs random **0.8412** OOF on a ResNet34 on this corpus, gap growing every epoch (verified [EXPERIMENTS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); language ≈ vendor (Cyrillic 100% Philips; Dutch/German/Greek 100% Siemens).
- Knee-specific external drop **0.05 ± 0.03** ([Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-12052-8)); the −0.097 MRI figure in [Guo et al.](https://arxiv.org/pdf/2409.04368) is prostate (critic item 31).
- Our audit: 9 languages, en 39% / es 15% / tr 12% / hr 9% / el 7% / de 6% / bg 5% / nl 3% / fr 2% — language is available for grouping today.
- **Hidden-test site mix (critic item 13):** if the test set is drawn from the same sites, site-grouped CV is *pessimistic*. Log the header-derived site distribution of the test studies at rerun (Manufacturer/Model/language-of-nothing — pixels only) next to train, and choose grouping strictness from it.
Measure: (1) |OOF_seedA − OOF_seedB| on fold 0, same config → recorded as the measured floor; (2) grouped vs random-split OOF macro over 4,407 plus per-site OOF; the gap is the deliverable.
Noise floor: (1) defines it; (2) a gap > 0.02 confirms inflation.
Cost: 0.5 session total (one extra fold-0 run; ~40 lines in `build_targets.py` for `site_key` from the manifest, language-only fallback).
If it works: honest model comparison; predicts private-LB behaviour; site-level OOF shows whether P-05/augmentation shrink per-site spread.
If it fails: (headers anonymised — no Manufacturer) group on language + report hash and note it.
Depends on: P-01 for the header manifest; the seed baseline needs only the cache for speed.

### P-03 Fine-tuning recipe — two arms
Status: **(a) LR 2e-5 + EMA + wd 0.02**: 🔧 shipped as the v02 baseline, effect PENDING on fold 0. **(b) LLRD 0.75 vs uniform 2e-5**: 💡 untested (LLRD 0.75 also shipped in v02, so the arm to run is the *uniform* control). Head-freeze warm-start: 💡, blog-grade, lowest priority. Checkpointing as shipped: `{version}_fold{k}_best.pt` = EMA weights after the last completed epoch (fixed-epoch selection; per-epoch score logged only); OOF written every epoch (`_ep{e}_oof.csv`) plus `_oof.csv` for the checkpointed epoch; no weight decay on bias/LayerNorm or on cls/mask/pos-embedding tokens.
Hypothesis: (a) 2e-5 peak with EMA is at least as good and more stable than 5e-5 uniform; (b) layer-wise decay 0.75 (block 0 at 0.75¹² ≈ 0.03× top) protects SSL low-level features enough to show in OOF.
Origin: peer-reviewed (a); MAE/BEiT convention with no medical ablation (b).
Evidence:
- Medical DINOv2 sweeps land at 1e-5 ([MedMNIST FM](https://arxiv.org/html/2501.14685v1)), ≤ 2e-5 ([DINOv2 radiology](https://arxiv.org/html/2312.02366v3)), 1e-6 at batch 2 ([MST](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)); lr 1e-3 collapse ([dinov2 #276](https://github.com/facebookresearch/dinov2/issues/276)). Public knee recipe 8e-6 @ batch 8, wd 0.02 ([pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)).
- LLRD 0.75 / wd 0.05 / 5% warmup is the MAE fine-tune recipe ([MAE](https://arxiv.org/html/2111.06377), [BEiT](https://arxiv.org/abs/2106.08254)); uniform LR framed as catastrophic forgetting ([2606.25989](https://arxiv.org/html/2606.25989)) — no medical A/B exists.
- EMA is more robust to label noise ([2411.18704](https://arxiv.org/html/2411.18704)); community uses 0.997–0.999 (hida1211; ISIC note +0.0015 pAUC, [zenn](https://zenn.dev/morim34/articles/bfa2465defee06)).
- **Already present before today** (critic item 2): 10% linear warmup + cosine, grad-clip 1.0 after `scaler.unscale_`, head dropout 0.1. Shipped today: `lr_backbone` 2e-5 with LLRD 0.75 (range 4.75e-7…2e-5), `weight_decay` 0.02 with none on bias/LayerNorm, EMA 0.998 validated and saved.
- Caveat (critic item 4): our effective batch is 1 study × accum 4 with ~30–36 slices through a shared encoder; the cited sweeps used batch 8–64. Batch 8 becomes possible only from the cache.
Measure: (a) fold-0 OOF macro vs teacher + per-epoch curve smoothness, gold reported not gated; (b) same config with `layer_decay=1.0` on fold 0 from cache.
Noise floor: 0.01 OOF; (b) is expected to be 🔁 INCONCLUSIVE — say so up front.
Cost: (a) done; (b) 0.3 session (one fold-0 run).
If it works: (a) the fixed-epoch selection already in v02 is confirmed safe; (b) LLRD stays.
If it fails: (a) OOF < what v01-config would have given is unknowable without a v01 fold — if pred_std collapses or a label sits at chance, sweep {1e-5, 2e-5, 5e-5} on fold 0; (b) uniform wins → drop LLRD, one fewer knob.
Depends on: (b) P-01.

### P-04 Fixed-epoch schedule (8 epochs) chosen from the fold-mean OOF curve
Status: 🔁 **INCONCLUSIVE — 8 epochs does not beat 4**, see [experiments.md](experiments.md).
`v05b` (concat, 8 ep) ends at 0.8471 vs `v04d` (concat, 4 ep) at 0.8528 — inside the floor, no
gain and possibly a small loss; the concat head overfits past epoch 4 even with jitter. **The
epoch count is not a global setting**: the attention head needs 8 (still climbing at 3), the
concat head wants 4. Tie the schedule to the head, not to the project.
Status (original): ⏳ RUNNING (kernel v13, `v05b`). Reopened by v11: without augmentation the OOF curve
peaked at epoch 2 and declined, which argued *against* more epochs; **with jitter it rises
monotonically and had not peaked at epoch 3** (0.8492 → 0.8528). So the question is no longer
"more epochs?" but "more epochs now that a regulariser exists?" — judged against `v04d`'s 0.8528.
Status (original): 💡 untested for the epoch *count*. Fixed-epoch *selection* is already what v02 does (`best.pt` = EMA weights after the last completed epoch, no best-epoch pick on gold); the per-epoch OOF csvs it writes are the input to this card.
Hypothesis: 8 epochs with cosine + EMA and one fixed epoch count for all folds beats 4 epochs on OOF without memorising teacher errors.
Origin: competition write-up / peer-reviewed.
Evidence: winners 10–40 epochs ([TheoViel](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma): maxvit 40 ep @ 4e-5, coatnet 20 ep @ 2e-5); public knee recipes 10–12 epochs ([pilkwang], [hida1211](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4)); label-noise training has a memorisation phase to stop before ([Label Wave](https://arxiv.org/html/2502.07551v1)); checkpoint averaging from one init matches the best checkpoint ([Model soups](https://arxiv.org/html/2203.05482)). Best-epoch on ~12 gold per fold is a coin flip (experiments.md).
Measure: per-epoch OOF-vs-teacher macro and per-label AUC over 4,407, fold-averaged; gold curve alongside to detect divergence (gold_weight bias — see P-06).
Noise floor: 0.01 OOF; gold 0.05.
Cost: 1 session (5 folds × 8 epochs from cache; per-epoch time for a fully unfrozen ViT-S to be measured).
If it works: one fixed epoch count for all folds; a converged model for P-06.
If it fails: (OOF peaks at 3–4) keep 4 epochs, spend budget on slices/members.
Depends on: P-01, P-03(a).

### P-05 Laterality normalisation from DICOM geometry
Status: ✅ **CONFIRMED — card closed, see [experiments.md](experiments.md)**. `lat_undo` costs
**−0.0147 OOF, ~1.9× the 0.008 floor**, and the cost lands on exactly the side-specific/focal
labels (Baker's −0.044, MCL −0.033, Medial Meniscus −0.032) while fluid labels are unmoved. This
also **resolves the v03 confound**: of v03's +0.022, laterality is ≈ +0.015 and the 130 mm crop
plus per-series normalisation ≈ +0.007 *together* — inside the floor, so still unproven.
Status (original): ⏳ ablation RUNNING (kernel v11, arm `v04b`, `lat_undo=True`). Rather than rebuilding a
21 GB cache, the arm re-applies the cache's own transforms to right knees at load time — both are
involutions, so it de-canonicalises from the existing cache for free (verified a clean involution,
no NumPy aliasing; fired on **2,288/4,407 studies = 51.9%** on Kaggle). This is a *cleaner* test
than v03-vs-v02 because the 130 mm crop is held constant instead of varying alongside. Indirect
evidence it will bite: in v8 every per-label gain was a side-specific or focal finding (Lateral
Meniscus +0.07, Medial Meniscus +0.05, MCL/Lateral OA/ACL +0.03) while Effusion, Synovitis,
Baker's and Fracture were flat or slightly down. As built: sagittal stacks sorted along +x (patient left) with a fixed sign and reversed for right knees; coronal/axial mirrored when (columns run to +x) == is_right.
Hypothesis: Canonicalising knee side (mirror W on COR/AX, reverse SAG slice order) removes a chirality the model must otherwise learn twice and lifts Medial/Lateral Meniscus, Medial/Lateral OA and MCL on OOF.
Origin: public consensus / competition write-up.
Evidence: Laterality tag missing on **12,367/24,371** series; centre-x sign rule **97.4%** (98.5% with 20 mm dead zone); IPP-corner rule 58.8% (verified [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); plane-specific operation ([pilkwang] source, pulled; [JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)); laterality metadata errors are common ([PMC6646614](https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/)); lateral meniscus ~0.1 harder than medial even at 18k studies ([Fritz](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)).
Rule (critic item 8): side = tag, else sign(median image-centre x) with 20 mm dead zone; **tag-vs-geometry conflict → no-op and counted**; unresolved → no-op and counted; `ImageOrientationPatient` row/column cosines canonicalised for all planes (sagittal reversal fixes stack direction only).
Measure: OOF AUC over 4,407 for the five side-specific labels before/after; tag-vs-geometry agreement on tagged series (assert ≥ 0.95); fraction unresolved and fraction conflicting; visual spot-check of ~20 knees.
Noise floor: **~0.03 per label OOF** (measured 2026-08-29); gold cannot resolve it. Judge on the macro plus the *sign pattern* across the side-specific labels, not one label.
Cost: 0.3 session, ~80 lines in the cache builder (the cache version string encodes the rule).
If it works: baked into the cache. **H-flip does not become legal** — see Rejected (critic item 23).
If it fails: (no OOF change) keep as anatomically correct; investigate the GE sub-sample where the rule is reported unreliable.
Depends on: P-01.

### P-06 Per-label failure analysis, gold_weight arm, slot-fill census
Status: 🔧 logging shipped in v02 (per-label AUC table, OOF csv per fold **every epoch** — `_ep{e}_oof.csv` plus `_oof.csv` for the checkpointed epoch — 2,000-rep bootstrap CI on gold, throughput); gold_weight arm and census 💡.
Hypothesis: One label at chance costs ~0.029 macro; per-label OOF AUC, pred_std and gold FN/FP composition identify the cheapest lever per weak label; gold_weight 8 may bias both the gold curve and the fold-mean OOF curve.
Origin: our hypothesis; peer-reviewed for the difficulty ordering.
Evidence:
- Literature difficulty order ACL < medial meniscus < OA/effusion < lateral meniscus ≈ BME < synovitis ([Eur Radiol 2024 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12021734/), [Astuto](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/), [Fritz], [synovitis DL](https://pubmed.ncbi.nlm.nih.gov/37951778/)) matches our teacher (Synovitis 0.788, Lateral OA 0.804, Fracture 0.825; experiments.md).
- Label collapse to base rate observed publicly ([JunhaoLiXD V02](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)).
- **Audit today:** prevalence gaps gold vs blend ≥ 0.5 — Synovitis 47% vs 13%, Lateral Meniscus 40% vs 16%, Fracture 31% vs 7%, ACL 41% vs 21%; silence (pilkwang UNK) Synovitis 84%, Fracture 56%, Baker's 46%, Lateral OA 33%. Co-occurrence φ (gold / weak; n=58, SE of gold φ ≈ 0.13): Effusion~Synovitis 0.40 / 0.28, Medial OA~Medial Meniscus 0.42 / 0.36, Contusion~Fracture 0.33 / 0.28, Medial OA~Lateral OA 0.32 / 0.49 (relevant to P-09's risk).
- **gold_weight (critic item 5):** gold is 1.3% of studies but ~10% of loss mass at 8×; with 8-epoch runs the gold curve risks becoming a training-set curve for the *other* folds' gold. Community uses 3–8× (hida1211, JunhaoLiXD); LP-FT literature says gold should be selection-only ([BoxWRENCH](https://arxiv.org/html/2501.07727), [Kumar et al.](https://arxiv.org/abs/2202.10054)). **Resolved role of gold: validation + this weight arm; no gold fine-tuning stage** (critic item 25).
- **Slot-fill census (critic item 14):** train fill SAG_FLUID_FS 100% / AX_FLUID_FS 100% / COR_FLUID_FS 95.8% / SAG_FLUID_NOFS 87.5% / COR_T1 62.5% / SAG_T1 50% on 24 studies (experiments.md); only 3,991/4,407 have fat-sat fluid in all three planes ([JunhaoLiXD]) — reconcile slot definitions (critic item 34).
Measure: per-epoch per-label OOF AUC and pred_std over 4,407; gold FN-vs-FP counts for Synovitis/Contusion/Fracture/Lateral OA; macro over the 6 weakest labels next to the 12; `gold_weight ∈ {1, 3, 8}` on fold 0 from cache → OOF macro and gold macro with CI; at inference, per-slot fill rate on test logged and compared to train, relaxed tier as fallback.
Noise floor: **per-label OOF ~0.03** (measured 2026-08-29); gold per-label 0.09 (diagnostic only); the gold_weight arm is likely 🔁 on OOF.
Cost: logging done; gold_weight arm 0.5 session (two extra fold-0 runs); census ~20 lines.
If it works: directs effort per label (synonym fix vs cut-point vs slot/resolution); a gold_weight that does not distort the OOF curve.
If it fails: n/a — diagnostics.
Depends on: nothing for the audit (done); P-01 for the arm.

### P-07 Synovitis ← Effusion back-fill and coverage audit (no new sources)
Status: 🔁 measured on gold today (`artifacts/label_audit.md`), **not adopted**; student-OOF arm PENDING. Verdict row to be appended to experiments.md.
Hypothesis: Replacing "unaddressed" Synovitis targets with the Effusion soft label improves the weakest teacher label more than any model change.
Origin: competition write-up / peer-reviewed.
Evidence:
- Public claim: Synovitis gold AUC 0.678 → 0.790 after back-fill ([stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)) — from a 0.678 baseline. **On our blend the same rule went 0.788 → 0.729**, paired-bootstrap delta **−0.059 [−0.164, +0.042]** → 🔁 INCONCLUSIVE (the CI spans zero in both directions). Of 58 gold, 41 are unaddressed for Synovitis and 14 of those are positive.
- Weights on the back-filled rows (critic item 6): mean Synovitis weight 0.69 without recomputation, 0.81 if `confidence_weights()` is recomputed from the back-filled values — so the back-fill would *not* be down-weighted away.
- Coverage the rule would touch: 84% overall (en 80%, es 81%, tr 92%, hr 93%, el 84%, de 76%, bg 97%, nl 92%).
- **Source independence is worse than the research assumed:** hans_v4 ~ sol56 make identical decisions at the 0.5 cut (agreement 99.45% over all 4,407 studies; error φ = **1.000 on gold for every label**); raw values differ, so this is consistent with — not proof of — the v4 blend including the sol56 table. hans_v4~pilkwang 95.4%, pilkwang~sol56 95.2%. Mean pairwise error φ 0.88 vs ~0.39 for frontier-LLM panels ([2605.29800](https://arxiv.org/html/2605.29800)). Either way our "3 sources" are ~1.5 effective votes and `confidence_weights()`' agreement term is inflated. Adding sources is already 🔁 INCONCLUSIVE (experiments.md).
- **Silence is barely down-weighted:** pilkwang is the only source that flags silence (UNK). On UNK rows hans_v4 averages ~0.25 (many distinct values), the blend averages ~0.18, and the confidence weight averages 0.69 vs 0.80–0.89 on addressed rows — a silent report looks like a confident negative (the docstring claim that silent reports "pull far less" does not hold).
- Reports help only where the finding is represented in text ([2510.24385](https://arxiv.org/abs/2510.24385)); Contusion ← bone-marrow-oedema synonyms cannot be audited from the tables (no matched terms exposed) → P-16.
Measure: student OOF-vs-teacher for Synovitis over 4,349 non-gold with vs without back-fill on fold 0 (coverage + OOF per experiments.md convention; critic item 35) — only if a cheap fold-0 slot appears; gold direction is already recorded.
Noise floor: gold per-label 0.09 (already inconclusive); OOF per-label 0.02.
Cost: ~30 lines exist in `label_audit.py`; a student arm is 0.3 session from cache.
If it works: (student OOF up) adopt; template for other silence-dominated labels.
If it fails: closed as ❌/🔁 in experiments.md; **do not** try Fracture ← Contusion without a gold read first. Also consider dropping sol56 from the blend or de-duplicating the agreement term (cheap follow-up, expected 🔁 on gold).
Depends on: P-06 OOF instrumentation (done), P-01 for the student arm.

### P-08 Slices per slot 6 → 12–16, per-plane bands, random offsets
Status: **K sweep re-raised 2026-08-30** as P-23 candidate #2a — `ARMS = [("v08k", {"slices_per_slot": 12, "cache_jitter": True, "epochs": 8})]`, ~2× a normal fold-0 arm (≈ 3.5 h), judged by `src/blend_check.py` against the 4-version blend (ρ < 0.80, gain > 0.008). Rationale in P-23. Jitter half: **jitter sub-arm ✅ KEEP — see [experiments.md](experiments.md)**, now **LB-confirmed**
(submission #4: **0.877**, a new best, from OOF 0.8528). `cache_jitter` gives
**+0.0113 OOF against a 0.008 floor**, removes the epoch-2 overfitting turn (train loss *rises*
to 0.4265 while OOF improves), and lifts **11 of 12 labels with none regressing** — a pattern no
seed change produces. It is now part of the default recipe and of every arm launched since. The
K sweep is still 💡 and **downgraded** — see the correction below.

⚠️ **Correction to this card's premise (2026-08-29, from reading the loader, not a run).** "More
slices" does not buy more pixels here. The cache stores K_cache=16 slices per slot, and
`array_to_tensor` at K=6 picks centres `[1,4,6,9,11,14]` whose `[c-1,c,c+1]` triplets already
tile **all 16** cached slices. Going to K=16 re-presents the same pixels as more, heavily
overlapping tokens, at ~2.7× GPU. The real effect is *token granularity in depth* — a CLS token
centred on a lesion is a stronger signal than one where the lesion is 1 of 3 — which is a genuine
but smaller claim than the card's "raises recall at linear cost", and it now costs linearly in GPU
time because the cache removed the decode bottleneck. Deeper coverage would need a **cache rebuild
at higher K_cache**, not a loader change. The cheap, live part of this card is therefore the
`random offsets` sub-arm, which is why `v04d` runs and the K sweep waits.

Status (original): 💡 untested as an A/B; the cache is built at **K=16 @ 224 in 2 shards** (≈21 GB total, ~10.6 GB per shard), so K ≤ 16 can be sampled from it without rebuilding.
Hypothesis: More slices per slot under attention pooling raise recall for focal findings (meniscus, Baker's, MCL) at linear cost.
Origin: competition write-up / peer-reviewed.
Evidence: winners used 24–32+ slices per volume ([Nischaydnk], [darraghdog/RSNA22](https://github.com/darraghdog/RSNA22), [brendanartley](https://github.com/brendanartley/RSNA-2024-Competition)); more views help on anisotropic volumes ([TomoGraphView](https://arxiv.org/html/2511.09605)); query attention benefits from more tokens ([AnyMC3D](https://arxiv.org/pdf/2512.12887)); per-plane bands sag 0.08–0.92 / ax 0.10–0.90 / cor 0.20–0.80 because menisci/MCL sit near sagittal stack ends (romanrozen, **notebook, not re-read**). No knee A/B of 6 vs 16 exists. Triplet gap in mm instead of index is **[our hypothesis]** and an *ablation*, not a default — `round(3 mm/spacing)` yields g=1 for 3–4 mm sagittal slices, a behaviour change (critic item 7).
Memory fallback order on a 15 GB T4 (critic items 19, 27): (1) HF `attn_implementation="sdpa"` — free speed/memory, try first; (2) `torch.utils.checkpoint` per block; (3) freeze patch-embed + first 4 blocks; (4) batch 1 with accumulation.
Measure: OOF macro and per-label AUC over 4,407 at K=6 vs K=12/16 on the same cache and folds; s/study on T4; peak memory.
Noise floor: 0.01 OOF.
Cost: 0.5 session; cache size drives K (K=12 @ 224 ≈ 15.9 GB; K=16 @ 224 ≈ 21 GB — built as **2 shards** of ~10.6 GB).
If it works: default K raised; random strided training subset doubles as augmentation.
If it fails: keep K=6; spend budget on resolution or a second member.
Depends on: P-01.

### P-09 Per-label masked attention head over slots (optionally all slice tokens)
Status: ✅ **KEEP — card closed, see [experiments.md](experiments.md)**. At a matched 8-epoch
schedule the attention head beats concat by **+0.0103**, 1.3× the 0.008 floor. **The card's
rationale was wrong even though its conclusion was right**: it predicted gains on the
plane-specific findings, and those are precisely where the head *loses* (MCL −0.040, Lateral
Meniscus −0.032, both beyond the 0.03 per-label floor). What it actually buys is **resistance to
overfitting** — it plateaus at 0.857 and holds while concat peaks at epoch 4 and decays. Verdict
is also **policy-dependent**: at each head's own best epoch they are indistinguishable, and attn
wins only under fixed-last-epoch checkpointing (see P-22). Do not reuse the plane-specialisation
argument as if it were confirmed.
Status (original): 🔁 INCONCLUSIVE at 4 epochs — retest ⏳ RUNNING (kernel v13, `v05a` attn+jitter vs
`v05b` concat+jitter, 8 epochs, only the head differs). v11 gave −0.0048, inside the 0.008 floor,
but the arm was **unconverged**: still rising at epoch 3 with train loss 0.4471 vs the concat
head's 0.3980, which is what a 27,732 → 9,300 parameter cut and a new initialisation do to a
budget tuned for the old head. The correlated-pair risk did **not** materialise. Not a dead end.
Status (original): 🔧 implemented and ⏳ RUNNING (kernel v11, arm `v04c`, `head_type="attn"`). As built:
12 learned label queries over the 6 slot vectors, a per-(label, slot) bias, absent slots masked to
`finfo.min` *before* the softmax so the context vector keeps its scale at 4 slots or 6, per-label
output projection. **9,300 parameters against the concat head's 27,732.** Unit-verified: absent
slots provably cannot influence the logits, an all-masked row cannot produce NaN, fp16-safe; and
the arm ran end to end through the *inference* path too. Slot dropout is wired but left at 0 so the
arm is a clean head-only A/B. Extra argument the card did not have when written: v8 showed train
loss falling monotonically while OOF turned over at epoch 2, so **removing** head parameters is
pointed at a measured problem, not just a specialisation argument.
Hypothesis: 12 learned queries attending over present slot/slice tokens beat concat → linear by letting each finding read its own sequences and handling missing slots by renormalisation.
Origin: peer-reviewed / competition write-up.
Evidence: per-finding plane dependence ([MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), [CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)); negative transfer with one shared vector ([Azcona](https://arxiv.org/abs/2010.01947)); query attention > Transformer > mean > LSTM on DINO features ([AnyMC3D]); public leaders use all-series cross-attention with slot-type embedding (mattiaangeli, **notebook, not re-read**). Risk (critic item 18): correlated pairs (gold φ Effusion~Synovitis 0.40, MOA~MMen 0.42, Contusion~Fracture 0.33; n=58, SE of φ ≈ 0.13) may lose the shared-vector benefit — report those pairs separately.
Measure: OOF macro and per-label AUC over 4,407 vs the concat head, same cache/folds; the three correlated pairs reported explicitly.
Noise floor: 0.01 OOF.
Cost: 0.3 session, ~80 lines, config flag; slot dropout 0.15 as a sub-arm.
If it works: default head; slot masks make P-20 trivial.
If it fails: keep concat; record in experiments.md.
Depends on: P-01.

### P-10 Second architecture family (timm ConvNeXt first; RadImageNet R50 behind a flag)
Status: 🔁 **MEASURED 2026-08-30 — see [experiments.md](experiments.md)** ("ConvNeXt-Tiny member `v06c`"):
own fold-0 OOF **0.8562** (parity with the best DINOv2 head), ρ 0.831 vs the two-head blend, blend gain
**+0.0059** with 10/12 labels up — fails the pre-registered P-23 thresholds narrowly; submitted anyway as
#8 on Tian's call to let the LB arbitrate: **0.900 (+0.004, 🔁 under the floor; best on the board)**. ConvNeXt-T is a strong *member* but only head-like diversity
at 224/6 slots; the RadImageNet half of this card is still gated on the licence. The card below is kept
as written. Previous status: 🔧 **implemented 2026-08-29 (late), effect pending** — `Config.backbone` ∈ {`dinov2`,
`convnext_tiny`}; HF `facebook/convnext-tiny-224` (Apache-2.0, ImageNet-1k, 27.8 M params,
LayerNorm-only so batch-of-1 is safe) mounted from the private Kaggle Dataset
`tiankljucanin/convnext-tiny-224-hf` (must be made public or replaced before a final submission
relies on it). Pooled 768-d output feeds the same slice attention pool + concat head; LLRD decays
per stage (4 stages) from `lr_backbone = 1e-4`. Arm `v06c`: concat head, jitter, 8 epochs,
`best_oof`, fold 0. Local CPU smoke green; Kaggle smoke pending as this was written. The arm's
verdict rule (from P-21's template, stricter than the card's original): adopt as a blend member
only if fold-0 rank correlation against **both** `v05a` and `v05b` is **< 0.77** *and* the
three-way rank-mean beats the two-head 0.8670 by **> 0.008**; its own OOF must be within ~0.02 of
DINOv2-S (`v05b` 0.8471 / `v05g` 0.8508) or the diversity is bought with too much bias.
**2026-08-30:** real run in flight (train v15). The 0.936 notebook's 45-gold panel ranks ConvNeXt-B/L
among its weakest single families (0.8754 / 0.8752 vs CoAtNet-384 0.9025), so expect a member that
is weaker than DINOv2-S on its own; the ρ criterion, not own OOF, decides (research.md §2.7.1). This
card is now candidate #1 of P-23.
Previous status (kept for the record): 💡 untested, **de-prioritised 2026-08-29**: kernel v13 showed a
*different head on the same backbone* already gives ρ = 0.773 and a +0.0096 blend gain at zero
extra training cost (P-21). Buy the free diversity first; this card only earns a session if P-21's
LB gain lands and more is still wanted. → P-21 landed (+0.019 LB) and folds added nothing (#6/#7),
so the card was re-prioritised the same evening.
Hypothesis: A CNN member adds error diversity an all-DINOv2 blend lacks; the rank-mean of two families is at least as good as five DINOv2 folds on OOF.
Origin: competition write-up / peer-reviewed.
Evidence: every RSNA winner blended families ([TheoViel], [Nischaydnk], [darraghdog/RSNA22]); ConvNeXt-B > ViT-B/16 on CXR ([2510.07191](https://arxiv.org/abs/2510.07191)); DINOv2 lost to ImageNet CNNs on clinical brain MRI ([2402.07595](https://arxiv.org/abs/2402.07595)); RadImageNet R50 > ImageNet R50 on knee tasks (ACL 0.97 vs 0.91, [RadImageNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)); public +0.003 from a RadImageNet head blend is inside noise (prvsiyan, **notebook, not re-read**). BatchNorm diverges at tiny effective batch ([ELNet](https://arxiv.org/abs/2005.02706)).
**Licence:** RadImageNet weights carry **no stated licence** on [GitHub](https://github.com/BMEII-AI/RadImageNet) (code MIT, article CC BY 4.0, data by request); the CC-BY-NC-SA claim in older docs was unverified. Treat as restrictive until radimagenet.com T&C is read. Default = 5-fold single family until this card reports (critic item 26).
Measure: member's own OOF over 4,407 — include only if within ~0.02 of DINOv2-S; blend OOF and gold; LB secondary.
Noise floor: 0.01 OOF, 0.005 LB — blend gain will likely be unmeasurable; justify by CV robustness.
Cost: 1 session for the CNN member (backbone LR 1e-4, BN frozen or GroupNorm, channels_last); RadImageNet frozen-feature head ≈ 0.3 session.
If it works: 2-family rank-mean submission.
If it fails: single-family 5-fold; RadImageNet dropped at final if the licence stays unclear.
Depends on: P-01, P-04.

### P-11 Resolution 224 vs 336 after the 130 mm crop
Status: 💡 untested. **Raised 2026-08-30**: every branch of the 0.936 notebook runs at 336–384 px; its 0.924 single model is CoAtNet-2 at 384 px over 64 slices (research.md §2.7.1). P-23 candidate #2 combines this with the P-08 K sweep in one arm: a hybrid backbone at 336–384 with K = 12–16, first probed by upsampling the 224 cache, then with a 336 cache shard if the probe clears the floor.
Hypothesis: Higher effective mm/px helps the small focal labels (Lateral Meniscus, PF OA, Fracture, Contusion).
Origin: peer-reviewed / competition write-up.
Evidence: 224 → 512 median +1.05 pp AUROC on CXR, concentrated in focal findings, but DINOv2-ViT nearly flat (VinDr 89.2 → 89.1) ([2510.07191]); 224 → 336 +0.017 LB at 2.25× FLOPs, preds correlate 0.90 (sadamtorres, **notebook, not re-read**); yu4u's crop+384 gain is confounded with cropping ([yu4u deck](https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui)). HF `Dinov2Model` needs `interpolate_pos_encoding` for non-224 input (critic item 20).
Measure: per-label OOF over 4,407 for the four focal labels at 224 vs 336, K fixed; adopt only if their mean moves > 0.02.
Noise floor: **~0.03 per label OOF** (measured 2026-08-29).
Cost: 1 session (2.25× tokens) **plus** a separate sharded cache at 336 (K=12 @ 336 ≈ 36 GB → ≥ 2 kernels); do not build it before P-08 fixes K.
If it works: 336 member (initialised from the 224 checkpoint) added to the blend.
If it fails: stay at 224; spend budget on slices/folds.
Depends on: P-01, P-08.

### P-12 Slice-window TTA (label-safe only) [our hypothesis]
Status: 🔁 **measured 2026-08-30 on the pod — not adopted** (4-member blend 0.8722 → 0.8738 with mean TTA; every member +0.003–0.006 alone; result in experiments.md 2026-08-30 P-12). Earlier: ⏳ measuring on the RunPod pod (2026-08-30, 13:00 →) — Kaggle `oof_eval` v2 finished only v05a (0.8574 → 0.8621 with (-1,0,1)/mean, 🔁 alone) before the OOM kill of traps 28; both pools now run on the pod (`/kaggle/working/tta_mean/`, `tta_focal/`). Previously: 🔧 **implemented 2026-08-30 (afternoon)**, effect pending: `Config.tta_offsets` (e.g. `(-1, 0, 1)`
shifts every K centre by the offset, clipped; one forward per view), `tta_pool` `"mean"` | `"focal"` (max
over views for Fracture / Contusion / both Menisci / Baker's, top-2 mean for ACL / MCL, mean else),
`INFER_OVERRIDES = {version: {member keys}}` to give members whose checkpoints predate the fields their
TTA at inference, and **`MODE="oof_eval"`**, which scores each `INFER_MEMBERS` fold-0 checkpoint on its
held-out studies from the cache with exactly those settings and writes `{version}_fold0_tta_oof.csv` for
`src/blend_check.py`. `(0,)`+`mean` is bit-identical to the pre-change code (local infer of v05a+v05b:
identical submission). **Measure next:** `oof_eval` on v05a / v05b / v05g / v06c with `(-1,0,1)` × {mean,
focal} (~0.5 h GPU) → blend_check against the untouched OOF files. Note from the cell-level re-read: the
0.936 notebook SUBMITS its no-jitter view, so only the window/offset half is copied. Previous status:
💡 untested. **Raised 2026-08-30**: no longer only our hypothesis — the 0.936 notebook's DINOv2 branch averages every window position over the cached slices, ×2 with an affine-jittered view, and pools **per label**: max for Fracture/Contusion/Menisci/Baker's, top-2 for ACL/MCL, mean for OA/Effusion/Synovitis (focal findings live on few slices). Est. +0.003–0.008 on that branch (research.md §2.7.1). Implement the focal/diffuse pooling split with the window offsets; measure on fold-0 OOF.
Hypothesis: Averaging logits over 2–3 slice-index offsets per slot reduces variance at small inference cost; geometric TTA is not attempted.
Origin: our hypothesis (no source shows slice-window TTA helping; critic item 30). The *exclusion* of geometric TTA is peer-reviewed.
Evidence: winners used some TTA ([brendanartley], [SeuTao](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)); geometric TTA degraded 11/12 MedMNIST pairs ([2604.09697](https://arxiv.org/html/2604.09697v1)); flips hurt knee OA even after mirroring ([2311.06118](https://arxiv.org/html/2311.06118)).
Measure: OOF macro over 4,407 with vs without TTA; inference s/100 studies.
Noise floor: 0.01 OOF (asserted; P-02 step 1 measures it) — expected gain is *below* the macro floor; justified by cost only.
Cost: 0.1 session, ~30 lines.
If it works: default for the accuracy track; off for the efficiency variant.
If it fails: drop.
Depends on: P-01 (decode-once inference).

### P-13 3 vs 5 folds under a fixed session budget [our hypothesis]
Status: 💡 untested.
Hypothesis: 3 folds + a second family beats 5 folds of DINOv2-S for the same T4 hours.
Origin: our hypothesis.
Evidence: no source quantifies seed vs fold vs architecture gains; winners ran 4–5 folds × several models ([darraghdog/RSNA22] 5 × 3 seeds; [Nischaydnk] 4 folds); fold soup (averaging weights across folds) matches best single at zero inference cost only for same-init soups — cross-fold souping is untested here (critic item 12).
Measure: OOF on the common held-out set and gold for {5-fold ViT-S} vs {3-fold ViT-S + 3-fold CNN}; LB secondary.
Noise floor: 0.01 OOF; likely 🔁.
Cost: 1–2 sessions.
If it works: budget rule for the final submission.
If it fails: default to 5 folds of the best single family.
Depends on: P-01, P-04, P-10.

### P-14 DINOv2-S vs DINOv2-B (LoRA if memory-bound); registers variant
Status: 💡 untested; low priority.
Hypothesis: ViT-B adds little over ViT-S at 4k studies; test only as a late diversity member.
Origin: peer-reviewed.
Evidence: ViT-T/S/B 0.952/0.954/0.954 on MRNet ACL ([SB-SSL](https://arxiv.org/abs/2208.13923)); size non-monotonic ([2509.06467](https://arxiv.org/html/2509.06467v3)); S ≈ B ([2402.07595], [2606.25989]); +0.008–0.029 on CT with LoRA ([AnyMC3D]); full FT > PEFT at scale ([2510.07191]). **Registers (critic item 20):** DINOv2-with-registers removes high-norm attention artefacts that can hurt attention pooling ([Darcet et al.](https://arxiv.org/abs/2309.16588)); `facebook/dinov2-with-registers-small` exists on Kaggle Models — a cheaper first swap than ViT-B, same cost as ViT-S.
Measure: OOF macro over 4,407 vs ViT-S at matched wall-clock.
Noise floor: 0.01 OOF.
Cost: registers-S 0.3 session; ViT-B 1 session (~3× ViT-S; gradient checkpointing or LoRA r=8).
If it works: extra blend member.
If it fails: confirmed dead end; frees budget.
Depends on: P-01, P-04; after P-10.

### P-15 DINOv3-S/16 as a diversity member (not a replacement)
Status: 💡 the DINOv3 half untested; the **16-channel re-scope was run 2026-08-30 as `v07s` and is ❌ DEAD END as built** (own OOF 0.7366, experiments.md) — a linear 16-ch patch embed at 224 cannot learn the stack; any retry needs a non-linear stem (`DepthCompress`-style) at a much higher LR. **Raised 2026-08-30** as P-23 candidate #3, re-scoped: the 0.936 notebook's DINOv3-class member feeds **16 slices as input channels** (`in_chans=16` patch embed, or a gated 16→3 `DepthCompress` stem) with the slot identity as an extra ViT token — the diversity comes from the input representation, not the weights (research.md §2.7.1). Test that representation on DINOv2-S first (no model mirror needed, no new cache); swap in DINOv3 weights only if the member is accepted.
Hypothesis: DINOv3 at 224 is not better than DINOv2 but is decorrelated enough to help a rank blend.
Origin: peer-reviewed / competition write-up.
Evidence: DINOv2 vs DINOv3 differences 0.002–0.008 in both directions ([AnyMC3D Table 8]); DINOv3 wins only at 512 px ([2510.07191]); public leaders blend 5 DINOv3-S folds (mattiaangeli, **notebook, not re-read**); custom licence, gated download, must mirror LICENSE.md ([dinov3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)).
Measure: member OOF and blend OOF over 4,407; gold.
Noise floor: 0.01 OOF.
Cost: 1 session + Kaggle Model mirror with the licence file.
If it works: third member.
If it fails: dropped; CNN member preferred.
Depends on: P-01, P-10.

### P-16 Re-labelling with an open-weights LLM inside Kaggle (graded, synonym-aware, native language)
Status: 💡 untested. The audit that scopes it is done.
Hypothesis: A 14–32B open model prompted per finding for a 4-level grade with explicit negation and synonym lists produces better soft targets than the existing binary tables, with the gain concentrated where the audit found silence and language gaps.
Origin: peer-reviewed / competition write-up.
Evidence:
- Open models ≈ GPT-4o on extraction, including German (Mistral-Large 92.6 vs 92.4 macro-F1, [Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.240895); [JIIM 2025](https://pubmed.ncbi.nlm.nih.gov/40325326/)); uncertainty-graded smoothing +4.5 mean AUC, biggest on rare classes ([Rep-GLS](https://arxiv.org/html/2508.02495)); image-aligned labels beat report labels ([VisualCheXbert](https://arxiv.org/abs/2102.11467)); native > translate-first for multilingual models ([2602.21374](https://arxiv.org/html/2602.21374), [2403.10258](https://arxiv.org/html/2403.10258)); Qwen3-14B-AWQ reached 0.881 gold (sadamtorres, **notebook, not re-read**); 70B does not fit 2×T4, 32B AWQ is the ceiling.
- **Aims fixed by today's audit:** (1) Spanish coverage — Fracture UNK 80%, Lateral Meniscus 40%, Effusion 29%; Turkish Lateral OA UNK 68%; (2) silence on Synovitis 84%, Fracture 56%, Baker's 46%; (3) graded (not binary) extraction; (4) native-language prompts; (5) on-platform open-weights model — no report text leaves Kaggle (rules risk in CLAUDE.md). Bulgarian has the lowest inter-source agreement (Spearman hans_v4~pilkwang 0.67 vs 0.83 English; near-duplicate pair excluded). Contusion ← bone-marrow-oedema synonym handling is only testable here.
- Existing sources are ~1.5 effective votes (φ 0.88), so a genuinely independent reader is the one thing more sources could add.
Measure: gold macro-AUC of new targets vs current blend (paired bootstrap, reported not gated); per-language UNK rate before/after; hand-read sample of flipped reports (aggregates only in docs); student OOF-vs-new-teacher on fold 0.
Noise floor: gold 0.05 macro (CLAUDE.md rule; a 3,000-rep study-level bootstrap of the teacher's gold macro-AUC gives SD 0.017) / per-label HM SE ≈ 0.09 — decide on coverage + flipped-report audit + student OOF.
Cost: 1–2 sessions (vLLM on 2×T4, weights as a Kaggle Model) + ~150 lines; a grade → soft-value rubric in `build_targets.py`.
If it works: replaces the blend as the teacher; P-07-style back-fills become unnecessary.
If it fails: keep the current blend (with P-00 scale); consider dropping sol56 as a near-duplicate.
Depends on: P-06 audit (done).

### P-17 Noise-robust loss / self-distillation / AUC-margin stage 2
Status: 💡 untested; bottom of the modelling backlog.
Hypothesis: Symmetric CE, or round-2 targets (0.5 teacher + 0.5 rank-normalised round-1 OOF), reduce teacher-error memorisation by ≥ 0.01 OOF.
Origin: peer-reviewed / competition write-up.
Evidence: only SCE/CDR marginally beat CE under imbalanced clinical noise ([LNMBench](https://arxiv.org/html/2512.09315v1)); pseudo-label relabelling was load-bearing for winners (yu4u bw 0.881 → 0.917; [brendanartley]); BCE within ~0.01 of ASL/RAL/focal ([RAL](https://arxiv.org/abs/2308.05542)); AUC-margin stage-2 won CheXpert ([Yuan et al.](https://arxiv.org/html/2012.03173)) but **needs hard labels** — thresholding our soft targets reintroduces the cut-point problem (critic item 39), so AUCM is parked unless a soft-target formulation is specified.
Measure: OOF-vs-teacher over 4,407 (and vs round-1 OOF for distillation); gold sanity only.
Noise floor: 0.01 OOF; self-distillation judged on gold is unresolvable.
Cost: 0.3 session each.
If it works: ≥ 0.01 OOF gain adopted.
If it fails: BCE stays.
Depends on: P-01, P-04, P-06.

### P-18 Efficiency-track variant, decode-once inference, submission robustness
Status: 🔧 inference path shipped in v02/kernel v4 (MODE=infer from mounted checkpoints; **no placeholder — loud failure**; refuse-to-submit-constants; image-root probing via a shallow glob that never descends into train_series/test_series; **(2026-08-29)** the infer path no longer downgrades preprocessing when no cache is mounted (traps 6d) and no longer decides `mode="train"` in real mode because only fold 0 has a checkpoint (traps 12c/12d) — the infer notebook is generated with `MODE="infer"` and `FORCE_SMOKE=False` sed'd in, the same pattern `cache-b` uses; infer mode reads slices_per_slot/triplet_gap/img_size from the checkpoint's saved config so FORCE_SMOKE cannot change the model's inputs; MODE=auto picks infer only if every configured fold has a mounted best.pt); efficiency variant 💡.
Hypothesis: A single DINOv2-S at 224 with decode-once inference and no TTA is competitive on the Efficiency LB at no accuracy cost we can measure.
Origin: competition write-up; our verified failure.
Evidence:
- **Submission #1 (kernel v2, smoke, v01) scored exactly 0.500 public** — constant predictions: the image-root/`.dcm` assumption failed silently at rerun. Fixed today: probe the root via glob, **no placeholder file** (a crash → missing submission → visible error, by design), then **fail loudly if < 90% of test studies are imaged and have ≥ 1 slot, or predictions are constant**. Added to traps.md (12b).
- Efficiency LB top 0.948 vs accuracy 0.952, non-monotone in score (verified CSV, [ryanholbrook LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)); **formula unread** — may score CPU-only runtime (critic item 12), which would make decode, not the ViT, binding.
- ViT-S inference ≪ decode (~4,617 img/s on a 3090 fp16, [timm CSV](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv)); community plans for ≥ 1,322 hidden studies ([pilkwang] log).
- Slot-fill census at inference (critic item 14) and a hidden-test-size-adaptive time budget (critic item 22) belong here: N_test measured at start → per-study budget; row order asserted equal to `test.csv`.
Measure: s/100 test studies (logged); slot fill per slot on test vs train; public LB of the lean variant.
Noise floor: 0.005 LB.
Cost: 0.2 session (`EFFICIENCY_MODE` flag) — **after** reading the formula in a browser.
If it works: second prize track with the same model.
If it fails: (formula penalises differently) adjust after reading it.
Depends on: P-01.

### P-19 Decoder wheels and TransferSyntax census
Status: 💡 untested (insurance).
Hypothesis: Compressed DICOMs may exist in train or test; without mounted pylibjpeg/GDCM wheels they vanish silently at rerun.
Origin: public consensus.
Evidence: pydicom needs plugins for JPEG Lossless/2000 ([plugin table](https://pydicom.github.io/pydicom/stable/guides/plugin_table.html)); Kaggle's image lacks them ([kaggle_requirements.txt](https://raw.githubusercontent.com/Kaggle/docker-python/main/kaggle_requirements.txt)); one notebook reports compressed series (jirkaborovec, pulled) vs a census claiming 100% Explicit VR Little Endian ([FINDINGS.md]); internet is off at rerun. Decode failures currently zero-fill and still train (critic item 21).
Measure: TransferSyntaxUID value counts over 24,371 series (manifest); decode-failure count at test (assert < 0.1%, logged, never silently zero-filled).
Noise floor: n/a.
Cost: 0.1 session, ~20 lines + a wheels dataset (`pip install --no-index`).
If it works: silent-failure trap closed; traps.md entry.
If it fails: n/a.
Depends on: P-01 header pass.

### P-20 Leave-one-slot-out ablation and T1 slot retirement
Status: 💡 untested.
Hypothesis: SAG_T1 (50% fill) and COR_T1 (62.5%) can be retired if no label drops > 0.02 OOF; Contusion, Fracture and the OA labels are the ones at risk.
Origin: peer-reviewed / our hypothesis.
Evidence: contusion 0.82 → 0.70 without T1W/T2W ([CoPAS]); fat-sat fluid carries oedema ([Maarek 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12362699/)); 2024 2nd dropped a view moving CV < 0.02 ([brendanartley]); plane preference is model-specific ([MRNet], [Azcona], [ELNet]). Works with the **current concat head** too — zero the slot vector and set mask = 0 (critic item 9); no dependency on P-09.
Measure: per-label OOF over 4,407 with each slot's presence forced absent at inference, no retraining.
Noise floor: **~0.03 per label OOF** (measured 2026-08-29).
Cost: 0.1 session (inference-time flag).
If it works: slot budget reallocated to K in fat-sat slots.
If it fails: keep all 6 slots.
Depends on: the first real model (v02 fold 0); P-01 for speed.

### P-21 Blend two heads on one backbone as the default ensemble axis
Status: ✅ **MEASURED 2026-08-29 — card closed, see [experiments.md](experiments.md)**
(Submissions #5). LB **0.896** vs 0.877 for the best single head: **+0.019, 3.8× the 0.005 floor**;
OOF 0.8670 → LB 0.896 keeps the +0.02–0.03 offset. Shipped as `INFER_MEMBERS` (every mounted fold
checkpoint of each listed version is a member; a missing version is fatal; heads per member;
geometry must agree) with decode-once test preprocessing (P-18). "If it works" clause applies:
head diversity is the default ensemble axis and P-10/P-14/P-15 drop in priority. Open follow-ups:
the blend gain on more than one fold (the 5-fold `v05g` run gives concat on folds 1–4; an attn
5-fold run would complete the 10-member blend), and per-version weights (untested; flat mean
shipped).
Hypothesis: Rank-meaning an attention-head and a concat-head model trained on the same backbone,
fold and seed beats either alone by more than the noise floor, and is a cheaper source of error
diversity than a second architecture family.
Origin: verified in our code+data (kernel v13), then generalised.
Evidence:
- **Measured on fold 0**: `v05a` 0.8574, `v05b` 0.8471, **rank-mean 0.8670** — +0.0096 over the
  best single arm (1.2× the 0.008 floor), +0.0142 over the submitted `v04d`. Plain rank-mean, no
  weights fitted, so it is a held-out estimate.
- **Mean rank correlation 0.773** between two models sharing backbone, data, fold, schedule and
  seed — differing only in the head. Least correlated exactly where each is weakest (Fracture
  0.664, Lateral Meniscus 0.695, MCL 0.715).
- The two heads have complementary per-label profiles: attn wins Fracture/LatOA/Contusion, concat
  wins MCL/Lateral Meniscus, both beyond the 0.03 per-label floor.
- Contrast with P-10/P-13, which assume diversity must be bought with a second family (a session
  each; RadImageNet also has an unresolved licence). This costs **nothing extra** when the two
  heads are already run as an A/B.
Measure: LB of the two-head rank-mean vs the best single arm's LB; OOF blend gain on more than one
fold before believing the magnitude.
Noise floor: 0.008 OOF macro; 0.005 LB. The expected LB gain is small — do not read a sub-0.005
LB movement as confirmation.
Cost: ~0.2 session of code. The infer path currently rank-means across **folds** for one
`cfg.version`; blending across *versions* needs `ckpt_paths` to accept a list of (version, fold)
pairs. No extra training.
If it works: head diversity becomes the default ensemble axis and P-10/P-14/P-15 drop in priority.
If it fails: (LB flat) the OOF blend gain did not transfer; keep the single best head and revisit
family diversity.
Depends on: P-09 (done).

### P-22 Checkpoint selection on OOF-vs-teacher instead of fixed last epoch
Status: ✅ **MEASURED 2026-08-29 — card closed, see [experiments.md](experiments.md)** ("P-22:
checkpoint on OOF-vs-teacher"). `src/oof_epoch_analysis.py`: split-half gain **+0.0128** for the
concat head (200/200 splits), **−0.0002** for attn, gold flat at the chosen epoch, no
teacher-chasing signature. Policy switched: `Config.ckpt_policy = "best_oof"`; P-09 re-read as a
tie (−0.0024 at best epochs). The card below is kept as written for the record.
Hypothesis: Selecting the epoch by OOF-vs-teacher over 882 held-out studies is enough less noisy
than the 11-gold selection we rejected that it beats fixed-last-epoch, and it removes the
policy-dependence that currently decides P-09.
Origin: our hypothesis, forced by kernel v13.
Evidence:
- **The problem is now concrete.** `v05b` peaks at 0.8600 (epoch 4) and is checkpointed at 0.8471
  (epoch 7) — fixed-last-epoch throws away 0.013, more than the floor. `v05a` peaks 0.8576 and
  ships 0.8574, so it loses nothing. **P-09's verdict flips depending on which policy is used**
  (attn +0.0103 under fixed-epoch; the two are indistinguishable under best-epoch).
- Fixed-epoch was adopted for a good reason that still holds: best-epoch on ~11 gold studies is a
  coin flip at Hanley–McNeil SE ≈ 0.09 (experiments.md, traps 3).
- But OOF-vs-teacher is measured on **882** studies, not 11, and its run-to-run floor is now
  **measured at 0.008** — two orders of decision quality apart from the gold selection we banned.
- The stated counter-argument (docs/research.md) is that selecting on soft targets rewards
  memorising the teacher. That is a real risk and is exactly what this card must measure, not
  assume: a head that overfits the teacher would show a *rising* OOF-vs-teacher while gold falls.
  `v05b` shows both falling together, which is ordinary overfitting rather than teacher-chasing.
Measure: for each existing arm, OOF at the best epoch vs OOF at the last epoch, and the gold curve
alongside to detect teacher-chasing; then whether best-epoch selection changes any recorded
verdict. All computable from the `_ep{e}_oof.csv` files already written — **no new training**.
Noise floor: 0.008 OOF macro; the decision is whether the gap between policies exceeds it.
Cost: ~0.1 session, analysis only, on files we already have.
If it works: switch the policy, and re-read P-09 and P-04 under it.
If it fails: (best-epoch tracks the teacher while gold diverges) keep fixed-epoch and record why,
with the evidence this time rather than the argument.
Depends on: nothing — the per-epoch OOF csvs from v11 and v13 are enough.

### P-23 Multi-family rank fusion: the ensemble is the product, not a member
Status (2026-08-30 16:45): ✅ **candidate #2b `v10c` measured** — fold-0 OOF 0.8641 (v08w 0.8648), ρ 0.73–0.81 vs the DINOv2 members, Lateral/Medial Meniscus 0.858/0.922 (best of all), ACL/MCL weakest; 4-member LB blend + v08w + v10c = **0.8795 (+0.0073)**; v08w+v10c pair +0.0088 over v08w alone; still rising at epoch 7 → a 10–12-epoch rerun is a candidate arm; **`v09h` (224 probe) = 0.8683 in 50 min — the best single, 7-member blend 0.8820** (experiments.md 2026-08-30 `v09h`) (experiments.md 2026-08-30 `v10c`). Earlier status (2026-08-30 afternoon): **candidate #2 implemented, not yet run** — `v09h` = `timm:coatnet_rmlp_1_rw_224`
(41.7 M, 7.8 GMACs) at 224 on the c02 cache with the P-25 window head (RunPod probe, ~4 h fold 0; tests
"hybrid backbone + all-window per-label attention" before the expensive one) and `v10c` (`ARM_V10C`) =
`timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (73.9 M, 47.7 GMACs) at 384 from the 336 cache, 24 train
windows, `eval_windows=42`, `grad_checkpoint` on 24 GB cards (RunPod 6–8 h fold 0) — the 0.936 notebook's
0.924 recipe minus its 140 mm crop / [2, 98] normalisation (kept at ours for a shared test-time builder)
and plus our laterality (theirs has none). Weights mounted as Datasets `timm-coatnet-rmlp-1-rw-224` /
`-2-rw-384`; offline load strict except the ImageNet head (`load_timm_backbone`). #2a (K=12) withdrawn —
superseded by P-25/P-26. Inference now groups members by cache geometry (`INFER_CACHE_KEYS`) and decodes
the test set once per group, so c01 and c02 members blend in one run. Compute: `v10c` at 42 eval windows
≈ 15–25 min per checkpoint on the hidden test — budget 1–2 checkpoints of it, not 5 folds.
Status (morning): 🔧 **two candidates measured 2026-08-30** (experiments.md): #1 `v06c` ConvNeXt-T — own 0.8562,
ρ 0.831, blend +0.0059, narrow reject by the rule; submission #8 scored **0.900 (+0.004, 🔁 under the 0.005
floor, best on the board)**; #3 `v07s`
16-channel DINOv2 — ❌ own 0.7366, blend −0.015 (this stem recipe is dead; the representation is not
disproven). Lesson so far: a member must be *both* strong and differently-shaped; ConvNeXt at the same
224/6-slot geometry is strong but head-like (ρ 0.77–0.83). Remaining candidates: #2 high-res many-slice
hybrid (needs quota: ~6 h left this week), #4 RadImageNet heads (licence). **Raised to the top of the backlog 2026-08-30** on Tian's decision
after reading `crazy_good_rsna.ipynb` in full (research.md §2.7.1). Member arms already have cards
(P-10 ConvNeXt-T in flight as `v06c`; P-11 336–384 px; P-15 16-channel member); this card is the
*programme* that sequences them and fixes the acceptance rule for each member.
Hypothesis: A rank-fusion of three or more architecture families — each within ~0.02 OOF of
DINOv2-S, each at ρ < 0.80 to the blend it joins — beats any further work on one family; every
+0.005 of the remaining 0.040 to the public top is cheaper to buy with a new family than with
folds, epochs or heads.
Origin: verified in a public artefact (the 0.936 notebook's own stage files) + our own #6/#7.
Evidence:
- 0.936 = DINOv2 branch 0.899 → + 16-ch ViT + RadImageNet heads + calibrator 0.920 → + CoAtNet-2@384
  0.935 → + LB tuning 0.936 (stated in-notebook; research.md §2.7.1). Our two-head DINOv2 blend is
  0.896 — parity with their DINOv2 branch; the entire gap is other families.
- Our own measurements point the same way: five folds of one head +0.009 alone, **+0.000** on top
  of a second head (#6/#7); two heads on one backbone +0.019 (P-21).
- Their in-notebook counter-example: CoAtNet + Swin-B + EffV2-L on the *same* 64-slice input blended
  to +0.001 over CoAtNet alone. Families that differ only in backbone name but share input geometry
  and pretraining regime add little. Diversity must be in the **input representation and
  pretraining** — SSL ViT on triplets vs supervised hybrid at 384/64 slices vs a 16-channel stack vs
  frozen RadImageNet features — which is also why two heads on one backbone (ρ 0.77) gave more than
  five folds (ρ ≈ 0.84).
Measure: fold-0 n-way rank-mean OOF over the 882 held-out studies vs the current best blend; ρ of
each new member against every existing member; then one LB submission per accepted member. A member
is **accepted** iff (a) own fold-0 OOF ≥ current best single − 0.02, (b) ρ < 0.80 against the
existing blend, (c) blend OOF gain > 0.008. A member that fails (b) is not run to five folds.
Noise floor: 0.008 OOF macro for blend gains; 0.005 LB.
Cost: one fold-0 arm (~1–2 h GPU) per candidate; 5 folds only for accepted members. Candidate
order by expected diversity ÷ cost:
1. `v06c` ConvNeXt-T (P-10, running) — cheapest; weakest family on their panel, so ρ decides.
2. **High-resolution, many-slice hybrid** (P-11 + P-08): timm `coatnet_rmlp_2_rw_384` or a
   MaxViT/CoAtNet-class backbone at 336–384 with K = 12–16 from the 16-slice cache, label attention
   over all windows. Probe first by upsampling the 224 cache (cheap, measures the K/attention part);
   build a 336 cache shard only if the probe clears the floor. Their single-model 0.924 lives here.
   **Re-scoped 2026-08-30 after `v06c`/`v07s`:** run **2a** first — DINOv2-S with `slices_per_slot` 12 from
   the existing 16-slice cache (no rebuild, no new weights, ~3.5 h; fits the ~6 h left this week) — because
   the two runs showed the missing diversity is in the *input*, not the backbone; **2b** (hybrid at 336–384)
   waits for P-24 compute.
3. **16-slices-as-channels member** (P-15 re-scoped): `in_chans=16` patch embed on DINOv2-S with a
   slot token — a new input representation with no new cache and no model mirror.
4. RadImageNet R50 frozen features + attention heads (~0.3 session) — behind the licence gate.
Zero-training alternative: mount the public checkpoints themselves (dreaddevelopment
`raptor-knee-*`, tonylica `rsna-knee-bend-dinov3-0917-repro-assets`) as members. Legal (public
datasets), fastest to ~0.93, but it makes us a fork of the shared ensemble whose author expects a
private shakeup, and their dataset licence fields are unread — Tian's call (brainstorm.md).
If it works: the final submission is an n-family `INFER_MEMBERS` blend with one vote per family
(`INFER_BLEND="by_version"` already does this); per-label weights only if a grouped-OOF measurement
clears 0.008 — never fitted on gold-58 or the LB.
If it fails: (every candidate ρ ≥ 0.80 or blend gain < 0.008) the residual gap is single-model
strength, and the budget goes to P-11/P-08 on DINOv2 alone.
Depends on: P-10 (running), P-11, P-15; `INFER_MEMBERS` (done, P-21).

### P-24 Compute expansion without paying: 2×T4 on Kaggle, and an off-Kaggle (Colab-free) runner
Status: 🚀 **first real run 2026-08-30** (secure RTX 4090, EUR-IS-2, $0.74/h; community 4090s had no public IP — traps 29): P-12 `oof_eval` first, then `v10c`, then `v09h`, checkpoints shipped as `rsna-knee-ckpt-<arm>`. Earlier: 💡 untested — written 2026-08-30 after the overnight runs left ≈ 6 h of the 30 h weekly Kaggle
quota and Tian ruled out paid GPUs. Two independent halves.
Hypothesis: (a) A Kaggle **GPU T4 ×2** session is charged to the quota once, so `DataParallel` over two
studies per step gives ≥ 1.5× training per quota hour at identical OOF. (b) Training needs only the
**21 GB uint8 cache + CSVs + weights, never the DICOMs**, so the same `src/kaggle_pipeline.py` can train
on free Colab (T4, ≤ 12 h sessions, unpublished fluctuating quota) with the cache staged from Google
Drive to the VM's local disk, and only inference stays on Kaggle — roughly doubling weekly T4 hours.
Origin: our hypothesis (a: Kaggle quota is counted per session — community-sourced, verify on the first
2×T4 run); verified facts for (b): a Google One storage plan includes **no** Colab compute units, Colab
Pro is a separate $9.99/100-unit subscription, the free student Colab Pro was US-only and is closed
(web search 2026-08-30, sources in the 10:30 handoff conversation); Colab free ≈ 12 h max session,
~90 min idle disconnect, ephemeral disk.
Evidence: our pipeline already runs as a plain script off-Kaggle (`ON_KAGGLE=False` → `data/`,
`artifacts/cache_local`, `models/`) and **resumes from `{version}_fold{k}_last.pt` every epoch**, so a
Colab disconnect costs ≤ 1 epoch (7–11 min), not the run. Throughput is loader-bound (6× fewer forwards
bought 1.7× in `v07s`), so an off-Kaggle box needs local NVMe + ≥ 8 workers; reading the cache through
the Drive FUSE mount would be *slower* than Kaggle. Checkpoints return via `kaggle datasets version`
into a private Dataset, which `find_mounted_checkpoints` already resolves (the `rsna-knee-ckpt-v05` pin
proved it 2026-08-30).
Measure: (a) s/study and fold-0 OOF of one 2×T4 arm vs the single-T4 twin (same seed). (b) wall-clock
of a fold-0 8-epoch DINOv2 arm on Colab vs 1.9 h on Kaggle; number of disconnects/resumes.
Noise floor: OOF must match within 0.008 (a is a speed change only); speed gain must be ≥ 1.5× to be
worth the code.
Cost: (a) ~30 lines + 1 smoke + 1 arm. (b) ~1 h of code, no GPU: env-var paths (`RSNA_CACHE_DIR`,
`RSNA_DATA_DIR`, `RSNA_MODELS_DIR`), CUDA requirements file, `FORCE_SMOKE` honoured off-Kaggle,
`num_workers` 8, `colab_bootstrap.ipynb` (mount Drive → copy cache to `/content` → clone → run →
`kaggle datasets version`); plus turning the two cache outputs into one **private** Kaggle Dataset
(browser, "create dataset from notebook output"). Rules note: private use of a derived cache on one's
own compute is ordinary Kaggle practice; it must never be published or shared outside the team.
If it works: ~3× the current experiment rate for $0; Kaggle's 30 h reserved for smokes, inference and
2×T4 arms; P-23 candidate #2 (336–384 px hybrid) becomes affordable.
If it fails: (a) if 2×T4 is charged 2× or DataParallel stalls the loader, drop it; (b) if Colab's
dynamic quota starves overnight runs, fall back to Kaggle-only with strict fold-0-first discipline.
Depends on: P-01 cache (done); P-23 for what to run with the hours.
**Status update 2026-08-30 (afternoon):** half (b) is **built** as a RunPod runner, not Colab — Tian
chose a paid-per-hour pod (no idle disconnects, local NVMe): `scripts/runpod_bootstrap.sh setup|train
<arm>|ship <arm>` mimics the `/kaggle/input` tree on the pod so `src/kaggle_pipeline.py` runs
unchanged (`ON_KAGGLE` flips true), pulls the four c02 shards with `kaggle kernels output` (~18
blobs each; verified by file count and bytes, never by exit status — traps 14), trains one arm via
`RSNA_ARM=<version>` + `RSNA_TRAIN_ONLY=1` (`RSNA_WORKERS=8`, `RSNA_RUNTIME_H=40`), and ships
`_best.pt` + `_oof.csv` as Dataset `rsna-knee-ckpt-<arm>`. `requirements-gpu.txt` pins the deps. Not yet
run on a pod. Half (a) 2×T4 stays 💡.

### P-25 Window-attention head + random-window training (every (slot, window) token, per-label softmax)
Status: ✅ **measured 2026-08-30** — `v08w` fold 0 OOF 0.8648 (+0.0074 vs v05a, 12/12 labels up; blend +0.0044 🔁, ρ 0.866): the recipe is the new default member, not a fifth blend member (experiments.md 2026-08-30 `v08w`). Earlier: 🔧 **implemented 2026-08-30**, effect pending. `Config.head_type="window_attn"` (`WindowAttnHead`:
learned slot embedding + LayerNorm + gate `Linear(dim,256)→Tanh→Dropout→Linear(256,12)`, softmax over the
study's windows **per label**, per-label output weight; padded/absent windows masked with `finfo.min`),
`window_mode="random"` (train: `train_windows`=24 (slot, centre) windows sampled without replacement,
stratified ≥ 2 per present slot; eval: all valid windows, or `eval_windows` equidistant ones — the SAME
value in `oof_eval` and infer). The Dataset ships the uint8 study + indices; the model gathers the
[c−1, c, c+1] triplets, scales, resizes and normalises on the GPU. Unit-verified (`src/window_head_test.py`:
60/84 window counts, no absent-slot picks, fp16-finite with masks, `param_groups` covers every parameter
once for all four backbones); local CPU smoke trained `v08w` + `v09h` end to end. First arm **`v08w`**
= DINOv2-S @224, c02 cache, 24 train windows, 8 ep, `best_oof` — Kaggle smoke, then fold 0 (~2 h).
Hypothesis: letting each label run its own softmax over every window of the study — with no
label-agnostic per-slot pooling in between — lifts the focal labels (menisci, MCL, Fracture, Contusion)
by > 0.008 OOF macro over the attention head at equal backbone, and random-window training replaces
jitter as the augmentation.
Origin: verified in a public artefact (the 0.936 notebook's `RaptorClassifier` pools 62 windows with a
per-label softmax; research.md §2.7.1) + verified in our code (`AttnPool` is label-agnostic).
Evidence: our `KneeNet` pools each slot's slices with one `AttnPool` whose scores do not depend on the
label, so a Fracture slice and a meniscus slice in the same sagittal stack compete for one 384-d
vector before any label reads it; P-09's plane-specialisation story failed (the head won by
overfit-resistance), which is consistent with the slot vector, not the head, being the bottleneck.
Their strongest member (0.924 alone) pools per label over all windows; est. ≈ +0.005 of its gain
(§2.7.1). Jitter's +0.011 (P-08) says slice-position augmentation pays; random window subsets are the
same idea with full coverage across epochs. Deviation from the notebook: we add a slot embedding and
keep windows inside a slot (theirs cross slot boundaries and carry no slot identity).
Measure: fold-0 OOF macro + per-label of `v08w` vs `v05a` (0.8574) over the 882 studies; then
`src/blend_check.py` vs the 4-version blend under the P-23 rule. **Confound, on purpose:** `v08w` also
moves to the c02 cache (P-26); one ~2 h arm measures the pair. If it wins, a `window_attn` arm on the
c01 cache (or an `attn` head on c02) splits them for ~2 h more.
Noise floor: 0.008 OOF macro; ~0.03 per label.
Cost: ~2 h GPU (T4) fold 0 at 224 with 24 train windows; eval ≈ 60 windows/study (2× the val time).
If it works: default head for every new member (the hybrids `v09h`/`v10c` already use it); 5 folds.
If it fails: keep attn/concat heads; run the c02 cache with the attn head to isolate P-26.
Depends on: P-26 (the c02 cache), P-01.

### P-26 Cache v2 (`c02`): band 2–98 %, ragged budgets 18/12/12/14/8/8, 336 px, 64-study blobs
Status: ✅/🔁 **measured 2026-08-30 on `v08w`** — MCL 0.795 → 0.823 (+0.028, the claim), Lateral Meniscus 0.818 → 0.827 (+0.009, not); blob loader 0.12 s/study on Kaggle (experiments.md 2026-08-30 `v08w`). Earlier: 🔧 **built 2026-08-30** — four CPU kernels `rsna-knee-cache2-a..d` (`SHARD = 0..3` sed'd,
`N_SHARDS = 4`) launched ~11:50, all complete by ~12:10: **4,407/4,407 studies, 70 blobs, 35.8 GB (8.5–9.5 GB
per shard), 0 decode failures, 0 GPU h** (experiments.md "Cache v2 built"); effect pending. Code shipped in `src/cache_pipeline.py`
(`SCHEME="c02"`, `SCHEME_DEFAULTS`, `build_study_flat`, `write_blob`, sidecars, manifest rebuilt from
sidecars every run) and `src/kaggle_pipeline.py` (`cache_scheme`, `cache_geom()`, `cache_version_for()`,
`read_cached()` header-offset blob read, `slot_stacks()`, per-version `CACHE_INDEX`). Local: the 3
sample studies built into one blob, resume-only rerun still writes the manifest, and
`src/cache_selftest.py` shows both modules **bit-identical for c01 and c02** (arrays, masks, slots, side,
version strings, blob rows). c01 rebuilt with the new code is byte-identical to the 2026-08-28 files.
Hypothesis: the outer 8 % (sagittal) / 20 % (coronal) of each stack that c01 discards carries the
collateral ligaments and the lateral meniscus body; a 2–98 % band with more slices in the
fluid-sensitive slots lifts MCL (0.836) and Lateral Meniscus (0.833) — our two weakest labels — by
> 0.03 each at equal model, and 336 px storage lets the hybrids run at 336–384 without a third cache.
Origin: verified in a public artefact (the 0.936 notebook's strongest member: budgets SAG-FS 18 / SAG 14 /
COR-FS 12 / COR 8 / AX 12 at span 0.02–0.98, and the docstring that cutting the outer slices "was
measurably costing accuracy on the collateral ligaments and the lateral meniscus") + our per-label
OOF table (experiments.md 2026-08-30).
Evidence: per-label OOF of the 4-version blend — MCL 0.836, Lateral Meniscus 0.833, Lateral OA 0.828,
PF OA 0.833 vs ≥ 0.86 for the other eight; sagittal series are ~26–32 slices, so c01's 16-of-(8–92 %)
already skips ~2 slices at each end; their WideDense (6–94 %) → MaxSpan (2–98 %) is the notebook's
own ablation in this direction. Our six slots map onto their five with the two T1 slots sharing the
non-fluid coronal budget (8). Costs: 8.1 MB/study (was 4.8) → 35.8 GB; the c01 version string did not
encode the band at all (traps 23), so the scheme carries its own `cache_version_of()`.
Measure: per-label OOF (MCL, Lateral Meniscus, Lateral OA, PF OA) of the first c02 member (`v08w`) vs
`v05a`; macro second. Blob loader throughput (`s/study`) vs 0.19 on the mounted input.
Noise floor: ~0.03 per label; 0.008 macro.
Cost: 0 GPU h (~1 h CPU wall); +70 % loader bytes per study.
If it works: c02 is the default cache for every new member; c01 stays mounted for the trained members
(mixed-geometry inference shipped: one decode-once pass per geometry group).
If it fails: (no per-label movement) keep c02 for the 336-px hybrids anyway; the band was not the lever.
Depends on: P-01 (the c01 design it extends).

---

## Rejected without testing

Merged from brainstorm.md and research.md §5; one line each. Do not resurrect without a new reason.

| Idea | Why | Source |
|---|---|---|
| Text branch at inference | `test.csv` has no `Report`; nothing to read | CLAUDE.md |
| Horizontal flip — **both variants** (with or without medial↔lateral swap) | undoes laterality normalisation; the swap is anatomically wrong (MCL has no lateral counterpart); P-05 does *not* make it legal | [pilkwang], traps.md, critic item 23 |
| Vertical flip; zoom-out with padding | off-distribution; fabricated tissue | [pilkwang], [Guo et al.] |
| Geometric TTA | degraded 11/12 medical pairs; flips hurt knee OA | [2604.09697], [2311.06118] |
| Calibration, Platt scaling, thresholds, `pos_weight`, label smoothing on soft targets | AUC reads rank order only | metric arithmetic, brainstorm.md |
| Averaging probabilities across folds/models | most confident model dominates; rank-mean instead | brainstorm.md |
| Full fine-tuning or best-epoch selection on 58 gold (~12/fold); a gold fine-tuning stage | SE 0.09, coin flip, our NaN-fold bug; gold role resolved as validation + weight arm | experiments.md, [Andre et al.], critic 25 |
| Forking the public 0.95 ensemble; tuning on public LB | author-labelled overfit; 0.001–0.003 movements; the 0.936 notebook's gold-58-tuned per-label weights + "clinical residual" are worth **+0.001** over its untuned 0.935 (read in full 2026-08-30) | mattiaangeli (not re-read), `crazy_good_rsna.ipynb` (research.md §2.7.1), CLAUDE.md |
| The 0.936 notebook's **88-feature stacking calibrator** (rank blocks + cross-view deltas + 12 protocol counts, w 0.4 on 7 labels) | decoded 2026-08-30: coefficients are ≈ a per-label reweighting of the same three views (protocol columns ≤ 0.003); est. +0.002–0.005 and fragile to a protocol-mix shift on private | cell-level re-read (research.md §2.7.1) |
| **Clinical residual** cross-label adjustments (`ACL −0.10 × mean rank(Contusion, Lateral Meniscus)` etc.) and correlation-guarded per-label fusion weights | the notebook itself: "an aggressive leaderboard experiment, not an unbiased estimate of private-test performance"; +0.001 stated | cell-level re-read |
| Per-label **max / top-2 pooling over 3 slice-offset views** as a default (P-12 `tta_pool="focal"`) before it is measured | with 3 views the max is a noise amplifier; measure `mean` and `focal` side by side in `oof_eval` first — the notebook applies it over ~14 windows, not 3 | our reading of P-12 |
| Evaluating CoAtNet-2 @384 on **all 60 windows** at rerun | 60 × 47.7 GMACs per study ≈ 0.6–1.2 s on a T4 before decode; the notebook caps at 42 (`K_EVAL`) — `eval_windows=42` in `v10c`, same value in `oof_eval` | design review 2026-08-30 |
| Random or report-only K-fold as the comparison metric | grouped vs random gap up to +0.136 | [EXPERIMENTS.md] |
| Native 3D CNN / nnU-Net / segmentation-first | 0.69 vs 0.85 (p=0.001); multi-A100 budgets | [MST], [CoPAS], [MIC-DKFZ 2025] |
| Frozen DINOv2 + head as the final model | 0.79 vs 0.85 knee; 0.776 vs 0.866 LB | [MST], sadamtorres (not re-read) |
| Backbone LR ≥ 5e-5 uniform | every medical recipe ≤ 2e-5; 1e-3 collapse | [2501.14685], [dinov2 #276] |
| ViT-B/L before a second family | S ≈ B in three sources; 3× cost | [SB-SSL], [2509.06467], [2402.07595] |
| DINOv2 → DINOv3 swap at 224 as an accuracy gain | ±0.002–0.008; wins only at 512 | [AnyMC3D], [2510.07191] |
| BiomedCLIP / MedSAM / RAD-DINO / OrthoFoundation | far below general ViTs; CXR-only; weights not public | [2501.14685], [2601.18250] |
| EfficientNet-B0 mean-pool | 0.664 vs 0.809 public | [JunhaoLiXD] |
| Laterality tag alone / default L / IPP-corner rule; pixel-flipping sagittal slots | tag missing 50.7%; corner 58.8%; SAG stacks are order-reversed | [FINDINGS.md], [pilkwang] |
| Filename / InstanceNumber slice ordering | ρ ≈ −0.01 | experiments.md |
| Crops ≥ 160 mm; resolution > 512 | skipped on 60% of series; ViT losses −6.6/−7.9 pp | [pilkwang], [2510.07191] |
| N4 / Nyul / VOI-LUT before normalisation | segmentation/radiomics evidence only; infeasible at 24k series | [2307.03827] |
| Decoding DICOM in the DataLoader each epoch; float32 caches; `.npz` + mmap; GPU decode at ≤ 512 px | 100× slower; 29.6 GB; mmap ignored; ~1–2.5× | [hida1211], [NumPy #5976], [nvImageCodec] |
| `pip install` at scoring time | internet off | [pydicom plugin table] |
| bf16 on T4; channels_last for ViT; `torch.compile` by default | no bf16 tensor cores; cuDNN-only; compile > gain (SDPA is the cheap win — P-08) | [PyTorch memory_format], critic 27 |
| More LLM label sources, Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2 in literature, **~1.5 here** (φ 0.88); our 0.002 spread | [2605.29800], [BoxWRENCH], experiments.md, label_audit.md |
| Co-teaching / DivideMix / DISC; focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE | [LNMBench], [RAL], [Xin et al.] |
| Calibrated priors with a 50% floor for zero-support states | OOF collapsed to base rate | [JunhaoLiXD V02] |
| Translate-then-extract; sub-3B extractors; 70B on 2×T4 | precision loss; F1 0.74; ~40 GB weights | [2602.21374] |
| Hosted LLM APIs for report text | plausible Rule 4.b violation; open-weights parity | CLAUDE.md, [Radiology 2025] |
| Extending epochs before the cache exists | 6–8 h/fold does not fit 9 h | experiments.md |
| In-domain SSL continued pretraining as a first priority | in-domain ViT still below ImageNet AlexNet on MRNet | [SB-SSL] |
| Auxiliary report-reconstruction head | same weak supervision as the targets; nothing new to learn | brainstorm.md #10 (speculative, unranked) |
| Treating gold deltas < 0.05 or LB < 0.005 as real | noise floor | CLAUDE.md |
| P100 accelerator | no Pascal kernels | CLAUDE.md |

---

## Open questions needing a browser

Kaggle pages are JS-rendered; the CLI exposes none of these. Each blocks a card.

| Question | Blocks | Where to read |
|---|---|---|
| Exact Rules text: Rule 4.b (data off-platform), winner licence clause, ≤ 9 h runtime, internet-off | P-16 (risk), P-10 (licence), P-18 | competition Rules page |
| Efficiency Prize formula and Base — is runtime CPU-only? | P-18 | Efficiency evaluation page; `ryanholbrook` notebook body |
| radimagenet.com Terms & Conditions for the weights | P-10 | https://www.radimagenet.com/ |
| Hidden test size and site/vendor mix (community: ≥ 1,322 studies, 16–19 sites) | P-02 grouping strictness, P-18 time budget | competition Data page; header pass at rerun |
| Discussion threads: TransferSyntax reports, hidden-test decode issues, gold severity protocol | P-19, P-16 grading rubric | Discussion tab; pilkwang notebook gold-protocol cells |
| Kaggle per-kernel output cap (assumed ~20 GB) and Datasets size limits | P-01 sharding, P-11 | Kaggle docs: Notebooks → Output; Datasets → limits |
