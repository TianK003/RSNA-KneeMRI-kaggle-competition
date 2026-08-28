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
| **OOF macro-AUC vs teacher** | all 4,407 studies (fold-out) | **~0.01 — asserted, not yet measured** (critic item 16). First cache-era run is a 2-seed repeat of fold 0 to measure it (P-02 step 1). | breakage, epoch choice, every A/B |
| Gold macro-AUC | 58 labelled studies | **0.05** macro (CLAUDE.md rule); per-label Hanley–McNeil SE ≈ 0.09 ([Andre et al.](https://arxiv.org/html/2601.17103)). A 3,000-rep study-level bootstrap of the teacher's gold macro-AUC gives SD 0.017. | direction only, never a gate |
| Public LB | hidden test (size unknown) | **0.005** (top ten span 0.006) | direction only; never tuned on |

Judge label changes on **coverage per language + OOF**, not gold alone (experiments.md
convention). Per-label OOF floors are wider (~0.015–0.02, asserted) than the macro floor.

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
| P-01 | Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised) | ⏳ running (shards A/B on Kaggle) | very high (unblocks P-02…P-15) | 1 session, ~300 lines | — |
| P-02 | Seed-noise baseline, then site-grouped folds + grouped-vs-random OOF | 💡 untested | very high (validity of every comparison) | 0.5 session, ~40 lines | P-01 header manifest |
| P-03 | Fine-tuning recipe: (a) LR 2e-5 + EMA shipped; (b) LLRD 0.75 vs uniform | (a) 🔧 shipped, effect pending · (b) 💡 | medium | (a) done · (b) 0.3 session | P-01 for (b) |
| P-04 | Fixed-epoch schedule, 8 epochs, chosen from fold-mean OOF curve | 💡 untested (fixed-epoch *selection* already shipped in v02) | medium-high | 1 session | P-01, P-03 |
| P-05 | Laterality normalisation from DICOM geometry | 🔧 implemented in cache kernel, effect pending | medium (5/12 labels side-specific) | 0.3 session, ~80 lines | P-01 |
| P-06 | Per-label failure analysis + gold_weight {1,3,8} arm + slot-fill census | 🔧 logging shipped · arms 💡 | high (finds the cheapest lever per weak label) | 0.1–0.5 session | — |
| P-07 | Synovitis ← Effusion back-fill (measured, not adopted) | 🔁 measured on gold; OOF pending | low-medium | done (audit) | P-06 OOF |
| P-08 | Slices per slot 6 → 12–16, per-plane bands, random offsets | 💡 untested | medium-high | 0.5 session | P-01 |
| P-09 | Per-label masked attention head over slots | 💡 untested | medium | 0.3 session, ~80 lines | P-01 |
| P-10 | Second architecture family (timm ConvNeXt first; RadImageNet behind a flag) | 💡 untested | medium (diversity) | 1 session | P-01, P-04 |
| P-11 | Resolution 224 vs 336 after the 130 mm crop | 💡 untested | low-medium | 1 session + sharded cache | P-01, P-08 |
| P-12 | Slice-window TTA (label-safe only) [our hypothesis] | 💡 untested | low | 0.1 session | P-01 |
| P-13 | 3 vs 5 folds under a fixed session budget [our hypothesis] | 💡 untested | low | 1–2 sessions | P-10 |
| P-14 | DINOv2-S vs DINOv2-B (registers variant noted) | 💡 untested | low | 1 session | P-10 |
| P-15 | DINOv3-S/16 as diversity member | 💡 untested | low | 1 session + model mirror | P-10 |
| P-16 | Re-labelling with an open-weights LLM inside Kaggle (graded, native language) | 💡 untested | high but slow (raises the teacher ceiling) | 1–2 sessions | P-06 audit (done) |
| P-17 | Noise-robust loss / self-distillation | 💡 untested | low | 0.3 session each | P-01, P-04, P-06 |
| P-18 | Efficiency-track variant + decode-once inference + slot census | 🔧 infer mode + loud-failure submission shipped · variant 💡 | medium (separate prize) | 0.2 session + browser | P-01 |
| P-19 | Decoder wheels + TransferSyntax census | 💡 untested | insurance | 0.1 session | P-01 header pass |
| P-20 | Leave-one-slot-out ablation, T1 slot retirement | 💡 untested | low-medium | 0.1 session | first real model |

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
Noise floor: 0.01 OOF macro; 0.015–0.02 per label. There is no v01 baseline fold, so the first read is a *sanity* read, not an A/B.
Cost: done. A/B would cost one extra fold-0 run from the cache (~0.3 session).
If it works: rare-label OOF up and negatives near 0; card closes in experiments.md as ✅.
If it fails: (OOF or pred_std no better) blend was never the bottleneck; keep the probability blend anyway as the semantically correct target and move on.
Depends on: nothing.

### P-01 Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised)
Status: ⏳ running — `src/cache_pipeline.py` shipped; shards A/B running on Kaggle. As built: **K=16 slices/slot, 224 px, 2 shards** (≈21 GB total, ~10.6 GB per shard); sagittal stacks sorted along +x (patient left) with a fixed sign and reversed for right knees; coronal/axial mirrored when (columns run to +x) == is_right; failed slices replaced by the nearest good neighbour (not zeros); presence mask written into `manifest_shard{k}.csv`; ordering drops unreadable files instead of falling back to filename order.
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
Status: 💡 untested.
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
Status: 💡 untested for the epoch *count*. Fixed-epoch *selection* is already what v02 does (`best.pt` = EMA weights after the last completed epoch, no best-epoch pick on gold); the per-epoch OOF csvs it writes are the input to this card.
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
Status: 🔧 implemented in the cache kernel, effect pending. As built: sagittal stacks sorted along +x (patient left) with a fixed sign and reversed for right knees; coronal/axial mirrored when (columns run to +x) == is_right.
Hypothesis: Canonicalising knee side (mirror W on COR/AX, reverse SAG slice order) removes a chirality the model must otherwise learn twice and lifts Medial/Lateral Meniscus, Medial/Lateral OA and MCL on OOF.
Origin: public consensus / competition write-up.
Evidence: Laterality tag missing on **12,367/24,371** series; centre-x sign rule **97.4%** (98.5% with 20 mm dead zone); IPP-corner rule 58.8% (verified [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); plane-specific operation ([pilkwang] source, pulled; [JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)); laterality metadata errors are common ([PMC6646614](https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/)); lateral meniscus ~0.1 harder than medial even at 18k studies ([Fritz](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)).
Rule (critic item 8): side = tag, else sign(median image-centre x) with 20 mm dead zone; **tag-vs-geometry conflict → no-op and counted**; unresolved → no-op and counted; `ImageOrientationPatient` row/column cosines canonicalised for all planes (sagittal reversal fixes stack direction only).
Measure: OOF AUC over 4,407 for the five side-specific labels before/after; tag-vs-geometry agreement on tagged series (assert ≥ 0.95); fraction unresolved and fraction conflicting; visual spot-check of ~20 knees.
Noise floor: 0.01–0.02 per label OOF; gold cannot resolve it.
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
Noise floor: per-label OOF 0.015–0.02; gold per-label 0.09 (diagnostic only); the gold_weight arm is likely 🔁 on OOF.
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
Status: 💡 untested as an A/B; the cache is built at **K=16 @ 224 in 2 shards** (≈21 GB total, ~10.6 GB per shard), so K ≤ 16 can be sampled from it without rebuilding.
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
Status: 💡 untested.
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
Status: 💡 untested.
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
Status: 💡 untested.
Hypothesis: Higher effective mm/px helps the small focal labels (Lateral Meniscus, PF OA, Fracture, Contusion).
Origin: peer-reviewed / competition write-up.
Evidence: 224 → 512 median +1.05 pp AUROC on CXR, concentrated in focal findings, but DINOv2-ViT nearly flat (VinDr 89.2 → 89.1) ([2510.07191]); 224 → 336 +0.017 LB at 2.25× FLOPs, preds correlate 0.90 (sadamtorres, **notebook, not re-read**); yu4u's crop+384 gain is confounded with cropping ([yu4u deck](https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui)). HF `Dinov2Model` needs `interpolate_pos_encoding` for non-224 input (critic item 20).
Measure: per-label OOF over 4,407 for the four focal labels at 224 vs 336, K fixed; adopt only if their mean moves > 0.02.
Noise floor: 0.015–0.02 per label OOF.
Cost: 1 session (2.25× tokens) **plus** a separate sharded cache at 336 (K=12 @ 336 ≈ 36 GB → ≥ 2 kernels); do not build it before P-08 fixes K.
If it works: 336 member (initialised from the 224 checkpoint) added to the blend.
If it fails: stay at 224; spend budget on slices/folds.
Depends on: P-01, P-08.

### P-12 Slice-window TTA (label-safe only) [our hypothesis]
Status: 💡 untested.
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
Status: 💡 untested; low priority.
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
Status: 🔧 inference path shipped in v02/kernel v4 (MODE=infer from mounted checkpoints; **no placeholder — loud failure**; refuse-to-submit-constants; image-root probing via a shallow glob that never descends into train_series/test_series; infer mode reads slices_per_slot/triplet_gap/img_size from the checkpoint's saved config so FORCE_SMOKE cannot change the model's inputs; MODE=auto picks infer only if every configured fold has a mounted best.pt); efficiency variant 💡.
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
Noise floor: 0.015–0.02 per label OOF.
Cost: 0.1 session (inference-time flag).
If it works: slot budget reallocated to K in fat-sat slots.
If it fails: keep all 6 slots.
Depends on: the first real model (v02 fold 0); P-01 for speed.

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
| Forking the public 0.95 ensemble; tuning on public LB | author-labelled overfit; 0.001–0.003 movements | mattiaangeli (not re-read), CLAUDE.md |
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
