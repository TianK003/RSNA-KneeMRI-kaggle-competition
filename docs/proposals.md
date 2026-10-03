# Proposals — ranked, testable cards

*Written 2026-08-28. The forward-looking half of the lab notebook: every idea we intend to
test, written as a falsifiable card **before** it is run. Companion to
[experiments.md](experiments.md) (things measured, with verdicts) and
[research.md](research.md) (the evidence base these cards draw on).*

**Rewritten 2026-09-27** after a four-reviewer audit: measured and retired cards are one-line pointers in *Closed cards*; the
full pre-rewrite text is `git show 8304c96:docs/proposals.md`.

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

| Metric | Set | Floor | Used for |
|---|---|---|---|
| **Public LB, solo** (`rsna-knee-infer`, one member or our own blend; **≈ 30 min** send→score, P-41) | hidden test (public part) | **0.004 for one-seed deltas** — P-44 measured the Kaggle-retrain spread once: `v09u` (seed 43) = `v09r` = 0.927, s = 0.000 (#24); one draw, σ poorly estimated, LB rounded to 3 decimals — keep **0.005** for cross-recipe claims | member quality and **every target-source change** (traps 39) |
| Public LB, fork (`rsna-knee-fork`, ≤ 8 h 06 min to score + hours of queue) | same | 0.005 | final candidates only — at β 0.10 it cannot read member quality (#13 = #15 = 0.942, #17 0.941) |
| Fold-0 OOF vs LLM teacher | 882 fold-0 studies | 0.008 macro / ~0.03 per label, measured (P-02) | recipe A/Bs **on LLM targets only**; inadmissible for target changes (traps 39); production (`train_all`) members have none (traps 32) |
| Gold-58 | 58 labelled studies (held out for production members: `split_studies` trains `is_gold == 0` only) | 0.05 macro unpaired rule; paired study-level bootstrap SD ≈ 0.007 for near-identical members | direction only, never a gate |

Judge target changes on the solo LB (traps 39); gold-58 is direction only.

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
result*, per unit of cost. "Depends on" lists hard blockers only. EVs are solo-LB priors from the 2026-09-27 audit
(reviewer B's pricing unless stated), not measurements.

### Live cards, ranked by expected value per cost

| rank | id | title | status | expected value | cost | depends on |
|---|---|---|---|---|---|---|
| 0d | P-56 | Dense-slice input c03 (24/24/24/14/8/8, 150 mm) `v11a` / `v11b` | ⏳ **trained** (session D = `rsna-knee-folds` v10, 3.53 h): gold-58 `v11a` 0.9204 / `v11b` 0.9167, c03 pair 0.9207 vs c02 pair 0.9065 (+0.014, SD 0.005, 10/12 up — direction only); Datasets `rsna-knee-ckpt-v11a` / `-v11b`; **🔁 solo reads (2026-09-29): #30 `v11a` 0.932 (best single) / #31 `v11b` 0.929 → m = 0.9305 (✅ bar 0.9315); #32 c03 pair 0.932 vs the c02 pair 0.930** — consistent sign, under every floor; not the production input by rule, the axis stays open; #38 c03 pair + `v13c` 0.932 = the pair | the only untested input axis since c02 | ~3.7-4.4 GPU-h, 3 submissions | — |
| 0f | P-58 | Local-CPU open-weights LLM relabel as a 4th vote (Scott's Gemma route) | 💡 future, not scheduled (Tian 2026-09-28) | 0..+0.002 | 0 GPU; overnight CPU | — |
| 0g | P-61 | CoAtNet learning-rate probe upward (`lr_backbone` 2e-4, LLRD 0.85) | ⏳ **RUNNING: arm `v11dl` in session B = `rsna-knee-train` v41 (12:35 UTC; smoke v40 green)** = v11d + lr 2e-4 / LLRD 0.85 on the Raptor + D4 targets; runs in session B beside `v11d` (same-session control, one seed each); the forum mining calls it ranked too low (research.md 2.7.3) | 0..+0.005 — the CoAtNet trains under the LLRD that under-trained the ResNet by 0.069 on gold | ≈ 3 GPU-h (2 arms, c03) or ≈ $5 RunPod; 2 solos | — |
| 0h | P-60 | "Noisy student" regularisation on the c03 CoAtNet (`v11n` / `v11n2`: drop-path 0.1, heavy aug, 12 epochs) | 🔁 **READ 2026-10-03: #39 `v11n` 0.932 / #40 `v11n2` 0.932 → m = 0.932 vs 0.9305 (+0.0015, band 0.926–0.935) → INCONCLUSIVE; not adopted (1.5× training time); the CoAtNet does not gain from regularisation + longer schedule.** Part 2 = `rsna-knee-train` v39 (2.84 h, first Kaggle resume green); gold-58 SWA `v11n` 0.9152 / `v11n2` 0.9166. Part 1 DONE = `rsna-knee-folds` v11 (2.61 h, green: both arms guard-stopped in epoch 5, `_last.pt` ×2; gold-58 EMA at epoch 4 `v11n` 0.9166 / `v11n2` 0.9129 = `v11a` / `v11b` at their epoch 4, direction only); **part 2 = `rsna-knee-train` v39, pushed 2026-10-03 09:08 UTC (≈ 3.0 h) ⏳; do not push `rsna-knee-folds` before it is green** | +0.002..0.008 (literature + every prior RSNA winner) | ≈ $4 RunPod, 2 solos | — |
| 0i | P-62 | Silence-aware teacher mix (Raptor 0.75 where the report is silent, 0.5 where it speaks) | ⏸ **parked 2026-10-03: the real run (`rsna-knee-train-b` v1, pushed 09:09 UTC) was STOPPED by Tian at ≈ 09:28 UTC (≈ 0.3 GPU-h spent, nothing usable)** — its expected gain (+0.001..0.002) cannot reach its own ✅ bar (+0.0045), so it does not get a session of its own; a candidate to bundle into the final retrain only. Implemented 2026-10-01 (arms `v11s` ‖ `v11s2`; Kaggle smoke `rsna-knee-train` v38 green); priced on gold-58 at target level: 0.9300 vs 0.9268 (+0.0032, SD 0.0018; flat mixes at the same mean Raptor weight +0.0004) | 0..+0.002 — likely under the 0.004 one-seed floor | one c03 session ≈ 3.5 GPU-h + 2 solos | — |
| 0j | P-63 | Per-finding spatial reader + slot-count correction (D4's head, ported at 224) | ⏳ **RUNNING as ONE seed (`v11p`) in session A = `rsna-knee-train-b` v4 (pushed 12:21 UTC; smoke v3 green) beside `v13h`** (2026-10-03: one seed per idea; the forum gives it a weak prior — Will's region tokens ≈ 0, Tucker 0.94+ with no attention). 🔧 implemented 2026-10-03 (`spatial_reader` / `slot_count_norm`, arms `v11p` ‖ `v11p2`; unit checks green incl. untrained reader == parent model to 4.8e-7; local CPU smoke green; Kaggle smoke `rsna-knee-train-b` v2 GREEN (0.11 h: `ok  arm` ×2, Raptor table read, `reseeded 43`, SWA); real build `artifacts/train_p63_real.py`) — the one structural head difference between D4 (gold 0.9302, gold-selected) and `v11a` (0.9204); D4's plain Global96 baseline with this reader reaches 0.925–0.927 on its gold-best epochs | unknown; a structural change, the kind that can clear the +0.0045 two-seed band | ≈ 60–80 lines + one c03 seed pair ≈ 4 GPU-h; 2 solos | P-45 first (GPU priority, Tian 10-03) |
| 0k | P-64 | Long, heavy-augmentation CNN (`v13h`: ResNet-34 on c03, CNN LR 3e-4 uniform, frozen BN, aug heavy, drop-path 0.1, 30 epochs) | ⏳ **RUNNING in session A = `rsna-knee-train-b` v4 (12:21 UTC; Kaggle smoke v3 green: frozen BN, drop-path 0.1) beside `v11p`**; local smoke green | the best-evidenced recipe in the forum: Tucker / Myo / Scott / CoolinLai / SpeedSci reach 0.936–0.954 with ResNet / EfficientNet @224 and long, heavily augmented schedules; ours `v13c` (12 ep, light aug) 0.921; also the ensemble's second family | ≈ 3.4 GPU-h in a shared session; 1 solo | — |
| 4 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15 | decides what the private LB scores | a browser session; ≤ 1 fork check | P-40 ✅ closed (#22 / #27), Rules page |
| 5 | P-18 | Efficiency track with the solo member | 💡 (robustness half shipped); candidates `v11a` 0.932 / `v13c` 0.921 at ⅓ the inference cost (#37) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 6 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low — P-49 priced it on Raptor itself: matched mix 0.75 − 0.5 = −0.002 (SD 0.003) on gold | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 7 | P-45 | Second image teacher | ⏳ **pass DONE 2026-10-03 (`rsna-knee-teacher-d4` v2, 2.37 h, 4,349/0 failed; in-sample check passed: ρ D4~LLM 0.695 vs Raptor 0.652); `d4_teacher.csv` in `rsna-knee-teacher-tables`; student `v11d` RUNNING in session B = `rsna-knee-train` v41 (12:35 UTC)**. Step 1 read 2026-10-03: 0.5 LLM + 0.25 Raptor + 0.25 D4 = 0.9331 vs 0.9268 on gold (+0.0063, SD 0.0041, 8/4 labels; D4 ~ Raptor ρ 0.757) — direction only, the largest target-level gain since P-49**; **Tian's go 10-03; builder `src/build_d4_teacher_pass.py` (56 checks); gold spike `rsna-knee-teacher-d4` v1 ✅ 0.9301 vs 0.9302 on D4's original grid (0.1 h); full pass = v2 pushed 09:58 UTC (≈ 2 GPU-h) ⏳; student arms `v11d` / `v11nd` ready.** Mapped 2026-10-01 → D4 alone (gold 0.9302, est. ≈ 2–3 h pass) | +0.001..0.004 (CoAt family) / −0.002..+0.004 (DINO + A5); fork ≈ 0 | spike 0.2–0.3 h, pass ≈ 8 GPU-h, 100–490 lines, 2.8 h arm | P-49, P-44, the 10-03 reset |
| 8 | P-46 | Upgrade the LLM half of the targets (absorbs P-16, P-30) | 💡 low | 0..+0.002 (dread vote) / +0.001..0.003 (re-label) | ≈ 2.8 h per arm; step 2 a new kernel | P-44 floor |
| 9 | P-48 | Final-member polish: gold-58 as training rows + seed averaging | 💡 contested, parked | +0.001..0.002, unreadable by construction | part of the final retrain | P-50 decision |
| 10 | P-51 | Teacher-aware confidence weights | 💡 low | 0..+0.002 | ≈ 20 lines + 2.8 h; 1 solo | P-44 floor |

### Closed cards

| id | title | final verdict | where the result lives |
|---|---|---|---|
| P-00 | Target scale: probability-space blend, not rank percentiles | ✅ the target of every member since v02 (confident negatives no longer at 0.28–0.39); never A/B'd | experiments.md 2026-08-28 "Rank-percentile targets put confident negatives at ~0.3" |
| P-01 | Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised) | ✅ LB 0.841 → 0.871, OOF 0.821 → 0.843 | experiments.md 2026-08-28 "Preprocessing cache built", "v03 fold 0 trained from the cache" |
| P-02 | Seed-noise floor, then site-grouped folds | ✅ step 1: OOF floor 0.008 macro / ~0.03 per label · step 2 (site folds) retired — every remaining decision is a solo-LB read; the LB-level floor → P-44 | experiments.md 2026-08-29 "Run-to-run noise floor MEASURED" |
| P-03 | Fine-tuning recipe (LR 2e-5 + EMA; LLRD) | retired — LR 2e-5 + EMA is the base; the LLRD question is from the DINOv2 era; the hybrid LR is settled by P-34 | experiments.md 2026-08-28 "v02 fold 0"; 2026-09-23 "Round 2" |
| P-04 | Fixed-epoch schedule | → P-29: production = 8 epochs | experiments.md 2026-08-29 "8-epoch matched head A/B"; 2026-09-22 "P-29 epoch-budget probe" |
| P-05 | Laterality normalisation from DICOM geometry | ✅ −0.0147 OOF when removed (1.9× the floor) | experiments.md 2026-08-29 "Laterality normalisation confirmed" |
| P-06 | Per-label failure analysis, `gold_weight` arm, slot census | retired — `gold_weight` is inert under `train_all`; the logging is standard; gold rows as training data → P-48 | no entry (never run as an arm); traps 32 |
| P-07 | Synovitis ← Effusion back-fill | retired — the Raptor table moved Synovitis gold 0.744 → 0.823 (`v09a` → `v09r`) | experiments.md 2026-08-28 "Synovitis ← Effusion back-fill"; 2026-09-26 "`v09r`" |
| P-08 | Slices per slot, per-plane bands, random offsets | ✅ jitter (+0.0113 OOF, LB 0.877); the K sweep superseded by c02 + random windows | experiments.md 2026-08-29 "Slice jitter as augmentation" |
| P-09 | Per-label masked attention head over slots | ✅ +0.0103 at matched 8 ep (the c01 slot head); superseded by `window_attn` (P-25) | experiments.md 2026-08-29 "8-epoch matched head A/B" |
| P-10 | Second architecture family (ConvNeXt; RadImageNet) | 🔁 ConvNeXt-T measured (#8 0.900, +0.004) · RadImageNet half retired (already the anchor's stage 3; licence) | experiments.md 2026-08-30 "ConvNeXt-Tiny member `v06c`" |
| P-11 | Resolution 224 vs 336 | superseded by P-43 — the `v10c` @384 (0.8641) vs `v09h` @224 (0.8683) comparison confounded backbone size | experiments.md 2026-08-30 "⭐ What made the 0.936 notebook good" |
| P-12 | Slice-window TTA | 🔁 +0.003–0.006 per c01 member, +0.0016 on the blend; window members evaluate every window | experiments.md 2026-08-30 "P-12 slice-offset TTA, measured on the RunPod pod" |
| P-13 | 3 vs 5 folds | retired → P-44 — production is one all-data model; folds were replicates (#6 +0.009 alone, #7 +0.000, #11 +0.001) | Scoreboard #6 / #7 / #11 |
| P-14 | DINOv2-S vs DINOv2-B | retired — P-42: a second backbone on the same targets adds nothing | experiments.md 2026-09-27 "Submission #23" |
| P-15 | DINOv3-S/16 / 16-channel member | retired — 16-ch dead as built (`v07s` OOF 0.7366); DINOv3 in *Rejected*; the anchor's A5 stage is that representation | experiments.md 2026-08-30 "16-slices-as-channels DINOv2-S member `v07s`" |
| P-16 | Open-weights LLM re-label inside Kaggle | → P-46 step 2 | P-46 below |
| P-17 | Noise-robust loss / self-distillation / AUC-margin | retired — self-distillation = P-38 ❌; losses in *Rejected*; its OOF measure is invalid (traps 39) | experiments.md 2026-09-24 "Submission #19" |
| P-19 | Decoder wheels + TransferSyntax census | retired as de-risked — 0 decode failures over 4,407 (c02 build) + 4,349 (Raptor pass) studies; public and private come from one rerun | experiments.md 2026-08-30 "Cache v2 (`c02`) built"; 2026-09-26 "Raptor teacher pass complete" |
| P-20 | Leave-one-slot-out ablation, T1 slot retirement | retired — needs OOF; production members have none | traps 32 |
| P-21 | Two heads on one backbone as the ensemble axis | ✅ LB 0.896 (+0.019, #5) — head diversity | experiments.md 2026-08-29 "Two heads rank-blend to 0.8670"; Scoreboard #5 |
| P-22 | Checkpoint on OOF-vs-teacher, not the last epoch | ✅ (+0.0128 split-half, concat head); moot in production (`ckpt_policy="last"`) | experiments.md 2026-08-29 "P-22: checkpoint on OOF-vs-teacher" |
| P-23 | Multi-family rank fusion | ✅ LB 0.913 own blend (#11); the c02 lane is mined out; its fold-0 ρ rule cannot judge `train_all` members | Scoreboard #9–#11; experiments.md 2026-08-30 "⭐ What made the 0.936 notebook good" |
| P-24 | Off-Kaggle runner (RunPod) + 2×T4 | ✅ runner (shipped `v09t` / `v09r` / `v08r`); the 2×T4 half → P-31 | experiments.md 2026-09-23 "RunPod arms on a 4090"; `scripts/runpod_bootstrap.sh`, `scripts/runpod_chain.sh` |
| P-25 | Window-attention head + random-window training | ✅ `v08w` OOF 0.8648, 12/12 labels up — the member recipe | experiments.md 2026-08-30 "`v08w` fold 0" |
| P-26 | Cache v2 (`c02`) | ✅/🔁 built (4,407 studies, 35.8 GB, 0 decode failures); MCL +0.028 (claim holds), Lateral Meniscus +0.009 (does not) | experiments.md 2026-08-30 "Cache v2 (`c02`) built", "`v08w` fold 0" |
| P-27 | Public-stack fork + our arm | ✅ 0.913 → 0.942; our arm ±0.000 at β 0.10, −0.003 at β 0.20; now infrastructure, reads in P-40 | experiments.md 2026-09-22 "P-27 read-out with the control" |
| P-28 | Production training regime | ✅ all 4,349 studies, 8 epochs, SWA 5–7; `v09a` ≈ 2.6 h on Kaggle, `v08a` ≈ 1.2 h | experiments.md 2026-09-23 "S2 production retrain" |
| P-29 | Epoch-budget probe | ❌ 16 epochs over-train (OOF peak 0.8731 at epoch 8, 0.8607 at epoch 15) | experiments.md 2026-09-22 "P-29 epoch-budget probe" |
| P-30 | `dreaddevelopment/rsna-knee-labels` as a 4th source | → P-46 step 1 (died only for lack of a judge; the solo LB is one now) | experiments.md Label sources 2026-09-22 "P-30" |
| P-31 | Two arms per Kaggle session (`PARALLEL_ARMS`) | ✅ two arms in 3.22 h wall at 1.00× / 1.12× solo speed | experiments.md 2026-09-23 "S1 A/B on both T4s" |
| P-32 | Multi-study BatchNorm batches (`batch_studies=2`) | 🔁 `v09b` 0.8690 vs `v09h` 0.8683; rides in the production recipe | experiments.md 2026-09-23 "S1 A/B on both T4s" |
| P-33 | Light train-time augmentation | 🔁 `v09c` +0.0039 over `v09b`; rides in the production recipe | experiments.md 2026-09-23 "S1 A/B on both T4s" |
| P-34 | CoAtNet backbone LR 3e-5 | ❌ `v09d` 0.8596 vs `v09c` 0.8730 (10/12 down) | experiments.md 2026-09-23 "Round 2" |
| P-35 | Light augmentation on the DINOv2 arm | 🔁 `v08c` 0.8650 vs `v08w` 0.8648 | experiments.md 2026-09-23 "Round 2" |
| P-36 | CoAtNet 16 epochs × LR 3e-5 | retired — dropped by rule after P-34; every recipe knob is now measured ≈ 0, so not re-openable | experiments.md 2026-09-23 "Round 2" |
| P-37 | Per-label `pos_weight` [1, 10] | 🔁 `v09f` 0.8717 vs `v09c` 0.8730 | experiments.md 2026-09-23 "RunPod arms on a 4090" |
| P-38 | Self-distillation targets | ❌ in production: #19 `v09t` 0.917 vs 0.918 (the fold-0 +0.0109 was agreement with the LLM teacher) | experiments.md 2026-09-24 "Submission #19"; traps 39 |
| P-39 | Raptor teacher pass | ✅ #20 `v09r` 0.927 vs #18 0.918; the pass took 7.76 GPU-h. Note: pilkwang's public `oof.npz` = honest 5-fold DINOv2-S@336 OOF on all 4,407 studies, gold 0.840 — a free sanity reference for any future image teacher | experiments.md 2026-09-26 "Raptor teacher pass complete"; 2026-09-27 "Submission #20" |
| P-41 | Faster solo scoring (threaded scan + 8 decode workers) | ✅ solo scores in 28.3–29.3 min (#19: ≤ 42 min) | experiments.md Infrastructure 2026-09-27; 2026-09-27 "Submission #23" |
| P-42 | Two-family Raptor-distilled solo blend | 🔁 #23 = 0.927 = `v09r` alone; gold within-class ρ 0.861 vs 0.777 for the LLM-target pair | experiments.md 2026-09-27 "Submission #23" |
| P-43 | Student resolution: CoAtNet-1 @320 on the `v09r` targets (`v09x`) | 🔁 #25 `v09x` 0.929 vs m 0.927 (✅ needed ≥ 0.932); 224 stays the production resolution; `v09x` = our strongest single member, 1.4× inference / 2.1× training | experiments.md 2026-09-28 "rsna-knee-train v30", "Submissions #24–#26" |
| P-44 | Kaggle-retrain seed spread + a 2-seed production member (`v09u`) | ✅ s = 0.000 (#24 `v09u` 0.927 = `v09r`) → one-seed deltas need ≥ 0.004; P-39 re-confirmed on Kaggle; #26 `v09r` + `v09u` 0.930 = the production member (+0.003, under the floor — adopted, not proven) | experiments.md 2026-09-28 "Submissions #24–#26"; traps 42 |
| P-59 | ResNet-34 pipeline control: CNN learning rate (`v13b`) + frozen BatchNorm (`v13c`) | ✅ the optimiser was the defect: gold-58 `v13a` 0.8306 → `v13b` 0.8992 (uniform 3e-4, 12 ep; train loss 0.471 → 0.391); 🔁 frozen BN (`v13c` 0.9014, +0.002); ρ to `v09r` 0.86, gold blends flat — LB (2026-09-30): **#37 `v13c` solo 0.921 ❌ (≤ 0.922), #38 `v11a` + `v11b` + `v13c` 0.932 = the c03 pair ❌** → P-57 closes (Datasets `rsna-knee-ckpt-v13b` / `-v13c`) | experiments.md 2026-09-29 "P-59" |
| P-40 | Raptor-distilled members into the fork | 🔁 closed: #22 (β 0.10) 0.942, #27 (β 0.20) 0.941 vs 0.942 — the fork does not register a 0.927 member at either weight; the pre-registered β 0.20 retry is spent → the fork is for final-selection builds only (P-50). Step B: `v08r` 0.918 solo (#21), not a fork member | experiments.md 2026-09-27 "Submission #22"; 2026-09-28 "Submissions #27–#28" |
| P-49 | Raptor over the 58 gold studies | 🔁 read (direction only): Raptor 0.9254, the 0.5/0.5 target 0.9268 (12/12 above the LLM), within-class ρ `v09r` ~ Raptor 0.835 (`v09a` ~ Raptor 0.742, `v09r` ~ LLM 0.408) → the fork cannot read `v09r`; matched mix 0.5 is the best on gold (0.75 −0.002, 1.0 −0.017) | experiments.md 2026-09-28 "P-49" |
| P-52 | Three-member production blend `v09r` + `v09u` + `v09x` | 🔁 closed: #29 = 0.931 vs #26 0.930 (band 0.931–0.933) — the pair stays (the third member costs 1.4× inference and read like a seed) | experiments.md 2026-09-29 "Submissions #29–#32" |
| P-54 | 5-fold cross-fit of the `v09r` recipe (`v09k0` … `v09k4`) | ❌ as a member: #33 fold ensemble 0.928 vs #26 0.930 (≤ 0.930; five 80 %-data models ≈ one all-data member); its OOF table `xfit_v09k` (gold 0.9028) fed P-55, which is ❌ too | experiments.md 2026-09-29 "P-54 cross-fit table", "Submissions #29–#32" addendum |
| P-55 | OOF soft-bootstrapped student `v09o` / `v09o2` (0.25 LLM + 0.375 Raptor + 0.375 `xfit_v09k`) | ❌ DEAD END: #34 / #35 0.927 / 0.927 → m = 0.927 < 0.9285; pair #36 0.928 ≤ 0.930 — the third null for OOF-derived targets (P-38, Nicolai); the OOF-target line closes (gold-58 had +0.007) | experiments.md 2026-09-30 "Submissions #34–#38"; traps 39 |
| P-57 | ResNet-34 on the `v09r` recipe (`v13a` → `v13b` / `v13c`) | ❌ closed: `v13a` gold 0.8306 was the ViT-tuned optimiser (P-59); with a CNN LR `v13c` reads 0.921 solo (#37, ≤ 0.922) and adds nothing to the c03 pair (#38 0.932) — the small-CNN route closes for accuracy; Efficiency track only (P-18) | experiments.md 2026-09-29 "Sessions C ‖ D", "P-59"; 2026-09-30 "Submissions #34–#38" |
| P-53 | Per-label public probe (ACL / MCL / PF OA / Lat Men at 0.5) | ✅ a measurement: #28 = 0.785 → those four average 0.935 on the public test vs 0.9275 for the other eight — the gold-58 "structural deficit" (+0.028) is gold-specific | experiments.md 2026-09-28 "Submissions #27–#28" |

---

## Cards

### P-52 Three-member production blend `v09r` + `v09u` + `v09x`
Status:       💡 new 2026-09-28 — no training needed; all three checkpoints are shipped (`rsna-knee-ckpt-v09r` / `-v09u` / `-v09x`,
              all mounted on `rsna-knee-infer`). **2026-09-29: placeholder `rsna-knee-infer` v25 pushed 11:36 UTC**
              (`artifacts/infer_p52_trio.py` = `src` + 3 seds).
Hypothesis:   adding the 320-px member to the 2-seed pair lifts the solo read above #26 (0.930): the 320 student sees detail the
              224 pair does not (Lateral Meniscus 0.901 vs 0.853 / 0.870 on gold), so it adds more than a third seed would.
Origin:       P-43 "If it works" (read `v09x` + `v09r` (+ `v09u`) as a blend); #24–#26.
Evidence:     for: #25 `v09x` 0.929 is our strongest single member; #26 showed same-recipe averaging reads +0.003; gold-58 rank-means
              `v09r` + `v09x` 0.9126, all three 0.9110 vs the pair 0.9065 (direction only, and gold-58 has had same-recipe
              directions wrong — traps 39). Against: P-43 was 🔁 (+0.002); ρ(`v09x`, `v09r`) 0.949 ≈ the seed twins' 0.952, so
              `v09x` may be just another seed; 320 raises the member's inference 1.4×.
Measure:      `INFER_MEMBERS = ["v09r", "v09u", "v09x"]` (flat rank-mean, `by_version`, one vote each), solo vs #26 0.930.
Noise floor:  the P-44 floor: **≥ 0.934 ✅ the 3-member blend is the production member / 0.931–0.933 🔁 (keep #26's pair — cheaper at
              inference) / ≤ 0.930 ❌** (bands as restated in the 2026-09-28 single-model plan, Step 5, before any read; the
              card's first draft had 0.927–0.933 / ≤ 0.926).
Cost:         ≈ 0.1 h T4 placeholder (one c02 decode shared by all three) + 1 solo (≈ 30 min to score).
If it works:  the 3-member blend is the production member and the fork candidate for P-50.
If it fails:  🔁 → the pair (#26) stays; resolution adds nothing a seed does not.
Depends on:   P-43 / P-44 (both read).

### P-56 Dense-slice input c03 (`v11a` / `v11b`)
Status:       ⏳ **session D = `rsna-knee-folds` v10** (pushed 15:36 UTC; `kernel_sources` = `rsna-knee-cache3-a..d` only, so c02 and c03 are
              never mounted together). Cache `c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20`: 4,407 studies over 4 shards,
              12.1–13.4 GB each (50.7 GB), 0 decode failures, 14–43 min per shard (CPU).
              **2026-09-29: trained** (3.53 h, both `ok  arm`, 0.35 s/study ≈ 25 min/epoch): gold-58 SWA `v11a` **0.9204**, `v11b`
              **0.9167**; c03 pair 0.9207 vs the c02 pair (`v09r` + `v09u`) 0.9065 — +0.014 (paired SD 0.005), 10/12 labels up; direction
              only (gold-58 had same-recipe directions wrong twice, traps 39). Datasets `rsna-knee-ckpt-v11a` / `-v11b`. **Solo reads 🔁:** #30 `v11a` 0.932, #31 `v11b`
              0.929 → m = 0.9305 (0.001 under the ✅ bar); #32 the c03 pair 0.932 vs the c02 pair 0.930 (+0.002). Both c03 seeds read ≥ both
              c02 seeds; neither card branch fires (experiments.md 2026-09-29 "Submissions #29–#32").
Hypothesis:   the c03 input (fluid slots 24 / 24 / 24, SAG no-FS 14, T1 8 / 8, 150 mm crop; `train_windows` 34) lifts the `v09r` recipe
              solo above `v09r` / `v09u`.
Origin:       our census (experiments.md 2026-09-28 "Gold-58 per-label diagnosis + c02 slice census": c02 keeps 12 of ≈ 30 native cor/ax
              fluid slices → 7.5–9 mm stored spacing; median FOV 160 mm vs the 130 mm crop).
Evidence:     for: the one input axis untested since c02 (the c02 lane was +0.009 LB, #9). Against: **#28 — the four labels c03 was
              aimed at (ACL / MCL / PF OA / Lat Men) are already our strongest on the public test (0.935 vs 0.9275)**; `v09x` (320
              px) read only +0.002 — more pixels have not paid so far.
Measure:      m = mean(`v11a`, `v11b`) solo vs 0.927; the c03 pair vs 0.930.
Noise floor:  **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285**.
Cost:         ≈ 3.7–4.4 GPU-h + 3 solos; a c03 member decodes the test at its own geometry (a second decode pass beside c02 members).
If it works:  c03 becomes the production input; a 320-px member and seeds follow in week 2.
If it fails:  c02 stays; the input axis closes.
Depends on:   —.

### P-58 Local-CPU open-weights LLM relabel as a 4th vote
Status:       💡 future, not scheduled (Tian 2026-09-28: written down only).
Hypothesis:   a fourth label vote from an open-weights LLM run locally (Scott Willis's Gemma route) lifts the LLM half of the targets.
Origin:       Scott Willis, discussion 735304; P-46 step 2.
Evidence:     a 14B open model reached 0.881 gold vs our blend's 0.8948 (P-46); the three sources are ≈ 1.5 effective votes (φ 0.88).
Measure:      gold-58 of the 4-vote blend (direction only), then a `v09r`-recipe arm solo vs 0.927.
Noise floor:  the P-44 floor.
Cost:         0 GPU; an overnight CPU run over 4,349 reports. The report text stays on this machine or inside Kaggle — never a hosted
              API (Tian, 2026-09-28; CLAUDE.md "Rules").
If it works:  → P-46 (the LLM half upgraded).
If it fails:  the LLM half stays three sources.
Depends on:   a free machine for a night; P-46.

### P-60 "Noisy student" regularisation on the c03 CoAtNet (`v11n` / `v11n2`)
Status:       🔁 **CLOSED 2026-10-03: #39 / #40 = 0.932 / 0.932 → m 0.932 vs 0.9305 (+0.0015) → INCONCLUSIVE, not adopted** (experiments.md 2026-10-03 "P-60 part 2"). ⏳ was: **part 2 = `rsna-knee-train` v39, pushed 2026-10-03 09:08 UTC** (`artifacts/train_p60_kaggle_part2.py` +
              `artifacts/p60_part2_kernel-metadata.json`, c03 ×4 + `rsna-knee-folds`; ≈ 3.0 h). Part 1 done 2026-09-30 (green; bold
              2026-09-30 lines below). History —
              ⏸ blocked on compute (2026-09-29): the code and arms are in `src/` (unit + local CPU smoke green). Pod 1 (RTX 5090,
              EU-RO-1) had no bandwidth (traps 45); pod 2 (A100 80 GB, US-MD-1) ran 5 epochs — EMA gold-58 by epoch `v11n` 0.776 ·
              0.852 · 0.882 · 0.898 · 0.911, `v11n2` 0.773 · 0.842 · 0.881 · 0.900 · 0.912 (direction only; `v11a` ep 4 ≈ 0.906) —
              until **the RunPod balance ran out** (19:33 UTC, container removed, nothing kept; `402 Payment Required` on restart).
              Tian (2026-09-29): no top-up → run it on Kaggle after the 2026-10-03 quota reset (two c03 arms × 12 epochs ≈ 5 h on
              2 × T4 = ≈ 5 GPU-h; `PARALLEL_ARMS = ("v11n", "v11n2")`, `TEACHER_TABLES = ("raptor_teacher",)`, `rsna-knee-folds`
              with the c03 `kernel_sources`). Tian: "brainstorm … what would make more sense and do that".
              **2026-09-30: part 1 = `rsna-knee-folds` v11, pushed 12:10 UTC, RUNNING** (Tian: start now, finish after the reset):
              `artifacts/train_p60_kaggle_part1.py` = `src` + 4 seds (`FORCE_SMOKE = False`, `PARALLEL_ARMS = ("v11n", "v11n2")`,
              `TEACHER_TABLES = ("raptor_teacher",)`, `runtime_limit_hours` default 8.3 → **2.75**), c03 `kernel_sources` only. At ≈ 25.5
              min/epoch (session D) the children's guard fires ≈ ⅔ into epoch 5 (0-based); `_last.pt` keeps that partial epoch as done, so
              the resume skips ≈ ⅓ of one epoch of 12 (LR schedule continuous in steps). **Part 2:** the same build with the 8.3 default in
              `rsna-knee-train`, `kernel_sources` = the four c03 caches + `rsna-knee-folds` (traps 31: a kernel cannot mount its own output);
              green = `resume: copied v11n_fold0_last.pt` / `v11n2_fold0_last.pt` and `resumed fold 0 at epoch 6` in each child log — the first
              Kaggle resume ever (traps 31).
              **2026-09-30: part 1 COMPLETE, green (2.61 h; `kaggle quota` 0.58 h left):** both children guard-stopped in epoch 5
              (`v11n` ≈ 65 % through it, `v11n2` ≈ 37 % — cuda:1 runs 0.38 vs 0.36 s/study), `_last.pt` ×2 in the v11 output; gold-58
              EMA epochs 0–4 `v11n` 0.781 · 0.854 · 0.895 · 0.905 · 0.917, `v11n2` 0.774 · 0.851 · 0.884 · 0.903 · 0.913 = `v11a` /
              `v11b` at the same epochs (0.9156 / 0.9129 at epoch 4 — the "≈ 0.906" above was wrong) at ≈ 2× their LR; direction only
              (experiments.md 2026-09-30 "P-60 part 1"). The epoch-5 `_best.pt` files are NOT members (traps 47). **Part 2 ≈ 3.0 h**
              (6 epochs × 27–28.5 min + SWA), not 2.7 h. **Nothing may be pushed to `rsna-knee-folds` before part 2 has mounted
              v11's output** — `kernel_sources` reads the latest version, and a new version would replace the two `_last.pt`.
Hypothesis:   our student imitates its targets because it is barely regularised — no stochastic depth (`drop_path_rate`
              is never set), light augmentation, 8 epochs; with drop-path 0.1, heavier augmentation and 12 epochs the
              `v11a` recipe (c03, Raptor mix 0.5) reads above `v11a` / `v11b`.
Origin:       research agent 1 (Xie 2020 Noisy Student, Table 6: removing student noise 85.1 → 84.3; Beyer 2022: long
              schedules on teacher targets do not overfit); agent 2 (every prior RSNA MRI/CT winner: drop-path 0.1–0.2,
              heavy augmentation / mixup / cutmix, 20–75 epochs); the thread (Yann Majewski: "training longer with more
              regularization"; Tucker Arrants: "student consistently outperforms teacher").
Evidence:     P-29 (16 epochs over-train) and P-33 (light aug ≈ 0) were measured on the plain LLM targets with no added
              regularisation — neither tests "longer WITH more regularisation" on soft image-teacher targets. A bundle on
              purpose: longer alone over-trains (P-29), aug alone was flat (P-33); ablate only if it works.
Measure:      `v11n` = `v11a` + `drop_path` 0.1 + `aug` "heavy" (per window at p 0.9: rotation ±15°, zoom 0.90–1.15, shift
              ±8 %, gamma 0.7–1.4, contrast 0.8–1.25, gain 0.85–1.15, one cutout ≤ 25 % of the area at p 0.3; no flips) +
              12 epochs (SWA of 9–11); `v11n2` = the same, seed 43. RunPod RTX 4090 (c03 cache pulled to the pod). Solo
              reads: m = mean(`v11n`, `v11n2`) vs m(`v11a`, `v11b`) = 0.9305.
Noise floor:  **✅ m ≥ 0.935 / 🔁 0.926 < m < 0.935 / ❌ m ≤ 0.926** (the P-56 ±0.0045 band); gold-58 direction only.
Cost:         two pods ≈ 2.5 h each on an RTX 4090 (≈ 7 min/epoch at c03 + ≈ 40 min set-up) ≈ $4; 0 Kaggle GPU-h; 2 solos.
If it works:  the production recipe gains the regularisation; then an ablation (P-55's student targets are ❌, 2026-09-30).
If it fails:  regularisation is not the student-vs-teacher gap either; the recipe line closes and the label side (P-51,
              silence-aware mixing) is what is left.
Depends on:   — .

### P-61 CoAtNet learning-rate probe upward (`lr_backbone` 2e-4, `llrd_decay` 0.85)
Status:       💡 new 2026-09-29 (from P-59).
Hypothesis:   the CoAtNet is mildly under-trained by the same optimiser that under-trained the ResNet: LLRD 0.75 over its stages
              puts the stem at 2.4e-5 and the top block at 7.5e-5 (the audit: the top block is one decay step below the documented
              `lr_backbone`); a higher, flatter LR reads above the `v11a` recipe.
Origin:       P-59 (a CNN LR lifted the ResNet +0.069 on gold); tonight's audit finding 5; only a LOWER CoAtNet LR was ever probed
              (`v09d` 3e-5 ❌, −0.013 OOF).
Evidence:     for: P-59; prior RSNA CoAtNet/EffNet winners at 1e-4..2.3e-4 uniform. Against: the CoAtNet's train loss already
              reaches 0.383 (it is not visibly under-fit the way the ResNet was); TheoViel's 2023 CoAtNet ran at 2e-5 for 20 epochs.
Measure:      `v11a` + `lr_backbone` 2e-4, `llrd_decay` 0.85, two seeds; fold-0 OOF would be a valid read for a recipe change, but
              every current member is `train_all`, so read solo vs m(`v11a`, `v11b`) = 0.9305 (the P-60 band).
Noise floor:  ✅ m ≥ 0.935 / 🔁 0.926 < m < 0.935 / ❌ m ≤ 0.926.
Cost:         ≈ 3.5 GPU-h (c03, two T4s) or ≈ $5 on a RunPod A100; 2 solos.
If it works:  the production recipe's LR moves up; re-read P-60 on it.
If it fails:  the CoAtNet optimiser is settled at 1e-4 / 0.75.
Depends on:   P-60's read (if P-60 ✅, probe on top of it).

### P-50 Final selection and publishability
Status:       💡 new 2026-09-27; decide by the 2026-10-15 entry deadline.
Hypothesis:   two private-scored picks that are different tickets beat two LB-maximising forks, because ≈ 690 teams share the
              anchor and private rank is decided by components they do not have (reviewer C).
Origin:       reviewer C (final picks, publishability); reviewer A (missing card M5).
Evidence:     #13 = #15 = 0.942 (β 0.10 left the score unchanged — the same ticket); #16 flat 0.60 = 0.940: the LB-tuned
              per-label outer map is worth ≈ +0.002 publicly and is the component the anchor's own author calls likeliest to
              give back (experiments.md 2026-09-23 "Submission #16"; 2026-09-22 night "The public frontier re-read").
Decision:     never pick #13 and #15 together. Pick 1: the anchor with our member (#13, or #22 if ≥ 0.942). Pick 2: the #16
              flat-map hedge, ideally rebuilt as "flat + `v09r` at β 0.10" after one public check ≥ 0.939.
Measure:      none for the private LB; one public read of the rebuilt hedge. Rules to read in a browser: which assets must be
              public for a winning submission (checkpoints) — **never publish `rsna-knee-teacher-tables`** (keyed by
              StudyInstanceUID, which CLAUDE.md bans from public locations; regenerable from public CC0 checkpoints +
              competition data); RadImageNet / the anchor's Rad stage licence (CC-BY-NC-SA?) vs the winner licence.
Noise floor:  the rebuilt hedge must read ≥ 0.939 public before it replaces #16.
Cost:         a browser session; ≈ 0.2 h T4 + 1 fork submission for the rebuilt hedge.
If it works:  both picks are set before 2026-10-22.
If it fails:  the rebuilt hedge reads < 0.939 → pick 2 = #16 as sent.
Depends on:   P-40 (#22), the Rules page (*Open questions*), P-48 (the final-member decision).

### P-18 Efficiency track with the solo member
Status:       💡 untested. The submission-robustness half shipped long ago (`MODE="infer"`, loud failure instead of a
              placeholder, refuse-to-submit constants, decode-once inference — experiments.md 2026-08-28 "Submission #1 scored
              exactly 0.500", 2026-08-29 "Cross-version rank blend + decode-once inference shipped").
              **2026-09-30:** two candidates now — `v11a` (c03 CoAtNet-1, 0.932, scored in 20–28 min) and `v13c` (ResNet-34 at
              a CNN LR, **0.921** at 0.12 s/study vs 0.29–0.35 for the CoAtNets, scored in ≈ 16 min — #37).
              **Step 1 done 2026-09-30:** the Efficiency LB lists us at rank 2,533 with the 0.942 fork (#22) — it seems to read the
              best-public-score (or selected) submission, not the fastest; top 100 span 0.917–0.958 (experiments.md 2026-09-30 "P-18 step 1").
Hypothesis:   our solo member (`v09r`, 0.927, ≈ 29 min send→score with P-41) is competitive on the Efficiency LB with no model
              change — the candidate is the member we already have, not a lean DINOv2-S variant.
Origin:       the Efficiency Prize track (CLAUDE.md); P-41.
Evidence:     the efficiency LB is published as a notebook (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`) and its leader is
              also top-5 on accuracy (CLAUDE.md); **the prize formula is unread** — it may score CPU-only runtime, which would
              make decode, not the model, binding.
Measure:      step 1 (no GPU): `kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb -p <dir>` + the formula
              in a browser → where `v09r` (0.927, ≈ 29 min) would place.
Noise floor:  0.005 LB on the accuracy half; the runtime half unknown until the formula is read.
Cost:         step 1: 0 GPU h; an `EFFICIENCY_MODE` flag only after the formula is read.
If it works:  a second prize track with the same member, no retrain.
If it fails:  the formula penalises what we spend → adjust after reading it, or leave the track.
Depends on:   the Efficiency formula (browser, *Open questions*); P-41.

### P-47 Teacher-mix bracket: mix 0.75 only
Status:       💡 low (new 2026-09-27). Mix 1.0 is PARKED, not rejected.
Hypothesis:   `TEACHER_MIX` 0.75 (Raptor-heavier targets) reads above `v09r` by more than the P-44 floor solo.
Origin:       reviewers A and D proposed mix 1.0; reviewer B's gold-58 analogs moved the bracket to 0.75.
Evidence:     reviewer B's gold-58 analogs (rank-space LLM + a held-out CoAt reader; direction only, gold-selected): resgated mix
              1.0 − 0.5 = −0.018 (SD 0.010), D4 −0.004 (SD 0.008); mix 0.75 − 0.5 = −0.002 / +0.004. Mix 1.0 is not
              "Raptor-only" anyway: the `w__` weights stay LLM-agreement weights and quantile matching uses the LLM marginals.
Measure:      the mix-0.75 arm solo vs #20 0.927 (and `v09u`).
Noise floor:  the P-44 floor (0.004 for a one-seed delta, #24).
Cost:         needs code: a per-arm `TEACHER_MIX` (today module-level; targets are built once per process) and a guard that also
              checks the mix (the distilled-arm guard checks the table set only — traps 40); ≈ 2.8 h T4 + 1 solo.
If it works:  Raptor-heavier targets become the production mix; mix 1.0 is read next.
If it fails:  0.5 stays. Expected ≈ 0 — run only if a GPU slot is idle and P-44 has set the floor.
Depends on:   P-44 (the floor), an idle slot.

### P-45 Second image teacher
Status:       🔧 **2026-10-03: step 1 done (0 GPU)** — D4's own gold-58 predictions ship in its Dataset (`d4_gold58_reference.npz`,
              macro 0.9302); the target 0.5 LLM + 0.25 Raptor + 0.25 D4 reads **0.9331 vs 0.9268** (+0.0063, SD 0.0041, 8 up / 4
              down), 0.4 / 0.3 / 0.3 reads 0.9347; D4 ~ Raptor within-class ρ 0.757 (`v11a` ~ Raptor 0.824, `v11a` ~ D4 0.781);
              gold-rank matching checked on Raptor (0.9262 vs 0.9268) — experiments.md 2026-10-03 "P-45 step 1". Direction only;
              both teachers are gold-selected. Next: the builder (in progress, 0 GPU) → the 58-study gold spike (must reproduce
              0.9302) → full pass → a c03 seed pair on the three-source target, read m vs 0.9305 with the P-60 band. **Tian's go
              given 2026-10-03.** Spike ✅ (experiments.md 2026-10-03 "P-45 gold spike"): 0.9301 on the original grid, 0.9290 on
              the notebook grid → the pass runs `--grid original` (`rsna-knee-teacher-d4` v2, 09:58 UTC, ≈ 2 GPU-h). Then: merge
              (`merge_teacher.py --teacher d4`), the in-sample check (D4-train ~ LLM vs Raptor-train ~ LLM), a new version of
              `rsna-knee-teacher-tables` with `d4_teacher.csv`, a Kaggle smoke, and `PARALLEL_ARMS = ("v11d", "v11d2")` (or
              `v11nd` / `v11nd2` if P-60 ✅) with `TEACHER_TABLES = ("raptor_teacher", "d4_teacher")`, mix 0.5.
              Earlier: 💡 deferred to after the 2026-10-03 quota reset. **Mapped 2026-10-01 (read-only agent over `notebook_score_0.942.ipynb`,
              `src/build_teacher_pass.py` and `artifacts/kaggle_out/fork_v4/`) → recommendation: D4 alone.**
              D4 = `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` (CC0; ships its own runtime + a timm-1.0.22 wheel; backbone
              not stated — "CoAtNet @384" inferred from file names): one model, Global96 stack 27/21/18/12/18 at 384 px / 130 mm with a
              right-knee sagittal flip, rank-8 attention + a depth-zone adapter; gold 0.9302 (epochs chosen on gold twice — the most
              optimistic of the three) vs Raptor 0.9254; 1 DICOM preparation + 1 backbone pass per study (Raptor 3 + 4) → est. ≈ 2–3 h
              for 4,349 studies (inferred, no at-scale timing). resgated (same backbone as Raptor, gold 0.9095, 50/50 analog 0.9275 ≈
              Raptor's) and the DINO ×20 + A5 stack (≈ 1,500 lines, in-sample folds) are not recommended. Builder: generalise
              `build_teacher_pass.py` (all of cell 12, cell 14, the cv2-wheel install + the D4 block from cell 45's `_coat_substitute`
              re-indented; patches: chunk root as `competition`, read the raw `coatnet_d4_depthzone_swa3_predictions.npz` (1, N, 12),
              relax `_d4_check_outputs`' `fallback_studies == 0`, a view-count-generic `load_prior`; D4 flushes only per shard → loop
              sub-chunks of 300–500 studies) ≈ 150–250 lines incl. tests; `merge_teacher.py` unchanged with `view_weights=[1.0]`.
              **Cheapest first step (0 GPU):** get D4's gold-58 predictions (the source of reviewer B's 0.9346 analog; not under
              `artifacts/` — likely inside the D4 Dataset), score 0.5 LLM + 0.25 Raptor + 0.25 D4 on gold by the P-49 method and the
              D4~Raptor within-class ρ; then D4 over the 58 gold (must reproduce 0.9302; doubles as the timing spike). Needs Tian's go
              — the 2026-09-23 decision to keep the CoAt children out of the teacher still stands.
Hypothesis:   a second image-teacher table beside Raptor lifts the production member solo beyond `v09r`.
Origin:       reviewers A (M2), B and D (2026-09-27 audit).
Evidence:     options: the anchor's CoAt family (resgated top-3 + D4) or its transformer stack (DINO ×20 + A5). For: reviewer B's
              E1 — D4 50/50 gold analog 0.9346; D4 adds to `v09r` on gold (rank-mean 0.929 vs 0.909; ρ 0.887). Against: A — the
              CoAt family is Raptor's family (CoAtNet-2 on the same labels), little target diversity; D — DINO / A5 are
              out-of-fold only if the authors' fold splits are recoverable, else in-sample on pilkwang's LLM labels (the P-38
              failure shape), and the Rad stage is CC-BY-NC-SA; E2 — an honest weaker teacher (pilkwang `oof.npz`, gold 0.840)
              dilutes the targets (−0.003..−0.007 on gold).
Measure:      a 100-study spike (plausibility vs the LLM teacher and vs Raptor), then a production arm solo vs `v09r` / `v09u`.
Noise floor:  the P-44 floor.
Cost:         spike 0.2–0.3 h; a pass ≈ 8 GPU-h (the Raptor pass took 7.76); a builder of 100–490 lines; then a 2.8 h arm + 1
              solo. Fork value ≈ 0 (the teacher is inside the anchor).
If it works:  a two-teacher table in `rsna-knee-teacher-tables`; the production member retrains on it.
If it fails:  Raptor stays the only image half.
Depends on:   P-49 ✅ read (Raptor on gold: 0.9254; the analogs are in experiments.md 2026-09-28 "P-49"), P-44 (the floor), the 2026-10-03 reset.

### P-46 Upgrade the LLM half of the targets (absorbs P-16 and P-30)
Status:       💡 low (new 2026-09-27).
Hypothesis:   a better LLM half — a 4th vote (step 1) or an open-weights re-label (step 2) — lifts the Raptor-distilled member solo.
Origin:       P-30 (the dread table died only for lack of a judge — the solo LB is one now); P-16 (re-label inside Kaggle).
Evidence:     the dread table (`dreaddevelopment/rsna-knee-labels`, CC0, 4,349 rows, no gold rows) is ρ 0.817 with our blend ("a
              4th LLM reading"); the three sources are ≈ 1.5 effective votes (φ 0.88). Whether Raptor's Synovitis gain is
              dread-driven is disputed: at a raw 0.5 cut Raptor sides with dread on 78–80 % of Synovitis disagreements, at the
              LLM's positive rate 52 %, rank vs rank 50 % (A: an operating-point artefact); a rank-based test within
              disagreements gives AUC 0.71 on Synovitis and 0.50–0.74 on every label (C: Raptor leans to dread on all labels).
              Step 2: a 14B open model reached 0.881 gold vs our blend's 0.8948.
Measure:      step 1: the `v09r` recipe with dread as a 4th vote in the LLM blend (`build_targets.py --sources`), solo vs 0.927;
              step 2: score the new table on gold-58 first, then the same arm, solo.
Noise floor:  the P-44 floor; gold-58 direction only.
Cost:         ≈ 2.8 h per arm + 1 solo; step 2 is a new kernel (vLLM not in the image, T4 has no bf16) + ≈ 150 lines.
If it works:  the LLM half changes for every later member.
If it fails:  the LLM half is not binding; the image half (P-43 / P-45) is.
Depends on:   P-44 (the floor).

### P-48 Final-member polish: gold-58 as training rows + seed averaging
Status:       💡 CONTESTED, parked; decide at P-50.
Hypothesis:   for the final retrain only, the 58 gold studies as hard training rows (weight 8) and 2–3 seeds rank-meaned add
              +0.001–0.002.
Origin:       reviewer B; absorbs P-06's `gold_weight` arm (inert under `train_all`).
Evidence:     reviewer C rejects it (it removes the only held-out truth for the member that matters; +0.001–0.002 is
              unreadable); A and B accept it only as the very last retrain.
Measure:      none possible — it spends the last held-out truth (gold-58 is the production members' only validation).
Noise floor:  n/a (unreadable by construction).
Cost:         part of the final retrain (≈ 2.8 h per seed).
If it works:  n/a — adopted or not on judgement at P-50.
If it fails:  n/a.
Depends on:   P-50 (decision), P-44 (seed averaging).

### P-62 Silence-aware teacher mix
Status:       ⏸ **PARKED 2026-10-03** — Tian stopped the real run (`rsna-knee-train-b` v1) ≈ 20 min in, on the GPU-budget rule "no session for a change that cannot clear its own read band" (expected +0.001..0.002 vs the +0.0045 ✅ bar). Re-open only bundled into the final retrain (P-50). Was: ⏳ **real run = `rsna-knee-train-b` v1** (a new third slug, so `rsna-knee-folds` stays untouched while P-60 part 2 may
              still need it), pushed 2026-10-03 09:09 UTC from `artifacts/train_p62_real.py` with the c03-only metadata (Tian 10-03:
              "push our work"); ≈ 3.5 h. 🔧 **implemented 2026-10-01, Kaggle smoke `rsna-knee-train` v38 GREEN** (0.04 h: both children `teacher table raptor_teacher: 4349`, the `P-62: report-silent cells … mix at 0.75` line, `reseeded 43 for arm v11s2`, SWA, `ok  arm` ×2): `TEACHER_SILENT_MIX` (config cell, sed'd per
              session) + `silence_mask` / per-cell `mix_teacher` in `src/kaggle_pipeline.py` and `src/build_targets.py`
              (`--teacher-silent-mix`; AST-identical, checked in `window_head_test.py`); arms `v11s` / `v11s2` (seeds 42 / 43);
              guards both ways (`DISTILLED_SILENT_MIX`); `targets_test.py` green, default teacher md5 `29f641ed` unchanged, pipeline
              yt == `build_targets.py` yt exactly; local CPU smoke green (`artifacts/local_p62/smoke.log`). Real run = one
              `PARALLEL_ARMS` session after P-60 part 2 — Tian's go.
Hypothesis:   the LLM half has no information where the report is silent (it reads as a confident negative), so the image teacher
              should carry more of the target exactly there; `w_sil` 0.75 on pilkwang-`UNK` cells and `w_addr` 0.5 elsewhere lifts
              the c03 member solo above the flat-0.5 recipe.
Origin:       label audit 2026-08-28 (silence ≈ 0.18, weight 0.69; Synovitis UNK 84 %); P-49's target-level pricing; thread 735304
              (Archit's open question: "applied to all cells or only report-silent ones?"); Tian 2026-09-30 ("the labels?").
Evidence:     experiments.md 2026-09-30 "Silence-aware teacher mix": target gold 0.9300 vs 0.9268 (+0.0032, SD 0.0018, 5 / 1 labels);
              flat mixes at the same mean Raptor weight +0.0004 → the gain is placement, not amount. Inside silent cells the LLM ranks
              gold positives at 0.17 (Fracture) / 0.48 (Baker's) / 0.75 (Synovitis), Raptor at 0.74 / 0.70 / 0.82. Against: 1.8 SD on 58;
              Raptor is optimistic on gold; Lateral OA drops 0.021 (22 silent negatives); the student shrinks target gains (P-49:
              0.9268 → 0.909).
Measure:      arms `v11s` ‖ `v11s2` = the `v11a` / `v11b` recipe (c03, all 4,349, 8 ep, SWA; seeds 42 / 43) with
              `TEACHER_SILENT_MIX = 0.75` (the silence mask from `report_labels_v2.csv` `__verdict == "UNK"`, already mounted); one
              `PARALLEL_ARMS` session (the silent mix is per session, so it pairs only with its own seed twin); two solos, read
              m = mean(`v11s`, `v11s2`) vs m(`v11a`, `v11b`) = 0.9305. If P-60 wins, re-read on top of the P-60 recipe.
Noise floor:  **✅ ≥ 0.935 / 🔁 0.926–0.935 / ❌ ≤ 0.926** (the P-56 band, as P-60); gold-58 direction only.
Cost:         code done (above); one c03 session ≈ 3.5 GPU-h (session D: 3.53 h for `v11a` ‖ `v11b`) + 2 solos.
If it works:  every later member trains on the silence-aware target.
If it fails:  the target is not binding at this resolution; the label side closes except a second image teacher (P-45).
Depends on:   GPU after the 2026-10-03 reset; P-60 part 2 first (it holds the resume).

### P-63 Per-finding spatial reader + slot-count correction (D4's head, ported at 224)
Status:       🔧 implemented 2026-10-03: `Config.spatial_reader` (`SpatialFindingPool`, zero-initialised query, applied to
              `enc.forward_features`; timm + window path only, refuses anything else) and `Config.slot_count_norm`
              (`WindowAttnHead(count_norm=True)`: logits − log #valid windows of the token's slot); `WindowAttnHead` takes
              (B, W, L, dim) per-label tokens; both keys in `INFER_MEMBER_KEYS` (read from the checkpoint). `window_head_test.py`:
              zero query = GAP, 12 identical tokens = the 3-D path, count norm gives 0.5 / 0.5 slot mass (padding not counted),
              spatial model = parent + 1 tensor and equals it untrained (max |diff| 4.8e-7), query at `lr_head` with a non-zero
              gradient. Local CPU smoke green (`artifacts/local_p63/smoke.log`); Kaggle smoke = `rsna-knee-train-b` v2 **GREEN** (0.11 h; both children `ok  arm`, `teacher table raptor_teacher: 4349`, `reseeded 43 for arm v11p2`, arms carry `spatial_reader: True`, SWA written); real build
              `artifacts/train_p63_real.py` (`PARALLEL_ARMS = ("v11p", "v11p2")`, Raptor 0.5). Real run waits behind P-45.
Hypothesis:   our window head global-average-pools each window to one 768-d vector before any label sees it, which loses small
              findings (Fracture, Contusion, the menisci); one attention query per finding over the backbone's patch grid (85 %
              attention + 15 % average pool, logit cap 2) plus a log slot-count correction on the window-attention logits lifts the
              c03 member solo above m(`v11a`, `v11b`) = 0.9305.
Origin:       read of D4's shipped training + inference code (`artifacts/d4_ds/`, Dataset
              `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`, CC0; subagent read 2026-10-03, file:line cited there):
              per-finding spatial reader `coatnet_global96_inference_model.py:23-83`, count correction `:175-247`; its ablation note
              calls the 85/15 reader "successful" and a learned attention/GAP mixture "failed" (no numbers).
Evidence:     D4's plain Global96 baseline (same trainer lineage, the reader, no FSX / depth adapter) reads 0.9274 / 0.9267 / 0.9250 on
              its gold-best epochs (0.9305 averaged) — the FSX + depth-zone stack adds ≈ 0 on gold, so the reader + input are what
              separates it from our 0.9204; but every one of those numbers is gold-selected. Other D4 differences, NOT in this card:
              CoAtNet-2 @384 / 130 mm (our `v10c` @384 ≈ `v09h` @224 on fold 0, `v09x` @320 +0.002 LB), a 0.075 pairwise rank loss on
              detached features + 24-study label-balanced groups, 24 epochs OneCycle at flat backbone 3e-5 (our 3e-5 at 8 epochs was
              −0.013, P-34), no EMA, almost no augmentation. With our 22/22/22/12/6/6 windows each fluid slot gets ≈ 3.7× the
              attention prior of a T1 slot — the count correction removes that.
Measure:      arms on the `v11a` recipe (c03, Raptor 0.5, 8 ep, SWA) + the reader + count correction, seeds 42 / 43, one PARALLEL_ARMS
              session; two solos, m vs 0.9305.
Noise floor:  ✅ m ≥ 0.935 / 🔁 0.926 < m < 0.935 / ❌ m ≤ 0.926 (the P-56 band).
Cost:         ≈ 60–80 lines (encoder returns the feature map; the window head takes per-label vectors) + unit checks in
              `window_head_test.py`; ≈ 4 GPU-h; 2 solos.
If it works:  the reader joins the production recipe; then the rank loss + label-balanced groups as one bundled arm, and 384 px only
              if both read positive.
If it fails:  the head is not where D4's lead comes from — most of that lead is gold selection.
Depends on:   P-45 (GPU priority).

### P-64 Long, heavy-augmentation CNN (`v13h`)
Status:       ⏳ 2026-10-03: arm `v13h` (local smoke green, unit-checked) runs in session A beside `v11p` (P-63), Raptor targets.
Hypothesis:   our CNN reads 0.921 (`v13c`, 12 epochs, light aug, c02) because it is under-regularised and under-trained, not because
              ResNets are weak here; ResNet-34 on c03 with heavy augmentation, drop-path 0.1 and 30 epochs reads ≥ 0.930 solo.
Origin:       forum mining 2026-10-03 (research.md 2.7.3): Tucker (H, "ResNet34 or EffNetB0. No attention … 0.94+"), Myo ("all about
              adding aug and regularization", 50 epochs, best at 28, 5-fold 0.94), Scott (H, small ResNet single fold 0.949), CoolinLai
              (5-fold ResNet-50 @224 0.954), SpeedSci (ResNet-50 0.940), Will (25 → 50 epochs +0.004).
Evidence:     P-59 fixed our CNN's optimiser (+0.069 gold); P-60 (heavy aug + drop-path + 12 ep on the CoAtNet) reads tonight. Against:
              P-29 (16 CoAtNet epochs over-trained on LLM targets without added regularisation).
Measure:      `v13h` solo vs `v13c` 0.921 (one seed; the P-44 floor 0.004) and vs the CoAtNet `v11a` 0.932; as an ensemble member, the
              pair `v11a` + `v13h` vs `v11a` alone.
Noise floor:  one-seed delta ≥ 0.004 (P-44).
Cost:         ≈ 3.4 GPU-h inside a two-arm session (ResNet-34 ≈ 6 min/epoch on c03, est.); 1 solo.
If it works:  a second family for the 3–5-member ensemble; then EfficientNet-B0 / ResNet-50 variants and 40–50 epochs. **Staged
              2026-10-03:** arms `v13r` (ResNet-50 a1) / `v13e` (EfficientNet-B0 ra) = `v13h` with the backbone swapped; private weight
              Datasets `timm-resnet50-a1` / `timm-efficientnet-b0-ra` mounted in train / train-b / infer; unit + local smoke green
              (`artifacts/train_sC_*.py`). The case for the CNN line: SpeedSci's CNNs read 0.940 on the LB vs their CoAtNet's 0.926,
              and Tucker reports "low 0.95s" with ResNet-34 / EfficientNet-B0 on plain Qwen labels (745214).
If it fails:  the CNN family stays an Efficiency-track candidate only (P-18).
Depends on:   —.

### P-51 Teacher-aware confidence weights
Status:       💡 low (new 2026-09-27).
Hypothesis:   recomputing the `w__` weights from the mixed target (not LLM agreement alone) lifts the solo member by 0..+0.002.
Origin:       reviewer A (M4), reviewer B (E4).
Evidence:     the `w__` weights come from LLM agreement / decisiveness only (`build_targets()` in `kaggle_pipeline.py`, the
              `agree` / `decisive` block) and never see the teacher table; they are mildly lower where Raptor disagrees (0.744 vs
              0.809, reviewer B) — the rows where Raptor may know most.
Measure:      solo vs `v09r` / `v09u`.
Noise floor:  the P-44 floor — the expected gain is likely below it.
Cost:         ≈ 20 lines + 2.8 h + 1 solo.
If it works:  mixed-target weights for every distilled member.
If it fails:  LLM-agreement weights stay.
Depends on:   P-39 (the mixed target), P-44 (the floor).

## Rejected without testing

Merged from brainstorm.md and research.md §5; one line each. Do not resurrect without a new reason.

| Idea | Why | Source |
|---|---|---|
| Text branch at inference | `test.csv` has no `Report`; nothing to read | CLAUDE.md |
| Horizontal flip — **both variants** (with or without medial↔lateral swap) | undoes laterality normalisation; the swap is anatomically wrong (MCL has no lateral counterpart); P-05 does *not* make it legal | [pilkwang], traps.md, critic item 23 |
| Vertical flip; zoom-out with padding | off-distribution; fabricated tissue | [pilkwang], [Guo et al.] |
| Geometric TTA | degraded 11/12 medical pairs; flips hurt knee OA | [2604.09697], [2311.06118] |
| Calibration, Platt scaling, thresholds, label smoothing on soft targets | AUC reads rank order only; `pos_weight` measured 🔁 (P-37), rejection confirmed | metric arithmetic, brainstorm.md |
| Averaging probabilities across folds/models | most confident model dominates; rank-mean instead | brainstorm.md |
| Full fine-tuning or best-epoch selection on 58 gold (~12/fold); a gold fine-tuning stage | SE 0.09, coin flip, our NaN-fold bug; gold's role is validation (gold is not trained under `train_all`); the contested final-retrain variant is P-48 | experiments.md, [Andre et al.], critic 25 |
| Tuning weights on the public LB | author-labelled overfit; 0.001–0.003 movements; the 0.936 notebook's gold-58-tuned per-label weights + "clinical residual" are worth **+0.001** over its untuned 0.935 (read in full 2026-08-30) | mattiaangeli (not re-read), `crazy_good_rsna.ipynb` (research.md §2.7.1), CLAUDE.md |
| The 0.936 notebook's **88-feature stacking calibrator** (rank blocks + cross-view deltas + 12 protocol counts, w 0.4 on 7 labels) | decoded 2026-08-30: coefficients are ≈ a per-label reweighting of the same three views (protocol columns ≤ 0.003); est. +0.002–0.005 and fragile to a protocol-mix shift on private | cell-level re-read (research.md §2.7.1) |
| **Clinical residual** cross-label adjustments (`ACL −0.10 × mean rank(Contusion, Lateral Meniscus)` etc.) and correlation-guarded per-label fusion weights | the notebook itself: "an aggressive leaderboard experiment, not an unbiased estimate of private-test performance"; +0.001 stated | cell-level re-read |
| Random or report-only K-fold as the comparison metric | grouped vs random gap up to +0.136 | [EXPERIMENTS.md] |
| Native 3D CNN / nnU-Net / segmentation-first | 0.69 vs 0.85 (p=0.001); multi-A100 budgets | [MST], [CoPAS], [MIC-DKFZ 2025] |
| Frozen DINOv2 + head as the final model | 0.79 vs 0.85 knee; 0.776 vs 0.866 LB | [MST], sadamtorres (not re-read) |
| Backbone LR ≥ 5e-5 uniform on DINOv2 / SSL ViTs | every medical recipe ≤ 2e-5; 1e-3 collapse. Not for the hybrid: our CoAtNet trains at 1e-4, and 3e-5 was harmful (P-34) | [2501.14685], [dinov2 #276] |
| More backbones on the same teacher table and cache — DINOv2-B, DINOv3, ConvNeXt, a third family | P-42: a second family on the Raptor table read 0.927 = `v09r` alone; within-class ρ 0.861 (LLM-target pair 0.777) — a shared teacher raises cross-family agreement | experiments.md 2026-09-27 "Submission #23", reviewer C (2026-09-27 audit) |
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
| More *report-label* LLM sources beyond P-46's one test (dread as a 4th vote), Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2 in literature, **~1.5 here** (φ 0.88); our 0.002 spread. Image-grounded tables are the demonstrated lever (P-39); the LLM half is P-46 | [2605.29800], [BoxWRENCH], experiments.md, label_audit.md |
| Co-teaching / DivideMix / DISC; focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE | [LNMBench], [RAL], [Xin et al.] |
| Calibrated priors with a 50% floor for zero-support states | OOF collapsed to base rate | [JunhaoLiXD V02] |
| Translate-then-extract; sub-3B extractors; 70B on 2×T4 | precision loss; F1 0.74; ~40 GB weights | [2602.21374] |
| Hosted LLM APIs for report text | plausible Rule 4.b violation; open-weights parity | CLAUDE.md, [Radiology 2025] |
| In-domain SSL continued pretraining as a first priority | in-domain ViT still below ImageNet AlexNet on MRNet | [SB-SSL] |
| Auxiliary report-reconstruction head | same weak supervision as the targets; nothing new to learn | brainstorm.md #10 (speculative, unranked) |
| Judging a target-source change on fold-0 OOF vs the LLM teacher | rewards agreement with the teacher, not truth: P-38's fold-0 +0.0109 read −0.001 in production (#19) | traps 39, experiments.md 2026-09-24 "Submission #19" |
| A weighted `v09r` 0.75 / `v08r` 0.25 solo | a sub-floor bet that `v08r` pays for its weakness; the flat pair already read 0.927 = `v09r` (P-42) | handoff 2026-09-27 |
| Re-anchoring the fork on "Speedy Raptors 0.943" | it is our anchor + two serial CoAt readers; ≤ +0.001 by its own claim (0.2× the floor), more serial rerun work | experiments.md Infrastructure 2026-09-27 |
| Members below ≈ 0.90 solo into the fork | #12–#14: our arm moved 0.942 by 0.000 to −0.003; even a 0.918 solo member read −0.001 (#17) — at β 0.10 the fork cannot read member quality | experiments.md 2026-09-22 "P-27 read-out", 2026-09-23 "Submission #17 read 0.941" |
| A CoAtNet-2 @384 production arm on Kaggle T4 | 9–15 GPU-h per arm, a 2-session resume, batch 2 OOMs; P-43 tests resolution at 320 on CoAtNet-1 instead | reviewer D (2026-09-27 audit), P-43 |
| Treating gold deltas < 0.05 or LB < 0.005 as real | noise floor | CLAUDE.md |
| P100 accelerator | no Pascal kernels | CLAUDE.md |

---

## Open questions needing a browser

Kaggle pages are JS-rendered; the CLI exposes none of these. Each blocks a card.

| Question | Blocks | Where to read |
|---|---|---|
| Exact Rules text: Rule 4.b (data off-platform), winner licence clause, which assets a winning submission must make public, ≤ 9 h runtime, internet-off | P-50 (publishability, licence), P-46 step 2 (risk), P-18 | competition Rules page |
| Efficiency Prize formula and Base — is runtime CPU-only? | P-18 | Efficiency evaluation page; `ryanholbrook` notebook body |
| radimagenet.com Terms & Conditions for the weights (the anchor's Rad stage) | P-50 | https://www.radimagenet.com/ |
| Hidden test size and site/vendor mix (community: 16–19 sites; the "1,322 studies" figure is a public notebook's cache-sizing assumption, 0.3 × train — experiments.md Infrastructure 2026-09-27) | P-18 time budget, P-44 (LB sampling SD) | competition Data page; header pass at rerun |
| Discussion threads: hidden-test decode issues, gold severity protocol | P-46 step 2 grading rubric | Discussion tab; pilkwang notebook gold-protocol cells |
| Kaggle per-kernel output cap (assumed ~20 GB) and Datasets size limits | P-45 (pass sharding) | Kaggle docs: Notebooks → Output; Datasets → limits |
