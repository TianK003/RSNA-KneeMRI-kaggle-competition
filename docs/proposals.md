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
| 0b | P-54 | 5-fold cross-fit of the v09r recipe (`v09k0-4`): honest OOF + a 5-fold member | ⏳ **all 5 folds ✅** (A ‖ B 2026-09-28; fold 4 = C re-push `rsna-knee-train` v34, 2.39 h); table `xfit_v09k` built (gold-58 0.9028 vs LLM 0.8948, 4/12 below → loose gate OPEN); Dataset `rsna-knee-ckpt-v09k` (5 ckpts); **fold-ensemble solo ⏳** | enables P-55; fold ensemble +0.000..0.005 | ~7.7 GPU-h (3 sessions) | — |
| 0c | P-55 | OOF soft-bootstrapped student `v09o` / `v09o2` (0.25 LLM + 0.375 Raptor + 0.375 xfit, mix 0.75) | ⏳ **session E = `rsna-knee-train` v35 RUNNING** (pushed 2026-09-29 11:37 UTC; loose gate OPEN; `rsna-knee-teacher-tables` now carries `xfit_v09k.csv`; local CPU smoke of the exact build green) | 0..+0.005 (Nicolai: no LB transfer) | ~2.9 GPU-h, 3 submissions | P-54; loose gate |
| 0d | P-56 | Dense-slice input c03 (24/24/24/14/8/8, 150 mm) `v11a` / `v11b` | ⏳ **trained** (session D = `rsna-knee-folds` v10, 3.53 h): gold-58 `v11a` 0.9204 / `v11b` 0.9167, c03 pair 0.9207 vs c02 pair 0.9065 (+0.014, SD 0.005, 10/12 up — direction only); Datasets `rsna-knee-ckpt-v11a` / `-v11b`; **🔁 solo reads (2026-09-29): #30 `v11a` 0.932 (best single) / #31 `v11b` 0.929 → m = 0.9305 (✅ bar 0.9315); #32 c03 pair 0.932 vs the c02 pair 0.930** — consistent sign, under every floor; not the production input by rule, the axis stays open | the only untested input axis since c02 | ~3.7-4.4 GPU-h, 3 submissions | — |
| 0e | P-57 | ResNet-34 on the v09r recipe (`v13a`) | ❌ **on gold-58**: `v13a` 0.8306 vs `v09r` 0.9093 (−0.079, beyond the 0.05 floor; 8/12 labels down, ACL / MCL / both menisci 0.75–0.77); solo read deprioritised (Dataset `rsna-knee-ckpt-v13a` shipped, submit only into a spare slot) | tests the Scott Willis / CoolinLai route | ~0 extra (rides with `v09k4`) | — |
| 0f | P-58 | Local-CPU open-weights LLM relabel as a 4th vote (Scott's Gemma route) | 💡 future, not scheduled (Tian 2026-09-28) | 0..+0.002 | 0 GPU; overnight CPU | — |
| 4 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15 | decides what the private LB scores | a browser session; ≤ 1 fork check | P-40 ✅ closed (#22 / #27), Rules page |
| 5 | P-18 | Efficiency track with the solo member | 💡 (robustness half shipped) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 6 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low — P-49 priced it on Raptor itself: matched mix 0.75 − 0.5 = −0.002 (SD 0.003) on gold | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 7 | P-45 | Second image teacher | 💡 deferred to after 2026-10-03 | +0.001..0.004 (CoAt family) / −0.002..+0.004 (DINO + A5); fork ≈ 0 | spike 0.2–0.3 h, pass ≈ 8 GPU-h, 100–490 lines, 2.8 h arm | P-49, P-44, the 10-03 reset |
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
| P-40 | Raptor-distilled members into the fork | 🔁 closed: #22 (β 0.10) 0.942, #27 (β 0.20) 0.941 vs 0.942 — the fork does not register a 0.927 member at either weight; the pre-registered β 0.20 retry is spent → the fork is for final-selection builds only (P-50). Step B: `v08r` 0.918 solo (#21), not a fork member | experiments.md 2026-09-27 "Submission #22"; 2026-09-28 "Submissions #27–#28" |
| P-49 | Raptor over the 58 gold studies | 🔁 read (direction only): Raptor 0.9254, the 0.5/0.5 target 0.9268 (12/12 above the LLM), within-class ρ `v09r` ~ Raptor 0.835 (`v09a` ~ Raptor 0.742, `v09r` ~ LLM 0.408) → the fork cannot read `v09r`; matched mix 0.5 is the best on gold (0.75 −0.002, 1.0 −0.017) | experiments.md 2026-09-28 "P-49" |
| P-52 | Three-member production blend `v09r` + `v09u` + `v09x` | 🔁 closed: #29 = 0.931 vs #26 0.930 (band 0.931–0.933) — the pair stays (the third member costs 1.4× inference and read like a seed) | experiments.md 2026-09-29 "Submissions #29–#32" |
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

### P-54 5-fold cross-fit of the `v09r` recipe (`v09k0` … `v09k4`)
Status:       ⏳ folds 0–3 ✅ (sessions A ‖ B = `rsna-knee-train` v32 ‖ `rsna-knee-folds` v9, 2026-09-28, 2.29 / 2.26 h; experiments.md
              2026-09-28 "P-54 sessions A ‖ B"); **fold 4 = `v09k4`: session C (`rsna-knee-train` v33) died from outside at 2.2 h, 0 files saved (traps 44) — re-push.**
              **2026-09-29: fold 4 ✅** (C re-push = `rsna-knee-train` v34, 2.39 h; OOF vs LLM 0.8811, gold 0.9432 on n = 11); (a) the table
              `xfit_v09k.csv` built — gold-58 pooled **0.9028** vs LLM 0.8948 (+0.008, SD 0.015), 4/12 labels below the LLM → **loose gate
              OPEN** (experiments.md 2026-09-29 "P-54 cross-fit table"); all five checkpoints in Dataset `rsna-knee-ckpt-v09k`; (b) ⏳.
Hypothesis:   a 5-fold cross-fit of the production recipe gives honest out-of-fold predictions for every training study — the
              precondition for OOF soft bootstrapping (P-55) — and its 5-fold flat rank-mean reads above the #26 pair solo.
Origin:       Archit Konde, Kaggle discussion 735304 (out-of-fold soft bootstrapping); research.md 2.7.2.
Evidence:     for: Archit's single-fold CoAtNet @224 at 0.950 used OOF targets. Against: P-38 ❌ — our first self-distillation used
              a teacher weaker than the labels (gold 0.873 < 0.895); this is the first cross-fit of a `v09r`-generation member.
              Each fold model sees 80 % of the data, so a fold ensemble starts below a `train_all` member's prior.
Measure:      (a) the table: `build_distill_table.py --sets "artifacts/kaggle_out/xf_*/v09k[0-4]_fold[0-9]_oof.csv" --per-fold-rank
              --out artifacts/teacher/xfit_v09k.csv` (one set of five folds, 4,407 rows); gold diagnostic — pooled per-label
              AUC vs the LLM blend (0.8948), Raptor (0.9254) and the 0.5/0.5 target (0.9268) from P-49; ρ vs Raptor. (b) the
              fold ensemble solo, `INFER_MEMBERS = ["v09k0", …, "v09k4"]`, vs #26 0.930.
Noise floor:  (b) **≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌** (five members ≈ 1–1.5 h to score). (a) is never a verdict (traps 39).
Cost:         ≈ 7 GPU-h (A ‖ B + half of C); 1 solo; every fold arm `eval_final_only` (the per-epoch held-out pass costs ≈ 1 h a session).
If it works:  the fold ensemble is a production / final candidate; the table feeds P-55.
If it fails:  the table still feeds P-55 — the loose gate decides.
Depends on:   —.

### P-55 OOF soft-bootstrapped student (`v09o` / `v09o2`)
Status:       🔧 arms `v09o` (seed 42) / `v09o2` (seed 43), `TEACHER_PATHS["xfit_v09k"]`, `DISTILLED_ARMS` + the `DISTILLED_MIX` guard
              (refuses any mix but 0.75 for these arms) in `src/`; waits for P-54's fold 4 and table.
              **2026-09-29: session E = `rsna-knee-train` v35 RUNNING** (pushed 11:37 UTC; `artifacts/train_student_E.py` = `src` + 4
              seds; loose gate OPEN; the new `rsna-knee-teacher-tables` version carries `xfit_v09k.csv`; local CPU smoke of the exact
              build green — both tables read, `(1 - 0.75) * LLM + 0.75 * quantile-matched ['raptor_teacher', 'xfit_v09k']`).
Hypothesis:   the `v09r` recipe on 0.25 LLM + 0.375 Raptor + 0.375 honest cross-fit OOF reads above `v09r` / `v09u` solo by more
              than the one-seed floor.
Origin:       Archit Konde's OOF soft bootstrapping, "heavier than 50/50" (discussion 735304).
Evidence:     for: Archit ("50/50 jumped" — LB or CV unknown; question in brainstorm). Against: P-38 ❌ (#19), Nicolai Karcher (gold /
              CV up, LB flat), and P-49: on Raptor alone the matched mix 0.75 is −0.002 vs 0.5 on gold (the xfit table is a
              different teacher, so this prices only the Raptor half).
Measure:      session E = `PARALLEL_ARMS = ("v09o", "v09o2")` with `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`,
              `TEACHER_MIX = 0.75` (mix_teacher averages the two matched tables); m = mean of the two solos vs 0.927; the pair vs 0.930.
Noise floor:  **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285**; pair ✅ ≥ 0.935.
              **Loose gate (Tian, 2026-09-28):** E runs unless the pooled cross-fit gold < 0.875 AND ≥ 8/12 labels below the LLM.
Cost:         ≈ 2.9 GPU-h (two seeds in one session) + 3 solos; a new PRIVATE version of `rsna-knee-teacher-tables`.
If it works:  an attribution control (0.25 LLM + 0.75 Raptor) in week 2; a second cross-fit round on the student.
If it fails:  the third null (P-38, Nicolai) → the OOF-target line closes.
Depends on:   P-54 (the table), the loose gate.

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

### P-57 ResNet-34 on the `v09r` recipe (`v13a`)
Status:       ❌ **on gold-58 (2026-09-29)**: `v13a` (C re-push, `rsna-knee-train` v34; 8 epochs in 0.7 h at 0.07 s/study) SWA **0.8306**
              vs `v09r` 0.9093 — −0.079, beyond the 0.05 gold floor; 8/12 labels down, 3 up (Medial OA / Effusion / Baker's), and the
              structure labels collapse (ACL 0.772, MCL 0.753, Med Men 0.768, Lat Men 0.752); within-class ρ with `v09r` 0.667. The
              solo read is deprioritised: Dataset `rsna-knee-ckpt-v13a` is shipped and mounted, submit it only into a spare slot.
              Caveat: the `v09r` optimiser (backbone LR 1e-4, 8 ep) was not re-tuned for a CNN — this closes "ResNet-34 on our
              recipe", not "a small CNN" (experiments.md 2026-09-29 "Sessions C ‖ D").
              (was: lost with session C v33, traps 44; weights = private Dataset `timm-resnet34-a1`, smoke v31 green.)
Hypothesis:   the backbone class is not what separates us from the thread's single models: `v13a` reads ≥ `v09r` solo.
Origin:       discussion 735304 — Scott Willis (small ResNet, single fold, 0.949), CoolinLai (5-fold ResNet @224, 0.954).
Evidence:     against: P-42 (a second backbone on the same targets added nothing); Archit Konde reaches 0.950 with our model class.
Measure:      `v13a` solo vs 0.927.
Noise floor:  **✅ ≥ 0.932 / 🔁 0.923–0.931 / ❌ ≤ 0.922**.
Cost:         ≈ 0 extra wall (rides with `v09k4`); 1 solo.
If it works:  a cheaper member (efficiency, P-18) and a family for the blend.
If it fails:  the small-CNN route closes.
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
Status:       💡 deferred to after the 2026-10-03 quota reset.
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
