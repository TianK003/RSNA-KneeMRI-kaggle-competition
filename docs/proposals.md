# Proposals — ranked, testable cards

*Written 2026-08-28. The forward-looking half of the lab notebook: every idea we intend to
test, written as a falsifiable card **before** it is run. Companion to
[experiments.md](experiments.md) (things measured, with verdicts) and
[research.md](research.md) (the evidence base these cards draw on).*

**Rewritten 2026-09-27** after a four-reviewer audit: measured and retired cards are one-line pointers in *Closed cards*; the
full pre-rewrite text is `git show 8304c96:docs/proposals.md`. **Pruned 2026-10-04:** P-45, P-56, P-58, P-60, P-61, P-63, P-64
moved to *Closed cards* and their bodies (and P-52's) deleted; the text before the prune is `git show c2b0e32:docs/proposals.md`.

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

**GPU rule (Tian, 2026-10-03):** a GPU session only for a change whose expected gain can clear its own read band (two-arm mean ± 0.0045; one-seed delta ≥ 0.004, P-44) — one seed per idea, two ideas per session, seed twins only for ensemble members; **RunPod (2026-10-04):** every run needs a written justification checked by a critic subagent, then Tian's go.

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
| 0i | P-62 | Silence-aware teacher mix (Raptor 0.75 where the report is silent, 0.5 where it speaks) | ⏳ **session D TRAINED green 2026-10-04 (`rsna-knee-train-b` v6, 5.94 h): gold-58 SWA `v13es` 0.9107 / `v13rs` 0.9160 vs 0.9126 / 0.9111 (direction only); shipped `rsna-knee-ckpt-v13es` / `-v13rs`; solos 10-05 = `rsna-knee-infer` v49 / v50; read m vs 0.9345 (✅ ≥ 0.9390)** | 0..+0.002 — likely under the 0.004 one-seed floor | one c03 session ≈ 3.5 GPU-h + 2 solos | — |
| 0l | P-65 | Grading-aware Claude relabel of the reports (an independent, severity-aware LLM vote) | ⏳ **full pass DONE 2026-10-04: 4,349 rows, ≈ 25 min, ≈ 6.9 M tokens; `claude_v1` / `claude_rap_v1` published in `rsna-knee-teacher-tables`; session E (`v13ec` ‖ `v13rc` on `claude_rap_v1` at mix 0.75) staged — runs on Kaggle after the 2026-10-10 reset, after session D's P-62 read; read m vs 0.9345 (✅ ≥ 0.9390)** — pilot 🔁 (Opus 0.9062 alone / 0.9397 with Raptor, bars 0.910 / 0.945 not met; Haiku 0.8639 ❌) | 0..+0.005 (literature + forum); the policy-misaligned labels are the upside | session E ≈ 6 GPU-h + 2 solos | — (Tian's go given) |
| 4 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15; **own candidate = B6 #52 0.942** (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, = the public-stack fork, replaces #48 0.938); fork candidate = the public stack with our trio at β 0.45 (#49, `rsna-knee-fork` v11, ⏳), then C2 with B6 as the leg | decides what the private LB scores | a browser session; ≤ 1 fork check | P-40 🔁 closed (#22 / #27), the fork v11 read, Rules page |
| 5 | P-18 | Efficiency track with the solo member | 💡 low priority — Tian 2026-10-04: score over efficiency; robustness half shipped; the natural fast candidate is `v13e` (0.935 solo, ≈ 15 min to score, 17 MB) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 6 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low — P-49 priced it on Raptor itself: matched mix 0.75 − 0.5 = −0.002 (SD 0.003) on gold | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 7 | P-46 | Upgrade the LLM half of the targets (absorbs P-16, P-30) | 💡 low — step 2 (re-label) superseded by P-65; step 1 (dread as a 4th vote) left | 0..+0.002 (dread vote) | ≈ 2.8 h per arm + 1 solo | P-65 session E's read |
| 8 | P-48 | Final-member polish: gold-58 as training rows + seed averaging | 💡 contested, parked | +0.001..0.002, unreadable by construction | part of the final retrain | P-50 decision |
| 9 | P-51 | Teacher-aware confidence weights | 💡 low | 0..+0.002 | ≈ 20 lines + 2.8 h; 1 solo | P-44 floor |

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
| P-16 | Open-weights LLM re-label inside Kaggle | → P-46 step 2 → P-65 (the re-label ran as the hosted Claude pass) | P-65 below |
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
| P-45 | Second image teacher (D4) | 🔁 not adopted as a training target: student `v11d` #43 = 0.930 vs `v11a` 0.932 (−0.002) — the target-level gold +0.0063 did not transfer; the D4 table (4,349 / 0 failed) stays in `rsna-knee-teacher-tables`; later members stay on 0.5 LLM + 0.5 Raptor | experiments.md 2026-10-03 "P-45 step 1", "P-45 gold spike", "P-45 D4 pass complete", "Submissions #41–#43" |
| P-56 | Dense-slice input c03 (`v11a` / `v11b`) | 🔁 as a lever: #30 / #31 0.932 / 0.929 → m 0.9305 (✅ bar 0.9315); c03 pair #32 0.932 vs c02 pair 0.930 — a consistent small edge, so c03 became the base input of every later member | experiments.md 2026-09-29 "Sessions C ‖ D", "Submissions #29–#32" |
| P-58 | Local-CPU open-weights LLM relabel as a 4th vote | → P-65 — the 4th LLM vote ran as the hosted Claude relabel once host rule 2.6.b permitted hosted LLMs; the local route existed only because hosted APIs were ruled out | P-65 below |
| P-60 | "Noisy student" regularisation on the c03 CoAtNet (`v11n` / `v11n2`) | 🔁 #39 / #40 = 0.932 / 0.932 → m 0.932 vs 0.9305 (band 0.926–0.935); not adopted (1.5× training time); the first Kaggle resume worked (traps 31) | experiments.md 2026-09-30 "P-60 part 1"; 2026-10-03 "P-60 part 2" |
| P-61 | CoAtNet learning-rate probe upward (`v11dl`: `lr_backbone` 2e-4, LLRD 0.85) | 🔁 not adopted: #44 `v11dl` 0.927 vs `v11d` 0.930 (−0.003; gold −0.005 the same way) — the CoAtNet optimiser stays at 1e-4 / 0.75 | experiments.md 2026-10-03 "Sessions A ‖ B"; 2026-10-04 "Submissions #44–#47" |
| P-63 | Per-finding spatial reader + slot-count correction (`v11p`) | 🔁 not adopted: #41 = 0.929 vs `v11a` 0.932 (−0.003; gold 0.9185 vs 0.9204); the code stays, inert at the defaults | experiments.md 2026-10-03 "Sessions A ‖ B", "Submissions #41–#43" |
| P-64 | Long, heavy-augmentation CNN (`v13h`: c03, CNN LR 3e-4, frozen BN, aug heavy, drop-path 0.1, 30 ep) | ✅ KEEP — the CNN recipe: #42 ResNet-34 `v13h` 0.931 vs `v13c` 0.921 (+0.010); on bigger CNNs #46 EfficientNet-B0 `v13e` 0.935 (best solo), #45 ResNet-50 `v13r` 0.934; trio `v11a` + `v13r` + `v13e` #48 0.938; gold called it flat (traps 39). Bigger CNN → P-66 (B3 0.940 ✅) | experiments.md 2026-10-03 "Submissions #41–#43"; 2026-10-04 "Session C", "Submissions #44–#47", "Submission #48" |
| P-66 | Bigger CNN (`v13b3` EfficientNet-B3 @ 288) + the first CNN seed twin (`v13e2`), trained on RunPod (≈ $2.6, critic-reviewed) | ✅ KEEP — #50 `v13b3` **0.940** = our best solo (+0.005 vs `v13e` 0.935, bar 0.004; +0.0035 vs the B0 seed mean); #51 `v13e2` 0.938 → CNN seed spread s = 0.003, the one-seed bands stand. B3 = a member and a final-retrain candidate; its seed twin is candidates.md T2 | experiments.md 2026-10-04 "P-66 on RunPod", "P-66 complete"; 2026-10-05 "Submissions #49–#53" |

---

## Cards

### P-50 Final selection and publishability
Status:       💡 new 2026-09-27; decide by the 2026-10-15 entry deadline. **2026-10-05: the own candidate is B6 = #52 0.942**
              (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, flat rank-mean), equal to the public-stack fork and replacing #48. The
              fork candidate is #49 (⏳), then C2 = the stack + B6 at β 0.45 (experiments.md "Submissions #49–#53").
              **2026-10-04: the two candidates were set** — our best
              own ensemble (#48 0.938, `v11a` + `v13r` + `v13e`) and the public-stack fork with that trio as our leg at β 0.45
              (`rsna-knee-fork` v11, green placeholder, read on 2026-10-05). The full lineup of candidate ensembles and fork legs, with
              read rules, is the queue in candidates.md (sections B, C).
Hypothesis:   two private-scored picks that are different tickets beat two LB-maximising forks, because ≈ 690 teams share the
              anchor and private rank is decided by components they do not have (reviewer C).
Origin:       reviewer C (final picks, publishability); reviewer A (missing card M5); research.md 2.7.4 (the 0.946–0.947 teams
              blend their own leg into the public stack at rank weight 0.45–0.5).
Evidence:     #13 = #15 = 0.942 (β 0.10 left the score unchanged — the same ticket); #16 flat 0.60 = 0.940: the LB-tuned
              per-label outer map is worth ≈ +0.002 publicly and is the component the anchor's own author calls likeliest to
              give back (experiments.md 2026-09-23 "Submission #16"; 2026-09-22 night "The public frontier re-read"). Our
              Raptor-distilled leg read flat at β 0.10–0.20 (P-40: it correlates with the stack); the trio is a cross-family leg,
              and cross-family blends have read above every member (#47 0.934, #48 0.938).
Decision:     never pick #13 and #15 together. Pick 1: our best own ensemble (#48 0.938, or an own blend that beats it on the
              solo LB by 10-15). Pick 2: the public-stack fork with our trio at β 0.45 (`rsna-knee-fork` v11).
Measure:      none for the private LB; one public read of the v11 fork vs #13 0.942: ✅ ≥ 0.945 / 🔁 0.941–0.944 / ❌ ≤ 0.940
              (handoff 2026-10-04). Rules to read in a browser: which assets must be
              public for a winning submission (checkpoints) — **never publish `rsna-knee-teacher-tables`** (keyed by
              StudyInstanceUID, which CLAUDE.md bans from public locations; regenerable from public CC0 checkpoints +
              competition data); RadImageNet / the anchor's Rad stage licence (CC-BY-NC-SA?) vs the winner licence.
Noise floor:  0.005 on the fork (decision-metric hierarchy); the bands above.
Cost:         a browser session; 1 fork submission (built: `rsna-knee-fork` v11; ≤ 8 h to score, #17).
If it works:  both picks are set before 2026-10-22.
If it fails:  the v11 fork reads ❌ → pick 2 falls back to an anchor fork already scored (#13 0.942 or the #16 flat hedge
              0.940) — Tian's call.
Depends on:   the v11 fork read (2026-10-05), the Rules page (*Open questions*), P-48 (the final-member decision).

### P-18 Efficiency track with the solo member
Status:       💡 untested. The submission-robustness half shipped long ago (`MODE="infer"`, loud failure instead of a
              placeholder, refuse-to-submit constants, decode-once inference — experiments.md 2026-08-28 "Submission #1 scored
              exactly 0.500", 2026-08-29 "Cross-version rank blend + decode-once inference shipped").
              **2026-09-30:** two candidates now — `v11a` (c03 CoAtNet-1, 0.932, scored in 20–28 min) and `v13c` (ResNet-34 at
              a CNN LR, **0.921** at 0.12 s/study vs 0.29–0.35 for the CoAtNets, scored in ≈ 16 min — #37).
              **Step 1 done 2026-09-30:** the Efficiency LB lists us at rank 2,533 with the 0.942 fork (#22) — it seems to read the
              best-public-score (or selected) submission, not the fastest; top 100 span 0.917–0.958 (experiments.md 2026-09-30 "P-18 step 1").
              **2026-10-04: low priority — Tian: "score over efficiency"; the track no longer steers anything.** The natural fast
              candidate is now `v13e` (EfficientNet-B0, 0.935 solo, ≈ 15 min to score, 17 MB). research.md 2.7.4: a formula read
              from another team's code (not the competition page) is (score − max) / (0.5 − max) + seconds / 32,400, and the board
              seems to list *selected* submissions — so for P-18 to count, one final pick must be a fast solo (P-50).
Hypothesis:   our solo member (now `v13e`, 0.935, ≈ 15 min send→score) is competitive on the Efficiency LB with no model
              change — the candidate is the member we already have, not a lean DINOv2-S variant.
Origin:       the Efficiency Prize track (CLAUDE.md); P-41.
Evidence:     the efficiency LB is published as a notebook (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`) and its leader is
              also top-5 on accuracy (CLAUDE.md); **the prize formula is unread** — it may score CPU-only runtime, which would
              make decode, not the model, binding.
Measure:      step 1 (no GPU): `kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb -p <dir>` + the formula
              in a browser → where `v13e` (0.935, ≈ 15 min) would place.
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

### P-46 Upgrade the LLM half of the targets (absorbs P-16 and P-30)
Status:       💡 low (new 2026-09-27). **2026-10-04: step 2 (the re-label) is superseded by P-65** — host rule 2.6.b permits
              hosted LLMs, and the grading-aware Claude relabel of all 4,349 reports is done (`claude_v1`); its session E reads
              the "better LLM half" hypothesis with a stronger vote than either step here. Step 1 (dread as a 4th vote; the table
              has no gold rows, so it cannot be priced on gold) is the only untested part left, and it waits for E: if the Claude
              vote does not move the LB, the LLM half is not binding and step 1 closes with it.
Hypothesis:   a better LLM half — a 4th vote (step 1) or an open-weights re-label (step 2) — lifts the Raptor-distilled member solo.
Origin:       P-30 (the dread table died only for lack of a judge — the solo LB is one now); P-16 (re-label inside Kaggle).
Evidence:     the dread table (`dreaddevelopment/rsna-knee-labels`, CC0, 4,349 rows, no gold rows) is ρ 0.817 with our blend ("a
              4th LLM reading"); the three sources are ≈ 1.5 effective votes (φ 0.88). Whether Raptor's Synovitis gain is
              dread-driven is disputed: at a raw 0.5 cut Raptor sides with dread on 78–80 % of Synovitis disagreements, at the
              LLM's positive rate 52 %, rank vs rank 50 % (A: an operating-point artefact); a rank-based test within
              disagreements gives AUC 0.71 on Synovitis and 0.50–0.74 on every label (C: Raptor leans to dread on all labels).
              Step 2: a 14B open model reached 0.881 gold vs our blend's 0.8948.
Measure:      step 1: the production recipe (`v09r` when written; `v13e` 0.935 now) with dread as a 4th vote in the LLM blend
              (`build_targets.py --sources`), solo vs that recipe's read;
              step 2: → P-65.
Noise floor:  the P-44 floor; gold-58 direction only.
Cost:         ≈ 2.8 h per arm + 1 solo.
If it works:  the LLM half changes for every later member.
If it fails:  the LLM half is not binding; the image half is.
Depends on:   P-65 session E's read (after the 2026-10-10 reset), P-44 (the floor).

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
Status:       ⏳ **2026-10-04: RE-OPENED on the CNN line (Tian's go) — session D = `rsna-knee-train-b` v6 (pushed 10:15 UTC; Kaggle smoke v5 green: `P-62: report-silent cells … mix at 0.75` in both child logs, Raptor table read, frozen BN): `v13es` (EfficientNet-B0) ‖ `v13rs` (ResNet-50) = `v13e` / `v13r` + `TEACHER_SILENT_MIX = 0.75` (`artifacts/train_sD_real.py`). Read m(`v13es`, `v13rs`) vs m(`v13e`, `v13r`) = 0.9345: ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299 (the two-family mean is a two-seed read of the idea, band ± 0.0045).** Why now: the literature + forum rank image-teacher pseudo-labels on silent cells as the only label lever with measured LB transfer (research.md 2.7.5, lever (d)). History: ⏸ **PARKED 2026-10-03** — Tian stopped the real run (`rsna-knee-train-b` v1) ≈ 20 min in, on the GPU-budget rule "no session for a change that cannot clear its own read band" (expected +0.001..0.002 vs the +0.0045 ✅ bar). Re-open only bundled into the final retrain (P-50). Was: ⏳ **real run = `rsna-knee-train-b` v1** (a new third slug, so `rsna-knee-folds` stays untouched while P-60 part 2 may
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
Measure:      (since 2026-10-04, on the CNN line) `v13es` ‖ `v13rs` = `v13e` / `v13r` exactly + `TEACHER_SILENT_MIX = 0.75` (the
              silence mask from `report_labels_v2.csv` `__verdict == "UNK"`), one `PARALLEL_ARMS` session (session D); two solos,
              read m(`v13es`, `v13rs`) vs m(`v13e`, `v13r`) = 0.9345. The two families at one seed each act as a two-seed read.
              (The CoAtNet design `v11s` ‖ `v11s2` vs 0.9305 was parked on 10-03 and is kept in git history.)
Noise floor:  **✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299** (band ± 0.0045); gold-58 direction only. P-66's seed twin `v13e2`
              re-measured the CNN seed spread behind this band: s = 0.003 on 10-05 (#51), so the band stands.
Cost:         session D ≈ 6 GPU-h (session C, the same two arms flat: 5.87 h) + 2 solos.
If it works:  the final members (P-50) and session E consider the silent mix; E needs `DISTILLED_SILENT_MIX` entries for its arms.
If it fails:  the target is not binding at this resolution. The label side rests on P-65 (session E, after 10-10).
Depends on:   — (session D trained green 2026-10-04, gold-58 `v13es` 0.9107 / `v13rs` 0.9160, experiments.md "Session D";
              solos 2026-10-05 = `rsna-knee-infer` v49 / v50).

### P-65 Grading-aware Claude relabel of the reports (a genuinely independent LLM vote)
Status:       ⏳ session E staged (Kaggle, after the 2026-10-10 reset); read m(`v13ec`, `v13rc`) vs m(`v13e`, `v13r`) = 0.9345:
              ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299 (experiments.md 2026-10-04 "P-65 full pass"; two arms, so this replaces
              the one-arm read in Measure (2)).
              🔁 2026-10-04: pilot read — Opus 0.9062 alone (LLM blend 0.8948), 0.9397 with Raptor (0.9324): under both
              pre-registered bars; Haiku 0.8639 ❌. Fracture (acute only) 0.815 → 0.924 and PF OA 0.903 → 0.967 gain; Effusion
              −0.030 / Baker's −0.092 lose (size thresholds collapse the ranking). experiments.md 2026-10-04 "P-65 gold-58 BLIND
              pilot". **Full pass DONE 2026-10-04 (Tian's go): 4,349 / 4,349, ≈ 25 min, ≈ 6.9 M tokens; `claude_v1` + composite
              `claude_rap_v1` published; session E (`v13ec` ‖ `v13rc`, TEACHER_TABLES ("claude_rap_v1",) at mix 0.75 ≈ 0.25 LLM
              + 0.25 Claude + 0.5 Raptor) staged, `artifacts/train_sE_real.py`; GPU after 2026-10-10 (or a RunPod top-up).** **2026-10-04:
              the RunPod top-up went to P-66 instead (critic-reviewed, Tian's choice); E runs on Kaggle after 10-10, after session D's
              P-62 read decides whether E also gets `TEACHER_SILENT_MIX`.**
              Expected LB 0..+0.004.
Hypothesis:   our LLM half is ≈ 1.5 effective votes (hans_v4 ~ sol56 agree 99.45 % at 0.5; label audit §3) and ignores the
              host's severity thresholds; a grading-aware graded relabel (moderate/large effusion and Baker's, high-grade ACL,
              acute MCL / fracture, ≥ 1 cm > 50 % cartilage loss for OA, `pos` / `sub` / `neg` / `unk` + calibrated p) lifts
              the labels exactly where the image models already beat them on gold (Effusion 0.880, Fracture 0.815,
              Contusion 0.861, Medial OA 0.931) and adds a second real vote.
Origin:       the host's 2.6.b rule update (hosted LLMs permitted); Tian 2026-10-04 ("you could be labelling them yourself?");
              literature: frontier models beat cheap ones on severity grading (GPT-4o 98 % vs mini 69 % on knee OA severity;
              artifacts/research_1004/literature.md); host grading rules (topic 733343).
Evidence:     against: extraction is near its ceiling (forum 743148: 22 of 33 gold positives an LLM misses are never named);
              Tucker / Yann / tennogh: better extraction did not move the LB; P-46: a 14B open model read 0.881 gold. For:
              SpeedSci (GPT + Claude + Gemini vote + teachers) 0.931 → 0.942 on DINOv2; Yann +0.015 from combining label sets.
Measure:      (1) pilot, target level on gold-58, read ONCE: AUC of p per label vs the LLM blend (0.8948) and the rank mix
              0.5 Claude + 0.5 Raptor vs 0.5 LLM + 0.5 Raptor (0.9324). **Pre-registered:** promising = Claude alone ≥ 0.910 OR
              the Raptor mix ≥ 0.945, with the gain on the policy-misaligned labels (not scattered); flat = within ± 0.01
              (D4 +0.006 and xfit +0.008 did not transfer, Raptor +0.017 did). (2) if promising: label all 4,407 (≈ 1.5 M
              input tokens), build `claude_v1` as a label source, then ONE arm (`v13e` recipe, targets = rank-blend of LLM +
              Claude, then 0.5 with Raptor) solo vs `v13e` 0.935.
Noise floor:  gold-58 SE ≈ 0.03–0.09 per label; one-seed LB floor 0.004 (P-44).
Cost:         pilot ≈ 2 × 30k tokens, minutes. Full pass ≈ 1.5 M input + ≈ 0.4 M output tokens in parallel subagents
              (≈ 1–2 h wall) — at API prices ≈ $6–8 (Haiku, batch) to ≈ $25–70 (Opus) per pass, under "minimal cost". Then
              ≈ 3 GPU-h (one EfficientNet arm) + 1 solo. Measured: the full pass took ≈ 25 min wall and ≈ 6.9 M subagent tokens;
              session E is two arms, ≈ 6 GPU-h + 2 solos.
If it works:  the LLM half becomes LLM + Claude for every later member.
If it fails:  the label line closes; image-side (recipe, families, P-62 silent-cell teacher) only.
Depends on:   — (Tian's go for the full pass given 2026-10-04; the pass ran).

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
| More *report-label* LLM sources beyond P-46's one test (dread as a 4th vote), Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2 in literature, **~1.5 here** (φ 0.88); our 0.002 spread. Image-grounded tables are the demonstrated lever (P-39); the LLM half is P-46. Exception on a new reason: P-65's grading-aware Claude vote (the host's severity thresholds) | [2605.29800], [BoxWRENCH], experiments.md, label_audit.md |
| Co-teaching / DivideMix / DISC; focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE | [LNMBench], [RAL], [Xin et al.] |
| Calibrated priors with a 50% floor for zero-support states | OOF collapsed to base rate | [JunhaoLiXD V02] |
| Translate-then-extract; sub-3B extractors; 70B on 2×T4 | precision loss; F1 0.74; ~40 GB weights | [2602.21374] |
| ~~Hosted LLM APIs for report text~~ — **SUPERSEDED 2026-10-04 by host rule 2.6.b** (hosted LLMs permitted at minimal cost); it ran as P-65 | was: plausible Rule 4.b violation; open-weights parity | CLAUDE.md "Rules", [Radiology 2025] |
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
| Exact Rules text: ~~Rule 4.b (data off-platform)~~ (**superseded 2026-10-04 by host rule 2.6.b**: hosted LLMs permitted, CLAUDE.md "Rules"), winner licence clause, which assets a winning submission must make public, ≤ 9 h runtime, internet-off | P-50 (publishability, licence), P-18 | competition Rules page |
| Efficiency Prize formula and Base — is runtime CPU-only? | P-18 | Efficiency evaluation page; `ryanholbrook` notebook body |
| radimagenet.com Terms & Conditions for the weights (the anchor's Rad stage) | P-50 | https://www.radimagenet.com/ |
| Hidden test size and site/vendor mix (community: 16–19 sites; the "1,322 studies" figure is a public notebook's cache-sizing assumption, 0.3 × train — experiments.md Infrastructure 2026-09-27) | P-18 time budget, P-44 (LB sampling SD) | competition Data page; header pass at rerun |
| Discussion threads: hidden-test decode issues; ~~gold severity protocol~~ **ANSWERED** — research.md §2.7.3 (host grading rules, topic 733343; P-65's prompt was written from them) | was: P-46 step 2's grading rubric (now P-65) | Discussion tab |
| Kaggle per-kernel output cap (assumed ~20 GB) and Datasets size limits | any future teacher pass (sharding; P-45 is closed) | Kaggle docs: Notebooks → Output; Datasets → limits |
