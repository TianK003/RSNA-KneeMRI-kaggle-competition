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
| **Public LB, solo** (`rsna-knee-infer`, one member or our own blend; **≈ 30 min** send→score, P-41) | hidden test (public part) | **0.005 by rule — the seed/retrain spread of a production member is UNMEASURED** (P-44 measures it; reviewer C: a paired gold-58 bootstrap scaled to ≈ 400 public studies (unverified) plus an unknown seed spread puts the SD of one solo-vs-solo delta at ≈ 0.004–0.006) | member quality and **every target-source change** (traps 39) |
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
| 1 | P-43 | Student resolution: CoAtNet-1 @320 on the `v09r` targets (`v09x`) | ⏳ **trained** — `rsna-knee-train` v30 COMPLETE (5.88 h, SWA, 0.59 s/study); gold-58 0.9094 (= `v09r`, +0.005 over mean(`v09r`, `v09u`), 10/12 labels up); shipped `rsna-knee-ckpt-v09x`; **solo read next** | +0.002..0.006 solo — the only input-side lever left | one ≈ 6.2 GPU-h session shared with P-44; 1 solo | P-39 ✅, P-31, P-44 (its read) |
| 2 | P-44 | Kaggle-retrain seed spread + a 2-seed production member (`v09u`) | ⏳ **trained** — same session, cuda:1, gold-58 0.8995 (−0.010 vs `v09r`); shipped `rsna-knee-ckpt-v09u`; **solo + blend reads next** | sets the floor every later read needs; 2-seed rank-mean +0.001..0.003 | ≈ 2.8 h on the second T4 (≈ 0 extra quota); 2 solos | — |
| 3 | P-40 | Raptor-distilled members into the fork | 🔁 **#22 = 0.942** (= #13 / #15); β 0.20 retry is an open decision for Tian | low — the fork at β 0.10 is flat over member strength 0.913 → 0.927 | spent; a β 0.20 retry = 1 fork submission (hours) | P-39 ✅, P-27 |
| 4 | P-49 | Raptor over the 58 gold studies — a correlation diagnostic | 💡 | validity: fork redundancy without an 8-h fork read; gates P-45 / P-47 | ≈ 20 lines + 0.2 h T4 (second slot) | P-39 (`build_teacher_pass.py`) |
| 5 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15 | decides what the private LB scores | a browser session; ≤ 1 fork check | P-40 (#22), Rules page |
| 6 | P-18 | Efficiency track with the solo member | 💡 (robustness half shipped) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 7 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 8 | P-45 | Second image teacher | 💡 deferred to after 2026-10-03 | +0.001..0.004 (CoAt family) / −0.002..+0.004 (DINO + A5); fork ≈ 0 | spike 0.2–0.3 h, pass ≈ 8 GPU-h, 100–490 lines, 2.8 h arm | P-49, P-44, the 10-03 reset |
| 9 | P-46 | Upgrade the LLM half of the targets (absorbs P-16, P-30) | 💡 low | 0..+0.002 (dread vote) / +0.001..0.003 (re-label) | ≈ 2.8 h per arm; step 2 a new kernel | P-44 floor |
| 10 | P-48 | Final-member polish: gold-58 as training rows + seed averaging | 💡 contested, parked | +0.001..0.002, unreadable by construction | part of the final retrain | P-50 decision |
| 11 | P-51 | Teacher-aware confidence weights | 💡 low | 0..+0.002 | ≈ 20 lines + 2.8 h; 1 solo | P-44 floor |

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

---

## Cards

### P-43 Student resolution: CoAtNet-1 @320 on the `v09r` targets (`v09x`)
Status:       ⏳ trained: `rsna-knee-train` v30 COMPLETE 2026-09-28 (5.88 h, SWA, no guard stop, 0.59 s/study); gold-58 SWA
              0.9094 vs `v09r` 0.9093 / `v09u` 0.8995 (direction only; Lateral Meniscus 0.901); shipped as Dataset
              `rsna-knee-ckpt-v09x`; solo read after `v09u`'s (experiments.md 2026-09-28 "rsna-knee-train v30"). Arm `v09x` = the `v09r`
              dict at `img_size` 320. Runs in ONE
              `PARALLEL_ARMS = ("v09x", "v09u")` session with P-44 (`rsna-knee-train`, both children on
              `TEACHER_TABLES=("raptor_teacher",)`, mix 0.5 — the two children of a session share the table set and the mix).
Hypothesis:   now that half the target comes from a 384-px image teacher (Raptor = CoAtNet-2 @384), the student's 224-px input
              (the 336-px c02 cache downsampled) is the binding student-side constraint: `v09x` reads ≥ the P-44 baseline +
              max(0.005, 2s) solo.
Origin:       reviewer B (2026-09-27 audit); our hypothesis.
Evidence:     for: `v10c` @384 was the meniscus specialist (Lateral Meniscus 0.858) and still rising at epoch 7; the only earlier
              resolution read (P-11: `v10c` CoAtNet-2 @384 0.8641 vs `v09h` CoAtNet-1 @224 0.8683) confounded backbone size, was
              fold-0, under-trained and on LLM targets — and report-derived targets cannot reward image detail the reports never
              mention. Against: P-11; the public 0.924 member at 384 is no better than our 224 member on better targets;
              resolution's LB effect was never isolated.
Mechanics:    (verified by the reviewers) `timm.create_model(..., img_size=320)` loads the 224 weights strictly (RelPosMlp `cr`
              mode, 0 missing / unexpected); without `img_size` a 320 input crashes (400 vs 196 positions); 336 is not /32;
              windows are resized on the GPU (`KneeNet`); inference rebuilds from the saved cfg and shares `v09r`'s c02 decode.
              `batch_studies` 1 × `grad_accum` 4 (≈ 7 GiB; batch 2 @320 ≈ 13–14 GiB = OOM risk; P-32 priced batch composition
              at +0.0008).
Measure:      `v09x` solo (`rsna-knee-infer`) vs m = mean(#20 `v09r` 0.927, `v09u`), with s = |`v09u` − 0.927| from P-44;
              gold-58 direction only.
Noise floor:  pre-registered: **✅ `v09x` − m ≥ max(0.005, 2s) / ❌ `v09x` − m ≤ −max(0.005, 2s) / 🔁 otherwise.**
Cost:         ≈ 0.55–0.63 s/study ≈ 5.3–6.1 h for 8 epochs on one T4, under the ≈ 8.0 h child budget (break-even ≈ 0.80
              s/study); the session costs ≈ 6.2 GPU-h (quota charges session wall-clock); 1 solo. Inference ≈ 2.1–2.3× the
              member's model pass; a fork rerun +≈ 25 min. A smoke trains 4 studies, so it cannot time the real run; a
              guard-stopped `_best.pt` is the last epoch, not SWA — never submit one (check the log for the `SWA of last` line).
If it works:  read `v09x` + `v09r` (+ `v09u`) as a blend next; 320 becomes the production resolution.
If it fails:  ❌ / 🔁 → stay at 224; resolution is not the lever.
Depends on:   P-39 ✅ (the table), P-31 (two children), P-44 (`v09u` is half the baseline). Supersedes P-11.

### P-44 Kaggle-retrain seed spread + a 2-seed production member (`v09u`)
Status:       ⏳ trained (`rsna-knee-train` v30, cuda:1, `reseeded 43 for arm v09u`; gold-58 SWA 0.8995 vs `v09r` 0.9093, rank-mean
              0.9065; shipped as Dataset `rsna-knee-ckpt-v09u`); solo + blend reads next (experiments.md 2026-09-28 "rsna-knee-train v30"). `v09u` = the exact `v09r`
              dict with `"seed": 43`, plus the per-arm reseed fix
              (`seed_all(cfg.seed)` in the arm loop when the arm's seed differs from the base config's — before it an arm-dict
              `seed` only changed the banner, and every production member trained on the seed-42 stream). Same session as P-43.
Hypothesis:   the retrain spread of a production member on the public LB is < 0.004, and a 2-seed rank-mean adds ≥ +0.002.
Origin:       three of the four 2026-09-27 reviewers picked a seed twin; reviewer C's paired gold-58 bootstrap.
Evidence:     every verdict since P-39 is a sub-0.01 solo delta judged against an unmeasured floor. #18 (`v09a`, Kaggle T4, 2
              loader workers) vs #20 (`v09r`, RunPod 4090, 8 workers) — the headline +0.009 — was itself cross-platform, and
              window draws depend on the worker count (per-worker numpy seeds), so `v09u` is the first same-platform twin of
              `v09a`'s setting. On fold 0 the same config moved 0.004–0.008 OOF on seed alone (P-02).
Measure:      (1) `v09u` solo → s = |`v09u` − 0.927| = one draw of the Kaggle-retrain spread (seed + platform); (2) a third
              submission: the `v09r` + `v09u` rank-mean solo. **Submission order (after v30, on Tian's go): `v09u` → `v09x` →
              the `v09r` + `v09u` blend** — `v09u` first, because the `v09x` read needs its s; each ≈ 30 min to score, 3 of the
              day's 5. Ship first: `rsna-knee-ckpt-v09x` / `-v09u` Datasets, added to `rsna-knee-infer`'s `dataset_sources`.
Noise floor:  s ≤ 0.002 → one-seed deltas need ≥ 0.004; 0.003–0.005 → ≥ 0.008 or two seeds per side; ≥ 0.006 → nothing under
              ≈ 0.01 is readable from one seed, and P-39 stands only if mean(`v09r`, `v09u`) − 0.918 ≥ 0.006. **If `v09u` ≤
              0.920 → re-open P-39.** Blend: **≥ 0.930 ✅** seed ensembling is a lever; otherwise 🔁. Caveat: a 3-point range
              estimates σ poorly (≈ 50 % CV) and LB scores are rounded to 3 decimals.
Cost:         ≈ 2.8 h on the second T4, inside P-43's wall-clock (≈ 0 extra quota); 2 submissions (solo + blend).
If it works:  the floor for every later read is known; the 2-seed rank-mean becomes the production member.
If it fails:  blend 🔁 → it becomes the production member anyway (lower private-LB variance; no LB tuning).
Depends on:   nothing (the reseed fix is in). Supersedes P-02 step 1 at LB level and P-13.

### P-40 Raptor-distilled members into the fork
Status:       🔁 **#22 = 0.942** (read 2026-09-28) — fork v9 = the public 0.942 graph + `v09r` alone at β 0.10 (ref 56614068,
              sent 2026-09-27 16:36 UTC after a ≈ 3 h 40 min queue, traps 41) = #13 / #15; the "If it fails" branch (β 0.20 once,
              or drop fork member reads) is **open for Tian** (experiments.md 2026-09-27 "Submission #22"). Step B done: `v08r` 0.918 solo (#21), below the ≈ 0.920 bar → not a fork
              member; P-42: `v09r` + `v08r` = 0.927.
Hypothesis:   the fork registers a member of ours once it is at the public stack's best-member level: β 0.10 with `v09r`
              (0.927 solo) reads ≥ 0.947, where #17 (the 0.918-level `v09a` / `v08a`) read 0.941.
Origin:       P-39 "If it works"; #20.
Evidence:     for: #20's +0.009 solo from the targets alone. Against: #17 — the fork did not move with members 0.024 below the
              anchor; reviewer C: within-class ρ `v09r` / `v08r` 0.861 vs 0.777 for the LLM-target pair → `v09r`'s errors track
              Raptor, which the anchor already carries at 0.40–0.60. Reviewer prior: #22 ≈ 0.942 ± 0.001.
Measure:      #22 (`v09r` alone in our arm, β 0.10) public LB vs #13 / #15 0.942.
Noise floor:  **≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌**.
Cost:         spent: `v08r` 31 min on a RunPod A100 (≈ $0.95 for the pod); fork placeholder ≈ 0.2 h T4; 1 submission (forks
              score in ≤ 8 h 06 min, plus hours of queue).
If it works:  our arm counts → a final-selection candidate (P-50).
If it fails:  🔁 → the pre-registered branch is β 0.20 once; reviewers B and C recommend dropping it (expected ≤ 0.002; it is
              weight-tuning on the public LB) and using the fork only for final-selection builds (P-50) — **open decision for
              Tian**. ❌ → our arm hurts even at 0.927. The old "If it works" items (mix 1.0, a second teacher) live in P-47 / P-45.
Depends on:   P-39 ✅, P-27 (`src/build_fork.py`), traps 39 / 40.

### P-49 Raptor over the 58 gold studies — a correlation diagnostic
Status:       💡 untested (new 2026-09-27).
Hypothesis:   within-class ρ(`v09r`, Raptor) on gold-58 says how redundant our member is inside the anchor — without an 8-h fork read.
Origin:       reviewer C (the Raptor-on-gold diagnostic); reviewer B (the gate for a second teacher).
Evidence:     the Raptor-distilled pair `v09r` / `v08r` is within-class ρ 0.861 vs 0.777 for the LLM-target pair (12/12 labels) —
              the shared teacher raised cross-family agreement (the same backbone on two teachers, `v09r` / `v09a`, is still 0.883, so
              the architecture has not stopped mattering — reviewer A's round-2 check); the teacher table has no gold rows (the pass excluded them), so Raptor's own
              gold predictions have never been in our hands.
Measure:      Raptor's view-weighted gold-58 predictions → within-class ρ vs `v09r` (and `v09a`, `v08r`) per label; and reviewer
              B's gold-58 "teacher-mix analogs" (E1) computed on Raptor itself, before any mix or second-teacher spend.
Noise floor:  correlation read only, never a verdict: Raptor's epoch / SWA were chosen on gold-58, so its gold AUC is optimistic.
Cost:         ≈ 20 lines (the `build_teacher_pass.py` pattern over the gold UIDs only) + ≈ 0.2 h T4; the output goes to a file
              name `TEACHER_PATHS` never reads, so gold rows cannot reach training. Can run in the second Kaggle slot.
If it works:  a high within-class ρ confirms the fork cannot read `v09r` → the fork is for final-selection builds only (P-50);
              the analogs price P-47 / P-45 on Raptor before any GPU spend.
If it fails:  a low ρ says our member is not redundant inside the anchor → one more fork read is worth a submission.
Depends on:   P-39 (`src/build_teacher_pass.py`, the Raptor checkpoints).

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
Noise floor:  the P-44 floor (≥ 0.005 by rule).
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
Depends on:   P-49 (Raptor on gold, analogs first), P-44 (the floor), the 2026-10-03 reset.

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
