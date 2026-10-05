# Proposals — ranked, testable cards

*Written 2026-08-28. The forward-looking half of the lab notebook: every idea we intend to
test, written as a falsifiable card **before** it is run. Companion to
[experiments.md](experiments.md) (things measured, with verdicts) and
[research.md](research.md) (the evidence base these cards draw on).*

**Rewritten 2026-09-27** after a four-reviewer audit: measured and retired cards are one-line pointers in *Closed cards*; the
full pre-rewrite text is `git show 8304c96:docs/proposals.md`. **Pruned 2026-10-04:** P-45, P-56, P-58, P-60, P-61, P-63, P-64
moved to *Closed cards* and their bodies (and P-52's) deleted; the text before the prune is `git show c2b0e32:docs/proposals.md`. **Restructured 2026-10-05 (Tian: live cards only):** the *Closed cards* table moved to the end of experiments.md ("Closed cards index"); *Rejected without testing* became **Dropped directions**, each with the reason from the forum, the literature or our own reads; the file before this move is `git show fd96082:docs/proposals.md`.

## How a card moves

1. An idea enters here as `💡 untested`. It needs a hypothesis, a measure, a noise floor and a
   cost — if it cannot be written in the template it is not ready to run.
2. When code ships it becomes `🔧 implemented, effect pending` (the change exists in
   `src/`, nothing has been measured yet).
3. While a run is live it is `⏳ running`.
4. When the number comes back the result goes to **experiments.md** (append-only) with a
   verdict — ✅ KEEP / ❌ DEAD END / 🔁 INCONCLUSIVE — and the card here is **deleted** (its index row
   too); experiments.md's closed-cards index gets its one-line pointer. Untried ideas never go to
   experiments.md; measured ones never stay here.
5. A direction we have decided against goes to **Dropped directions** (last section) with the reason.
   A new card on a dropped direction must answer that row with a new reason.

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
| 0n | P-67 | Fast-proxy 5-fold CV ruler + single-variable recipe ablation loop | 🔧 **implemented, effect pending: `v14p` / `v14p2` in src, Kaggle smoke v45 green (10-05); the 5-fold floor run starts at the 10-10 reset** — approved by Tian (research.md 2.7.6 / 2.7.7 / 2.10) | + 0.003–0.005 per production member if ≥ half transfers → B6 ≈ 0.945–0.947 | ≈ 16 proxy variants per 30-h Kaggle week, or ≈ $0.6 each on a 4090; + 1 transfer arm | a measured pooled-OOF floor (2 seeds) |
| 0o | P-68 | Different-family (CNN OOF) image teacher at ≥ 0.5 on report-silent cells | 💡 proposed 2026-10-05 | 0..+ 0.004 solo (forum claims + 0.011; same-family was flat for us) | OOF free from P-67's best proxy, or ≈ $4 / 15 session-h; + 1 arm | the P-62 read; P-67 |
| 0p | P-69 | Sixth family for B6: ConvNeXt-T on the `v13h` recipe | ✅ approved with the loop 2026-10-05 (the week-1 family arm); loader check first | B6 + 0.002–0.003 (cross-family rule) | 1 Kaggle arm ≈ 6 h + a loader check | the 10-10 quota |
| 0i | P-62 | Silence-aware teacher mix (Raptor 0.75 where the report is silent, 0.5 where it speaks) | ⏳ **session D TRAINED green 2026-10-04 (`rsna-knee-train-b` v6, 5.94 h): gold-58 SWA `v13es` 0.9107 / `v13rs` 0.9160 vs 0.9126 / 0.9111 (direction only); shipped `rsna-knee-ckpt-v13es` / `-v13rs`; solos 10-05 = `rsna-knee-infer` v49 / v50; read m vs 0.9345 (✅ ≥ 0.9390)** | 0..+0.002 — likely under the 0.004 one-seed floor | one c03 session ≈ 3.5 GPU-h + 2 solos | — |
| 0l | P-65 | Grading-aware Claude relabel of the reports (an independent, severity-aware LLM vote) | ⏳ **session E trained on RunPod (Tian's go): `v13ecp` (0.5 Claude) gold-58 0.9063, `v13ec` (0.25 Claude) 0.9116 (🔁 direction only); infer v53 / v54; B0, seed 42; both solos 10-06 vs the B0 seed mean 0.9365** — ⏳ **full pass DONE 2026-10-04: 4,349 rows, ≈ 25 min, ≈ 6.9 M tokens; `claude_v1` / `claude_rap_v1` published in `rsna-knee-teacher-tables`; session E (`v13ec` ‖ `v13rc` on `claude_rap_v1` at mix 0.75) staged — runs on Kaggle after the 2026-10-10 reset, after session D's P-62 read; read m vs 0.9345 (✅ ≥ 0.9390)** — pilot 🔁 (Opus 0.9062 alone / 0.9397 with Raptor, bars 0.910 / 0.945 not met; Haiku 0.8639 ❌) | 0..+0.005 (literature + forum); the policy-misaligned labels are the upside | session E ≈ 6 GPU-h + 2 solos | — (Tian's go given) |
| 4 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15; **own candidate = B6 #52 0.942** (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, = the public-stack fork, replaces #48 0.938); fork candidate = the public stack with our trio at β 0.45 (**#49 0.943**, 🔁 +0.001, rank 373; `rsna-knee-fork` v11), next C2 with B6 as the leg | decides what the private LB scores | a browser session; ≤ 1 fork check | P-40 🔁 closed (#22 / #27), the fork v11 read, Rules page |
| 5 | P-18 | Efficiency track with the solo member | 💡 low priority — Tian 2026-10-04: score over efficiency; robustness half shipped; the natural fast candidate is `v13e` (0.935 solo, ≈ 15 min to score, 17 MB) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 6 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low — P-49 priced it on Raptor itself: matched mix 0.75 − 0.5 = −0.002 (SD 0.003) on gold | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 7 | P-46 | Upgrade the LLM half of the targets (absorbs P-16, P-30) | 💡 low — step 2 (re-label) superseded by P-65; step 1 (dread as a 4th vote) left | 0..+0.002 (dread vote) | ≈ 2.8 h per arm + 1 solo | P-65 session E's read |
| 8 | P-48 | Final-member polish: gold-58 as training rows + seed averaging | 💡 contested, parked | +0.001..0.002, unreadable by construction | part of the final retrain | P-50 decision |
| 9 | P-51 | Teacher-aware confidence weights | 💡 low | 0..+0.002 | ≈ 20 lines + 2.8 h; 1 solo | P-44 floor |
| 10 | P-70 | Per-study prediction fallback at the hidden rerun (train prior instead of a crash) | 💡 low, infra, proposed 2026-10-05 | robustness of both final picks | ≈ 20 lines + 1 smoke | — |

## Cards

### P-67 Fast-proxy 5-fold CV ruler + single-variable recipe ablation loop
Status:       🔧 **implemented, effect pending (2026-10-05, 09:12 UTC): `PROXY` preset + arms `v14p` / `v14p2` (seeds 42 / 43) in
              `src/kaggle_pipeline.py`; Kaggle smoke `rsna-knee-train` v45 green (PARALLEL pair, cache mounted, teacher table loaded,
              `_best.pt` + `_oof.csv` per child).** Approved by Tian ("go with the loop"): the real 5-fold floor run goes out at the
              10-10 reset — `sed PARALLEL_ARMS = ("v14p", "v14p2")`, `TEACHER_TABLES = ("raptor_teacher",)`, `FORCE_SMOKE = False`
              (≈ 3.75 h session, both T4s). Research.md 2.7.6 / 2.7.7 / 2.10.
Hypothesis:   a cheap CNN (ResNet-34 or EfficientNet-B0 @ 224 on c03 with the windows subsampled, 10–12 epochs, 5 folds) gives a
              pooled report-label OOF whose seed floor is ≈ 0.003–0.004 macro, and ablating one training variable at a time against
              it finds ≥ +0.005 on the proxy, of which ≥ half transfers to the 30-epoch production members.
Origin:       forum: Tucker (740610: 5-fold report-label CV, 0.003 threshold, aug stacks ablated singly), Scott (≈ 900 models), Myo
              (50 ep, patience 8, best at 28); our P-02 (fold-0 seed floor 0.008 → pooled 5-fold ≈ 0.008 / √5 ≈ 0.0036); the
              aneurysm 1st-place aug list (low-resolution simulation, grid distortion, intensity ops).
Evidence:     P-64 gave +0.010 from one heavy-aug + schedule setting on ResNet-34 and nothing has been ablated since; no forum
              post publishes a stack, so the stack has to be found. Image-side changes may be judged on the LLM-target OOF
              (traps 39 forbids that only for target changes).
Measure:      pooled 5-fold OOF macro-AUC vs `y__*` (the LLM blend) over the 4,349 report-only studies (`src/fold_oof_summary.py`);
              gold-58 reported, not read. Variables, in order: (1) aug components added one at a time to the P-64 "heavy" stack —
              rotation ± 10–25°, scale / shear ± 10 %, grid distortion, low-resolution (slice-thickness) simulation, Gaussian noise /
              blur, contrast / sharpen, gamma; (2) 30 vs 50 epochs with best-epoch selection; (3) head: window-attn vs GAP / max
              pooling; (4) drop-path 0.1 vs 0.2, EMA on / off; (5) mixup / cutmix across studies; (6) slot layout within c03
              (Dread: non-fluid slots weighted, blank slices dropped); (7) label smoothing / target temperature.
Noise floor:  measured first: two seeds of the baseline proxy → the pooled seed delta (expected ≈ 0.0036). Ablation bar = 1.5 × the
              measured floor. A variable that clears it on the proxy is confirmed by ONE production arm before the final retrains.
Cost:         proxy 5-fold ≈ 3.3 T4-GPU-h (≈ 40 min / fold) → ≈ 5 variants per 9-h Kaggle session on both T4s → ≈ 16 per 30-h week;
              or ≈ $0.6 per variant on a 4090. Setup ≈ 1 arm dict + `FIVE_FOLD` + the existing OOF summary; first session = baseline
              × 2 seeds + 3 variables. The transfer check = 1 production arm (≈ 6 h).
If it works:  the winning stack goes into every final retrain (P-50 week of 10-17: B3 × 2 seeds, B0 × 2, R50, the P-69 family) →
              each member + 0.003–0.005 → B6 ≈ 0.945–0.947 and the fork leg with it.
If it fails:  nothing clears 1.5 × the floor, or the proxy's wins do not transfer in the production check → week 2 = plain retrains
              with more seeds / families (blend rule: + 0.001–0.003).
Depends on:   Tian's go on the 10-10 week; the P-62 read does not block it (image-side only).

### P-68 Different-family image teacher on the report-silent cells (a CNN OOF table at ≥ 0.5)
Status:       💡 proposed 2026-10-05.
Hypothesis:   mixing a CNN-family OOF teacher (5-fold cross-fit of the B0 / B3 recipe) into the training targets at ≥ 0.5 on
              report-silent cells, beside Raptor (a CoAtNet) and the LLM blend, lifts a production solo by ≥ 0.004.
Origin:       forum 2.7.6: Archit (OOF image predictions > 50 % on silent cells, "only because the predictions were properly out of
              fold"), Raymond + 0.011 and SpeedSci + 0.011 from multi-source teachers; same-encoder OOF flat for Myo and for us
              (P-54 / P-55, the CoAtNet cross-fit on CoAtNet students). Literature 2.7.7: per-label uncertainty policies (CheXpert),
              soft pseudo-distillation + 0.03 CV (aneurysm 4th).
Evidence:     our only teacher gain is cross-family (Raptor CoAtNet → DINOv2 / CoAtNet students + 0.009); the Claude table's `m` field
              gives a second, finer silence mask than pilkwang `UNK`.
Measure:      solo LB of one production arm on the new mix vs its flat twin (one-seed ≥ 0.004), or a pair read ± 0.0045; gold
              direction only. The table itself: `src/build_distill_table.py` + quantile matching (both exist), plausibility by
              `src/teacher_plausibility.py`.
Noise floor:  ≥ 0.004 one-seed.
Cost:         the OOF table is free from P-67's best proxy 5-fold (a ≈ 0.92-level teacher), or ≈ 5.5 4090-h ≈ $4 / ≈ 15 Kaggle
              session-h for a 30-epoch B0 cross-fit (a ≈ 0.935-level teacher); then one production arm ≈ 6 h.
If it works:  all final retrains train on it. If it fails: the label side is closed; Raptor 0.5 (± the P-62 silent weight) stays.
Depends on:   the P-62 read (A3, 10-06: the silent-cell weight); P-67 for the free OOF.

### P-69 A sixth family for B6: ConvNeXt-T on the `v13h` recipe
Status:       ✅ approved with the loop 2026-10-05: the week-1 family arm, in the same session as the first proxy runs.
Hypothesis:   a ConvNeXt-T member at ≥ 0.935 lifts B6 by + 0.002–0.003 (the cross-family blend rule at six members), against + 0.001
              for another EfficientNet seed.
Origin:       experiments.md 2026-10-05 (blend rule); literature 2.7.7(c): ensembles of different pretrained models beat seed
              ensembles; Dread 0.941 → 0.944 with a ConvNeXt added.
Evidence:     `v06c` ConvNeXt-T was a weak early member (c01, DINOv2-era recipe: OOF 0.8562, blend + 0.004) and has never run on the
              `v13h` recipe; the weight Dataset `convnext-tiny-224-hf` (Apache-2.0) exists. Needs a loader check (HF-format weights
              vs `backbone="timm:convnext_tiny"`; `src/window_head_test.py`).
Measure:      solo LB (a member if ≥ 0.933) and B6 + ConvNeXt vs B6 0.942 (✅ ≥ 0.946 / 🔁 0.939–0.945).
Noise floor:  blend + 0.004 over B6; solo ≥ 0.004.
Cost:         one Kaggle arm ≈ 6 h on one T4 (pairs with another arm in the same session) + the loader check + a smoke.
If it works:  a member and part of the fork leg. If it fails: the families stay at three.
Depends on:   the 10-10 quota.

### P-70 Per-study prediction fallback at the hidden rerun (train prior instead of a crash)
Status:       💡 low, infrastructure; proposed 2026-10-05.
Hypothesis:   the hidden-test rerun can hit a decode / shape failure on a study we never saw; writing the training-prevalence (or
              OOF-mean) probabilities for that study, instead of raising or writing 0.5, keeps the whole submission scoring.
Origin:       the RSNA 2025 1st place falls back to OOF-mean probabilities on any pipeline failure (research.md 2.7.7).
Evidence:     our "loud-failure submission" (kernel v4) is designed to be visible in a placeholder run; what the inference loop does
              per study at the rerun has not been reviewed since.
Measure:      code review of the per-study exception path; one smoke with an injected corrupt study; the submission must still be
              complete and the failure logged.
Noise floor:  n/a (robustness).
Cost:         ≈ 20 lines + one smoke. Depends on: nothing.

### P-50 Final selection and publishability
Status:       💡 new 2026-09-27; decide by the 2026-10-15 entry deadline. **2026-10-05: the own candidate is B6 = #52 0.942**
              (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, flat rank-mean), equal to the public-stack fork and replacing #48. The
              fork candidate #49 (the stack + that trio at β 0.45) read **0.943** (🔁 +0.001; rank 373 of 5,187: the 0.942
              plateau is ≈ 1,000 forks wide). Next: C2 = the stack + B6 at β 0.45 (experiments.md "Submissions #49–#53").
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
Status:       ⏳ **2026-10-05: session E TRAINED — `v13ecp` (0.5 Claude) gold-58 SWA 0.9063, `v13ec` (0.25 Claude) 0.9116, vs the
              B0 seed pair 0.9126 / 0.9151 (🔁 direction only); both shipped, pod deleted (≈ $1.9); placeholders infer v53 / v54;
              solos 10-06 vs the B0 seed mean 0.9365** (experiments.md 2026-10-05 "Session E on RunPod, chain 1" and "chain 2").
              ⏳ **session E RUNNING on RunPod pod `j4obvfotdbdudo` (RTX 4090, $0.74/h) since 2026-10-05 09:10 UTC, on Tian's go ("Go E")
              — as a two-dose design on one backbone instead of the B0 + R50 pair:** `v13ecp` (new arm: 0.5 Raptor + 0.5 Claude,
              no LLM-blend share; `TEACHER_TABLES = ("raptor_teacher", "claude_v1")` at mix 1.0) then `v13ec` (the registered
              0.25 LLM + 0.5 Raptor + 0.25 Claude). Read each solo vs the B0 seed mean 0.9365 (`v13e` 0.935 / `v13e2` 0.938):
              ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932; `v13ecp` takes a 10-06 slot, `v13ec` the fifth slot or 10-07. The
              earlier plan — session E staged (Kaggle, after the 2026-10-10 reset); read m(`v13ec`, `v13rc`) vs m(`v13e`, `v13r`) = 0.9345:
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

## Dropped directions — do not propose again (with the why)

Merged from brainstorm.md, research.md §5 and the 2026-10-05 research (forum re-read 2.7.6; literature + RSNA 2025 2.7.7;
synthesis 2.10). One line each: the direction, why it is closed (forum / literature / our own reads), the source. **A new card on
any of these must answer its row with a new reason.** The 2026-10-05 rows are first.

| Direction | Why it is closed | Source |
|---|---|---|
| **EfficientNet-B4-class or larger CNNs; any resolution above 288 for capacity** | every ≥ 0.949 single on the forum is ResNet-50-class, a small ResNet / EffNet or a CoAtNet at 224–288 ("bigger is null": 735154, 738096); our own ladder R34 0.931 → R50 0.934 → B0 0.935 / 0.938 → B3 @ 288 0.940 ended at +0.0035 over the B0 seed mean (under the 0.004 bar); B4 @ 336 ≈ 25 GB VRAM and ≈ $3.7 (critic 10-05); a single bigger model only replicates an ensemble's gain [2202.06985] | research.md 2.7.6 / 2.7.7 (c), experiments.md 2026-10-05, candidates.md (T5 dropped) |
| **MIL / bag-of-slices / all-slice transformers** | forum: Tom @392 bag-of-32 0.915, Pand ConvNeXt-T MIL 0.935, nobody ≥ 0.94; coverage saturates (Ziad flat after ≈ 31 windows); 2025 3rd place: "MIL and LSTMs did not help" | research.md 2.7.6 / 2.7.7 |
| **DINOv3 / RadImageNet / BiomedCLIP / medical foundation backbones as members** | at ≤ 288 px ImageNet timm init ≥ any SSL or radiology init: DINOv2 weaker on clinical MRI [2402.07595], DINOv3 frozen at parity [2509.06467], RadiologyNET ≈ ImageNet [s41598-025-05009-w]; RadImageNet's licence is unclear (CLAUDE.md); the only large pretraining effect found anywhere is in-task dense pretraining (2025 1st) which needs labels we lack | research.md 2.7.7 (d) |
| **More seeds of the same recipe as a *score* lever** | blend rule 10-05: same-recipe members add +0.001–0.003, cross-family +0.004–0.006 (#26 / #29 / #32 / #36 vs #47 / #48 / #52 / #53); seeds are for the final members' robustness (P-50), not for the LB | experiments.md 2026-10-05 "Submissions #49–#53" |
| **External datasets** (OAI, MRNet, fastMRI+, SKM-TEA, KneeCoT) | KneeCoT banned; OAI needs institutional sign-off ("a no", 741819); the one forum user who tried external data: "not really" (743416); none carries our 12 labels; the licence fit is a winners' problem | CLAUDE.md Rules, research.md 2.7.6 |
| **Multimodal-LLM / VLM labelling of the images** (hengck23 745861, Deotte) | we never download the images in bulk (570 GB, hard constraint 2), inference has no internet, 18 days; OmerZalman: 80–90 % correct on 800 studies | research.md 2.7.6, CLAUDE.md |
| **Gold-58 as a member or recipe judge** | it inverted the LB direction several times for us (traps 39) and for SpeedSci / Lê / Raymond on the forum; direction only. The ruler for image-side changes is the pooled 5-fold report-label OOF (P-67) | traps 39, research.md 2.7.6 (B) |
| Text branch at inference | `test.csv` has no `Report`; nothing to read | CLAUDE.md |
| Horizontal flip — **both variants** (with or without medial↔lateral swap) | undoes laterality normalisation; the swap is anatomically wrong (MCL has no lateral counterpart); P-05 does *not* make it legal | [pilkwang], traps.md, critic item 23 |
| Vertical flip; zoom-out with padding | off-distribution; fabricated tissue | [pilkwang], [Guo et al.] |
| Geometric TTA | degraded 11/12 medical pairs; flips hurt knee OA ; our P-12 read +0.0016 on the blend; the 2025 winners' flip-TTA needs a label swap we cannot do | [2604.09697], [2311.06118], research.md 2.7.7 |
| Calibration, Platt scaling, thresholds, label smoothing on soft targets | AUC reads rank order only; `pos_weight` measured 🔁 (P-37), rejection confirmed | metric arithmetic, brainstorm.md |
| Averaging probabilities across folds/models | most confident model dominates; rank-mean instead | brainstorm.md |
| Full fine-tuning or best-epoch selection on 58 gold (~12/fold); a gold fine-tuning stage | SE 0.09, coin flip, our NaN-fold bug; gold's role is validation (gold is not trained under `train_all`); the contested final-retrain variant is P-48 | experiments.md, [Andre et al.], critic 25 |
| Tuning weights on the public LB | author-labelled overfit; 0.001–0.003 movements; the 0.936 notebook's gold-58-tuned per-label weights + "clinical residual" are worth **+0.001** over its untuned 0.935 (read in full 2026-08-30) | mattiaangeli (not re-read), `crazy_good_rsna.ipynb` (research.md §2.7.1), CLAUDE.md; **10-05:** fork β and blend weights stay flat — the 2025 aneurysm team that tuned weights partly on the public LB fell on private; our flat B6 (0.942) = the LB-tuned public stack |
| The 0.936 notebook's **88-feature stacking calibrator** (rank blocks + cross-view deltas + 12 protocol counts, w 0.4 on 7 labels) | decoded 2026-08-30: coefficients are ≈ a per-label reweighting of the same three views (protocol columns ≤ 0.003); est. +0.002–0.005 and fragile to a protocol-mix shift on private | cell-level re-read (research.md §2.7.1) |
| **Clinical residual** cross-label adjustments (`ACL −0.10 × mean rank(Contusion, Lateral Meniscus)` etc.) and correlation-guarded per-label fusion weights | the notebook itself: "an aggressive leaderboard experiment, not an unbiased estimate of private-test performance"; +0.001 stated | cell-level re-read |
| Random or report-only K-fold as the comparison metric | grouped vs random gap up to +0.136 | [EXPERIMENTS.md] |
| Native 3D CNN / nnU-Net / segmentation-first | 0.69 vs 0.85 (p=0.001); multi-A100 budgets ; **10-05:** nobody ≥ 0.94 on the knee forum uses 3D; the RSNA 2025 winners' 3D worked only with a vessel-segmentation-pretrained backbone (+0.11 in their ablation) and localisation labels we do not have | [MST], [CoPAS], research.md 2.7.6 / 2.7.7 |
| Frozen DINOv2 + head as the final model | 0.79 vs 0.85 knee; 0.776 vs 0.866 LB | [MST], sadamtorres (not re-read) |
| Backbone LR ≥ 5e-5 uniform on DINOv2 / SSL ViTs | every medical recipe ≤ 2e-5; 1e-3 collapse. Not for the hybrid: our CoAtNet trains at 1e-4, and 3e-5 was harmful (P-34) | [2501.14685], [dinov2 #276] |
| ~~More backbones on the same teacher table and cache~~ **SUPERSEDED 2026-10-05:** cross-family members at ≥ 0.93 *do* pay (+0.004–0.006 over the members' mean: #47 / #48 / #52; P-69 ConvNeXt-T is live). What stays dead: a member under ≈ 0.93, or a seed twin sold as diversity (+0.001–0.003)  More backbones on the same teacher table and cache — DINOv2-B, DINOv3, ConvNeXt, a third family | P-42: a second family on the Raptor table read 0.927 = `v09r` alone; within-class ρ 0.861 (LLM-target pair 0.777) — a shared teacher raises cross-family agreement | experiments.md 2026-09-27 "Submission #23", reviewer C (2026-09-27 audit) |
| DINOv2 → DINOv3 swap at 224 as an accuracy gain | ±0.002–0.008; wins only at 512 ; **10-05 literature:** DINOv2 weaker than ImageNet CNNs on clinical brain MRI [2402.07595]; at ≤ 288 px ImageNet timm init is as good as any SSL or radiology init | [AnyMC3D], [2510.07191], research.md 2.7.7 (d) |
| BiomedCLIP / MedSAM / RAD-DINO / OrthoFoundation | far below general ViTs; CXR-only; weights not public | [2501.14685], [2601.18250] |
| EfficientNet-B0 mean-pool | 0.664 vs 0.809 public | [JunhaoLiXD] |
| Laterality tag alone / default L / IPP-corner rule; pixel-flipping sagittal slots | tag missing 50.7%; corner 58.8%; SAG stacks are order-reversed | [FINDINGS.md], [pilkwang] |
| Filename / InstanceNumber slice ordering | ρ ≈ −0.01 | experiments.md |
| Crops ≥ 160 mm; resolution > 288 | skipped on 60% of series; ViT losses −6.6/−7.9 pp ; **10-05:** no gain above 288 anywhere on the forum (Raymond, tennogh, Tucker; Less @384 0.937, KalyanG17 CoAtNet @384 0.935); our P-43 (320 px) +0.002 🔁 at 1.4× the inference time | [pilkwang], [2510.07191], research.md 2.7.6 |
| N4 / Nyul / VOI-LUT before normalisation | segmentation/radiomics evidence only; infeasible at 24k series | [2307.03827] |
| Decoding DICOM in the DataLoader each epoch; float32 caches; `.npz` + mmap; GPU decode at ≤ 512 px | 100× slower; 29.6 GB; mmap ignored; ~1–2.5× | [hida1211], [NumPy #5976], [nvImageCodec] |
| `pip install` at scoring time | internet off | [pydicom plugin table] |
| bf16 on T4; channels_last for ViT; `torch.compile` by default | no bf16 tensor cores; cuDNN-only; compile > gain (SDPA is the cheap win — P-08) | [PyTorch memory_format], critic 27 |
| More *report-label* LLM sources beyond P-46's one test (dread as a 4th vote), Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2 in literature, **~1.5 here** (φ 0.88); our 0.002 spread. Image-grounded tables are the demonstrated lever (P-39); the LLM half is P-46. Exception on a new reason: P-65's grading-aware Claude vote (the host's severity thresholds) | [2605.29800], [BoxWRENCH], experiments.md, label_audit.md |
| Co-teaching / DivideMix / DISC; focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE ; **10-05 literature:** no medical multi-label win over soft targets (BoMD 89.7 vs co-teaching 80.1) | [LNMBench], [RAL], [Xin et al.], research.md 2.7.7 (b) |
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
