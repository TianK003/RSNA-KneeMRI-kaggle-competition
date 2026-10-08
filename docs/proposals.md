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

**Focus (Tian, 2026-10-06): single-model strength and new family members for the blend; labels are de-emphasised.** The 10-06
reads back it: every target change except Raptor read flat or worse (P-65 closed), and the blend gain grows with the number of
families (B13 / B11). Label-side cards (P-68, P-47, P-51) run only on an otherwise idle slot; P-62 read 🔁 on 10-07
and is closed (experiments.md). The 10-06 brainstorm added P-71 … P-74; the ideas it set aside are rows in Dropped directions.

**Read 2026-10-08: P-69 closed ✅.** ConvNeXt-T `v15c` on its own optimiser reads **0.942 solo** (our best single model, = the
five-member B6) and B17 = B6 + it **0.943** (experiments.md "Submissions #64–#66"). The follow-up is **P-80** (the seed twin, then
the final ConvNeXt vote). It also takes most of P-71's case away.

**Tian's decisions, 10-06 night (they supersede the evening's "no ConvNeXt"):**
- **P-69 is re-opened: one ConvNeXt-T arm with its own recipe**, built from the literature and other teams' experience
  (research.md 2.7.9), trained on RunPod under a $2.5 cap (critic-checked), 288 px, and a second seed only if the 10-07
  B4 / B5 reads free the B3 money and both fit under the cap. Read on 10-08: the solo, and B17 = B6 + it. (Done: ✅, above.)
- **Single-model research → cards only, Tian picks later:** P-75 … P-79 below, plus amendments to P-62 / P-67 / P-68 / P-73.
- Unchanged from the evening: run the pseudo-label pair (P-68 `v13ex` ‖ `v13ex2`) at the 10-10 reset; the rest is decided later.

**Tian's decisions, 10-06 evening: run the pseudo-label pair (P-68 `v13ex` ‖ `v13ex2`) and NO ConvNeXt training (P-69 dropped);
everything else is decided later.** (The ConvNeXt half is superseded by the night's decision above.)

**Research later on 10-06 (research.md 2.7.8): improve the families we have first.**
- A fourth family is worth ≈ +0.0008 LB at our level (gold-58 model: experiments.md 2026-10-06 "Gold-58: what a blend gains by pair
  type").
- No forum read shows ConvNeXt helping beyond one tick.
- Per-member levers moved us +0.0035–0.010 each.
- Recommended: P-67, P-73 and P-74 first; at most one hedged ConvNeXt arm (Tian: none that evening, then one that night — P-69); P-72 dropped; P-71
  demoted. P-68's `v13ex` seed pair is surfaced as the single-model lever with the most
  forum evidence.

| rank | id | title | status | expected value | cost | depends on |
|---|---|---|---|---|---|---|
| 1 | P-68 | Different-family OOF image teacher (cross-family pairs; no silent-cell weight: A3 read 🔁 on 10-07) | ⏳ **APPROVED 2026-10-06 evening (Tian): one session at the 10-10 reset, the seed pair `v13ex` ‖ `v13ex2` (B0 on Raptor + the CoAtNet cross-fit OOF `xfit_v09k`); `v13ex2` (seed 43) added 10-06 night, pair smoke `rsna-knee-train-b` v8 green. Read the pair mean vs the B0 seed mean 0.9365: ✅ ≥ 0.9410 / 🔁 0.9320–0.9409 / ❌ ≤ 0.9319.** Earlier 10-06: demoted ("don't focus on labels that much"). Was: implemented 2026-10-05, effect pending; Tian's top priority ("especially this"); redesigned after the critic the same day**: `v13ex` (B0 student on Raptor + `xfit_v09k`, the CoAtNet cross-fit OOF; trainable at the 10-10 reset), `v11o` (CoAtNet student on Raptor + `cnnoof_v1`, the B0 floor pair's OOF; after the floor run), `v13eo` (B0 on `cnnoof_v1`, the same-family control), all at the flat mix 0.5; RunPod on 10-07 = NO-GO (critic: ≈ $3.8, no decision unlocked before 10-17) | 0..+ 0.004 solo (forum claims + 0.011; same-family read flat for us) | the P-67 floor run + 1–2 Kaggle arms (≈ 6 T4-h each) | A3 (10-07) for any silent mix; the P-67 floor run for `cnnoof_v1` |
| 2 | P-80 | ConvNeXt as the lead family: the seed twin `v15c2`, then the final ConvNeXt vote | 💡 new 2026-10-08 (P-69's "if it works" branch fired: B17 0.943); **needs Tian's go for where it trains** (RunPod ≈ $1.46 after a critic check, or Kaggle after the 10-10 reset) | the seed spread of our strongest family; a two-seed vote ≈ +0.001–0.003 on the solo (#26: +0.003) | one arm ≈ $1.46 on a 4090 (or ≈ 6.5–10 T4-h); ≈ 15 lines for `INFER_VOTE_GROUPS`; 1–2 sends | — |
| 3 | P-71 | An independently trained public reader as a blend member (B16: goodpjw2008's 2.5D ConvNeXt-T, Apache-2.0) | 💡 new 2026-10-06; **recommended DROP (pending Tian), 10-08:** our own ConvNeXt-T `v15c` reads 0.942 solo (#64), so B16 would add a second, weaker (0.929) ConvNeXt vote, plus third-party code at the rerun. Was: recommended DEMOTE (≈ +0.0005 LB) | B16 ≈ 0.941–0.943 vs B6 0.942; upside if its independence beats our families' (gold ρ 0.83–0.89) | ≈ 60 lines + 1 placeholder (≈ 0.2 GPU-h) + 1 send; ≈ 20 min more scoring | — |
| 4 | P-72 | A seventh family on the `v13h` recipe (`eca_nfnet_l0` by default) + a family-diversity ruler on the proxy OOF (Δ_div) | 💡 new 2026-10-06; **recommended DROP (pending Tian): no read on this task, in1k only** | B6 + 0.001–0.003 if member-grade (≥ 0.933 solo); the ruler ranks families on 4,349 studies, not LB slots | weight Dataset + loader check; 1 Kaggle arm ≈ 6 h; ≈ 1 proxy per screened family | the 10-10 quota; the P-67 floor for Δ_div |
| 5 | P-67 | Fast-proxy 5-fold CV ruler + single-variable recipe ablation loop | 🔧 **implemented, effect pending: the floor pair `v14p` / `v14p2` and, since 10-05 (Tian: "focus on these"), ten one-variable arms `v14lr` / `v14th` / `v14gd` / `v14bl` / `v14ns` / `v14sh` (augmentation components), `v14mx` (mixup), `v14r288` (B0 @ 288), `v14db` (blank windows), `v14ep20` (longer schedule); unit checks, a local CPU smoke and a Kaggle GPU smoke (v46) green; the floor run starts at the 10-10 reset** — approved by Tian (research.md 2.7.6 / 2.7.7 / 2.10) | + 0.003–0.005 per production member if ≥ half transfers → B6 ≈ 0.945–0.947 | ≈ 16 proxy variants per 30-h Kaggle week, or ≈ $0.6 each on a 4090; + 1 transfer arm | a measured pooled-OOF floor (2 seeds) |
| 6 | P-73 | Study-level token mixer before the per-label attention head | 💡 new 2026-10-06; a P-67 proxy variable (`v14tx`) | 0..+0.004 per member if it transfers | ≈ 40 lines + a unit check + 1 proxy | the P-67 floor |
| 7 | P-74 | CNN throughput on a T4: `channels_last`, and B3 @ 288 by progressive resizing | 💡 new 2026-10-06, infrastructure | + 8–35 % arms per week; B3 seeds on Kaggle instead of RunPod | ≈ 10 + 20 lines; smoke timings; 1 proxy (`v14prog`) | — |
| 8 | P-75 | Noisy-Student JFT weights for the EfficientNet members (`tf_efficientnet_b0 / b3.ns_jft_in1k`) | 💡 new 2026-10-06 night (research.md 2.7.9); **Tian picks later** | 0..+0.003 solo, plus blend diversity (different pretraining data) | weights Dataset + loader check + 1 proxy (`v14jft`) | the P-67 floor |
| 9 | P-76 | CNN learning rate 3e-4 → 6e-4 (then weight decay 0.05 as its own arm) | 💡 new 2026-10-06 night; **Tian picks later** | 0..+0.004 solo, can be negative | 1 proxy (`v14blr6`) | the P-67 floor |
| 10 | P-77 | A real SWA tail: LR held at 0.25× over the last 25 %, ≥ 6 snapshots averaged | 💡 new 2026-10-06 night; **Tian picks later** | 0..+0.003 solo (our SWA averages near-identical points today) | ≈ 20 lines + 1 proxy (`v14swa`) | the P-67 floor |
| 11 | P-78 | SAM (ρ 0.05) around AdamW | 💡 new 2026-10-06 night; **Tian picks later** | 0..+0.004 solo (label-noise robustness) | ≈ 40 lines; ≈ 2× compute per proxy (`v14sam`) | the P-67 floor |
| 12 | P-79 | Bias-field augmentation (a smooth multiplicative intensity field) | 💡 new 2026-10-06 night; **Tian picks later** | 0..+0.002 solo | ≈ 15 lines + 1 proxy (`v14bf`) | the P-67 floor |
| 13 | P-50 | Final selection and publishability | 💡 decide by 2026-10-15; **own candidate since 10-08 = B17 #65 0.943** (B6 + ConvNeXt-T `v15c`; 🔁 +0.001 over B6, pick 1 by the house convention; `v15c` alone #64 0.942; B12 #66 0.941 closed); **C3's gate is open** (B17 ≥ 0.943), but the public plateau moved to **0.950** on 10-08, so the fork's anchor choice is open again (brainstorm.md); was: own candidate = B6 #52 0.942 (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`; B13 #55 0.940 and B11 #56 0.941 did not beat it, 10-06); **fork candidate = C2 #54 0.944** (the public 0.942 stack + B6 at β 0.45, `rsna-knee-fork` v12; 🔁 +0.001 vs #49, rank 337; level with the public 0.944 notebook); C3 closed (no own blend ≥ 0.943); 10-07 drop-one reads: B14 (no CoAtNet) 0.942 = B6, B4 0.941, B5 0.940 (#59 / #62 / #63; week 2 drops the CoAtNet retrain); B12 (B6 + `v13es` + `v13rs`) passed its gate; the shortlist sends are candidates.md's priority table | decides what the private LB scores | a browser session | the Rules page |
| 15 | P-70 | Per-study prediction fallback at the hidden rerun (train prior instead of a crash) | 💡 low, infra, proposed 2026-10-05 | robustness of both final picks | ≈ 20 lines + 1 smoke | — |
| 16 | P-18 | Efficiency track with the solo member | 💡 low priority — Tian 2026-10-04: score over efficiency; robustness half shipped; the natural fast candidate is `v13e` (0.935 solo, ≈ 15 min to score, 17 MB) | a separate prize; unknown until the formula is read | 0 GPU h (CLI + browser) | Efficiency formula (browser) |
| 17 | P-47 | Teacher-mix bracket: mix 0.75 only | 💡 low, label-side (parked by the 10-06 focus) — P-49 priced it on Raptor itself: matched mix 0.75 − 0.5 = −0.002 (SD 0.003) on gold | ≈ 0 (+0.000..0.002) | per-arm `TEACHER_MIX` code + ≈ 2.8 h; 1 solo | P-44 floor, an idle slot |
| 18 | P-48 | Final-member polish: gold-58 as training rows + seed averaging | 💡 contested, parked | +0.001..0.002, unreadable by construction | part of the final retrain | P-50 decision |
| 19 | P-51 | Teacher-aware confidence weights | 💡 low, label-side (parked by the 10-06 focus) | 0..+0.002 | ≈ 20 lines + 2.8 h; 1 solo | P-44 floor |

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
              **2026-10-05 (Tian: "focus on these"), implemented as arms, each = PROXY + one key, seed 42, on raptor_teacher:**
              `v14lr` lowres (in-plane down / up-sample, the aneurysm winner's low-resolution simulation), `v14th` thick
              (through-plane blend of the window's three slices), `v14gd` grid distortion, `v14bl` blur, `v14ns` noise, `v14sh`
              sharpen (each `aug_extra` component at p 0.3 after the heavy stack), `v14mx` study-level mixup (p 0.5, the two
              studies' cache arrays blended voxel for voxel), `v14r288` B0 @ 288 (resolution vs B3's capacity), `v14db` blank
              windows dropped (`drop_blank_frac` 0.3, train = infer; likely null: the c03 band already trims the stack ends, and
              four local studies' darkest centre slices sit at 0.31–0.96 of their slot median), `v14ep20` 20 proxy epochs
              (≈ 50 production epochs; the per-epoch pooled OOF is the curve). Order, by evidence: lowres, grid, mixup, r288,
              ep20, blur / noise / sharpen, thick, db. **Amendment 2026-10-06 night (research.md 2.7.9 B2):** sharpen is the
              best-supported image-quality component (BigAug's single best transform: unseen prostate 64.4 → 77.4 Dice, source
              + 1.0), while blur and low-resolution cost the source domain there (prostate 89.6 → 86.1), so `v14sh` moves ahead of
              `v14bl` / `v14lr` when a slot frees up; and `v14mx`'s α 0.4 folded to λ ≥ 0.5 leaves ≈ 48 % of the mixed batches at
              λ ≥ 0.9, so a second mixup arm at α ≈ 2 (`v14mx2` = `v14mx` + `"mixup_alpha": 2.0`, an existing Config key; not
              coded yet) is worth adding (ChestX-ray14 mixup at α 3: 74.2 → 76.7). Production knobs for the transfer and final arms: `snapshot_every` (EMA
              snapshots `_ep{e}_ema.pt`, submittable) and `train_gold` (P-48). Unit checks (`src/window_head_test.py`) and a local
              CPU smoke with every switch on (mixup on a two-study batch, all six components at p 1, blank dropping, snapshots,
              SWA, inference) green; **Kaggle GPU smoke `rsna-knee-train` v46 green** (PARALLEL `v14mx` ‖ `v14lr` on the c03 cache: rc 0 per
              child, `_best.pt` = SWA, peak 1.11 GiB at batch 2 × 12 windows, the two arms' losses differ).
Noise floor:  measured first: two seeds of the baseline proxy → the pooled seed delta (expected ≈ 0.0036). **2026-10-05 (critic):** one
              pooled |Δ| is a single draw; estimate the floor from the five per-fold paired seed differences instead (their SD / √5
              for the pooled macro), or add a third baseline seed. Ablation bar = 1.5 × the
              measured floor. A variable that clears it on the proxy is confirmed by ONE production arm before the final retrains.
Cost:         proxy 5-fold ≈ 3.3 T4-GPU-h (≈ 40 min / fold) → ≈ 5 variants per 9-h Kaggle session on both T4s → ≈ 16 per 30-h week;
              or ≈ $0.6 per variant on a 4090. Setup ≈ 1 arm dict + `FIVE_FOLD` + the existing OOF summary; first session = baseline
              × 2 seeds + 3 variables. The transfer check = 1 production arm (≈ 6 h).
If it works:  the winning stack goes into every final retrain (P-50 week of 10-17: B3 × 2 seeds, B0 × 2, R50, the CoAtNet) →
              each member + 0.003–0.005 → B6 ≈ 0.945–0.947 and the fork leg with it.
If it fails:  nothing clears 1.5 × the floor, or the proxy's wins do not transfer in the production check → week 2 = plain retrains
              with more seeds / families (blend rule: + 0.001–0.003).
Depends on:   Tian's go on the 10-10 week; the P-62 read does not block it (image-side only).

### P-68 Different-family image teacher on the report-silent cells (a CNN OOF table at ≥ 0.5)
Status:       ⏳ **APPROVED 2026-10-06 evening (Tian: "I want a run of pseudo-labels"): one Kaggle session at the 10-10 reset,
              `v13ex` ‖ `v13ex2`** = the `v13e` B0 recipe at seeds 42 / 43 on `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`,
              flat mix 0.5 (0.5 LLM + 0.25 Raptor + 0.25 CoAtNet cross-fit OOF). Same seeds as `v13e` / `v13e2`, so each arm has a
              same-seed control. **Read:** mean(`v13ex`, `v13ex2`) vs the B0 seed mean 0.9365: ✅ ≥ 0.9410 / 🔁 0.9320–0.9409 /
              ❌ ≤ 0.9319 (the two-arm band ± 0.0045). Gold-58 not readable (the OOF table trained the gold rows). **`v13ex2`
              added 2026-10-06 (= `v13ex` + `"seed": 43`, pinned to `("raptor_teacher", "xfit_v09k")`, unit check + local smoke
              green); Kaggle smoke of the pair `rsna-knee-train-b` v8 green** (experiments.md Scoreboard) — the 10-10 real push
              is the same build with `FORCE_SMOKE = False`. If ✅: every final member trains on the
              mix, and `v11o` (the CoAtNet student on the CNN OOF) is the next arm. Earlier 10-06:
              🔧 **DEMOTED (Tian: "don't focus on labels that much", single models and a new family first).** It is a
              target change, and every target change except Raptor has read flat or worse on the LB (P-38, P-45, P-55, P-65). It runs
              only on an otherwise idle GPU slot, `v13ex` first; `v11o` / `v13eo` are paused. **10-06 evening (research.md 2.7.8):**
              the forum's largest measured single-model jumps (Raymond +0.011, SpeedSci +0.011, Archit) came from exactly this kind of
              teacher: different-source image models, out of fold, heavy on silent cells. It is the one target change we have not
              tested. If Tian gives targets one session, make it a `v13ex` ‖ `v13ex2` seed pair (± 0.0045 two-arm band). Its tables
              (Raptor + `xfit_v09k`) cannot share a session with a Raptor-only arm. Was:
              🔧 **implemented 2026-10-05, effect pending — Tian's top priority ("especially this one"); REDESIGNED the same day
              after the critic subagent** (artifacts/runpod_case_P68_1007.md + its review):
              - **Cross-family pairs only.** Same-family OOF teachers read flat for us (P-55: CoAtNet OOF into CoAtNet students)
                and for Myo; Archit's own-model setup is the one counter-example. So: `v13ex` = the `v13e` B0 student on
                `raptor_teacher` + `xfit_v09k` (the P-54 CoAtNet cross-fit OOF, which already exists → trainable at the 10-10 reset,
                first session); `v11o` = the `v11a` CoAtNet student on `raptor_teacher` + `cnnoof_v1` (the B0 floor pair's OOF,
                built after the P-67 floor run); `v13eo` = the B0 student on `cnnoof_v1`, the same-family control, lowest priority.
              - **One variable.** All three at the flat mix 0.5 (the two tables averaged after quantile matching: 0.5 LLM +
                0.25 Raptor + 0.25 OOF), so the LLM share stays 0.5 and the change is the image-teacher half. No silent-cell mix:
                A3 read 🔁 on 10-07 (P-62 closed, not adopted).
              - **Reads:** each solo vs its parent's seed mean (B0: 0.9365; CoAtNet `v11a` / `v11b` 0.932 / 0.929 → 0.9305), the
                one-seed bands (✅ ≥ parent + 0.004). **Gold-58 is not readable here, not even for direction:** both OOF tables come
                from k-fold runs that trained the gold rows at weight 8.
              - **RunPod 10-07 = NO-GO** (critic): recomputed ≈ 5 h ≈ $3.8, not $2.6 (per-study overhead does not shrink with the
                window count); the week-2 retrains start 10-17, so a 10-08 read unlocks nothing earlier than a 10-10 Kaggle read;
                it would leave ≈ $1.2, under one B3 retrain. On Kaggle, `v13ex` ‖ `v13ex2` go out at 10-10 00:00 and read 10-10
                evening; `v11o` follows the floor run.
              - **The table:** `src/build_distill_table.py --sets "<dir>/v14p_fold[0-9]_oof.csv" "<dir>/v14p2_fold[0-9]_oof.csv"
                --per-fold-rank --expect-folds 5 --expect-rows 4407 --out artifacts/teacher/cnnoof_v1.csv` (refuses a partial
                set: `mix_teacher` would silently fall back to the LLM value on uncovered rows), a plausibility read
                (`src/teacher_plausibility.py`), then publish to `rsna-knee-teacher-tables` (private). Later P-67 variants each add
                a five-fold OOF set, so `cnnoof_v2` can pool the best few proxies.
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
If it works:  all final retrains train on it. **Amendment 2026-10-06 night (research.md 2.7.9 C6):** Raptor and `xfit_v09k` are
              both CoAtNets, so their errors may correlate with each other; after a ✅ the production follow-up student is B3 (a
              student at least as large as its teacher gains more: Noisy Student B4 student +0.8, B5 +1.5 over a B4 teacher), and a
              second round switches family (`v11o`). With mixup in the same run, α ≥ 1. Realistic gain ≤ +0.004 (the forum's
              +0.011 came from weaker starting points).
              If it fails: the label side is closed; Raptor 0.5 stays (P-62 read 🔁 on 10-07).
Depends on:   — for `v13ex` / `v13ex2` (`xfit_v09k` exists). No silent mix (A3 read 🔁 on 10-07).
              `v11o` / `v13eo` need P-67's floor run for `cnnoof_v1`. Since 10-06 (P-65 ❌) this is the only live target-side card.

### P-80 ConvNeXt as the lead family: the seed twin `v15c2`, then the final ConvNeXt vote
Status:       💡 new 2026-10-08. P-69's pre-registered "more ConvNeXt training in week 2 only if B17 ≥ 0.943" fired (B17 #65 0.943).
              The arm `v15c2` (= `v15c` at seed 43) is built and smoke-green (`rsna-knee-train` v47, 10-06). Not scheduled:
              where it trains is Tian's call (RunPod needs a critic-checked case first).
Hypothesis:   `v15c`'s 0.942 is the family's level, not a lucky draw, and the two seeds as ONE vote make the strongest member of
              the final blends.
Origin:       experiments.md 2026-10-08 "Submissions #64–#66"; P-69 (closed ✅).
Evidence:     `v15c` 0.942 solo = B6; the CNN seed spread is s = 0.003 (#51), so one draw of 0.942 is consistent with a family
              level anywhere in ≈ 0.939–0.945. A same-recipe seed pair read +0.003 over each member on the CoAtNet (#26). B17 is only
              +0.001 over `v15c` alone, so the ConvNeXt is carrying the blend.
Measure:      (1) the `v15c2` solo LB; s_c = |`v15c2` − 0.942|. (2) `v15c` + `v15c2` as one vote (needs `INFER_VOTE_GROUPS`,
              ≈ 15 lines in the infer path: rank-mean the listed versions into one member before the flat blend; B13 showed
              that extra votes of one family cost), solo and inside the best own blend.
Noise floor:  one seed 0.004; the pair mean ± 0.0045; blends by the candidates.md section-B bands.
Read rules:   s_c ≤ 0.003 → the family level is ≈ 0.942 and the bands stand; s_c ≥ 0.005 → widen the ConvNeXt bands and treat
              #64 as a draw. The pair vote replaces `v15c` in every final blend if its solo ≥ 0.942.
Cost:         one arm ≈ $1.46 and 1.5–2 pod-h on a 4090 (P-69's measured run; RunPod ≈ $3.5 left), or ≈ 6.5–10 T4-h on Kaggle
              after the 10-10 reset (288 px × 20 epochs; two sessions with a resume). 1–2 sends.
If it works:  the ConvNeXt pair is the lead vote of both final picks (the own blend and the fork leg).
If it fails:  (s_c large, `v15c2` ≤ 0.938) `v15c` stays a single vote and B17 stays pick 1 on one draw.
Later, Tian picks (not cards yet): the clip-rate question (82 % of steps clipped at 1.0: a probe at clip 5.0 or none); 224 px
              for a cheaper vote. A bigger ConvNeXt is not proposed: the forum's "bigger is null" for CNNs (T5 dropped).
Depends on:   Tian's go (RunPod, or the 10-10 Kaggle quota).

### P-71 An independently trained public reader as a blend member (B16 first)
Status:       **Recommended DROP 2026-10-08 (pending Tian):** our own ConvNeXt-T `v15c` reads 0.942 solo (#64), so B16 would add a
              second ConvNeXt vote at 0.929. Earlier: 💡 new 2026-10-06 (brainstorm; Tian: single models and a new family for the blend). Inference only, so it can be read
              before the 10-10 reset. **Recommended 10-06 evening: DEMOTE** (research.md 2.7.8; pending Tian). A 0.929 member at
              ρ ≈ 0.90 is worth ≈ +0.0005 LB by the gold-58 model. It adds a third-party code path at the hidden rerun and
              ≈ 20 min of scoring, and makes pick 1 partly not ours. Build it only if the engineering time is truly spare.
Hypothesis:   the public 2.5D ConvNeXt-T reader (goodpjw2008, 0.929 solo with 3 folds) added to B6 as a sixth member lifts the flat
              rank-mean, because it shares neither our input (c03 windows at 150 mm) nor our targets (Raptor 0.5): B16 = B6 + reader ≥ B6.
Origin:       experiments.md 2026-10-06 "The public frontier moved"; the 10-06 blend refinement (new families pay, extra same-family
              members under the mean cost).
Evidence:     the reader lifts the 0.943 community stack to 0.944 at 15–30 % (45 % → 0.942). Its input: 0.4 mm/px, a 154 mm field of
              view, 256 px, 12 three-slice windows per series, a 2-layer study transformer, per-finding attention pooling. Its training:
              5 folds grouped by report text, soft BCE on the mean of four public label tables, no image teacher. The Dataset
              `goodpjw2008/rsna-knee-2-5d-convnext-reader` is **Apache-2.0** (read 10-06): three fold checkpoints (123 MB each),
              `preprocess.py` / `knee.py` / `infer.py`, offline pylibjpeg wheels. Against: a 0.929 member sits under B6's mean (0.9358),
              and the blend rule prices B16 at ≈ 0.941–0.943.
Measure:      (0, optional, direction only) the reader over the 58 gold studies in a Kaggle kernel → its within-class ρ to B6's members
              (our families sit at 0.83–0.89 to each other); ρ ≤ 0.80 would make it more independent than any family we have.
              (1) B16's solo LB vs B6 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938; ≥ 0.943 → pick 1 and the C3 leg.
Noise floor:  the blend bands above (LB rounded to 0.001).
Cost:         ≈ 60 lines in `src/kaggle_pipeline.py`: an external-member hook that runs the reader's `infer.py` in a subprocess on
              `test_series` after our members have freed the GPU, reads its CSV, checks rows and columns, and ranks it in as one
              member (its three folds averaged inside that vote). Mount the Dataset in `rsna-knee-infer`; one placeholder (≈ 0.2 GPU-h);
              ≈ 20 min more scoring on the hidden test.
If it works:  B16 replaces B6 as pick 1 and becomes the C3 leg. Pick 1 then holds one public model, still not part of the stack.
If it fails:  the reader stays out. (Our own ConvNeXt-T, P-69, was re-opened by Tian on 10-06 night, and became a member on 10-08: as a member,
              B16 would be a second ConvNeXt vote.)
Depends on:   — (licence read 10-06; Rules 2.6.b: a public Dataset is usable).

### P-72 A seventh family on the `v13h` recipe, and a family-diversity ruler on the proxy OOF
Status:       💡 new 2026-10-06 (brainstorm). **Recommended 10-06 evening: DROP the NFNet arm** (research.md 2.7.8; pending
              Tian): no read of NFNet / RegNet on this task anywhere, in1k pretraining only, the same recipe risk as ConvNeXt, and a
              fifth family is worth even less than a fourth (gold-58 model ≈ +0.0008 LB for the fourth). Δ_div stays as a tool.
Hypothesis:   a family that is none of ResNet, EfficientNet, CoAtNet or ConvNeXt reaches member grade (≥ 0.933 solo) on the `v13h`
              recipe and adds to B6 as the other families did (+ 0.001–0.003 at six or seven members). Default: `eca_nfnet_l0` (NFNet:
              no BatchNorm, scaled weight standardisation; ≈ 24 M parameters, ImageNet ≈ 82.6 %, the size of our ResNet-50).
              Fallback: `regnety_040` (BatchNorm, ResNet-like dynamics, so likely closer to `v13r`).
Origin:       the 10-06 refinement (the blend gain grows with the number of families); literature 2.7.7 (c): different pretrained models
              beat seeds; Dread 0.941 → 0.944 with a second family.
Evidence:     our three families sit at gold within-class ρ 0.83–0.89; same-recipe pairs at 0.88–0.95. No read of NFNet or RegNet on this
              task anywhere we have looked. Set aside for this slot: EfficientNetV2 / MobileNetV4 (the EfficientNet family again), MaxViT
              (CoAtNet's), SE-ResNeXt-50 (ResNet's), Swin / ViTs (DINOv2-S read 0.918 solo; no ViT ≥ 0.94 on the forum), DenseNet-121
              (ImageNet ≈ 74 %, likely under the member bar).
Measure:      (1) the production arm's solo LB: a member if ≥ 0.933; then B6 + it vs B6 (the section-B bands).
              (2) the ruler, on the P-67 proxy: **Δ_div** = pooled OOF of rank-mean(B0 proxy, family proxy) − pooled OOF of
              rank-mean(B0 proxy seed 42, B0 proxy seed 43). Both are pairs, so Δ_div separates diversity from "two models beat one".
              Same targets and an image-side change, so the OOF is a valid ruler (traps 39 bars only target changes). It ranks candidate
              families on 4,349 studies instead of LB slots, and it mirrors the LB rule (cross-family + 0.0025–0.0045 vs same-recipe
              + 0.001–0.003 for pairs).
Noise floor:  solo: the member bar 0.933 and the one-seed band (≥ 0.004 vs a parent). Δ_div: the P-67 per-fold paired floor.
Cost:         a weight Dataset (`timm-eca-nfnet-l0`; timm weights, Apache-2.0) + a CPU loader check (the `window_head_test.py` pattern)
              + a Kaggle smoke; frozen BN is a no-op on an NFNet (check the smoke's loss falls at LR 3e-4); one production arm ≈ 6 T4-h.
              A family proxy costs ≈ 1–2 × the B0 proxy (≈ 3.3 T4-h); time it in the smoke before screening more than one.
If it works:  a seventh member in the final retrains and the fork leg; Δ_div becomes the gate for any further family.
If it fails:  below 0.933 → the families stay at three; the screen, if run, names the next candidate.
Depends on:   the 10-10 quota; the P-67 floor run for Δ_div.

### P-73 Study-level token mixer before the per-label attention head
Status:       💡 new 2026-10-06 (brainstorm); a P-67 proxy variable (`v14tx`) once the floor is measured.
Hypothesis:   letting the window tokens of a study exchange information before the per-label attention pooling (2 transformer layers
              over all windows, with slot-type and slice-position embeddings) lifts a single model by ≥ 1.5 × the proxy floor.
Origin:       the public ConvNeXt-T reader (0.929 on report labels only, 3 folds) uses exactly this head; the RSNA 2025 aneurysm 1st
              place's ablation: without its location transformer 0.896 vs 0.902 (−0.006; research.md 2.7.7 (e)).
Evidence:     our `window_attn` head pools each finding independently over windows; no window sees the other planes, so cross-plane
              evidence (an ACL tear visible sagittally and coronally) meets only in the final weighted sum. Against: Tucker reaches the
              0.94s with "no attention, simple pooling"; 4,349 studies may be too few for a mixer; Ziad's −0.014 (mean vs attention)
              is about pooling, not mixing. **Amendment 2026-10-06 night (research.md 2.7.9):** a knee-MRI benchmark (AnyMC3D)
              ranks learnable-query attention, our head's type, above a transformer (0.962 vs 0.950; LSTM 0.903), and Will's region
              tokens added +0.000003 (740610), so the prior on a mixer is lower than the aneurysm ablation suggests.
Measure:      P-67 proxy pooled OOF, `v14tx` vs the floor pair; gold reported, not read.
Noise floor:  1.5 × the P-67 floor.
Cost:         ≈ 40 lines (`head_type="window_tx"`: project features to 256, `nn.TransformerEncoder` 2 layers × 4 heads, dropout 0.1,
              slot + position embeddings, then the existing per-label attention) + a unit check in `window_head_test.py` + one proxy
              (≈ 3.3 T4-h).
If it works:  one production transfer arm (B0 or B3), then the final retrains.
If it fails:  the head stays; P-67's pooling variable (window-attn vs GAP / max) covers the other direction.
Depends on:   the P-67 floor run.

### P-74 CNN throughput on a T4: `channels_last`, and B3 @ 288 by progressive resizing
Status:       💡 new 2026-10-06 (brainstorm), infrastructure.
Hypothesis:   (a) `channels_last` memory format on the CNN encoders under AMP speeds a 30-epoch arm by 8–35 % with predictions equal up to
              fp16 noise; (b) training B3 at 224 for epochs 0–23 and at 288 for 24–29 (SWA 27–29 at 288) brings a B3 arm under ≈ 8 T4-h
              (now 12–15 h, so RunPod-only) at the same solo LB.
Origin:       research.md 2.8 (the PyTorch memory-format tutorial: 8–35 % for cuDNN conv nets); progressive resizing (fast.ai;
              EfficientNetV2's progressive learning).
Evidence:     a 30-ep CNN arm takes ≈ 5.9 h on a T4, and `src/kaggle_pipeline.py` sets no `channels_last` (grep, 10-06). B3 @ 288 is our
              best single (0.940), and the final plan needs two B3 seeds (≈ $4 of RunPod).
Measure:      (a) s/study in two Kaggle smokes of the same arm with and without; inference equal to ≤ 1e-3 on the 3 test studies.
              (b) a proxy variable `v14prog` (B0 224 → 288) vs `v14r288` on the P-67 ruler before any B3 arm.
Noise floor:  (a) ≈ 5 % timing noise between runs (measure twice); (b) 1.5 × the P-67 floor.
Cost:         (a) ≈ 10 lines + one smoke; (b) ≈ 20 lines (a per-epoch `img_size`) + one proxy.
If it works:  (a) ≈ one more arm per week; (b) B3 seeds train on Kaggle, and the RunPod money stays for emergencies.
If it fails:  (a) stays off; (b) B3 stays on RunPod.
Depends on:   — ((b) needs the P-67 floor).

### P-75 Noisy-Student JFT weights for the EfficientNet members
Status:       💡 new 2026-10-06 night (single-model research, research.md 2.7.9); **Tian picks later** (cards only, no code).
Hypothesis:   the same EfficientNet-B0 started from `tf_efficientnet_b0.ns_jft_in1k` (ImageNet + 300 M unlabelled JFT images,
              Noisy Student) instead of `efficientnet_b0.ra_in1k` lifts the pooled proxy OOF by ≥ 1.5 × the P-67 floor, and
              its errors decorrelate from the in1k members'.
Origin:       literature + Kaggle (research.md 2.7.9): different pretraining data is the strongest error decorrelator
              (Gontijo-Lopes 2022); SIIM-ISIC 2020 1st place used `tf_efficientnet_*_ns` for 15 of its 18 models.
Evidence:     ImageNet top-1 +0.98 (B0: 78.68 vs 77.70) and +1.75 (B3: 84.05 vs 82.30); both Apache-2.0, ImageNet mean / std
              (the AdvProp `ap` weights would need 0.5 / 0.5 instead). Against: in-domain medical gains from bigger pretraining
              are small at this model size (Mustafa 2021, CheXpert 76.5 vs 76.6) and CheXtransfer finds no link between ImageNet
              accuracy and chest X-ray AUC.
Measure:      one P-67 proxy arm `v14jft` = PROXY with only the init changed (Tucker: "change nothing but the init"); time the
              TF "same"-padding variant in its smoke.
Noise floor:  1.5 × the P-67 floor.
Cost:         a weights Dataset + a loader check + 1 proxy (≈ 3.3–3.6 T4-h).
If it works:  one production B0-ns or B3-ns arm, and check its within-class ρ against the in1k twin for blend diversity.
If it fails:  the in1k weights stay.
Depends on:   the P-67 floor run.

### P-76 CNN learning rate upward (3e-4 → 6e-4)
Status:       💡 new 2026-10-06 night; **Tian picks later**.
Hypothesis:   the B0 proxy at `lr_backbone` 6e-4 beats 3e-4 by ≥ 1.5 × the floor; 3e-4 has never been bracketed from above.
Origin:       our P-59 (ResNet-34 1e-4 + LLRD 0.75 → 3e-4 uniform: gold 0.831 → 0.899, confounded with epochs); Li et al. 2020
              (target domains unlike ImageNet need larger effective LRs); RSNA 2023 1st place AdamW up to 4e-4.
Evidence:     research.md 2.7.9. Against: the optimum depends on the schedule (12 proxy vs 30 production epochs), so a proxy win
              needs a production check; ViTs collapse at 1e-3 (not CNNs).
Measure:      proxy arm `v14blr6`; if it wins, bracket 1e-3, then weight decay 0.05 as its own arm.
Noise floor:  1.5 × the P-67 floor.
Cost:         config only + 1 proxy (≈ 3.3 T4-h).
If it works:  one production transfer arm; then every CNN retrain uses it.
If it fails:  3e-4 stays.
Depends on:   the P-67 floor run.

### P-77 A real SWA tail (constant LR over the last quarter, ≥ 6 snapshots)
Status:       💡 new 2026-10-06 night; **Tian picks later**.
Hypothesis:   holding the LR at 0.25 × peak over the last 25 % of the steps and averaging ≥ 6 half-epoch snapshots beats today's
              "SWA" by ≥ 1.5 × the floor.
Origin:       Izmailov 2018 (SWA needs a constant or cyclic high LR; a decaying running average "does not perform very
              differently"); SWAD (DomainBed 63.3 → 66.9).
Evidence:     today `lr_at` decays the cosine to 0, so `swa_last 3` averages EMA weights taken at ≤ 3 % of the peak LR: nearly the
              same point, which matches our SWA ≈ 0 and Dread's six pairs ≈ 0 (737696). Frozen BN means no BN recompute is needed.
Measure:      proxy arm `v14swa`.
Noise floor:  1.5 × the P-67 floor.
Cost:         ≈ 20 lines (a schedule floor + a snapshot ring) + a unit check + 1 proxy.
If it works:  every final retrain uses the tail. If it fails: the cosine-to-0 + 3-epoch average stays.
Depends on:   the P-67 floor run.

### P-78 SAM around AdamW
Status:       💡 new 2026-10-06 night; **Tian picks later**.
Hypothesis:   sharpness-aware minimisation (ρ 0.05) makes the proxy more robust to the noisy report labels and lifts its pooled
              OOF by ≥ 1.5 × the floor.
Origin:       Foret 2021 (CIFAR-10 at 40 % label noise: SGD 68.8 → SAM 93.4); a medical SAM survey (PMC12121992: breast
              ultrasound ResNet-50 80.2 → 84.0, PathMNIST 86.1 → 88.4); averaging SAM iterates beats either alone (Kaddour 2022).
Evidence:     research.md 2.7.9. Against: those papers use hard noisy labels without soft targets, heavy augmentation or EMA, so
              expect the low end; no Kaggle or forum read found.
Measure:      proxy arm `v14sam` (two passes per step; unscale AMP before the ascent step, clip after).
Noise floor:  1.5 × the P-67 floor.
Cost:         ≈ 40 lines + a unit check; ≈ 2.1 × the proxy's compute (≈ 7 T4-h).
If it works:  one production transfer arm (≈ 12 T4-h at 2×, or RunPod). If it fails: AdamW stays.
Depends on:   the P-67 floor run.

### P-79 Bias-field augmentation
Status:       💡 new 2026-10-06 night; **Tian picks later**.
Hypothesis:   a random smooth multiplicative intensity field (log-polynomial, order 3, coefficients ± 0.3, shared by a window's three
              slices) at p 0.3 lifts the proxy by ≥ 1.5 × the floor: surface-coil inhomogeneity is the MRI artefact our global
              gain / gamma / contrast cannot imitate.
Origin:       BigAug (1906.03347): intensity perturbations were among the strongest transforms for unseen MRI domains (prostate
              Dice 64.4 → 71.5–73.2) at no source-domain cost; TorchIO `RandomBiasField`.
Evidence:     no isolated classification ablation found. Not the dropped "N4" row: that is bias-field *correction* in
              preprocessing, this is augmentation.
Measure:      proxy arm `v14bf` (a new `aug_extra` component `biasfield`).
Noise floor:  1.5 × the P-67 floor.
Cost:         ≈ 15 lines + a unit check + 1 proxy.
If it works:  it joins the heavy stack for the final retrains. If it fails: the stack stays.
Depends on:   the P-67 floor run.

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
Status:       💡 new 2026-09-27; decide by the 2026-10-15 entry deadline. **2026-10-06: pick 1 = B6 #52 0.942** (B13 #55 0.940
              and B11 #56 0.941 did not beat it); **pick 2 = C2 #54 0.944** (the public 0.942 stack + B6 at β 0.45, `rsna-knee-fork`
              v12, rank 337 of 5,293), replacing #49 (v11). C3 is closed until an own blend reads ≥ 0.943. Caveats from the 10-06
              public read (experiments.md "The public frontier moved"): the community stack now reads 0.943 alone, a public fork
              with a ConvNeXt-T leg 0.944, and that author measured the stack's run-to-run spread at one tick, so #49 vs #54 is
              weak evidence; pick 2 rests on construction (flat β, our best own blend as the leg).
              **2026-10-05: the own candidate is B6 = #52 0.942**
              (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, flat rank-mean), equal to the public-stack fork and replacing #48. The
              fork candidate #49 (the stack + that trio at β 0.45) read **0.943** (🔁 +0.001; rank 373 of 5,187: the 0.942
              plateau is ≈ 1,000 forks wide). Next: C2 = the stack + B6 at β 0.45 (experiments.md "Submissions #49–#53"), queued for 10-06. **2026-10-05 (Tian): submissions
              only until 10-10;** the shortlist sends (B12 = B6 + qualifying 10-06 solos, B14 / B13 = the CoAtNet weight, C3 = a better
              fork leg) are ranked by day in candidates.md.
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
Decision:     never pick #13 and #15 together. Pick 1: our best own ensemble (B6 #52 since 10-05, or an own blend that beats it
              on the solo LB by 10-15). Pick 2: the public-stack fork with our best own blend as the leg at β 0.45 (C2,
              `rsna-knee-fork` v12, since 10-06; was v11 with the #48 trio).
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
Depends on:   the Rules page (*Open questions*), P-48 (the final-member decision). The fork reads are in (#49 0.943, #54 0.944).

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

### P-48 Final-member polish: gold-58 as training rows + seed averaging
Status:       💡 CONTESTED, parked; decide at P-50. **2026-10-05: the switch exists** — `train_gold=True` on a `train_all` arm
              adds the 58 gold rows to training (gold labels at `gold_weight`; Tucker's 1.3 % of the loss ≈ gold_weight 1, ours
              is 8 ≈ 10 %); the gold-58 validation is then in-sample and the log says so. Tian listed it among the focus items
              for the final retrains.
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

## Dropped directions — do not propose again (with the why)

Merged from brainstorm.md, research.md §5 and the 2026-10-05 research (forum re-read 2.7.6; literature + RSNA 2025 2.7.7;
synthesis 2.10). One line each: the direction, why it is closed (forum / literature / our own reads), the source. **A new card on
any of these must answer its row with a new reason.** The 2026-10-06 brainstorm rows are first, then the 2026-10-05 rows.

| Direction | Why it is closed | Source |
|---|---|---|
| ~~**Training a ConvNeXt-T member (P-69)** — Tian, 2026-10-06 evening: "no convnext training"~~ **RE-OPENED 2026-10-06 night by Tian, and REVERSED by the read 2026-10-08: `v15c` = 0.942 solo, our best single model (P-69 closed ✅, follow-up P-80). Not a dropped direction any more; the prior below was right about the blend (+0.001) and wrong about the solo** | no forum read shows ConvNeXt helping beyond one LB tick (solos 0.929–0.939, always under the owner's other family; the public reader +0.001 = its rerun spread; Dread's +0.003 confounded); a fourth family is worth ≈ +0.0008 LB on B6 (gold-58 model); our CNN recipe (3e-4 uniform, 30 ep) is the configuration the literature flags for ConvNeXt; our own `v06c` peaked at epoch 3. Re-open only with a reason that beats one tick | research.md 2.7.8; experiments.md 2026-10-06 "Gold-58: what a blend gains by pair type", 2026-08-30 "`v06c`" |
| **Distilling B6 (or any ensemble) into one student** (10-06 brainstorm) | a target change, and Tian de-emphasised labels on 10-06; every student-of-our-own-models read flat (P-38 #19, P-55 #34–#36), and the public reader's author saw the same: OOF self-distillation lifted the single model 0.922 → 0.927 but lowered the 5-fold ensemble 0.930 → 0.929. A student is also at seed distance from its teachers, so it adds nothing to the blend that holds them | experiments.md 2026-09-24 "Submission #19", 2026-09-30 "Submissions #34–#38", 2026-10-06 "The public frontier moved" |
| **"New families" that share a family with a member we have**: EfficientNetV2 / MobileNetV4 (EfficientNet), MaxViT (CoAtNet), SE-ResNeXt-50 / ResNet-101 (ResNet); also Swin / plain ViTs as members (10-06 brainstorm) | 10-06: extra members of a family already in the blend cost (B13 #55 0.940 vs B6 0.942); the family slots go to ConvNeXt-T (P-69) and a non-BN CNN (P-72). ViTs: DINOv2-S read 0.918 solo (#21) and nobody ≥ 0.94 on the forum uses one | experiments.md 2026-10-06 "Submissions #54–#58"; research.md 2.7.6 |
| **Snapshot or checkpoint ensembles inside one run as a score lever** (10-06 brainstorm) | at best a same-recipe gain (+0.001–0.003, the 10-05 rule), smaller than a seed's; `snapshot_every` stays as the tool for epoch selection (P-67 `v14ep20`), not as members | experiments.md 2026-10-05 "Submissions #49–#53" |
| **A new input geometry before 10-22** (Dread's 140 mm / 80-slice cache) (10-06 brainstorm) | needs four new CPU cache kernels (≈ 50 GB) and fresh arms for every member; Dread's curve (0.917 → 0.932) was on CoAtNet-384, and Ziad's attention pipeline went negative at higher density; c03 is already at or above the forum's density (research.md 2.7.6 C). Slot-layout changes *inside* c03 remain P-67 variable (6) | research.md 2.7.6 |
| **EfficientNet-B4-class or larger CNNs; any resolution above 288 for capacity** | every ≥ 0.949 single on the forum is ResNet-50-class, a small ResNet / EffNet or a CoAtNet at 224–288 ("bigger is null": 735154, 738096); our own ladder R34 0.931 → R50 0.934 → B0 0.935 / 0.938 → B3 @ 288 0.940 ended at +0.0035 over the B0 seed mean (under the 0.004 bar); B4 @ 336 ≈ 25 GB VRAM and ≈ $3.7 (critic 10-05); a single bigger model only replicates an ensemble's gain [2202.06985] | research.md 2.7.6 / 2.7.7 (c), experiments.md 2026-10-05, candidates.md (T5 dropped) |
| **MIL / bag-of-slices / all-slice transformers** | forum: Tom @392 bag-of-32 0.915, Pand ConvNeXt-T MIL 0.935, nobody ≥ 0.94; coverage saturates (Ziad flat after ≈ 31 windows); 2025 3rd place: "MIL and LSTMs did not help" | research.md 2.7.6 / 2.7.7 |
| **DINOv3 / RadImageNet / BiomedCLIP / medical foundation backbones as members** | at ≤ 288 px ImageNet timm init ≥ any SSL or radiology init: DINOv2 weaker on clinical MRI [2402.07595], DINOv3 frozen at parity [2509.06467], RadiologyNET ≈ ImageNet [s41598-025-05009-w]; RadImageNet's licence is unclear (CLAUDE.md); the only large pretraining effect found anywhere is in-task dense pretraining (2025 1st) which needs labels we lack | research.md 2.7.7 (d) |
| **More seeds of the same recipe as a *score* lever** | blend rule 10-05: same-recipe members add +0.001–0.003, cross-family +0.004–0.006 (#26 / #29 / #32 / #36 vs #47 / #48 / #52 / #53); seeds are for the final members' robustness (P-50), not for the LB. **10-06:** two more CoAtNet-recipe votes on B6 lost 0.002 (B13 #55 0.940), the EfficientNet-only triple 0.001 (B11 #56 0.941): extra members of a family already in the blend do not pay | experiments.md 2026-10-05 "Submissions #49–#53"; 2026-10-06 "Submissions #54–#58" |
| **External datasets** (OAI, MRNet, fastMRI+, SKM-TEA, KneeCoT) | KneeCoT banned; OAI needs institutional sign-off ("a no", 741819); the one forum user who tried external data: "not really" (743416); none carries our 12 labels; the licence fit is a winners' problem. **10-08: the counter-example is public.** The 0.949 single model was trained with 2,399 OAI knees as masked labels for PF OA / Lateral OA / Synovitis: +0.005 in its author's table, the whole public gap above 0.943. Legality is still open: login.gov access with an email worked for some, failed from China; no host ruling since (experiments.md 2026-10-08 "The 0.949 / 0.950 public notebooks, read"). Stays dropped for our own training (NDA access + a new pipeline with 14 days left); using their public checkpoint is candidates.md C4, Tian's call | CLAUDE.md Rules, research.md 2.7.6 |
| **Multimodal-LLM / VLM labelling of the images** (hengck23 745861, Deotte) | we never download the images in bulk (570 GB, hard constraint 2), inference has no internet, 18 days; OmerZalman: 80–90 % correct on 800 studies | research.md 2.7.6, CLAUDE.md |
| **Gold-58 as a member or recipe judge** | it inverted the LB direction several times for us (traps 39) and for SpeedSci / Lê / Raymond on the forum; direction only. The ruler for image-side changes is the pooled 5-fold report-label OOF (P-67) | traps 39, research.md 2.7.6 (B) |
| Text branch at inference | `test.csv` has no `Report`; nothing to read | CLAUDE.md |
| Horizontal flip — **both variants** (with or without medial↔lateral swap) | undoes laterality normalisation; the swap is anatomically wrong (MCL has no lateral counterpart); P-05 does *not* make it legal | [pilkwang], traps.md, critic item 23 |
| Vertical flip; zoom-out with padding | off-distribution; fabricated tissue | [pilkwang], [Guo et al.] |
| Geometric TTA | **10-08:** the 0.949 author's anatomical L/R mirror (sagittal: reverse the slice channels; coronal / axial: width flip) read +0.001, a width flip on every plane −0.001: both one tick; degraded 11/12 medical pairs; flips hurt knee OA ; our P-12 read +0.0016 on the blend; the 2025 winners' flip-TTA needs a label swap we cannot do | [2604.09697], [2311.06118], research.md 2.7.7 |
| Calibration, Platt scaling, thresholds, label smoothing on soft targets | AUC reads rank order only; `pos_weight` measured 🔁 (P-37), rejection confirmed | metric arithmetic, brainstorm.md |
| Averaging probabilities across folds/models | most confident model dominates; rank-mean instead | brainstorm.md |
| Full fine-tuning or best-epoch selection on 58 gold (~12/fold); a gold fine-tuning stage | SE 0.09, coin flip, our NaN-fold bug; gold's role is validation (gold is not trained under `train_all`); the contested final-retrain variant is P-48 | experiments.md, [Andre et al.], critic 25 |
| Tuning weights on the public LB | author-labelled overfit; 0.001–0.003 movements; the 0.936 notebook's gold-58-tuned per-label weights + "clinical residual" are worth **+0.001** over its untuned 0.935 (read in full 2026-08-30) | mattiaangeli (not re-read), `crazy_good_rsna.ipynb` (research.md §2.7.1), CLAUDE.md; **10-05:** fork β and blend weights stay flat — the 2025 aneurysm team that tuned weights partly on the public LB fell on private; our flat B6 (0.942) = the LB-tuned public stack |
| **Per-label model routing** (send each finding to the member best on it) chosen on gold-58 | measured 2026-10-05 on 11 members + B6: in-sample oracle routing +0.012 over B6's flat rank-mean, but chosen on half the gold studies and scored on the other half **−0.006** (sd 0.007, worse in 83 % of 600 splits); per-label member gaps are inside the per-label SE ≈ 0.09. On the LB it is the per-label weight tuning above. Re-open only on a ruler with thousands of studies (the P-67 five-fold OOF), with shrinkage to equal weights | experiments.md 2026-10-05 "Per-label model routing on gold-58" |
| The 0.936 notebook's **88-feature stacking calibrator** (rank blocks + cross-view deltas + 12 protocol counts, w 0.4 on 7 labels) | decoded 2026-08-30: coefficients are ≈ a per-label reweighting of the same three views (protocol columns ≤ 0.003); est. +0.002–0.005 and fragile to a protocol-mix shift on private | cell-level re-read (research.md §2.7.1) |
| **Clinical residual** cross-label adjustments (`ACL −0.10 × mean rank(Contusion, Lateral Meniscus)` etc.) and correlation-guarded per-label fusion weights | the notebook itself: "an aggressive leaderboard experiment, not an unbiased estimate of private-test performance"; +0.001 stated | cell-level re-read |
| Random or report-only K-fold as the comparison metric | grouped vs random gap up to +0.136 | [EXPERIMENTS.md] |
| Native 3D CNN / nnU-Net / segmentation-first | 0.69 vs 0.85 (p=0.001); multi-A100 budgets ; **10-05:** nobody ≥ 0.94 on the knee forum uses 3D; the RSNA 2025 winners' 3D worked only with a vessel-segmentation-pretrained backbone (+0.11 in their ablation) and localisation labels we do not have | [MST], [CoPAS], research.md 2.7.6 / 2.7.7 |
| Frozen DINOv2 + head as the final model | 0.79 vs 0.85 knee; 0.776 vs 0.866 LB | [MST], sadamtorres (not re-read) |
| Backbone LR ≥ 5e-5 uniform on DINOv2 / SSL ViTs | every medical recipe ≤ 2e-5; 1e-3 collapse. Not for the hybrid: our CoAtNet trains at 1e-4, and 3e-5 was harmful (P-34) | [2501.14685], [dinov2 #276] |
| ~~More backbones on the same teacher table and cache~~ **SUPERSEDED 2026-10-05:** cross-family members at ≥ 0.93 *do* pay (+0.004–0.006 over the members' mean: #47 / #48 / #52; P-69 ConvNeXt-T: dropped by Tian on 10-06 evening, re-opened that night with its own recipe, research.md 2.7.8 / 2.7.9). What stays dead: a member under ≈ 0.93, or a seed twin sold as diversity (+0.001–0.003)  More backbones on the same teacher table and cache — DINOv2-B, DINOv3, ConvNeXt, a third family | P-42: a second family on the Raptor table read 0.927 = `v09r` alone; within-class ρ 0.861 (LLM-target pair 0.777) — a shared teacher raises cross-family agreement | experiments.md 2026-09-27 "Submission #23", reviewer C (2026-09-27 audit) |
| DINOv2 → DINOv3 swap at 224 as an accuracy gain | ±0.002–0.008; wins only at 512 ; **10-05 literature:** DINOv2 weaker than ImageNet CNNs on clinical brain MRI [2402.07595]; at ≤ 288 px ImageNet timm init is as good as any SSL or radiology init | [AnyMC3D], [2510.07191], research.md 2.7.7 (d) |
| BiomedCLIP / MedSAM / RAD-DINO / OrthoFoundation | far below general ViTs; CXR-only; weights not public | [2501.14685], [2601.18250] |
| EfficientNet-B0 mean-pool | 0.664 vs 0.809 public | [JunhaoLiXD] |
| Laterality tag alone / default L / IPP-corner rule; pixel-flipping sagittal slots | tag missing 50.7%; corner 58.8%; SAG stacks are order-reversed | [FINDINGS.md], [pilkwang] |
| Filename / InstanceNumber slice ordering | ρ ≈ −0.01 | experiments.md |
| Crops ≥ 160 mm; resolution > 288 | skipped on 60% of series; ViT losses −6.6/−7.9 pp ; **10-05:** no gain above 288 anywhere on the forum (Raymond, tennogh, Tucker; Less @384 0.937, KalyanG17 CoAtNet @384 0.935); our P-43 (320 px) +0.002 🔁 at 1.4× the inference time. **10-08:** the 0.949 author's 224 → 384 step read +0.003 (CoAtNet-2, with OAI, confounded with the architecture variant); their non-OAI 384 model read 0.940–0.942 | [pilkwang], [2510.07191], research.md 2.7.6 |
| N4 / Nyul / VOI-LUT before normalisation | segmentation/radiomics evidence only; infeasible at 24k series | [2307.03827] |
| Decoding DICOM in the DataLoader each epoch; float32 caches; `.npz` + mmap; GPU decode at ≤ 512 px | 100× slower; 29.6 GB; mmap ignored; ~1–2.5× | [hida1211], [NumPy #5976], [nvImageCodec] |
| `pip install` at scoring time | internet off | [pydicom plugin table] |
| bf16 on T4; channels_last for ViT; `torch.compile` by default | no bf16 tensor cores; cuDNN-only; compile > gain (SDPA is the cheap win — P-08) | [PyTorch memory_format], critic 27 |
| More *report-label* LLM sources beyond P-46's one test (dread as a 4th vote), Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2 in literature, **~1.5 here** (φ 0.88); our 0.002 spread. Image-grounded tables are the demonstrated lever (P-39). **10-06: the one exception ran and failed** — P-65's grading-aware Claude relabel read #57 0.935 / #58 0.932 vs the B0 seed mean 0.9365 (no lift at either dose), and P-46 step 1 (dread as a 4th vote) closed with it by its own rule. No report-side target change is open; a new card here needs a reason that is not "a better or different reader of the reports" | [2605.29800], [BoxWRENCH], experiments.md, label_audit.md |
| Co-teaching / DivideMix / DISC; focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE ; **10-05 literature:** no medical multi-label win over soft targets (BoMD 89.7 vs co-teaching 80.1) | [LNMBench], [RAL], [Xin et al.], research.md 2.7.7 (b) |
| Calibrated priors with a 50% floor for zero-support states | OOF collapsed to base rate | [JunhaoLiXD V02] |
| Translate-then-extract; sub-3B extractors; 70B on 2×T4 | precision loss; F1 0.74; ~40 GB weights | [2602.21374] |
| ~~Hosted LLM APIs for report text~~ — **SUPERSEDED 2026-10-04 by host rule 2.6.b** (hosted LLMs permitted at minimal cost); it ran as P-65, ❌ on the LB 2026-10-06 (the row above) | was: plausible Rule 4.b violation; open-weights parity | CLAUDE.md "Rules", [Radiology 2025] |
| In-domain SSL continued pretraining as a first priority | in-domain ViT still below ImageNet AlexNet on MRNet | [SB-SSL] |
| Auxiliary report-reconstruction head | same weak supervision as the targets; nothing new to learn | brainstorm.md #10 (speculative, unranked) |
| Judging a target-source change on fold-0 OOF vs the LLM teacher | rewards agreement with the teacher, not truth: P-38's fold-0 +0.0109 read −0.001 in production (#19) | traps 39, experiments.md 2026-09-24 "Submission #19" |
| A weighted `v09r` 0.75 / `v08r` 0.25 solo | a sub-floor bet that `v08r` pays for its weakness; the flat pair already read 0.927 = `v09r` (P-42) | handoff 2026-09-27 |
| Re-anchoring the fork on "Speedy Raptors 0.943" | it is our anchor + two serial CoAt readers; ≤ +0.001 by its own claim (0.2× the floor), more serial rerun work. **10-06, still closed:** that stack is now the public plateau (984 teams at 0.943, skarin's reproduction), and the stack + a 0.929 ConvNeXt-T leg reads 0.944 = our 0.942 anchor + B6 (#54); the stack's own run-to-run spread is one tick | experiments.md Infrastructure 2026-09-27; 2026-10-06 "The public frontier moved" |
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
