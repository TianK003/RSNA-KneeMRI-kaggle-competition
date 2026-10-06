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

Untried ideas do **not** belong here — they are cards in [proposals.md](proposals.md) (the ranked
backlog); open questions are in [brainstorm.md](brainstorm.md). This file is only for things that were
actually run and measured.

**The noise floor is the most important number in this file.** With 58 gold studies the
Hanley–McNeil SE of an AUC near 0.8 is ≈0.09 — a 95% interval of roughly ±0.17. Public-LB
deltas under ~0.005 are also noise (the top 10 public teams span 0.006 total). So:

> A gold-AUC difference under ~0.05, or a public-LB difference under ~0.005, is
> **INCONCLUSIVE, not a win.** Record it as such. Chasing those is exactly how the top
> public notebook admits it overfit the leaderboard.

Judge label changes on **coverage** (does the rule fire at all, per language) and on
**OOF over all 4,407 studies**, not on the 58 gold alone.

**CORRECTED 2026-10-04:** the working rules are in proposals.md, "Decision-metric hierarchy". Since traps 39,
label and target changes are judged by gold-58 direction plus the solo LB. OOF against the LLM targets rewards
agreement with the teacher, not truth. A one-seed LB delta needs ≥ 0.004 (P-44), not 0.005.

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
| 2026-08-30 | **P-23 six-version blend, submission #9 (infer v12)**: #8 set + v08w + v10c | fold-0 proxy **0.8795** | **0.909** | ✅ **KEEP: +0.009 over #8 = 1.8× the 0.005 LB floor**; OOF→LB offset +0.0295 (predicted 0.905–0.907, landed 0.909); the c02 band + window-attention recipe carries to the hidden test (Submissions #9) |
| 2026-08-30 | **P-23 seven-version blend, submission #10 (infer v13)**: #9 set + v09h | fold-0 proxy **0.8820** | **0.912** | ✅ **best on the board and the default blend** (tie rule: more members); the `v09h` *increment* is **+0.003 vs #9 → 🔁 under the 0.005 floor** (OOF said +0.0024); +0.012 vs #8's 0.900 = 2.4× the floor for the c02 lane as a whole; offset +0.030 (Submissions #10) |
| 2026-08-30 | **5-fold `v09h` (RunPod chain4, folds 1–4 added to fold 0; 3.4 h, ≈ $2.6)** | pooled OOF **0.8625** · gold-58 0.874 | — | ✅ the ensemble base: +0.016 over `v05g`'s pooled 0.8467; per-fold 0.8546–0.8683; infer v14 mounts all five (one vote) |
| 2026-08-30 | **Submission #11 (infer v14)**: the #10 blend with `v09h` as **5 folds** (still one vote) | fold-0 proxy 0.8820 · `v09h` pooled 0.8625 | **0.913** | 🔁 **as an increment, exactly as pre-registered** — +0.001 over #10 is 0.2× the 0.005 LB floor, and the rule written before sending was "0.912–0.916 = 🔁 by design, ≥ 0.917 = real". Folds are replicates inside one vote of seven, so this is confirmation, not disappointment. **Best on the board → the default blend** (it can only match or beat #10's members). Read 2026-09-03 (Submissions #11) |
| 2026-09-21 | **P-27 fork, submission #12 (`rsna-knee-fork` v3, sent 21:24, ref 56442019)**: the public 0.942 graph (cells 0–49 verbatim) + our arm (`v08w` + 5-fold `v09h`) at β = 0.20; placeholder run green (anchor graph 184 s, ours rc 0, `beta0.20`) | anchor's own public LB 0.942 · ours: fold-0 proxy of the two c02 members only | **0.939** (read 2026-09-22 07:50) | **below the pre-registered 0.940 line → the action rule fires (next: β 0.10, then the anchor alone)**; as *evidence* it is 🔁: −0.003 vs the author's 0.942 is 0.6× the 0.005 floor, and the anchor was never scored from our account (sources are unpinned Datasets; the arm is fail-soft, so a silently skipped arm would also read 0.942 — an anchor-only submission is the control). **Best number on the board** (Submissions #12) |
| 2026-09-21 | **`v09a` (P-28, kernel `rsna-knee-train`)**: CoAtNet-1 @224 on c02 + window_attn, **all 4,349 studies, 16 ep, SWA of the last 3 EMA snapshots** | gold-58 (all 58, reported only): **SWA 0.8768**, last EMA 0.8762, peak epoch 5 0.8925 | — | ✅ ran to completion (16 ep, 5.12 h, 0.25 s/study); SWA vs last EMA +0.0006 = 🔁 (no BatchNorm penalty, P-28 "if it fails" not triggered); gold peaks at epoch 5 and drifts −0.016 to epoch 15 — inside the gold SE, see the 2026-09-22 entry; LB via the fork ⏳ (#14) → read 2026-09-22: #14 **0.941** 🔁 |
| 2026-09-21 | **`v08a` (P-28, kernel `rsna-knee-folds` v6, pushed 21:26)**: DINOv2-S @224 on c02 + window_attn, same regime | gold-58: **SWA 0.8816**, last EMA 0.8802, peak epoch 6 0.8944 | — | ✅ ran to completion (16 ep, 2.38 h); SWA vs last EMA +0.0014 = 🔁; same peak-then-drift shape as `v09a` (−0.014 from epoch 6 to 15); LB via the fork ⏳ (#14) → read 2026-09-22: #14 **0.941** 🔁 |
| 2026-09-22 | **P-27 fork, submission #13 (`rsna-knee-fork` v4, sent 08:30, ref 56458837)**: #12's graph and members (`v08w` + 5-fold `v09h`) at **β = 0.10** | — | **0.942** | = the anchor-only control (#15, 0.942) → **our arm adds 0.000 at β 0.10** (🔁); vs #12 (β 0.20) +0.003, sub-floor but in the direction that says a heavier vote of our arm costs points. **Tied best on the board** |
| 2026-09-22 | **P-27 + P-28 fork, submission #14 (`rsna-knee-fork` v5, sent 08:41, ref 56459131)**: #13 + `v09a` + `v08a` in our arm (four votes), β = 0.10 | gold-58 of the new members 0.8768 / 0.8816 | **0.941** | **−0.001 vs #13 → 🔁**: the two all-data production members add nothing on the LB (P-28 not confirmed); consistent with P-29 — both were trained 16 epochs, which over-trains by ≈ 0.012 OOF |
| 2026-09-22 | **Anchor-only control, submission #15 (`rsna-knee-fork` v6, sent 10:08, ref 56461317)**: the public 0.942 graph verbatim, our arm not run (β 0, `submission.csv` byte-identical to the anchor) | — | **0.942** | ✅ **the anchor reproduces 0.942 from our account** (unpinned sources did not drift). The baseline for #12–#14: β 0.20 −0.003, β 0.10 ±0.000, β 0.10 + P-28 members −0.001. **Tied best on the board** |
| 2026-09-22 | **`v09p` (P-29, kernel `rsna-knee-train` v21, 5.64 h)**: the `v09h` recipe on fold 0 with the production schedule's **16 epochs**, per-epoch OOF | OOF peak **0.8731 at epoch 8**; epoch 15 **0.8607**; SWA-of-13–15 proxy 0.8611 | — | ❌ **the 16-epoch schedule over-trains**: peak − epoch 15 = **0.0124** (1.6× the 0.008 floor), **11/12 labels down**; the peak (+0.0048 vs `v09h` 8-ep 0.8683) is sub-floor. P-28 members lose ≈ 0.012 → epoch budget changes (P-29 entry) |
| 2026-09-22 | **Hedge, submission #16 (`rsna-knee-fork` v7, sent 20:37, ref 56471784)**: the public 0.942 graph with its own `PRESET = "parent"` (the per-label outer CoAtNet map — LatMen 1.00, ACL / LatOA / Fracture 0.75, MedMen 0.80 — flattened to 0.60), our arm not run (β 0). Placeholder: `preset=parent … outer CoAtNet weight per finding: flat 0.60`, audit diff vs v6 = every weight 0.6, `status: anchor_control` | — | **0.940** (read 2026-09-23 08:40) | **−0.002 vs the anchor's 0.942 (#15) → 🔁 INCONCLUSIVE as a score (0.4× the 0.005 floor), and exactly the pre-registered 0.939–0.941 band**: the per-label outer map the public authors tuned on this leaderboard is worth ≈ +0.002 publicly. Whether it gives that back privately is what this hedge is for → the validated flat-weights candidate for the second final slot (brainstorm; entry 2026-09-23 "#16 … why nothing of ours moves 0.942") |
| 2026-09-22 | **P-31 smoke (`rsna-knee-train` v22, 20:21 → 20:24)**: `PARALLEL_ARMS = ("v09b", "v09c")`, FORCE_SMOKE, full 24 windows | — | — | ✅ **the second T4 is usable**: children on `cuda:0` / `cuda:1`, both `_best.pt` written, rc 0, 0.03 h wall; **2 studies × 24 windows through CoAtNet-1 @224 peak 6.84 GiB** of 15 (no grad-checkpoint needed); `aug light` and `aug none` children diverge in loss (0.6834 vs 0.7001) as they should. Real S1 run staged, not pushed (needs Tian's go) |
| 2026-09-22 | **`v09b` (P-32, `rsna-knee-train` v23, pushed 20:54 on Tian's go, child on `cuda:0`)**: the `v09h` recipe (CoAtNet-1 @224, c02, window_attn, 8 ep, `best_oof`, fold 0) with **`batch_studies=2, grad_accum=2`** — 48 windows from two studies per BatchNorm batch, the same 4 studies per optimiser step | gold 0.926 (n=11) · OOF **0.8690** (epoch 7, still climbing +0.0005/ep) | — | **🔁 INCONCLUSIVE: +0.0008 vs `v09h` 0.8683 = 0.1× the 0.008 floor, 7/12 labels up** with seed-scatter signs (ACL +0.028, MedMen +0.017 / LatOA −0.032, LatMen −0.017) → the BatchNorm batch composition is *not* the CoAtNet gap (P-32). **P-31 ✅**: 0.25 s/study = 1.00× solo, 6.84 GiB peak, rc 0, `_best.pt`; both arms in **3.22 h wall** (entry 2026-09-23 "S1 A/B") |
| 2026-09-22 | **`v09c` (P-33, same kernel v23, child on `cuda:1`)**: `v09b` + **`aug="light"`** (per-window affine rot ±8° / zoom-in 1.00–1.08 / shift ±5 %, gamma 0.8–1.25, gain ±10 %, no flips, training only) | gold 0.919 (n=11) · OOF **0.8730** (epoch 6; plateau 6–7 at 0.8730) | — | **🔁 INCONCLUSIVE: +0.0039 over `v09b` (augmentation alone, 0.5× the floor, 8/12 up; LatOA +0.020, Fracture +0.014) and +0.0047 over `v09h` (7/12 up)**. Monotone `v09h` < `v09b` < `v09c` → by the pre-registered same-direction rule **both knobs ride into the S2 production `v09a`** (P-33). 0.28 s/study = 1.12× solo (GPU-side grid_sample) |
| 2026-09-23 | **S2 smoke (`rsna-knee-train` v25, 08:57 → 09:01; v24 = an accidental duplicate, traps 36)**: `PARALLEL_ARMS = ("v09a", "v08a")`, FORCE_SMOKE, the production `v09a` now with `batch_studies=2, grad_accum=2, aug="light"` | — | — | ✅ both children rc 0 with SWA `_best.pt` (`ok  arm v09a` / `ok  arm v08a`); `v09a` banner `batch 2 x accum 2 \| aug light \| train_all, swa_last`, peak **6.84 GiB**; `v08a` `batch 1 x accum 4 \| aug none`, 1.56 GiB; 0.05 h. Local CPU smoke of the same file (MODE train sed'd, traps 30) green first |
| 2026-09-23 | **S2 production retrain — `v09a` (P-28 + P-32 + P-33: CoAtNet-1 @224, c02, window_attn, all 4,349 studies, 8 ep, SWA of the last 3 EMA, `batch_studies=2, grad_accum=2, aug="light"`) ‖ `v08a` (P-28: DINOv2-S @224, same regime, 8 ep) — `rsna-knee-train` v26, pushed 09:05 on Tian's go, one arm per T4** | gold-58 (all 58, reported only; no OOF — traps 32): **`v09a` SWA 0.8922** (last EMA 0.8934; epochs 3–7: 0.873 → 0.891 → 0.891 → 0.8925 → 0.8934, still rising) · **`v08a` SWA 0.8850** (last EMA 0.8848; 0.879 → 0.881 → 0.883 → 0.884 → 0.885) | ⏳ via the fork (#17) → read 2026-09-23: #17 **0.941** 🔁 | ✅ **the regime ran end to end on both T4s in 2.64 h wall** (`ok  arm` ×2, `_best.pt = SWA`; `v09a` 0.26 s/study, 19 min/epoch, 6.84 GiB; `v08a` 0.12 s/study, 8.8 min/epoch). vs the 16-ep members' SWA (0.8768 / 0.8816): **+0.015 / +0.003 on gold-58 — direction only** (floor 0.05), but the curves no longer peak-and-drift (P-29 confirmed on the production data: at 8 epochs both are still climbing at epoch 7). Shipped as new versions of `rsna-knee-ckpt-v09a` / `-v08a` (11:44, `datasets status` ready); fork v8 built (β 0.10, four members) → #17 (entry 2026-09-23 "S2 production retrain") |
| 2026-09-23 | **P-27 + P-28 S2, submission #17 (`rsna-knee-fork` v8, sent 11:54, ref 56489906)**: #14's graph and blend with the S2 production members — `v08w` + 5-fold `v09h` + `v09a` (8 ep, batch 2, aug light) + `v08a` (8 ep), one vote each — at β 0.10 | gold-58 of the new members 0.8922 / 0.8850 | **0.941** (read 20:00) | 🔁 **INCONCLUSIVE: −0.001 vs #13 / #15 (0.942), identical to #14 (0.941, the 16-epoch members) — the 8-epoch S2 members (+0.015 gold-58; `v09a` 0.918 solo as #18) change the fork's public read by nothing; the fork at β 0.10 is not an instrument for member quality, the solo submission is (entry "Submission #17 read 0.941")**. Pre-registered read vs #13 / #15 (0.942): **≥ 0.947 ✅ our arm counts; 0.940–0.946 🔁; ≤ 0.939 the arm hurts**; vs #14 (0.941) = the epoch budget + the S1 knobs. Honest expectation 0.942 ± 0.001 — the member-quality wall (entry "#16 … why nothing of ours moves 0.942") |
| 2026-09-23 | **Round 2 — `v09d` (P-34: the `v09c` recipe with `lr_backbone=3e-5`) ‖ `v08c` (P-35: `v08w` + `aug="light"`) — `rsna-knee-folds` v8, fold 0, 8 ep, `best_oof`, one arm per T4, 3.24 h wall** | `v09d` gold 0.924 (n=11) · OOF **0.8596** · `v08c` gold 0.924 · OOF **0.8650** | — | **P-34 ❌ HARMFUL**: −0.0134 vs `v09c` 0.8730, 10/12 labels down (Lateral Meniscus −0.059, MCL −0.028) — the hybrid under-trains at 3e-5 in 8 epochs (still rising +0.0008/epoch at 7); **`v09e` (P-36, 16 ep × 3e-5) dropped by the pre-registered rule**. **P-35 🔁 INCONCLUSIVE**: +0.0002 vs `v08w` 0.8648, 6/12 up — light augmentation is worth nothing on the DINOv2 arm (entry 2026-09-23 "Round 2") |
| 2026-09-23 | **`v09f` (P-37: the `v09c` recipe + `pos_weight_max=10`, per-label `clip((1−p)/p, 1, 10)`) — RunPod RTX 4090 (secure, EU-CZ-1), fold 0, 8 ep, 33 min** | gold 0.910 (n=11) · OOF **0.8717** | — | 🔁 INCONCLUSIVE: −0.0013 vs `v09c` 0.8730, 3/12 up — `pos_weight` is not a training-dynamics lever either; stays 0 (entry "RunPod arms") |
| 2026-09-23 | **`v09s` (P-38: the `v09c` recipe on `TEACHER_TABLES=("selfdistill_v1",)`, mix 0.5 — self-distillation from the rank-mean of the `v09h` + `v05g` 5-fold OOF sets, quantile-matched onto the LLM blend; evaluation targets unchanged) — RunPod 4090, fold 0, 8 ep, 34 min** | gold 0.922 (n=11) · OOF **0.8839** | — | ❌ **superseded 2026-09-24 (traps 39: OOF vs the LLM targets rewards agreement with the teacher; the production twin `v09t` read 0.917 vs 0.918 on the LB, #19)** — was: ✅ **KEEP: +0.0109 vs `v09c` 0.8730 (1.4× the 0.008 floor), 12/12 labels up** (MCL +0.024, Medial OA +0.015, Baker's +0.015) — the first recipe change since `v09h` that clears the floor; leads `v09c`/`v09f` from epoch 1 (0.836 vs 0.807). Caveat: the table's fold-1..4 models saw fold 0's *targets* (second order); the LB solo read of a distilled production member is the arbiter (entry "RunPod arms") |
| 2026-09-23 | **Submission #18 — the S2 `v09a` ALONE (`rsna-knee-infer` v15, `INFER_MEMBERS=["v09a"]`; member-strength plan Task 11)** | gold-58 0.8922 (all-data SWA) | **0.918** | ✅ FINDING: one production member of ours reads 0.918 solo — above our whole 12-member blend (#11, 0.913) and inside the public stack's member range (0.90–0.928); the fold-0→LB offset (+0.02–0.03) under-predicted the all-data SWA member by ≈ 0.02. **Baseline for the Raptor-distilled retrain (P-39): ≥ 0.923 ✅ / ≤ 0.922 🔁 / < 0.918 ❌** (entry "Solo baseline #18") |
| 2026-09-23 | **Raptor teacher pass (P-39) — `rsna-knee-teacher` v2 smoke (LIMIT 6) and v3 spike (LIMIT 100)**: the 0.942 notebook's Raptor branch verbatim over a chunk of our training studies via `RSNA_COMP_ROOT`, both T4s | — | — | ✅ FEASIBLE: v1 red (the preamble looked for `train_images`; the tree is `train_series` — traps 37) → v2 green (6/6, 62 s) → v3 **100/100 studies, 0 failed, 5.11 s/study incl. setup** → full pass 4,349 × 5.1 s ≈ **6.2 GPU-h** (one session under the 8 h guard, or 2 shards × 3.1 h in the two slots). Plausibility on the 100: Raptor vs the hard LLM teacher macro AUC 0.914; operating point more positive (Fracture mean 0.47 vs 0.15) — quantile matching handles it (entry "Raptor teacher pass") |
| 2026-09-23 | **`v09t` (P-38 in production): the S2 `v09a` recipe (CoAtNet-1 @224, c02, window_attn, all 4,349 studies, 8 ep, SWA 5–7, `batch_studies=2, grad_accum=2, aug="light"`) trained on `TEACHER_TABLES=("selfdistill_v1",)`, mix 0.5 — its own version name; RunPod RTX 4090, 35 min (4.4 min/epoch)** | gold-58 (all 58, reported only): **SWA 0.9009** (CI95 0.867–0.929; last EMA 0.9016; epochs 0–7: 0.798 · 0.866 · 0.886 · 0.893 · 0.898 · 0.901 · 0.901 · 0.902) vs `v09a` 0.8922 → +0.0087, direction only | **0.917** (#19, read 2026-09-24 00:07) | **❌ by the pre-registered rule (< 0.918): −0.001 vs #18 (the same recipe on the LLM targets), 0.2× the 0.005 floor — self-distillation does not transfer to the production member; the fold-0 gain (`v09s` +0.011 OOF) was agreement with the LLM teacher, not truth (entry "Submission #19")**. ✅ the run: `train 4349 / val 58 studies`, `-> v09t_fold0_best.pt = SWA`; Kaggle smoke v28 green first (4 min); shipped as Dataset `rsna-knee-ckpt-v09t`; **solo submission #19 (`rsna-knee-infer` v16, ref 56504077, 23:25) — read vs #18 (0.918): ≥ 0.923 ✅ / 0.919–0.922 🔁 / < 0.918 ❌** (entry "`v09t`") |
| 2026-09-24 | **Raptor teacher pass, the full run (P-39) — shards 0/3 and 1/3 (`rsna-knee-teacher` v4, `rsna-knee-teacher-b` v1, pushed 14:54, one per GPU slot)**: the 0.942 notebook's Raptor branch verbatim over 2 × 1,450 gold-free training studies, raw per-view probabilities, partial flush every 5 min | — | — | ✅ **both green (read 17:12 / 17:25): 1,450 + 1,450 studies, 0 failed, 5.6 and 6.2 s/study (2.24 h and 2.50 h — the slower session ≈ 20 % over the spike's 5.1 s), k_eval 94; disjoint UIDs; merged 2,900 rows read macro AUC 0.906 vs the hard LLM teacher** (`src/teacher_plausibility.py`; spike 0.914; lowest MCL 0.844 / Effusion 0.845 / Synovitis 0.847, highest Baker's 0.950 / Fracture 0.949 / LatMen 0.947) — entry "Raptor pass shards 0–1". Shard 2/3 (1,449 studies, ≈ 2.5 h) after Saturday's reset; quota this week ≈ 1.5 h left. Re-sharded 2 → 3 to fit the week's last 6.3 h of quota (≈ 4.3 h used, ≈ 2 h margin — a quota kill loses nearly every row because a row counts only with all four views). RunPod cannot host the pass (it reads the DICOMs; hard constraint 2) |
| 2026-09-26 | **Raptor teacher pass — shard 2/3 (`rsna-knee-teacher` v5, pushed 17:26)**: the v4 render with `SHARD = 2` (the only diff; `teacher_pass_test.py` green), the last 1,449 gold-free studies; merge target `--expect-n 4349` | — | — | ✅ **green (COMPLETE 23:39): 1,449 studies, 0 failed, 7.50 s/study (3.02 h — the slowest of the three sessions), k_eval 94** after **3 h 10 min QUEUED** (17:26 → 20:35; Kaggle T4 capacity — a probe kernel queued too, `kaggle quota` 0.00 / 30 h; traps 41). **The pass is complete: `merge_teacher.py` s0+s1+s2 `--expect-n 4349` → `artifacts/teacher/raptor_teacher.csv`, 4,349 rows, 0 gold; Raptor vs the hard LLM teacher macro AUC 0.9075** (2,900 rows: 0.906; spike 0.914); published as a new version of Dataset `rsna-knee-teacher-tables` (23:40, beside `selfdistill_v1.csv`) — entry "Raptor pass complete" |
| 2026-09-26 | **`v09r` (P-39 / Task 12): the S2 `v09a` recipe (CoAtNet-1 @224, c02, window_attn, all 4,349 studies, 8 ep, SWA 5–7, `batch_studies=2, grad_accum=2, aug="light"`) trained on `TEACHER_TABLES=("raptor_teacher",)`, mix 0.5 — own version name; RunPod RTX 4090 (EUR-IS-1), 5.9 min/epoch, job 23:33 → 00:30** | gold-58 (all 58, reported only): **SWA 0.9093** (CI95 0.879–0.936; epochs 0–7 EMA: 0.770 · 0.860 · 0.887 · 0.898 · 0.904 · 0.910 · 0.909 · 0.908) vs `v09a` 0.8922 / `v09t` 0.9009 → +0.017 / +0.008, direction only; 9/12 labels up vs `v09a` (Synovitis 0.744 → 0.823, Fracture 0.897 → 0.943) | **0.927** (#20, `rsna-knee-infer` v17, read 2026-09-27 09:28) | **✅ KEEP — +0.009 vs #18 0.918 (1.8× the 0.005 floor; above the pre-registered 0.923 line): the first target-source change that transfers to the LB; our best single member, ≈ the best public member (0.928)** (entry "Submission #20") |
| 2026-09-27 | **`v08r` (P-40 step B): the S2 `v08a` recipe (DINOv2-S @224, c02, window_attn, all 4,349 studies, 8 ep, SWA 5–7, batch 1 × accum 4, aug none) on `TEACHER_TABLES=("raptor_teacher",)`, mix 0.5 — RunPod A100 SXM 80 GB (US-MD-1), 3.2 min/epoch, job 12:52 → 13:23 UTC** | gold-58 (all 58, reported only): **SWA 0.8981** (CI95 0.865–0.926; epochs 0–7 EMA 0.768 · 0.839 · 0.873 · 0.889 · 0.897 · 0.898 · 0.898 · 0.899) vs `v08a` 0.8850 → +0.013, direction only | **0.918** (#21, solo, `rsna-knee-infer` v19, ref 56610108, read 2026-09-27 16:24 UTC) | ✅ the run: shipped as Dataset `rsna-knee-ckpt-v08r` (ready); pod ≈ 36 min ≈ $0.95, deleted. 🔁 on gold (floor 0.05). **LB: −0.009 vs `v09r` 0.927 (same targets, CoAtNet-1; 1.8× the floor) and 0.002 under P-40's ≈ 0.920 bar for a second fork member → does not join the fork on solo strength (the pre-registered branch does not fire); = `v09a` 0.918 (#18). No `v08a` solo exists, so the Raptor gain on DINOv2 is unmeasured** (entry "Submission #21") |
| 2026-09-27 | **P-42 solo blend, submission #23 (`rsna-knee-infer` v20)**: flat rank-mean of the two Raptor-distilled families `v09r` (CoAtNet-1, 0.927 solo) + `v08r` (DINOv2-S, 0.918 solo), one vote each | gold-58 (hand-computed, direction only): rank-mean 0.9085 vs `v09r` 0.9093; ρ 0.924 | **0.927** (read 17:05 UTC) | **🔁 INCONCLUSIVE: ±0.000 vs `v09r` alone (the pre-registered 0.923–0.931 band)** — the second family neither lifts nor dilutes; `v08r` stays out of the fork. **P-41 ✅: scored 28.3–29.3 min after sending** (two members, 8 decode workers) vs #19's ≤ 42 min (entry "Submission #23") |
| 2026-09-27 | **P-40 step A, submission #22 (`rsna-knee-fork` v9)**: the public 0.942 graph + `v09r` alone at β 0.10 | gold-58 of `v09r` 0.9093 | **0.942** (first seen 2026-09-28 09:12 UTC) | **🔁 INCONCLUSIVE: ±0.000 vs #13 / #15 (0.942), inside the pre-registered 0.940–0.946 band** — a 0.927 member at β 0.10 moves the fork no more than #17's 0.918-level members (0.941) did; the fork does not read member quality at this weight (entry "Submission #22") |
| 2026-09-27 | **P-43 + P-44 — `v09x` (the `v09r` recipe at **320 px**, batch 1 × accum 4) ‖ `v09u` (`v09r` exactly, **seed 43** — the first honoured per-arm seed, traps 42), one `PARALLEL_ARMS` session on `rsna-knee-train`, both on `TEACHER_TABLES=("raptor_teacher",)` mix 0.5** — smoke v29 (17:59 → ≈ 18:05 UTC) green: both children `teacher table raptor_teacher: 4349 studies` (the first Kaggle training kernel to read it), `v09x` img 320 on cuda:0 **peak 7.03 GiB** (batch 1 × 24 windows), `v09u` `reseeded 43` on cuda:1 6.84 GiB, `ok  arm` ×2; **real run v30 pushed 18:08:24 UTC, RUNNING 18:08:48** (no queue); **COMPLETE in 5.88 h** (both `ok  arm`, SWA, no guard stop; `v09x` 0.59 s/study ≈ 43 min/epoch, `v09u` 0.27 s/study ≈ 19.5 min/epoch) | gold-58 SWA (direction only): **`v09x` 0.9094** (last EMA 0.9108), **`v09u` 0.8995** (last EMA 0.8984) vs `v09r` 0.9093; rank-mean `v09r` + `v09u` 0.9065, `v09r` + `v09x` 0.9126 (entry "rsna-knee-train v30") | #24 `v09u` **0.927** · #25 `v09x` **0.929** · #26 `v09r` + `v09u` **0.930** (rows below; shipped as `rsna-knee-ckpt-v09x` / `-v09u`) | read 2026-09-28 — verdicts in the three rows below; pre-registered in P-43 / P-44: `v09u` s = \|`v09u` − 0.927\| (≤ 0.920 → re-open P-39); `v09x` vs m = mean(0.927, `v09u`): ✅ ≥ m + max(0.005, 2s) / ❌ ≤ m − max(0.005, 2s); `v09r` + `v09u` blend ≥ 0.930 ✅ (production member either way). Est. 5.3–6.1 h for `v09x` (break-even 0.80 s/study under the ≈ 8.0 h child guard), ≈ 2.8 h for `v09u`; ≈ 6.2 GPU-h |
| 2026-09-28 | **P-44, submission #24 (`rsna-knee-infer` v21)**: `v09u` solo — the `v09r` recipe exactly, seed 43, Kaggle T4 | gold-58 SWA 0.8995 (−0.010 vs `v09r`) | **0.927** (read 09:48 UTC) | **✅ KEEP (a measurement): s = \|`v09u` − 0.927\| = 0.000 → by the card's rule one-seed deltas need ≥ 0.004 from now on** (one draw; LB rounded to 3 decimals). P-39 re-confirmed on the same platform as #18 (mean 0.927 − 0.918 = 0.009 ≥ 0.006) — the +0.009 was the targets, not RunPod. Gold-58's −0.010 did not transfer (entry "Submissions #24–#26") |
| 2026-09-28 | **P-43, submission #25 (`rsna-knee-infer` v22)**: `v09x` solo — the `v09r` recipe at 320 px | gold-58 SWA 0.9094 | **0.929** (read 09:59 UTC) | **🔁 INCONCLUSIVE: +0.002 vs m = mean(0.927, 0.927); ✅ needed ≥ 0.932 (m + max(0.005, 2s)), ❌ ≤ 0.922.** Our best single member, not by a readable margin; 1.4× the inference time of a 224 member (136 vs 98 s / 100 studies) |
| 2026-09-28 | **P-44, submission #26 (`rsna-knee-infer` v23)**: `v09r` + `v09u` flat rank-mean (seed 42 RunPod + seed 43 Kaggle, one recipe) | gold-58 rank-mean 0.9065 (−0.003 vs `v09r`); ρ 0.952 | **0.930** (read 10:01 UTC) — **our best solo** | **✅ KEEP as the production member — the pre-registered line (≥ 0.930) met exactly; 🔁 as evidence that seed averaging is a lever: +0.003 vs either member is under the 0.004 floor #24 sets.** Gold-58 had the sign wrong (−0.003) |
| 2026-09-28 | **P-40 β 0.20 retry, submission #27 (`rsna-knee-fork` v10)**: the public 0.942 graph + `v09r` alone at β 0.20 | gold-58 of `v09r` 0.9093 | **0.941** | **🔁 INCONCLUSIVE: −0.001 vs #13 / #15 (0.942), inside 0.940–0.946** — neither β 0.10 (#22 0.942) nor β 0.20 registers a 0.927 member; +0.002 over #12 (β 0.20, 0.87-level members) is sub-floor. **P-40 closes: the fork is for final-selection builds only (P-50)** (entry "Submissions #27–#28") |
| 2026-09-28 | **P-53 per-label probe, submission #28 (`rsna-knee-infer` v24)**: the #26 pair with ACL / MCL / PF OA / Lat Men = constant 0.5 — a diagnostic, never a candidate | gold-58: those 4 labels 0.886 vs the other 8 0.914 (Δ +0.028) | **0.785** | **✅ KEEP (a measurement): on the public test the 4 "structural" labels average 0.935 ± 0.003 and the other 8 0.9275 → Δ −0.0075: the gold-58 deficit is gold-specific (pre-registered: mean4 ≥ mean8 − 0.01).** The c03 input (P-56) keeps running on a lower prior (entry "Submissions #27–#28") |
| 2026-09-28 | **P-49 Raptor over the 58 gold studies (`rsna-knee-teacher-gold` v1, 58/58, 4.7 s/study, 0 failed)** | **Raptor 0.9254** (optimistic: its authors picked epochs on gold) · LLM blend 0.8948 · **the 0.5/0.5 training target 0.9268 (12/12 labels above the LLM; paired bootstrap +0.032, SD 0.008)**; within-class ρ `v09r` ~ Raptor **0.835** vs `v09r` ~ LLM 0.408 | — | **🔁 direction only (a correlation read, never a verdict):** our member is closer to Raptor than to anything else we have measured except its own seed twin (0.916) → the fork cannot read it (P-40 ✅ closed); matched-mix analogs on Raptor itself: 0.25 → 0.9187, **0.5 → 0.9268**, 0.75 → 0.9252 (−0.002, SD 0.003), 1.0 → 0.9098 (−0.017, SD 0.009) → mix 0.5 stays, P-47 priced at ≈ 0; the student trails its own target by 0.018 on gold (entry "P-49") |
| 2026-09-28 | **P-54 cross-fit, sessions A ‖ B (`rsna-knee-train` v32 ‖ `rsna-knee-folds` v9)**: the `v09r` recipe with `train_all` False on folds 0–3 (`v09k0` … `v09k3`), SWA 5–7, held-out eval once (`eval_final_only`), Raptor mix 0.5 | per-fold gold (11–12 studies each, noise): 0.9272 / 0.9136 / 0.8582 / 0.8883; OOF-vs-LLM (`auc_soft`, 881–882 each): 0.8816 / 0.8728 / 0.8707 / 0.8679 | — | ✅ **the runs** (`ok  arm` ×4, `fold k: train 3525–3526 / val 881–882`, `SWA of last 3`, no guard; 2.29 h / 2.26 h wall; checkpoints 157 MiB) — ⏳ fold 4 (`v09k4`, session C = `rsna-knee-train` v33) before the table (→ done 2026-09-29, row "Sessions C ‖ D"; table `xfit_v09k`, row "P-54 table"); the OOF-vs-LLM numbers are not a verdict on anything (traps 39) (entry "P-54 sessions A ‖ B") |
| 2026-09-29 | **Sessions C (re-push, `rsna-knee-train` v34, 2.39 h) ‖ D (`rsna-knee-folds` v10, 3.53 h)**: `v09k4` (P-54 fold 4) + `v13a` (P-57, ResNet-34 on the `v09r` recipe) ‖ `v11a` / `v11b` (P-56, the `v09r` recipe on the c03 input 24/24/24/14/8/8 at 150 mm, seeds 42 / 43) | gold-58 SWA: **`v11a` 0.9204, `v11b` 0.9167** (c03 pair 0.9207 vs c02 pair 0.9065: +0.014, SD 0.005, 10/12 up); **`v13a` 0.8306** (−0.079 vs `v09r`); `v09k4` OOF-vs-LLM 0.8811 | #30–#32 (rows below) | ✅ runs · 🔁 c03 direction only · **❌ `v13a` on gold-58** (beyond the 0.05 floor, 8/12 down) (entry "Sessions C ‖ D") |
| 2026-09-29 | **P-54 table `xfit_v09k`** (5-fold cross-fit OOF, ranked within each fold, 4,407 rows) → P-55 loose gate | gold-58 pooled **0.9028** vs LLM 0.8948 (+0.008, SD 0.015), 4/12 below the LLM | — | 🔁 direction only; **loose gate OPEN → session E (`rsna-knee-train` v35, `v09o` ‖ `v09o2`, mix 0.75 over Raptor + xfit) pushed 11:37 UTC** (entry "P-54 cross-fit table") |
| 2026-09-29 | **P-52, submission #29 (`rsna-knee-infer` v25)**: `v09r` + `v09u` + `v09x` flat rank-mean | gold-58 0.9110 | **0.931** | **🔁 INCONCLUSIVE: +0.001 vs #26 0.930 (band 0.931–0.933)** — the pair stays (cheaper); scored in ≈ 54 min (entry "Submissions #29–#32") |
| 2026-09-29 | **P-56, submissions #30 / #31 (`rsna-knee-infer` v26 / v27)**: `v11a` / `v11b` solo — the `v09r` recipe on the c03 input (24/24/24/14/8/8, 150 mm), seeds 42 / 43 | gold-58 0.9204 / 0.9167 | **0.932 / 0.929** | **🔁 INCONCLUSIVE: m = 0.9305 vs the ✅ bar 0.9315** (+0.0035 over the c02 twins 0.927 / 0.927); `v11a` = our best single member |
| 2026-09-29 | **P-56, submission #32 (`rsna-knee-infer` v28)**: the c03 pair `v11a` + `v11b` | gold-58 0.9207 (c02 pair 0.9065) | **0.932** | **🔁 INCONCLUSIVE: +0.002 vs the c02 pair #26 0.930** — ties #30 as the best solo read; every c03 reading points the same way, none clears its floor |
| 2026-09-29 | **P-54, submission #33 (`rsna-knee-infer` v29)**: the 5-fold cross-fit ensemble `v09k0` … `v09k4` (flat rank-mean, one vote per fold model) | — (OOF only vs the LLM, traps 39) | **0.928** | **❌ DEAD END as a member: −0.002 vs #26 0.930 (band ≤ 0.930)** — 5 × 80 %-data models ≈ one all-data member; the table still feeds P-55; scored in ≈ 66 min |
| 2026-09-29 | **P-55 session E (`rsna-knee-train` v35, 2.94 h)**: the OOF soft-bootstrapped student `v09o` ‖ `v09o2` — the `v09r` recipe on 0.25 LLM + 0.375 Raptor + 0.375 `xfit_v09k` (`TEACHER_MIX = 0.75` over two matched tables), all 4,349, seeds 42 / 43 | gold-58 SWA **`v09o` 0.9121, `v09o2` 0.9104**; student pair 0.9119 vs the c02 pair 0.9065 (+0.006, SD 0.004, 9/12 up) and the c03 pair 0.9207; seed-twin ρ 0.946 | ⏳ three solos after 00:00 UTC → read 2026-09-30: #34 / #35 / #36 **0.927 / 0.927 / 0.928** ❌ (row "P-55, submissions #34 / #35 / #36") — placeholders green: `rsna-knee-infer` **v30 = `v09o`, v31 = `v09o2`, v32 = the pair** (`smoke False`, `infer members` as named, decode-once verified, `constant labels 0`) | ✅ run · 🔁 direction only; shipped as `rsna-knee-ckpt-v09o` / `-v09o2` (entry "Session E") |
| 2026-09-29 | **P-59 ResNet-34 control (`rsna-knee-train` v37, 1.08 h)**: `v13b` = `v13a` (ResNet-34, c02, Raptor 0.5) + uniform `lr_backbone` 3e-4 (`llrd_decay` 1.0) + 12 epochs ‖ `v13c` = `v13b` + frozen encoder BatchNorm | gold-58 SWA **`v13b` 0.8992 / `v13c` 0.9014** vs `v13a` 0.8306 (+0.069, beyond the 0.05 floor; train loss 0.471 → 0.391); ρ to `v09r` 0.86 | — (shipped `rsna-knee-ckpt-v13b` / `-v13c`) | **✅ the optimiser was the ResNet's defect (LLRD 0.75 on 1e-4 = ViT rates); 🔁 frozen BN (+0.002)**; gold blends flat (entry "P-59") |
| 2026-09-30 | **P-55, submissions #34 / #35 / #36 (`rsna-knee-infer` v30 / v31 / v32)**: the OOF soft-bootstrapped student `v09o` / `v09o2` solo and their pair | gold-58 0.9121 / 0.9104; pair 0.9119 (c02 pair 0.9065) | **0.927 / 0.927 / 0.928** | **❌ DEAD END: m = 0.927 < 0.9285 (= `v09r` / `v09u` exactly); the pair −0.002 vs the c02 pair #26 0.930** — the third null for OOF-derived targets (P-38, Nicolai); the OOF-target line closes; gold-58's +0.007 did not transfer (traps 39) (entry "Submissions #34–#38") |
| 2026-09-30 | **P-59, submission #37 (`rsna-knee-infer` v33)**: ResNet-34 `v13c` (CNN LR 3e-4 uniform, 12 ep, frozen BN) solo | gold-58 0.9014 | **0.921** | **❌ as a member: −0.006 vs 0.927 (band ≤ 0.922)** — the small-CNN route closes for accuracy; an Efficiency-track candidate only (0.12 s/study, scored in ≈ 16 min) |
| 2026-09-30 | **P-59, submission #38 (`rsna-knee-infer` v34)**: the c03 pair `v11a` + `v11b` + `v13c` (flat rank-mean, two decode passes) | gold-58 0.9189 (c03 pair 0.9207) | **0.932** | **❌ DEAD END: = the c03 pair #32 0.932 (band ≤ 0.932)** — a 0.921 third family at within-class ρ 0.86 neither adds nor subtracts |
| 2026-09-30 | **P-60 part 1 (`rsna-knee-folds` v11, 2.61 h)**: `v11n` ‖ `v11n2` = the `v11a` recipe (c03, Raptor 0.5, all 4,349) + `drop_path` 0.1 + `aug` "heavy" + 12 epochs (SWA 9–11), seeds 42 / 43; runtime guard 2.75 h by design (quota) | gold-58 EMA at epoch 4: **`v11n` 0.9166 / `v11n2` 0.9129** vs `v11a` / `v11b` at their epoch 4 0.9156 / 0.9129 (pair +0.0005); guard stop in epoch 5 (0.9151 / 0.9148) | — (part 2 after the 2026-10-03 reset) | **✅ run (guard-stopped as planned, `_last.pt` ×2 for the resume) · ⏳ P-60** (→ read 2026-10-03: #39 / #40 **0.932 / 0.932** 🔁, row "P-60 part 2") — direction only, the regularised arms caught up with the 8-epoch curve at a higher LR; the epoch-5 `_best.pt` files are not members (traps 47) (entry "P-60 part 1") |
| 2026-09-30 | **Silence-aware teacher mix, priced on gold-58 at target level (0 GPU)**: Raptor weight 0.75 on pilkwang-`UNK` cells, 0.5 on addressed cells | target gold **0.9300 vs flat 0.5 0.9268** (+0.0032, SD 0.0018, 5 / 1 labels); flat mixes at the same mean Raptor weight +0.0004 | — | **🔁 direction only** — the gain is where Raptor speaks, not how much; Raptor is optimistic on gold and the student shrinks target gains → card P-62 (entry "Silence-aware teacher mix") |
| 2026-10-03 | **P-45 step 1 + spike + pass (0.1 + 2.37 GPU-h)**: D4 (public CoAtNet-2 @384) as a second image teacher — gold analog 0.5 LLM + 0.25 Raptor + 0.25 D4 **0.9331 vs 0.9268**; spike reproduces D4's gold reference (0.9301 vs 0.9302); full pass 4,349/4,349, in-sample check ρ D4~LLM 0.695 (Raptor 0.652) | target analog only | — | **🔁 direction only → the student pair `v11d` ‖ `v11dl` runs (session B, `rsna-knee-train` v41)** |
| 2026-10-03 | **P-60 part 2 (`rsna-knee-train` v39, 2.84 h)**: `v11n` ‖ `v11n2` resumed at epoch 6, SWA 9–11 | gold-58 SWA **0.9152 / 0.9166** (`v11a` / `v11b` 0.9204 / 0.9167) | **0.932 / 0.932** (#39 / #40) | **🔁 INCONCLUSIVE: m = 0.932 vs 0.9305 (+0.0015, band 0.926–0.935)** — heavy regularisation + 12 epochs = the plain recipe on the LB; seeds agree (0.932 / 0.932 vs 0.932 / 0.929); 1.5× the training time |
| 2026-10-03 | **Sessions A (`rsna-knee-train-b` v4, 4.40 h) ‖ B (`rsna-knee-train` v41, 3.82 h)**, c03, one seed each: `v11p` = `v11a` + P-63 spatial reader + slot-count norm · `v13h` = P-64 ResNet-34, heavy aug, drop-path 0.1, 30 ep · `v11d` = `v11a` recipe on 0.5 LLM + 0.25 Raptor + 0.25 D4 (P-45) · `v11dl` = `v11d` + lr 2e-4 / LLRD 0.85 (P-61) | gold-58 SWA **0.9185 / 0.9001 / 0.9184 / 0.9132** (`v11a` 0.9204, `v13c` 0.9014) | **0.929 / 0.931 / 0.930** / ⏳ (#41 / #42 / #43; `v11dl` 10-04 → read: #44 **0.927** 🔁, P-61 closed) | **✅ `v13h` KEEP: +0.010 vs `v13c` 0.921 (the CNN recipe transfers; gold said −0.001) · 🔁 `v11p` −0.003 and `v11d` −0.002 vs `v11a` 0.932, not adopted** (entries "Sessions A ‖ B", "Submissions #41–#43") |
| 2026-10-04 | **Session C (`rsna-knee-train` v43, 5.87 h)**: the `v13h` recipe (c03, CNN LR 3e-4 uniform, frozen BN, heavy aug, drop-path 0.1, 30 ep, Raptor 0.5) on ResNet-50 `v13r` ‖ EfficientNet-B0 `v13e` | gold-58 SWA **0.9111 / 0.9126** (`v13h` 0.9001) | **0.934 / 0.935** (#45 / #46) | **✅ `v13e` KEEP: 0.935 = our best solo (+0.004 vs `v13h`); 🔁 `v13r` +0.003** — blend `v11a` + `v13h` #47 0.934 (🔁 +0.002 over its best member); `v11dl` #44 0.927 (P-61 closed) (entries "Session C", "Submissions #44–#47") |
| 2026-10-04 | **Cross-family blends (flat rank-mean, `rsna-knee-infer` v43 / v44)**: `v11a` + `v13h` (#47) · `v11a` + `v13r` + `v13e` (#48) | gold-58 0.9170 / 0.9233 | **0.934 / 0.938** | **🔁 by rule (+0.002 / +0.003 over the best member, bars 0.936 / 0.939) — but both above every member, unlike four flat same-family blends; 0.938 = our best own-model score** (entry "Submission #48") |
| 2026-10-04 | **P-66 `v13b3` (RunPod RTX 4090, 2.2 h, ≈ $1.8)**: the `v13h` recipe on EfficientNet-B3 @ 288 (c03, Raptor 0.5, 30 ep, SWA 27–29) | gold-58 SWA **0.9222** (`v13e` 0.9126; 6 up / 5 down; menisci, ACL, Fracture up) | **0.940** (#50, 10-05) | **✅ KEEP: +0.005 vs `v13e` (bar 0.004), +0.0035 vs the B0 seed mean 0.9365; our best solo** (entry "Submissions #49–#53"). Was: ✅ run green; 🔁 direction only (+0.0096, floor 0.05) — read vs `v13e` 0.935: ✅ ≥ 0.939 / 🔁 0.931–0.938 / ❌ ≤ 0.930; seed twin `v13e2` training on the same pod (entry "P-66 on RunPod") |
| 2026-10-04 | **P-66 `v13e2` (RunPod RTX 4090, 65 min)**: `v13e` exactly at seed 43, the first CNN seed twin | gold-58 SWA **0.9151** (`v13e` 0.9126; within-class ρ 0.888, the same as B3 ~ B0) | **0.938** (#51, 10-05) | **✅ s = 0.003 → the CNN one-seed bands stand; our best solo, = #48** (entry "Submissions #49–#53"). Was: ✅ run green; 🔁 seed-level gold difference (+0.0025) — read s = \|`v13e2` − 0.935\|: ≤ 0.003 the CNN bands stand / ≥ 0.005 widen them; a final-ensemble member either way (entry "P-66 complete") |
| 2026-10-04 | **Session D (`rsna-knee-train-b` v6, 5.94 h), P-62 on the CNNs**: `v13es` / `v13rs` = `v13e` / `v13r` + Raptor 0.75 on report-silent cells | gold-58 SWA **0.9107 / 0.9160** (flat 0.9126 / 0.9111; pair mean +0.0015; ρ to the flat parents 0.932 / 0.948, closer than a seed twin's 0.888) | ⏳ (solos **10-07**, `rsna-knee-infer` v49 / v50; moved from 10-05 to 10-06, then 10-07 by Tian) | **✅ runs green; 🔁 direction only** — read m(`v13es`, `v13rs`) vs 0.9345: ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299 (entry "Session D") |
| 2026-10-05 | **B3 swap, submission #53 (`rsna-knee-infer` v52)**: flat rank-mean `v11a` + `v13r` + `v13b3` | gold-58 0.9254 | **0.940** | **🔁 +0.002 vs #48; = `v13b3` alone.** Every flat blend we have sent reads ≈ the members' mean + 0.002–0.005 (n = 7), so weak members now cost more than they add (entry "Submissions #49–#53") |
| 2026-10-05 | **B6, submission #52 (`rsna-knee-infer` v51)**: flat rank-mean of all five, `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` | gold-58 0.9256 | **0.942** | **✅ KEEP: +0.004 vs #48 (bar ≥ 0.941). Our own models = the public-stack fork (0.942); the own final-pick candidate (P-50).** Corrected blend rule: LB ≈ the members' mean + g(n), g(2) ≈ 0.0015–0.0045, g(3) ≈ 0.0045, g(5) ≈ 0.006 (entry "Submissions #49–#53") |
| 2026-10-05 | **P-50 fork, submission #49 (`rsna-knee-fork` v11, sent 00:41, ref 56838006)**: the public 0.942 stack + the #48 trio (`v11a` + `v13r` + `v13e`) as our leg at β 0.45 | — | **0.943** | **🔁 +0.001 vs #13 / #15 (band 0.941–0.944); our best public number: rank 373 of 5,187, up from ≈ 1,381 (the 0.942 plateau is ≈ 1,000 forks wide).** The first own leg to lift the stack (β 0.10–0.20 legs read 0.000 / −0.001); 5.3 h to score. Next: C2 = the stack + B6 (entry "Submissions #49–#53") |
| 2026-10-05 | **Session E chain 1, `v13ecp` (RunPod RTX 4090, 71 min, P-65)**: `v13e` exactly (B0, seed 42) on **0.5 Raptor + 0.5 Claude**, no LLM-blend share (`TEACHER_MIX` 1.0 over `raptor_teacher` + `claude_v1`) | gold-58 SWA **0.9063** (B0 seed pair 0.9126 / 0.9151; Baker's −0.046, Effusion −0.032 and MCL −0.041 down, Lateral Meniscus +0.029 and Lateral OA +0.020 up; within-class ρ to `v13e` 0.883, the seed pair's 0.888) | **0.935** (#57, 10-06) | **🔁 −0.0015 vs the B0 seed mean 0.9365 (band 0.933–0.940); = `v13e` at the same seed** (entry "Submissions #54–#58"). Was: ✅ run green, shipped; 🔁 direction only — read vs the B0 seed mean 0.9365: ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932 (entry "Session E on RunPod, chain 1") |
| 2026-10-05 | **Session E chain 2, `v13ec` (RunPod RTX 4090, 71 min, P-65)**: `v13e` exactly (B0, seed 42) on **0.25 LLM + 0.5 Raptor + 0.25 Claude** (`claude_rap_v1` at mix 0.75, the registered design) | gold-58 SWA **0.9116** (B0 seed pair 0.9126 / 0.9151; Baker's −0.037, MCL −0.032, Contusion −0.033 down at this dose too; Effusion back to −0.004; within-class ρ to `v13e` 0.886) | **0.932** (#58, 10-06) | **❌ by the pre-registered rule (≤ 0.932): −0.0045 vs the B0 seed mean. With #57, no lift at either dose → P-65 closed ❌** (entry "Submissions #54–#58"). Was: ✅ run green, shipped, pod deleted (E ≈ $1.9); 🔁 direction only — read vs the B0 seed mean 0.9365 (same bands). Fair epoch-selection test on gold: +0.007 (`v13ecp`) / −0.001 (`v13ec`), 🔁 (entry "Session E chain 2") |
| 2026-10-06 | **P-50 fork C2, submission #54 (`rsna-knee-fork` v12, sent 00:03, ref 56864525)**: the public 0.942 stack + **B6** (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`) as our leg at β 0.45 | — | **0.944** | **🔁 +0.001 vs #49 0.943 (band 0.942–0.945); +0.002 over the anchor alone (#13 / #15). Our best public number, rank 337 of 5,293 — level with the public notebooks now (the community stack reads 0.943, a public fork with a ConvNeXt-T leg 0.944; entry "The public frontier moved"). The fork pick (P-50 pick 2) moves to fork v12** (entry "Submissions #54–#58") |
| 2026-10-06 | **B13, submission #55 (`rsna-knee-infer` v55)**: flat rank-mean of B6 + `v11n` + `v11n2` (three CoAtNet votes, four CNN) | 0.9256 | **0.940** | **🔁 −0.002 vs B6 0.942 (band 0.939–0.945); the "≥ 0.943 → add a CoAtNet seed" branch does not fire.** Two more votes of a family B6 already has, both under the mean: +0.0053 over the members' mean 0.9347 (B6 +0.0062); predicted 0.941–0.942 |
| 2026-10-06 | **B11, submission #56 (`rsna-knee-infer` v56)**: flat rank-mean of `v13b3` + `v13e2` + `v13e` (the EfficientNet triple) | 0.9196 | **0.941** | **🔁 −0.001 vs B6; "strength over diversity" does not hold (< 0.943).** +0.0033 over the members' mean 0.9377 = the same-recipe gain; predicted 0.941–0.942. **B6 stays the own pick** |
| 2026-10-06 | **P-65 solo, submission #57 (`rsna-knee-infer` v53)**: `v13ecp` = `v13e` (B0, seed 42) on 0.5 Raptor + 0.5 Claude | 0.9063 | **0.935** | **🔁 −0.0015 vs the B0 seed mean 0.9365 (band 0.933–0.940); = `v13e` at the same seed (0.935)** |
| 2026-10-06 | **P-65 solo, submission #58 (`rsna-knee-infer` v54)**: `v13ec` = `v13e` on 0.25 LLM + 0.5 Raptor + 0.25 Claude | 0.9116 | **0.932** | **❌ by the pre-registered rule (≤ 0.932): −0.0045 vs the seed mean, −0.003 vs `v13e` at the same seed. P-65 ❌ DEAD END as a target lever (pair −0.003, no dose-response); P-46 closes with it** |
| 2026-10-06 | **P-69 Kaggle smoke (`rsna-knee-train` v47, pushed 14:44 UTC, 0.04 h)**: `PARALLEL_ARMS = ("v15c", "v15c2")` (ConvNeXt-T `in12k_ft_in1k` @ 288, LR 1e-4 + per-stage decay 0.9), `TEACHER_TABLES = ("raptor_teacher",)`, FORCE_SMOKE, full 34 training windows | — | — | ✅ **green: this slot's image ships timm 1.0.26** (train-b's 1.0.29, local / RunPod 1.0.28, traps 54; it loads the weights: `loaded 180 tensors`, `head.fc` dropped); `backbone LR range 5.90e-05 .. 1.00e-04 over 4 blocks`; `teacher table raptor_teacher: 4349`; **peak 7.03 GiB at 288 px, batch 2 × 34 windows → the arm fits a T4**; finite loss (0.734 / 0.818 after one smoke step); `reseeded 43` for `v15c2`; `-> _best.pt = SWA`; `ok  arm` × 2. The new `optimiser:` line printed; its single step was GradScaler's usual first-step fp16 overflow (`skipped 1`, scale 65536 → 32768). A PARALLEL parent runs no inference by design: 288-px inference was checked in the local CPU smoke and gets checked again by the A6 placeholder |
| 2026-10-06 | **P-68 Kaggle smoke (`rsna-knee-train-b` v8, 15:03 → 15:09 UTC, 0.03 h; v7 = a transient session ERROR, traps 54)**: `PARALLEL_ARMS = ("v13ex", "v13ex2")`, `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`, FORCE_SMOKE, full 34 windows; `v13ex2` (= `v13ex` at seed 43) new today | — | — | ✅ **green: both children read both tables** (`raptor_teacher: 4349`, `xfit_v09k: 4407`; `training targets = (1 - 0.5) * LLM + 0.5 * quantile-matched ['raptor_teacher', 'xfit_v09k']`); `reseeded 43 for arm v13ex2`; `freeze_bn: 49 encoder BatchNorm modules`; peak 2.95 GiB; finite loss; `optimiser: 1 steps … GradScaler scale 65536, skipped 0` (B0 does not overflow on its first step; ConvNeXt did); SWA `_best.pt`, `ok  arm` × 2. This slot's image: timm 1.0.29, Python 3.13. Unit checks (`v13ex2` = `v13ex` + seed 43 = `v13e2`'s recipe) and a local CPU smoke green first. **The 10-10 session B is ready** |

**External reference points** (not ours — for calibrating ambition):

| Score | What |
|---|---|
| 0.964 | Public LB #1 (2026-10-06); 10th 0.960; 123 teams ≥ 0.950 of 5,293 |
| **0.944** | `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` (2026-10-06): the 0.943 community stack + its author's 2.5D ConvNeXt-T reader (0.929 solo) at 30 %; 194 teams at 0.944 = our #54 (entry "The public frontier moved") |
| **0.943** | The community stack as of 10-06 (skarin's reproduction; = our anchor + two CoAt readers); 984 teams at exactly 0.943 |
| 0.958 | Public LB #1 (2026-09-21); #2–#5 at 0.955–0.956 — the top moved +0.006 in three weeks |
| 0.952 | Public LB #1 (2026-08-28); top 10 span 0.946–0.952 |
| **0.942** | `notebook_score_0.942.ipynb` ("DINOsaur V5"), read cell by cell 2026-09-21 — trains nothing; ~35 public checkpoints rank-fused (entry below); forked as P-27 |
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

### 2026-09-22 — P-30: the public 0.924 member's soft labels (`dreaddevelopment/rsna-knee-labels`) as a 4th source ❌ DEAD END as a measurable idea — the file has no gold rows, so it cannot be validated; NOT adopted

`labels_llm_soft.csv` (CC0, 4,349 rows × 12 labels, a coarse probability grid 0.05 / 0.15–0.2 / 0.45 / 0.8 / 0.95)
covers **exactly the 4,349 report-only studies and none of the 58 gold ones** — its author used gold-58 as validation
and published the training labels only. `python src/build_targets.py --sources hans_v4,pilkwang,sol56,dread`
(the new flag; the default teacher is byte-identical, md5 `29f641ed…`, 0.8948) therefore prints `dread nan` for the
source and an **unchanged BLEND 0.8948** — on gold rows the 4th source is NaN and `nanmean` ignores it. The
pre-registered measure (source alone ≥ 0.893 on gold-58) is impossible, not failed.

What can be measured (agreement on the 4,349 common studies, Spearman per label, mean over 12):

| pair | mean ρ | note |
|---|---|---|
| dread vs hans_v4 | **0.811** | worst Synovitis 0.52, Fracture 0.61; best PF OA 0.95, Medial OA 0.93 |
| dread vs our 3-source teacher | 0.817 | |
| hans_v4 vs pilkwang (our own two best sources, same studies) | 0.834 | the reference for "how much do good sources agree" |

Threshold agreement at 0.5 with hans_v4 is 0.87 on average, but dread is systematically **less positive** (Effusion 26 %
vs 59 %, MCL 5 % vs 15 %, Medial OA 25 % vs 37 %) — a different operating point, not a different reading; only Synovitis
is read *more* often (25 % vs 12 %). So it is a fourth reading of the same reports at pilkwang-level agreement with
hans_v4, with no way to know whether its disagreements are right.

**Verdict: ❌ DEAD END for adoption by the P-30 rule** — the only measurement that could validate it is a training A/B
(an arm on the 4-source teacher vs its twin), and that A/B has no neutral judge: OOF-vs-teacher is circular and gold-58
cannot see ±0.02. The `--sources` flag stays (it costs nothing and keeps the default teacher byte-identical); the
`data/llm_labels/dread/` file stays for a possible P-16/P-17 use as *extra soft votes on the report-only rows*, where no
gold validation is needed. **CORRECTION to the P-28 "morning item B5" plan**: "adopt if ≥ 0.893 on gold" was written
without knowing the file excludes gold.

### 2026-10-04 — P-65 gold-58 BLIND pilot: a grading-aware Claude relabel (Opus 5.5) reads **0.9062** alone vs the LLM blend 0.8948, and **0.9397** mixed 0.5 with Raptor vs 0.9324 · Haiku 4.5 reads **0.8639** · 🔁 (under both pre-registered bars) / ❌ for Haiku

Tian's go (2026-10-04, "Yes, run the gold pilot"). The host's 2.6.b update permits hosted LLMs (CLAUDE.md "Rules").
- **Setup.**
  - The prompt `artifacts/claude_labels/prompt_v1.md` was written from the host's grading rules only (topic 733343: borderline
    = negative, high-grade ACL, acute MCL and fracture, moderate/large effusion and Baker's, ≥ 1 cm > 50 % cartilage loss for
    OA). Per finding it outputs `m` ∈ pos / sub / neg / unk and a calibrated `p`.
  - The input `gold_reports.jsonl` holds the 58 gold reports, shuffled, with an index only. The UID key is kept separate and the
    labellers never saw labels or other files.
  - Two subagents labelled the same 58 reports. The scorer is `score_pilot.py`.
  - Opus took 5.4 min and ≈ 139k tokens (≈ 2.4k per report, all overhead included). Haiku took 5.5 min and ≈ 114k tokens.
- **Pre-registered** in card P-65 before the read: promising = alone ≥ 0.910 OR + Raptor ≥ 0.945; flat = ± 0.01. **Read once; the
  prompt is not iterated on gold.**

| label | LLM blend | Raptor | Claude Opus | Claude Haiku | 0.5 LLM + 0.5 Rap | 0.5 Opus + 0.5 Rap | LLM + Opus + Rap |
|---|---|---|---|---|---|---|---|
| ACL | 0.990 | 0.980 | 0.993 | 0.981 | 0.991 | 0.994 | 0.996 |
| MCL | 0.980 | 0.993 | 0.977 | 0.984 | 1.000 | 1.000 | 0.995 |
| Medial Meniscus | 0.955 | 0.969 | 0.953 | 0.912 | 0.974 | 0.982 | 0.980 |
| Lateral Meniscus | 0.881 | 0.863 | 0.904 | 0.855 | 0.911 | 0.921 | 0.914 |
| Medial OA | 0.931 | 0.984 | 0.913 | 0.888 | 0.974 | 0.960 | 0.957 |
| Lateral OA | 0.808 | 0.836 | 0.849 | 0.798 | 0.825 | 0.867 | 0.837 |
| PF OA | 0.903 | 0.835 | **0.967** | 0.879 | 0.905 | 0.929 | 0.942 |
| Effusion | 0.880 | 0.973 | 0.850 | 0.802 | 0.955 | 0.946 | 0.932 |
| Synovitis | 0.788 | 0.824 | 0.806 | 0.732 | 0.842 | 0.875 | 0.851 |
| Baker's | 0.947 | 0.978 | 0.855 | 0.807 | 0.976 | 0.933 | 0.953 |
| Contusion | 0.861 | 0.931 | 0.883 | 0.877 | 0.920 | 0.926 | 0.921 |
| Fracture | 0.815 | 0.938 | **0.924** | 0.851 | 0.916 | 0.945 | 0.932 |
| **macro** | **0.8948** | **0.9254** | **0.9062** | **0.8639** | **0.9324** | **0.9397** | **0.9341** |

**Where the grading rules helped and where they hurt.**
- **Fracture (acute only) is the clear win.** Inside the report-silent cells the Opus labeller ranks gold fractures at 0.987
  (n = 23, 4 positives), vs 0.138 for the LLM blend and 0.934 for Raptor. The blend's fracture signal is anti-correlated
  there, because it counts old fractures.
- PF OA +0.064, Lateral OA +0.041 and Lateral Meniscus +0.023 also gained.
- **The size thresholds hurt.** Effusion fell −0.030 and Baker's −0.092. 39 of the 58 effusions came out as `sub`, which
  collapses the ranking, and 30 Baker's cells were `unk`. Medial OA fell −0.018.
- Haiku is worse than the blend on 9 of 12 labels and worse than Opus on 11 of 12 (−0.042 macro). Grading-aware extraction
  needs a frontier model, as the literature says (artifacts/research_1004/literature.md).

**Reading.**
- The two gains, +0.011 alone and +0.007 in the Raptor mix, are the size of the D4 (+0.006) and cross-fit (+0.008) target gains
  that never reached the LB (traps 39). Only Raptor's +0.017 transferred.
- The per-label pattern is principled (the policy rules) but split: 6 labels up, 3 down.
- **Verdict: 🔁 INCONCLUSIVE for the Opus relabel (under both bars); ❌ for a cheap-model relabel.** A full pass would be judged by
  one solo LB read only. Picking labels per source on gold is not allowed (12 choices on 58 studies).

### 2026-10-04 — P-65 full pass: the grading-aware Claude (Opus 5.5) relabel of all 4,349 report-only studies — 4,349 / 4,349 rows, 0 missing / duplicate / malformed; ≈ 25 min wall, ≈ 6.9 M subagent tokens · ✅ table published (`claude_v1`, composite `claude_rap_v1`) · LB ⏳ (session E staged)

On Tian's go ("Run the full Opus pass now"). The reports were split into 44 shuffled batches of 100
(`artifacts/claude_labels/full/batch_NN.jsonl`, key `full_key.csv`). There was one blind subagent per batch, running the same
protocol as the pilot (`AGENT_TASK.md`): `prompt_v1.md` only, in chunks of ≈ 20 reports, self-verified output. About 15 ran in
parallel, each batch took 4.3–5.4 min (one 8.3 min), and each used 141k–179k tokens.
`merge_full.py` → `artifacts/teacher/claude_v1.csv` (4,349 × 12 p) + `claude_gold.csv` (the pilot's 58) + `claude_v1_m.csv` (codes).

Plausibility on the report-only rows (no ground truth; direction only):

| label | ρ vs LLM blend | ρ vs Raptor | AUC vs the hard LLM label | pos @ 0.5 (Claude / LLM) | `unk` share |
|---|---|---|---|---|---|
| ACL | 0.789 | 0.684 | 0.984 | 0.106 / 0.206 | 0.079 |
| MCL | 0.638 | 0.583 | 0.967 | 0.014 / 0.153 | 0.093 |
| Medial Meniscus | 0.928 | 0.818 | 0.998 | 0.387 / 0.402 | 0.054 |
| Lateral Meniscus | 0.783 | 0.717 | 0.999 | 0.138 / 0.153 | 0.098 |
| Medial OA | 0.901 | 0.747 | 0.961 | 0.148 / 0.368 | 0.211 |
| Lateral OA | 0.856 | 0.679 | 0.951 | 0.064 / 0.263 | 0.263 |
| PF OA | 0.918 | 0.776 | 0.949 | 0.178 / 0.455 | 0.177 |
| Effusion | 0.905 | 0.812 | 0.895 | 0.181 / 0.593 | 0.098 |
| Synovitis | 0.758 | 0.723 | 1.000 | 0.122 / 0.124 | 0.827 |
| Baker's | 0.806 | 0.580 | 0.915 | 0.078 / 0.247 | 0.459 |
| Contusion | 0.810 | 0.723 | 0.975 | 0.134 / 0.170 | 0.234 |
| Fracture | 0.422 | 0.613 | 0.994 | 0.055 / 0.067 | 0.437 |

- **The host's thresholds show up as lower positive rates where they bind:** MCL 1.4 % vs 15.3 % (acute high-grade only), Effusion
  18 % vs 59 % (moderate or large), Medial OA 15 % vs 37 % and PF OA 18 % vs 46 % (> 50 % cartilage loss).
- Rank agreement with the LLM blend is high where the reports are explicit (Medial Meniscus, PF OA, Effusion ρ ≈ 0.91–0.93).
- **Fracture is the outlier: ρ 0.42 with the LLM blend but 0.61 with the image teacher.** Acute-only fracture agrees with Raptor
  more than with the report-extraction labels, as on gold (pilot 0.815 → 0.924).
- The `unk` share matches pilkwang's silence pattern: Synovitis 83 %, Baker's 46 %, Fracture 44 %, Lateral OA 26 %.

**Training use (staged, not run): session E = `v13ec` (EfficientNet-B0) ‖ `v13rc` (ResNet-50) = `v13e` / `v13r` on
`TEACHER_TABLES = ("claude_rap_v1",)` at `TEACHER_MIX = 0.75`.**
- `claude_rap_v1` is a composite: per label, ⅔ Raptor rank + ⅓ Claude rank (ρ 0.970 with Raptor, 0.851 with Claude).
- In rank terms the target is ≈ 0.25 LLM + 0.25 Claude + 0.5 Raptor. That is `v13e` / `v13r`'s target with only the LLM half
  replaced by 0.5 LLM + 0.5 Claude, which is what P-65 pre-registered.
- `build_targets.py --teacher-tables claude_rap_v1 --teacher-mix 0.75` builds it (4,349 covered); `window_head_test.py` is green.
- Read rule: m(`v13ec`, `v13rc`) vs m(`v13e`, `v13r`) = 0.9345. ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299.
- It needs ≈ 6 GPU-h, which do not fit before the 2026-10-10 reset once session D is done (≈ 3.6 h would be left).

**Verdict: ✅ the table is complete and published** (private Dataset `rsna-knee-teacher-tables`: + `claude_v1.csv`,
`claude_rap_v1.csv`; the four older tables are md5-identical). Its value is ⏳ until session E is read on the LB.

**Read 2026-10-06: ❌ no value as a training target.** Session E ran as two doses on B0 instead of the B0 + R50 pair: `v13ecp` (0.5
Claude) **0.935**, `v13ec` (0.25 Claude, the design above) **0.932**, vs the B0 seed mean 0.9365. P-65 closed (entry "Submissions
#54–#58").

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

**Scored 2026-08-30 ≈ 19:00 — #9 = 0.909** (was 0.900): +0.009 = 1.8× the 0.005 LB floor → ✅ the mechanism carries to
the hidden test; offset +0.0295, inside the +0.02–0.03 band (n=6 calibration points now). #10 (seven versions) ⏳.
**#10 = 0.912 (≈ 19:45)**: +0.003 over #9 for adding `v09h` — 🔁 as an increment (OOF predicted +0.0024, the LB floor is
0.005), exactly the "one family, small ensemble gain" reading above; but it is the best number and becomes the default
blend (`INFER_MEMBERS` = the seven). The two-day c02 lane is **0.900 → 0.912 (+0.012, 2.4× the floor)** ✅. Offset
+0.030 (n=7). What is left is new *input representations*, as reading 4 says.

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

### 2026-08-30 — 5-fold `v09h` (RunPod chain4): pooled OOF 0.8625, +0.016 over `v05g`'s 0.8467; gold-58 0.874 ✅ KEEP as the ensemble base

**Config:** the `v09h` arm exactly (CoAtNet-1 @224 on c02, window_attn, 24 random train windows, all windows at
eval, 8 ep, `best_oof`, EMA 0.998, `lr_backbone` 1e-4, seed 42); folds 1–4 trained back to back on the RunPod 4090
(`ARM_FOLDS` sed on the pod copy; fold 0 reused from the 15:30 run — `_last.pt` at epoch 7 resumed at epoch 8 =
skipped, `_best.pt` untouched). ≈ 50 min/fold at 0.09 s/study; chain4 16:03–19:37 UTC; ≈ $2.6.

| fold | n | OOF | gold (n) | ckpt epoch |
|---|---|---|---|---|
| 0 | 882 | **0.8683** | 0.923 (11) | 7 |
| 1 | 882 | 0.8668 | 0.889 (12) | 7 |
| 2 | 881 | 0.8589 | 0.831 (12) | 6 |
| 3 | 881 | 0.8546 | 0.847 (12) | 6 |
| 4 | 881 | 0.8653 | 0.919 (11) | 5 |

- **Pooled over all 4,407: 0.8625** fold-rank-normalised (raw 0.8623); mean of folds 0.8628; **gold over all 58:
  0.874** (raw 0.873). `v05g`, the previous 5-fold base: pooled 0.8467, gold 0.848 → **+0.016 / +0.026**.
- **Fold spread 0.0137 (0.8546–0.8683) — 1.7× the 0.008 floor**, wider than `v05g`'s 0.0079. Fold difficulty only
  partly matches `v05g` (fold 0 easiest and folds 2–3 hard in both; fold 1 was `v05g`'s worst but is second best
  here), so it is fold difficulty *plus* seed noise, and the mid-run "every fold is getting worse" appearance was
  a coincidence fold 4 broke. Fold-0 A/Bs stay flattered by ≈ +0.006 vs the fold mean — fine for deltas, not levels.
- **Per-label pooled**: Medial Meniscus 0.903, Fracture 0.893, Baker's/Synovitis 0.892 … **Lateral Meniscus 0.847
  vs `v05g`'s 0.789 (+0.058)** — the band + window-head claim survives at 5-fold scale; the floor is now MCL 0.799
  and PF OA 0.826.
- **`ckpt_policy="best_oof"` bit for the first time on this recipe**: folds 2/3/4 peaked at epochs 6/6/5 (fold 4:
  0.8653 at ep5 vs 0.8642 at ep7). The fixed-last-epoch policy would have cost ~0.001 macro on three folds.
- **Ship**: the pod's `ship` step failed **silently** on the 3-h-expired OAuth token (`kaggle datasets version`
  exited 0 in < 2 s with no upload — the upload-side twin of traps 20); the Dataset was versioned from the local
  pull instead: `rsna-knee-ckpt-v09h` now holds 5 × `_best.pt` + 5 × `_oof.csv` (766 MB compressed, file list
  verified). Everything else (40 per-epoch csvs, all logs) pulled and md5-verified to
  `artifacts/{ship_v09h_5fold,kaggle_out/pod_v09h_5fold}/`; **pod deleted 22:05 local** (day's pod spend ≈ $7).
- **Verdict: ✅ KEEP** — the `v09h` vote in the default blend is now 5 folds (infer v14). Expected LB effect of the
  folds alone is below the floor (folds are replicates: `v05g`'s five folds bought +0.009 as a *lone* version, and
  less inside a 7-vote blend), so no submission was spent on it; it firms up the best member for the private set
  and gives `v09h` a full-corpus OOF — the target P-17 self-training needs.
- **MEASURED 2026-09-03 (a submission *was* spent on it that evening, Tian's call — #11):** LB **0.913**,
  **+0.001** over #10's 0.912. The prediction above was right to 0.004 and the verdict is unchanged: 🔁 as an
  increment, ✅ as the default blend. So the fold-count question is closed on the LB too — inside a seven-vote
  blend, five folds of the best member are worth ~+0.001 (Submissions #11, and P-13's index row).

### 2026-09-21 — Anatomy of the public 0.942 notebook ("DINOsaur V5"): a 2×T4 inference graph over ~35 public checkpoints, nothing trained ✅ read in full · forked as P-27

`notebook_score_0.942.ipynb` (53 cells, ~5,300 lines; Tian confirmed its 0.942). The successor of the 0.936
notebook decomposed in research.md §2.7.1; same author lineage (BTKD / mattiaangeli / dreaddevelopment).
**Every member is a mounted public checkpoint; several were trained on A6000/H100 boxes** (the resgated
dataset ships `trainer_a6000_epochs01_06.py` / `trainer_h100_epochs07_10.py`).

| Stage | Members | Fusion | Stated LB |
|---|---|---|---|
| 1 | `pilkwang/rsna-knee-weights`: **20 DINOv2-S** members, 336 px, 3 slices × 6 slots (SAG/COR/AX fluid-FS, SAG fluid-noFS, COR T1, SAG T1), overlapping windows, per-label pooling (max Fracture/Contusion/menisci/Baker's, top-2 ACL/MCL); shared 6-block prefix for speed | rank mean, 20/20 fingerprint gate | ≈ 0.899 alone |
| 2 | `mattiaangeli/knee-mri-fold-weights`: **A5 = 5 folds of a 16-slice timm model** (`DepthCompress` gated stem or `SlotDepthMixer`, `xattn`/`xres`/`clsadd` readouts, 336 px, band 0.12–0.88) | rank blend at **w 0.52** (0.941 lineage: 0.45) | — |
| 3 | RadImageNet R50 frozen GAP features + 3 head bundles (`v52` public, `E13`, E13-on-E11 layout) | α **0.55** (was 0.50) on 10 labels, second pass **0.20** (was 0.15), 88-feature calibrator at 0.40 on 7 labels | ≈ 0.920 cumulative (0.936 lineage) |
| 4 | **Raptor CoAtNet-2@384** × 4 views: `v5 swa` (64 slices, 2–98 %, 336 source, w 0.60), `v10` native-384 dense (0.10), `v5 reverse` slice order (0.10), `v8` 44 slices 6–94 % (0.20) at **94 dense windows** (capacity-aware allocation over 96 slices; released arms carried 62/42); then a CoAt family — `resgated` epochs 4/6/8 (3 ckpts) + `D4` depth-zone SWA3 adapter on a rank-8 SWA2 parent — equal rank mix, blended into the Raptor arm at **0.40**; two child processes, each fail-soft | per-label outer weights CoAtNet vs transformer stack: default 0.60, ACL 0.75, MedMen 0.80, **LatMen 1.00**, LatOA 0.75, Fracture 0.75 | 0.924 alone (`v5`); 0.941–0.942 fused |
| 5 | "FineSpacing v9" residual (80 slices / 78 windows, scale 0.10) | — | **not in the scored run**: its dataset was not attached (17 sources), the cell's own branch says "exact 0.942 anchor retained" |

Its header states the 0.941 → 0.942 step was **half blend-weight tuning** (the four numbers above) and half
the CoAt family. Its diagnostics cell itself warns that the per-finding outer weights (LatMen 1.00 discards
three of four stages) are "the likeliest place to give back points privately".

**The 0.924 member's training recipe** (`dreaddevelopment/knee-mri-training-the-twelve-finding-model`, read):
CoAtNet-2@384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, ImageNet-pretrained), **all 4,349 report-labelled
studies, gold-58 as the only validation**, 16 epochs, bs 8 studies × 12 random 3-slice windows (eval 24),
OneCycle (bb 3e-5 / head 1e-3, pct_start 0.15), AdamW wd 0.02, BCE with `pos_weight` (1−p)/p ∈ [1,10],
bf16, clip 3.0, ±10 % intensity gain as the only augmentation, **no laterality normalisation**, 140 mm crop,
2–98 % span, windows may straddle slot boundaries, no slot embedding; **best epoch by gold-58 AUC + SWA of the
top-3 epochs**; ~3 h on a 4090; corpus 3,155 → 4,349 studies was +0.013 gold / +0.010 LB for them. Their soft
labels are public (`dreaddevelopment/rsna-knee-labels`, CC0, `labels_llm_soft.csv`) — P-28 morning item B5.

**Differences to our best recipe (`v09h`, OOF 0.8683 ≈ LB 0.895–0.90)** that could matter, in the order we
now act on them: all-data + 16 epochs + SWA (P-28, tonight); bs 8 × 12 windows vs our 1 × 24 (BatchNorm in
the CoAtNet conv stages sees one study per step — a fold-0 A/B); their soft labels as a 4th source; 384 px
(measured −0.004 for us at 8 ep, and the stack already has 5+ of these). Not copied: no-laterality (P-05 is
+0.015 for us and is our diversity), slot-crossing windows, gold-58 epoch selection (rejected).

**Licences** (`kaggle datasets metadata`, closes the brainstorm.md item): Raptor ×3 datasets, `rsna-knee-labels`,
A5, resgated, D4, pilkwang weights = **CC0-1.0**; `resnet-50-radimagenet-marwan`, `e11-diverse-heads-v20`,
`e9-radimagenet-heads-v15` = **CC-BY-NC-SA-4.0**; `v52-radimagenet-heads-20260812` = "other". The Rad stage is
the only licence-encumbered part; decision deferred to the final submission.

**Decision (Tian, 2026-09-21):** fork it verbatim and add our members as one more vote (P-27), keep training our
own under the production regime (P-28). Verdict on the fork = ⏳ Scoreboard row (#12).

### 2026-09-22 — P-28 production arms `v09a` (CoAtNet-1) and `v08a` (DINOv2-S) trained on all 4,349 studies, 16 ep, SWA of the last 3 EMA snapshots (kernels `rsna-knee-train` v19 / `rsna-knee-folds` v6) ✅ the regime runs end to end · SWA vs last EMA 🔁 · gold peaks at epoch 5–6 ⏳ (a question, not a finding) · LB ⏳

Both kernels ran to `all folds complete: True` with the two P-28 lines the handoff asked for (`SWA of last 3 EMA
snapshot(s)`, `-> <arm>_fold0_best.pt = SWA, <arm>_fold0_lastema.pt = last EMA`); `train 4349 / val 58 studies
[train_all: val = gold rows]`. No resume was needed: `v09a` 5.12 h (0.25 s/study, 18 min/epoch on the T4 — the
CoAtNet-1 arm is only 2× the DINOv2 arm), `v08a` 2.38 h. The `_last.pt` of `v09a` is 1.15 GB (the SWA ring), as
the card predicted. **These members have no OOF** (traps 32): every number below is gold-58, SE ≈ 0.04 macro.

| arm | backbone | epoch 0 | peak (epoch) | epoch 15 (last EMA) | **SWA of 13–15 = `_best.pt`** | SWA − last EMA | hours |
|---|---|---|---|---|---|---|---|
| `v09a` | `timm:coatnet_rmlp_1_rw_224`, c02, window_attn, lr 1e-4 | 0.746 | **0.8925 (5)** | 0.8762 | **0.8768** (CI 0.841–0.910) | +0.0006 | 5.12 |
| `v08a` | DINOv2-S @224, c02, window_attn | 0.733 | **0.8944 (6)** | 0.8802 | **0.8816** (CI 0.846–0.913) | +0.0014 | 2.38 |

Per-label gold of the SWA checkpoints (both arms; the 7 labels the log prints in the table): ACL 0.941 / 0.898,
MCL 0.921 / 0.914, Medial Meniscus 0.935 / 0.930, Lateral Meniscus 0.819 / 0.848, Medial OA 0.986 / 0.963,
Lateral OA 0.739 / 0.812, PF OA 0.825 / 0.821 (`v09a` / `v08a`). Lateral OA is the weakest label for the
CoAtNet arm and Lateral Meniscus for both — the same two labels that were weakest for the fold-0 members.

Readings:

1. **SWA of the last three EMA snapshots ≈ the last EMA** (+0.0006 / +0.0014, far inside any floor) — SWA neither
   helps nor hurts on gold; in particular the CoAtNet arm shows **no BatchNorm-averaging penalty**, so the P-28
   "if it fails → `update_bn`" branch is not triggered. 🔁 as a gain; ✅ as a safe default (it can only smooth).
2. **Both gold curves peak at epoch 5–6 and drift down by ≈ 0.015 to epoch 15** (0.8925 → 0.8762; 0.8944 →
   0.8802) while the training loss keeps falling (0.45 → 0.32). Each drift alone is well inside the 58-row SE
   (≈ 0.04), so this is **not evidence that 16 epochs over-train** — but the *same sign on two independent
   arms* is the pattern the noise-floor rule says to watch, and it is exactly what the public 0.924 member's
   author saw (they select the best gold epoch + SWA the top 3, which the docs reject). It is a **question for a
   proper measurement**, not a finding: a fold-0 16-epoch twin of `v09h` (OOF on 871 studies, floor 0.008) would
   say whether the OneCycle tail over-fits the soft targets. Logged as an open question; the epoch budget stays 16
   until that number exists (P-28 hypothesis unchanged: the LB is the measure).
3. Compared with their fold-0 twins on gold (`v09h` fold 0 gold 0.923 on n=11; 5-fold pooled gold-58 0.874,
   `v08w` 0.927 on n=11) nothing can be read — different n, and gold-58 was *inside* the twins' training folds'
   validation only 11 at a time. The only admissible comparison is the LB via P-27 with and without these members.

**Verdict: ✅ the regime works as built (both arms, no resume, SWA written); 🔁 SWA vs last EMA; ⏳ the LB value
(fork v5 = `v08w` + `v09h` + `v09a` + `v08a`).** Checkpoints: `artifacts/kaggle_out/train_v19/` (`v09a_fold0_best.pt`
164 MB, `_lastema.pt`, `_last.pt` 1.15 GB, 16 per-epoch `_ep*_oof.csv` = gold-58 predictions) and, once the Kaggle
token is renewed, `folds_v6/` for `v08a`; Datasets `rsna-knee-ckpt-v09a` / `-v08a` (traps 20 hit at 07:55 — the
pull of `v08a` failed with the "wrong slug" message).

### 2026-09-22 — `build_fork.py --beta 0.0` is now a true anchor-only control (our arm is not launched) ✅ KEEP the code · fork v6 pushed as the control · LB ⏳ on Tian's go

Before this change β 0 would still have run our ~80-minute arm on the hidden test and then written
`rank_pct(rank_pct(anchor))` — AUC-identical to the anchor but with every failure mode of the arm attached and a
different byte content. Now the arm cell raises a `_ForkControl` before the runtime gate when `_FORK_BETA <= 0`,
the `except` branch sets `status = "anchor_control"`, and the `finally` block enforces `sha256(submission.csv) ==
sha256(anchor)` for that status exactly as it does for a failed arm. The markdown cell documents it; the builder's
`--check` determinism, the `selftest_blend` rank identities and the fork-v5 build are unchanged (v5 rebuilt to
`artifacts/fork_v5b/` with the new builder: same members/sources, the only diff is the new branch). Kernel version
**6** = `--beta 0.0 --members v08w v09h` (the same 20 sources as v3/v4 — the `v08w` / `v09h` Datasets stay mounted
although the arm never runs, so the anchor graph's inputs are exactly #12's/#13's). Verdict: ✅ the code; the control's LB read is ⏳
(brainstorm.md "Does the 0.942 anchor reproduce 0.942 from our account?").

### 2026-09-22 — P-29 epoch-budget probe (`v09p`, train v21): the 16-epoch production schedule **over-trains** — OOF peaks at epoch 8 (0.8731) and falls to 0.8607 by epoch 15 (−0.0124, 11/12 labels down) ❌ the P-29 hypothesis · ✅ KEEP the finding (changes P-28)

`v09h` recipe (CoAtNet-1 @224, c02, window_attn, lr_backbone 1e-4), fold 0 (train 3,525 / val 882), OneCycle
stretched to **16 epochs**, per-epoch OOF-vs-teacher on the 882 held-out studies; 5.64 h on one T4.

| epoch | 0 | 2 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OOF | 0.743 | 0.836 | 0.861 | 0.868 | 0.870 | 0.872 | **0.8731** | 0.871 | 0.868 | 0.865 | 0.863 | 0.862 | 0.861 | 0.8607 |
| train loss | 0.573 | 0.512 | 0.471 | 0.453 | 0.434 | 0.416 | 0.395 | 0.374 | 0.355 | 0.339 | 0.329 | 0.320 | 0.316 | 0.314 |

Per label, epoch 8 → epoch 15: **11 of 12 down** (MCL −0.031, Fracture −0.019, Synovitis −0.018, Lateral OA −0.017,
Medial OA / Effusion −0.016, the rest −0.002 … −0.009; ACL +0.001). A rank-mean of the epoch 13–15 predictions — the OOF
analogue of the production `swa_last=3` checkpoint — scores **0.8611**: SWA of the tail does not rescue it.

Readings:

1. **The pre-registered rule fires**: peak − epoch 15 = 0.0124 ≥ the 0.008 floor **and** the peak is before epoch 10 →
   the P-28 schedule over-trains the soft targets. The consistent sign over 11 labels is the evidence the floor rule asks
   for. It also explains the gold-58 drift of `v09a` / `v08a` (peak epoch 5–6, −0.015): gold was right in direction.
2. **16 epochs does not beat 8 at the peak**: 0.8731 vs `v09h`'s 8-epoch end point 0.8683 = +0.0048, under the floor.
   Stretching the schedule buys nothing; ending it at 16 costs ≈ 0.012.
3. **The shipped production members** `v09a` / `v08a` are 16-epoch SWA-of-13–15 checkpoints, so by this curve they sit
   ≈ 0.012 below what the same data could give — consistent with #14 adding nothing (−0.001).
4. The next production budget is **8 epochs** (the `v09h` schedule, whose end point is within the floor of this peak),
   `swa_last=3` over epochs 5–7. That is a P-28 card change, not yet measured.

**Verdict: ❌ P-29 hypothesis ("16 epochs is fine") · ✅ KEEP the finding: P-28 `epochs` 16 → 8 for future members.**
Files: `artifacts/kaggle_out/train_v21/` (16 per-epoch csvs, `v09p_fold0_oof.csv` = epoch 8 by `best_oof`, the log).

### 2026-09-22 — P-27 read-out with the control: the anchor reproduces 0.942 from our account; our arm is worth **±0.000 at β 0.10, −0.003 at β 0.20**; the P-28 members −0.001 🔁 (all sub-floor)

| # | what | public LB | vs the control |
|---|---|---|---|
| 15 | anchor only (β 0, our arm not run) | **0.942** | — |
| 13 | + `v08w` + 5-fold `v09h` at β 0.10 | **0.942** | ±0.000 |
| 14 | + `v09a` + `v08a` as well, β 0.10 | 0.941 | −0.001 |
| 12 | + `v08w` + 5-fold `v09h` at β 0.20 | 0.939 | −0.003 |

1. **The anchor is stable**: unpinned public sources + a rerun from our account = the author's 0.942. Every β read is
   now against a number we own.
2. **Our arm never helps on the public split**, and the only non-zero readings are negative and grow with our weight.
   Each delta is under the 0.005 floor, but the pattern is the P-27 "if it fails" branch: our c02 members (own blend
   0.913) are redundant with a stack that already holds several CoAtNet-2 @384 c02-like views.
3. **What β 0.10 is still for**: no measurable public cost, and one vote that was never tuned on the public LB. Whether
   that is worth anything privately cannot be measured before the deadline — Tian's decision for the two final
   selections (brainstorm).

**Verdict: 🔁 on every increment (all < 0.005); ✅ the control (anchor = 0.942). Best on the board: 0.942 (#13 and #15, tied).**

### 2026-09-22 (night) — The public frontier re-read: our anchor is a *superset* of the best public notebook; "0.957" is a gold-58 number; 690 teams sit at 0.941–0.942

**Verdict: ✅ FINDING (read-only, no run).** 12 public notebooks pulled with `kaggle kernels pull` and dumped cell by cell,
the full public LB csv (4,183 teams, 17:33 UTC) and Kaggle's score-sorted kernel listing.

- **LB distribution:** 0.958 ×1, 0.956 ×1, 0.955 ×8, 0.954 ×7, 0.953 ×7, 0.952 ×10 … 0.945 ×29, 0.944 ×28, **0.943 ×98,
  0.942 ×270 (us, rank 246), 0.941 ×420**, 0.940 ×184. 119 teams ≥ 0.945, 49 ≥ 0.950, 10 ≥ 0.955. Everything above ~0.945
  is private work on top of the shared stack.
- **Score-sorted public notebooks** (`kaggle kernels list --competition … --sort-by scoreDescending`): 1 `mattiaangeli/
  bend-the-knee-to-speedy-raptors-the-original` (0.943 by the DINOsaur V5 author's note), 2 `evgendvorkin/rsna-baseline`,
  3 `maverickss26/rsna-knee-0942-restructured`, 4 `jiweiliu/rsna-knee-fast-2xt4-inference` (0.942), 5 `romantamrazov/
  rsna-knee-dinosaur-v5` (our anchor's source). **Cell-level diff:** Speedy Raptors, DINOsaur V5 and the 2×T4 notebook share
  every member and weight our anchor has — 20 DINO, A5 0.52, Rad 0.55 / 0.20 + calibrator 0.40, the four Raptor views
  0.60 / 0.10 / 0.10 / 0.20 at 94 capacity-aware windows, the resgated + Global96 + D4 CoAt family at 0.40, the probe22
  outer map (LatMen 1.00). Our anchor is a superset (it also carries the repair-v1 child when pinned). **→ re-anchoring
  the fork is worth ≤ +0.001 — not a lever.**
  **CORRECTED 2026-09-27:** `notebook_score_0.942.ipynb` runs **two** CoAt readers (resgated top-3 + D4) and mounts neither the
  Global96 nor the Repair-v1 artifacts (its only "global96" strings are the D4 preparation function's name). The haideptry
  "0943" Speedy Raptors build (2026-09-24) is our anchor's cells **plus** Global96 + Repair-v1 — our anchor is the subset there.
  The ≤ +0.001 conclusion stands (it is that notebook's own claim). Infrastructure entry 2026-09-27 "Speedy Raptors".
- **"RSNA Fast Parent 0.957"** (kminsher / mekduy, published today): the *0.941* recipe (A5 0.45, Rad 0.50 / 0.15, 62/42
  windows, resgated only, flat outer 0.60). Its own config cell states `"parent" … (0.939 public)` and `"probe22" … (0.941
  public)`; the 0.957385 is its "local diagnostic" on the 58 gold studies. Nothing in it is missing from our anchor
  (traps 35).
- **"Why public forks stop at 0.941"** (starkhushi): ladder 0.937 (sources silently dropped) → 0.940 → 0.941 (0.55/0.15
  view weights); "injecting my own 2.5D members (ResNet34 / ConvNeXt / RadImageNet, holdout 0.85–0.87) at 15 %: **0.936**.
  Below ~0.90 solo, a member costs more than its diversity buys" — the same reading as our P-27 ±0.000 with 0.86–0.87 OOF
  members. It recommends a flat-0.60 outer-map build as the second final pick ("LatMen 1.00 discards three of four stages
  for that column") — built as the hedge (`build_fork.py --anchor-preset parent`).
- **Public training recipes** (the 0.957 notebook's fallback trainer cell; the 0.924 script; the DINOsaur V5 "train"
  notebook): DINO members — 10 epochs, **batch 8 studies**, OneCycle (head 1e-3 / backbone 8e-6), augmentation rot ±8° /
  scale +0–8 % / shift ±5 % / intensity ±10 %, gold weight 3.0, best epoch on an md5-report holdout; Raptor CoAtNet — 16
  epochs, **8 studies × 12 windows**, OneCycle 3e-5 / 1e-3, pos_weight [1, 10], intensity ±10 %, best gold epoch + top-3
  SWA. The DINOsaur V5 "train" notebook trains a ConvNeXt-T *guarded arm* (β ≤ 0.12), not a DINO member. Ours: 1 study ×
  24 windows, cosine, Gaussian noise only.
- **Consequence:** the public increment of any member of ours stays ≈ 0 until it is a ≥ 0.90-solo member; the recipe
  differences never A/B'd are the batch composition (P-32), train-time augmentation (P-33) and the backbone LR (round 2).
  A 16-channel retry and a RadImageNet member are already stages of the anchor (A5, stage 3). And the machine we train on
  has had a second, idle T4 all along (traps 34, P-31).

### 2026-09-23 — S1 A/B on both T4s (P-31 / P-32 / P-33, `rsna-knee-train` v23, 3.22 h): `v09b` (two-study BatchNorm batches) **0.8690**, `v09c` (+ light augmentation) **0.8730** vs `v09h` 0.8683 — both 🔁 INCONCLUSIVE (under the 0.008 floor) · P-31 two arms per session ✅ KEEP

The first real two-arm session: one child process per GPU (`v09b` on `cuda:0`, `v09c` on `cuda:1`), fold 0 (3,525 train /
882 val, 11 gold), 8 epochs, `best_oof`, seed 42, the `v09h` recipe otherwise (CoAtNet-1 @224, c02, window_attn, 24 random
train windows, `lr_backbone` 1e-4). `v09b` = `batch_studies=2, grad_accum=2` (48 windows from two studies per BatchNorm
batch, the same 4 studies per optimiser step); `v09c` = `v09b` + `aug="light"` (affine rot ±8° / zoom-in 1.00–1.08 / shift
±5 %, gamma 0.8–1.25, gain ±10 %, no flips). Both children rc 0 with `_best.pt` (parent: `ok  arm v09b` / `ok  arm v09c`);
**3.22 h wall for both** (≈ 5.6 h if run one after the other). Outputs `artifacts/kaggle_out/train_v23/` (children's logs,
per-epoch OOF csvs); the read-out script reproduces the kernel's macros (0.8683 / 0.8690 / 0.8730) on the same 882 studies.

**P-31 (throughput) ✅ KEEP:** `v09b` 0.25 s/study (1.00× the solo 0.25), `v09c` 0.28 s/study (1.12×; the GPU-side
grid_sample), val 6.0 / 7.2 min per epoch; peak **6.84 GiB per child** (2 × 24 windows, AMP); no host-RAM or loader stall in
the heartbeat. 1.75× arms per quota hour → the default for every session from S2 on.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `v09h` (1 × 24, no aug; RunPod) | 0.759 | 0.814 | 0.836 | 0.853 | 0.864 | 0.867 | 0.868 | **0.8683** |
| `v09b` (2 × 24, accum 2) | 0.7498 | 0.8128 | 0.8389 | 0.8534 | 0.8613 | 0.8665 | 0.8685 | **0.8690** |
| `v09c` (`v09b` + `aug="light"`) | 0.7514 | 0.8116 | 0.8397 | 0.8542 | 0.8646 | 0.8703 | **0.8730** | 0.8730 |

Per label (OOF-vs-teacher AUC, fold 0, all 882 rows; per-label floor ≈ 0.03):

| label | `v09h` | `v09b` | `v09c` | b − h | c − h | c − b |
|---|---|---|---|---|---|---|
| ACL | 0.871 | 0.899 | 0.890 | +0.028 | +0.019 | −0.009 |
| MCL | 0.799 | 0.807 | 0.810 | +0.008 | +0.011 | +0.003 |
| Medial Meniscus | 0.899 | 0.916 | 0.920 | +0.017 | +0.021 | +0.004 |
| Lateral Meniscus | 0.868 | 0.850 | 0.856 | −0.017 | −0.011 | +0.006 |
| Medial OA | 0.879 | 0.880 | 0.879 | +0.001 | 0.000 | −0.001 |
| Lateral OA | 0.848 | 0.816 | 0.836 | −0.032 | −0.012 | +0.020 |
| PF OA | 0.821 | 0.817 | 0.819 | −0.004 | −0.001 | +0.003 |
| Effusion | 0.854 | 0.861 | 0.864 | +0.007 | +0.010 | +0.002 |
| Synovitis | 0.894 | 0.890 | 0.889 | −0.004 | −0.006 | −0.002 |
| Baker's | 0.898 | 0.897 | 0.905 | −0.002 | +0.007 | +0.008 |
| Contusion | 0.870 | 0.875 | 0.874 | +0.005 | +0.003 | −0.001 |
| Fracture | 0.918 | 0.921 | 0.935 | +0.003 | +0.017 | +0.014 |
| **macro** | **0.8683** | **0.8690** | **0.8730** | **+0.0008** (7/12 up) | **+0.0047** (7/12 up) | **+0.0039** (8/12 up) |

1. **P-32 (BatchNorm batch composition) 🔁 — the hypothesis is not supported.** +0.0008 is 0.1× the floor, 7/12 labels up,
   and the per-label pattern (ACL +0.028 and Medial Meniscus +0.017 against Lateral OA −0.032 and Lateral Meniscus −0.017)
   is the seed-scatter signature (the floor run moved Fracture 0.028 on seed alone), not a normalisation effect. Two
   studies per BN batch is not what separates our CoAtNet from the public 0.928 one. It is free (0.25 s/study, 6.84 GiB),
   which is the only reason it rides along.
2. **P-33 (light augmentation) 🔁 — half the floor, right sign.** +0.0039 over its same-session control, 8/12 up, the
   largest moves on Lateral OA (+0.020) and Fracture (+0.014); `v09c` plateaus at epochs 6–7 (0.8730 / 0.8730) where
   `v09b` is still climbing, so augmentation did not need more epochs. Costs 12 % time.
3. **Both arms beat `v09h` in the same direction, monotone `v09h` < `v09b` < `v09c`.** The S2 rule pre-registered in the
   handoff (a 🔁 knob stays out *unless both A/B arms beat `v09h` in the same direction*) therefore puts **both knobs into
   the production `v09a` retrain**. Honest expectation: +0.00–0.005 OOF — nothing the fork can see at β 0.10.
4. **Epoch budget confirmed under both knobs:** the honest split-half best-epoch-minus-last-epoch is −0.0001 (`v09b`) /
   −0.0002 (`v09c`) over 200 splits, so `ckpt_policy="last"` + SWA over epochs 5–7 loses nothing at 8 epochs (P-29 holds).
5. **Diversity:** mean per-label Spearman ρ(`v09c`, `v09h`) 0.898, ρ(`v09b`, `v09h`) 0.889, ρ(`v09c`, `v09b`) 0.919 — the
   same member with another seed, not a new family (P-23's rule: a new member needs ρ < 0.80 against the blend).

**Verdict: ✅ KEEP P-31 (two arms per session is the default from S2 on); 🔁 INCONCLUSIVE P-32 and P-33 (both under 0.008;
both carried into the S2 `v09a` by the same-direction rule, at zero score risk).** Round-2 candidates unchanged: the
backbone LR (3e-5 OneCycle vs our 1e-4 cosine — the last never-A/B'd difference to the public 0.928 recipe) and `aug="light"`
on the DINOv2 arm.

### 2026-09-23 — Submission #16 (the flat-0.60 outer-map hedge, arm not run) read **0.940**: −0.002 vs the anchor's 0.942 🔁 INCONCLUSIVE (0.4× the floor, inside the pre-registered 0.939–0.941 band) · why nothing of ours moves the public 0.942 ✅ FINDING

The ladder from our own account, all on the same 20 public sources:

| # | what changed vs the anchor | public LB | Δ |
|---|---|---|---|
| 15 | nothing (anchor only) | 0.942 | — |
| 13 | + our c02 arm (`v08w`, 5-fold `v09h`) at β 0.10 | 0.942 | ±0.000 |
| 14 | + the 16-epoch `v09a` / `v08a` in the arm too, β 0.10 | 0.941 | −0.001 |
| 12 | + our c02 arm at β 0.20 | 0.939 | −0.003 |
| **16** | **per-label outer CoAtNet map → flat 0.60**, our arm not run | **0.940** | **−0.002** |

1. **0.940 is the public price of the flat map, not a defect.** The anchor's per-label map (LatMen 1.00 / ACL, LatOA,
   Fracture 0.75 / MedMen 0.80) was tuned *on this leaderboard* by the public authors; their own ladder priced it at
   +0.001–0.002 ("why public forks stop at 0.941"), we measured +0.002. Whether that is fitted noise that gives itself back
   privately is exactly what #16 hedges — the public number cannot tell, and −0.002 is under the 0.005 floor either way.
2. **Why no submission of ours exceeds 0.942.** (a) The anchor is a superset of the best public notebook (2026-09-22 night
   entry) — no public component is missing. (b) Our members are 0.86–0.87 OOF-vs-teacher, ≈ 0.89–0.90 solo on the LB by
   the measured +0.02–0.03 offset, while the stack's own members are 0.90–0.928 solo; a weaker vote that is also
   *correlated* with the stack (our c02 lane is the stack's own CoAtNet + window-attention family) can only perturb its
   ranks, and every non-zero reading is negative and grows with β (#13 0.000 → #14 −0.001 → #12 −0.003). (c) At β 0.10 a
   vote correlated ≈ 0.9 with the anchor overturns almost no pairwise orderings, which is why #13 is *exactly* the
   anchor. (d) 690 teams sit at 0.941–0.942 for the same reason; everything ≥ 0.945 is private members trained to ≥ 0.92
   solo (A6000/H100 boxes, several families). S1 moved our best single from 0.8683 to 0.8730 — a tenth of the gap to a
   member the stack would notice.
3. **Consequence for S2:** the retrained `v09a` (8 epochs, both S1 knobs) ships and is submitted at β 0.10 as planned, with
   the pre-registered read **≥ 0.947 = our arm counts, 0.940–0.946 = 🔁, ≤ 0.939 = the arm hurts**; the honest expectation
   is 0.942 ± 0.001. A public gain needs a member the stack does not already hold at ≥ 0.92 solo, which is a different
   input representation or self-training (P-23 #3 / #4, P-17), not another c02 CoAtNet.

**Verdict: 🔁 INCONCLUSIVE for #16 as a score (−0.002 < 0.005); ✅ FINDING for the mechanism (the public frontier is a
member-quality wall at ≈ 0.90 solo, not a blend-weight problem). #16 is the validated flat-weights candidate for the second
final slot (brainstorm).**

### 2026-09-23 — S2 production retrain (`rsna-knee-train` v26, 2.64 h, both T4s): `v09a` = 8 ep + the S1 knobs, gold-58 SWA **0.8922** (16-ep: 0.8768); `v08a` = 8 ep, **0.8850** (16-ep: 0.8816) ✅ the 8-epoch regime · gold deltas direction-only · LB ⏳ (#17)

Both production members retrained under `PROD` (all 4,349 report-labelled studies, 8 epochs, SWA of the last three EMA
snapshots, `ckpt_policy="last"`), one per GPU (P-31): `v09a` = CoAtNet-1 @224, c02, window_attn, `lr_backbone` 1e-4, **plus the
S1 knobs `batch_studies=2, grad_accum=2, aug="light"`** (P-32 / P-33, both 🔁 but same-direction — experiments.md 2026-09-23
"S1 A/B"); `v08a` = DINOv2-S @224, same regime, unchanged recipe. Kaggle smoke v25 and a local CPU smoke (MODE train sed'd)
green first; pushed 09:05 on Tian's go; both children rc 0, `-> {arm}_fold0_best.pt = SWA`. Outputs `artifacts/kaggle_out/train_v26/`.

| | 16-epoch (2026-09-22, train v19 / folds v6) | **8-epoch S2 (train v26)** | Δ gold-58 |
|---|---|---|---|
| `v09a` SWA | 0.8768 (last EMA 0.8762; peak epoch 5 0.8925, then −0.016) | **0.8922** (last EMA 0.8934; epochs 3–7: 0.8726 · 0.891 · 0.891 · 0.8925 · 0.8934) | +0.015 |
| `v08a` SWA | 0.8816 (last EMA 0.8802; peak epoch 6 0.8944, then −0.014) | **0.8850** (last EMA 0.8848; 0.8789 · 0.8809 · 0.8833 · 0.8836 · 0.8848) | +0.003 |
| wall | 5.12 h + 2.38 h, two sessions | **2.64 h, one session** (`v09a` 19 min/epoch at 0.26 s/study, 6.84 GiB; `v08a` 8.8 min/epoch at 0.12 s/study, 1.56 GiB) | −4.9 session-hours |

1. **The 8-epoch budget behaves as P-29 predicted on the production data**: neither curve peaks and drifts any more — both
   are still rising at epoch 7, so the SWA over epochs 5–7 averages the *best* part of the run instead of the tail past the
   peak. The 16-epoch members were built ≈ 0.015 past their gold peak; these are not.
2. **The gold-58 deltas (+0.015 / +0.003) are direction only** — the gold floor is 0.05 (per-label SE ≈ 0.09) and gold rows
   are training data under `train_all` (weight 8), so this number is optimistic by construction. It cannot separate the
   epoch budget from the S1 knobs on `v09a` either (two changes, one number; the fold-0 A/B already priced the knobs at
   +0.005). The measurement that counts is the fork at β 0.10 vs 0.942 (#17).
   **CORRECTED 2026-09-27:** gold rows are *not* trained under `train_all` — `split_studies` (`kaggle_pipeline.py`) trains
   `is_gold == 0` only and the 58 gold rows are the held-out validation (`window_head_test.py`: "train 7 non-gold, val … gold");
   production members' gold-58 is held out, not optimistic by construction (found by the 2026-09-27 proposals review).
3. **P-31 in production**: the pair that took 7.5 session-hours in two sessions took 2.64 h in one — 2.8× the arms per quota hour.
4. Shipped at 11:44 as new versions of the Datasets `tiankljucanin/rsna-knee-ckpt-v09a` / `-v08a` (`_best.pt` = SWA +
   gold-58 `_oof.csv`; `kaggle datasets status` ready, files re-listed); the fork mounts them unversioned, so #17 reads the
   S2 members. The 16-epoch checkpoints survive only in `artifacts/kaggle_out/train_v19/` and `folds_v6/`.

**Verdict: ✅ KEEP the 8-epoch production regime (the P-28 recipe is now `PROD` = 8 ep + SWA 5–7, CoAtNet member with the S1
knobs); 🔁 the gold-58 gains (sub-floor, direction only); ⏳ the LB (#17, fork v8, β 0.10 vs 0.942 — ≥ 0.947 = our arm counts;
honest expectation 0.942 ± 0.001, entry "#16 … why nothing of ours moves 0.942").**

**READ 2026-09-23 (20:00): #17 = 0.941 → 🔁 INCONCLUSIVE** — −0.001 vs #13 / #15 and identical to #14 (the 16-epoch members): the
epoch budget and the S1 knobs left the fork's public read unchanged at β 0.10. Entry "Submission #17 read 0.941".

### 2026-09-23 — Round 2 (`rsna-knee-folds` v8, 3.24 h, both T4s): `v09d` (P-34, backbone LR 3e-5) **0.8596** ❌ HARMFUL vs `v09c` 0.8730 · `v08c` (P-35, DINOv2-S + light aug) **0.8650** vs `v08w` 0.8648 🔁 INCONCLUSIVE · `v09e` (P-36) dropped by rule

**Setup.** Pushed 12:45 on Tian's go, COMPLETE 16:05 (11,671 s wall, both children rc 0, 6.84 GiB peak). `v09d` = the
`v09c` recipe (CoAtNet-1 @224, c02 `window_attn`, 24 random train windows, 8 ep, `best_oof`, `batch_studies=2,
grad_accum=2, aug="light"`) with `lr_backbone=3e-5` (the public 0.928 member's LR) on `cuda:0`; `v08c` = the `v08w`
recipe (DINOv2-S @224) + `aug="light"` on `cuda:1`. Read on all 882 fold-0 studies, hard = `y__ > 0.5` (the unchanged
LLM teacher), vs the controls `v09c` 0.8730 (`train_v23`) and `v08w` 0.8648 (`v17`). Outputs `artifacts/kaggle_out/folds_v8/`.

**`v09d` vs `v09c`** — 10/12 labels down, `best_oof` = epoch 7; the curve 0.739 / 0.805 / 0.831 / 0.845 / 0.852 / 0.857 /
0.859 / 0.860 is still rising at +0.0008/epoch, i.e. the hybrid *under-trains* at 3e-5 in 8 epochs (its loss 0.42 at epoch 7
vs `v09c`'s 0.40):

| label | v09c | v09d | delta |
|---|---|---|---|
| ACL | 0.8895 | 0.8935 | +0.0040 |
| MCL | 0.8100 | 0.7820 | -0.0281 |
| Medial Meniscus | 0.9198 | 0.8980 | -0.0219 |
| Lateral Meniscus | 0.8561 | 0.7971 | -0.0590 |
| Medial OA | 0.8793 | 0.8679 | -0.0115 |
| Lateral OA | 0.8362 | 0.8201 | -0.0161 |
| PF OA | 0.8193 | 0.8128 | -0.0065 |
| Effusion | 0.8637 | 0.8553 | -0.0084 |
| Synovitis | 0.8888 | 0.8942 | +0.0053 |
| Baker's | 0.9049 | 0.8964 | -0.0085 |
| Contusion | 0.8737 | 0.8675 | -0.0061 |
| Fracture | 0.9345 | 0.9309 | -0.0037 |
| **macro** | **0.8730** | **0.8596** | **-0.0134** (2/12 up) |

**`v08c` vs `v08w`** — 6/12 up, +0.0002: augmentation is worth nothing on the DINOv2 arm (on the CoAtNet, P-33 read +0.004,
also 🔁):

| label | v08w | v08c | delta |
|---|---|---|---|
| ACL | 0.8648 | 0.8732 | +0.0083 |
| MCL | 0.8226 | 0.8228 | +0.0002 |
| Medial Meniscus | 0.9031 | 0.9101 | +0.0070 |
| Lateral Meniscus | 0.8266 | 0.8309 | +0.0042 |
| Medial OA | 0.8782 | 0.8774 | -0.0009 |
| Lateral OA | 0.8330 | 0.8286 | -0.0043 |
| PF OA | 0.8249 | 0.8189 | -0.0060 |
| Effusion | 0.8555 | 0.8489 | -0.0067 |
| Synovitis | 0.8877 | 0.8841 | -0.0036 |
| Baker's | 0.8935 | 0.8959 | +0.0024 |
| Contusion | 0.8743 | 0.8832 | +0.0089 |
| Fracture | 0.9136 | 0.9059 | -0.0077 |
| **macro** | **0.8648** | **0.8650** | **+0.0002** (6/12 up) |

**Verdicts.** P-34 ❌ **HARMFUL** (−0.0134, 1.7× the floor the wrong way) — at 8 epochs the public LR is not our recipe's
gap; by its own pre-registered rule this **drops `v09e` (P-36, 16 ep × 3e-5)** — 16 epochs might close part of the gap
(the curve was still rising), but the rule was written before the read and stands; re-openable on Kaggle quota for 1.7 h
if nothing better competes for it. P-35 🔁 **INCONCLUSIVE** (+0.0002). **With P-32 (BN batch ≈ 0), P-33 (+0.004),
P-29 (16 ep harmful) and now P-34, every recipe knob between our c02 CoAtNet and the public 0.928 member is measured
and none clears the floor — the recipe is exhausted as a lever; the target source is not (next entry).**

### 2026-09-23 — RunPod arms on a 4090 (member-strength plan Task 10): `v09f` (P-37 `pos_weight`) **0.8717** 🔁 INCONCLUSIVE · `v09s` (P-38 self-distillation) **0.8839** ✅ KEEP (+0.0109, 12/12 labels up) · `v09e` (P-36) not run

**SUPERSEDED 2026-09-24 (the `v09s` ✅):** the +0.0109 was measured as OOF against the LLM targets, and the self-distillation table is
the rank-mean of models trained on those same targets — the student agreed with the teacher better without moving toward the truth; the
production twin `v09t` read **0.917 vs 0.918** on the public LB (#19). P-38 → ❌; see entry 2026-09-24 "Submission #19" and traps 39.

**Pod.** RTX 4090 (secure, EU-CZ-1, $0.74/h; the EUR-IS-2 4090s were out of stock), 32 vCPU / 125 GB, created 13:40
over the RunPod MCP, terminated 16:45 (≈ 3.1 h, ≈ $2.3). `/workspace` there is a MooseFS network mount, so the 36 GB c02
cache was pulled onto RAM (`/dev/shm`, 58 GB) — four shard pulls in parallel (~30 MB/s each; one died on an
`IncompleteRead` and was re-pulled after deleting the truncated blob — traps 38); 71 blobs verified against their csvs.
Throughput **0.05–0.06 s/study** (Kaggle T4 0.28), 3.3 min train + 0.9 min val per epoch, **≈ 34 min per 8-epoch
CoAtNet-1 fold-0 arm** (the 2026-08-30 pod: 50 min). Branch `member-strength` was pushed into the pod's clone over SSH;
`scripts/runpod_bootstrap.sh train <arm>` (now per-arm `artifacts/runpod_train_<arm>.py`, `ulimit -n`, unbuffered) ran
the arms back to back; `ship <arm>` published `rsna-knee-ckpt-v09f` / `-v09s` (`_best.pt` + `_oof.csv`, `datasets
status` ready). Read-outs on all 882 fold-0 studies, hard = `y__ > 0.5` (the unchanged LLM teacher), vs `v09c` 0.8730.

**`v09f` (P-37)** = `v09c` + `pos_weight_max=10` (per-label `clip((1−p)/p, 1, 10)` from the training rows; the public
0.924 member's loss). Printed `pos_weight [1, 10]: ACL 3.7, MCL 5.4, MedMen 1.5, LatMen 5.3, MedOA 1.8, LatOA 2.8, PF OA 1.2,
Effusion 1.0, Synovitis 6.9, Baker's 3.1, Contusion 4.8, Fracture 10.0`. Curve 0.744 / 0.807 / 0.835 / 0.854 / 0.865 /
0.870 / 0.8715 / 0.8717; gold 0.910 (n=11). **−0.0013, 3/12 up → 🔁 INCONCLUSIVE**; the "rejected without testing"
reasoning (AUC reads ranks; a positive-term weight changes calibration, not ranking) is now a measurement:

| label | v09c | v09f | delta |
|---|---|---|---|
| ACL | 0.8895 | 0.8700 | -0.0195 |
| MCL | 0.8100 | 0.8210 | +0.0109 |
| Medial Meniscus | 0.9198 | 0.9184 | -0.0015 |
| Lateral Meniscus | 0.8561 | 0.8781 | +0.0220 |
| Medial OA | 0.8793 | 0.8749 | -0.0045 |
| Lateral OA | 0.8362 | 0.8272 | -0.0090 |
| PF OA | 0.8193 | 0.8089 | -0.0104 |
| Effusion | 0.8637 | 0.8600 | -0.0037 |
| Synovitis | 0.8888 | 0.8831 | -0.0057 |
| Baker's | 0.9049 | 0.9046 | -0.0003 |
| Contusion | 0.8737 | 0.8820 | +0.0084 |
| Fracture | 0.9345 | 0.9320 | -0.0026 |
| **macro** | **0.8730** | **0.8717** | **-0.0013** (3/12 up) |

**`v09s` (P-38)** = `v09c` trained on `yt = 0.5 · LLM + 0.5 · quantile_match(selfdistill_v1)` (`TEACHER_TABLES =
("selfdistill_v1",)` sed'd; `selfdistill_v1.csv` = per-label rank-mean of the `v09h` (pooled OOF 0.8625) and `v05g`
(0.8467) 5-fold OOF sets, 4,407 studies, gold-58 of the table alone 0.8733; `src/build_distill_table.py`); gold rows keep
their hard 0/1; the **evaluation** targets `y__*` are the unchanged 3-source LLM teacher, so this OOF is comparable with
every earlier arm. Curve 0.772 / 0.836 / 0.861 / 0.872 / 0.879 / 0.881 / 0.8835 / 0.8839 — ahead of `v09f` from epoch 1
(0.836 vs 0.807); gold 0.922 (n=11):

| label | v09c | v09s | delta |
|---|---|---|---|
| ACL | 0.8895 | 0.9005 | +0.0110 |
| MCL | 0.8100 | 0.8343 | +0.0242 |
| Medial Meniscus | 0.9198 | 0.9229 | +0.0030 |
| Lateral Meniscus | 0.8561 | 0.8586 | +0.0025 |
| Medial OA | 0.8793 | 0.8940 | +0.0146 |
| Lateral OA | 0.8362 | 0.8491 | +0.0129 |
| PF OA | 0.8193 | 0.8323 | +0.0130 |
| Effusion | 0.8637 | 0.8755 | +0.0118 |
| Synovitis | 0.8888 | 0.8988 | +0.0100 |
| Baker's | 0.9049 | 0.9195 | +0.0146 |
| Contusion | 0.8737 | 0.8846 | +0.0109 |
| Fracture | 0.9345 | 0.9370 | +0.0025 |
| **macro** | **0.8730** | **0.8839** | **+0.0109** (12/12 up) |

**Verdict: ✅ KEEP — +0.0109 macro (1.4× the 0.008 floor), 12/12 labels up, the largest and only-consistent gain of any
recipe change since `v09h`.** Reading: a student trained on *smoothed* targets (half its own out-of-fold ranks, half the
LLM blend) ranks the LLM teacher's hard labels better than a student trained on the LLM blend alone — target noise, not
optimisation, was the binding constraint, exactly the member-strength diagnosis (spec 2026-09-23). **Caveats:** (a) the
OOF-vs-teacher metric rewards agreement with the LLM teacher, and the self-distill table is itself a function of that
teacher, so part of the gain may be "learning the teacher's noise more smoothly"; the 58 gold rows (0.922 vs 0.919 for
`v09c`, n=11, floor 0.05) cannot tell; (b) the fold-1..4 models behind the table saw fold 0's *targets* (not its images)
— a second-order leak into the fold-0 read. **Both caveats are settled by the LB**: the plan's Task 12 trains the
production `v09a` on a mixed teacher and reads it solo against #18 (0.918). **Next:** the production retrain on
`("selfdistill_v1",)` can run *now* (Kaggle both T4s, ≈ 2.7 h, next week's quota) without waiting for the Raptor table;
the Raptor table (P-39) is the second, stronger teacher for the same mechanism.

`v09e` (P-36, 16 ep × 3e-5) was **not run**: P-34's `v09d` read < 0.865 (entry above) and the card's rule drops it.
Cards: P-36 ❌ dropped, P-37 🔁, P-38 ✅ (proposals.md). OOF csvs + per-label tables: `artifacts/kaggle_out/pod_v09f/`,
`pod_v09s/`; logs there too; the `v09s` checkpoint also at `artifacts/ckpt_pod/v09s/`.

### 2026-09-23 — Submission #18: the S2 `v09a` ALONE reads **0.918** public — one member of ours above our own 12-member blend (#11, 0.913) ✅ FINDING · the baseline for the distilled retrain

`rsna-knee-infer` v15 = `MODE="infer"`, `INFER_MEMBERS = ["v09a"]`, Dataset `rsna-knee-ckpt-v09a` (the S2 member:
CoAtNet-1 c02 `window_attn`, all 4,349 studies, 8 ep, SWA of the last 3 EMA, `batch_studies 2, grad_accum 2, aug light`,
gold-58 0.8922). Placeholder 14:33 → 14:35 (2.2 min; `infer members (1): v09a/fold0 … [epoch 7, score 0.8922]`,
`constant labels 0`); sent 14:36, ref 56493264, scored **0.918** at ≈ 16:00.

**What it says.** (1) The fold-0-OOF → LB offset (+0.02–0.03, measured on fold-0 members) under-predicts an *all-data
SWA* member by ≈ 0.02: the "why nothing of ours moves 0.942" entry put our members at ≈ 0.89–0.90 solo; the S2 `v09a`
is **0.918**, inside the public stack's member range (0.90–0.928) and 0.006 under its best single member (Raptor
CoAtNet-2, 0.924). (2) A single model of ours beats the 12-member blend we submitted as #11 (0.913) — the c01-era members
in that blend were dragging it. (3) The member-quality wall is therefore thinner than estimated; whether a 0.918 member
*counts* in the 0.942 stack is exactly what #17 (fork v8, β 0.10, pending) reads. **Rule for the distilled retrain
(P-39 / Task 12), pre-registered: solo LB ≥ 0.923 = ✅ the teacher works (retrain `v08a` the same way, rebuild the fork);
0.919–0.922 = 🔁; < 0.918 = ❌.** Submissions table row 18.

### 2026-09-23 — Raptor teacher pass (P-39): the 0.942 notebook's Raptor branch as `kaggle/rsna-knee-teacher` — smoke green, spike **100 studies at 5.1 s/study → full pass ≈ 6.2 GPU-h** ✅ FEASIBLE · teacher vs LLM teacher macro AUC 0.914 on the 100

**Built (member-strength plan Tasks 7–8).** `src/build_teacher_pass.py` slices `notebook_score_0.942.ipynb` verbatim
(cell 16 helpers 104..195 + `rsna_phase` 274..294, cell 12 asset finder 123..165 + the memoised finder 308..330, cell 14
dense sampler, cell 45 Raptor branch 2..8 + 14..628 with two token patches asserted once — `_KE_TEACHER_OUTPUTS` /
`_KE_TEACHER_IDS` expose the running arrays) behind our chunk preamble (`SHARD` / `N_SHARDS` / `LIMIT`; the 4,349
report-labelled UIDs sorted and sharded, **gold excluded**; `chunk/test.csv` + `test_series.csv` + `sample_submission.csv`
in the competition schema; `chunk/test_series` **and** `test_images` symlinked to the mounted `train_series/`;
`RSNA_COMP_ROOT` → the chunk in `/tmp`, never under `/kaggle/working`), a 5-min flush thread (atomic write; suppressed
once `raptor_raw.npz` exists so the runner's in-place NaN fill can never be published as "complete"), and a final writer
(`raptor_teacher_shard{K}.npz` with `study_uids`, `raw_probabilities` 4 × N × 12, `view_names`, `view_weights`,
`checkpoint_sha256`; `.csv` with the view-weighted mean; `teacher_receipt.json`). Resume: `done_uids()` / `load_prior()`
read any mounted shard or partial npz (skipping the competition tree), prior complete rows are carried into the new
outputs; `--slug` / `--kernel-source` build a sibling kernel that mounts the previous run (a kernel cannot mount its own
output — traps 31). `src/merge_teacher.py` merges shards (partial files without weights borrow a sibling's; `.tmp.npz`
skipped; duplicate UID fatal; 4,349 expected unless `--allow-partial`) → `artifacts/teacher/raptor_teacher.csv`.
Checks: `src/teacher_pass_test.py` (32 checks) and a byte-for-byte assertion that the extracted code equals the notebook.

**Runs.** v1 (LIMIT 6) **red** in cell 1 — the preamble required `train_images/`; the competition mounts image trees
`train_series/` + `test_series/` (traps 37); fixed with a shallow-glob root finder. v2 (LIMIT 6) **green**: 6/6 studies,
62 s, both T4s (`balanced arm groups cuda:0=[0,2], cuda:1=[1,3]`), probabilities varied in (0, 1). v3 (LIMIT 100)
**green**: 100/100, 0 failed, **511 s = 5.11 s/study including ≈ 13 s setup**, k_eval 94. Outputs
`artifacts/kaggle_out/teacher_smoke_v2/`, `teacher_spike_v3/`; `merge_teacher.py --allow-partial` →
`artifacts/teacher/raptor_spike100.csv`.

**Plausibility (the 100 studies, Raptor's view-weighted mean vs the LLM blend, hard = LLM > 0.5):**

| label | LLM positive rate | Raptor mean | corr | AUC (Raptor vs hard LLM) |
|---|---|---|---|---|
| ACL | 0.23 | 0.42 | 0.84 | 0.946 |
| MCL | 0.16 | 0.46 | 0.52 | 0.772 |
| Medial Meniscus | 0.44 | 0.46 | 0.86 | 0.948 |
| Lateral Meniscus | 0.16 | 0.44 | 0.70 | 0.972 |
| Medial OA | 0.36 | 0.38 | 0.78 | 0.912 |
| Lateral OA | 0.26 | 0.37 | 0.71 | 0.904 |
| PF OA | 0.44 | 0.40 | 0.75 | 0.912 |
| Effusion | 0.47 | 0.49 | 0.76 | 0.906 |
| Synovitis | 0.25 | 0.52 | 0.54 | 0.884 |
| Baker's | 0.25 | 0.32 | 0.63 | 0.897 |
| Contusion | 0.25 | 0.46 | 0.81 | 0.965 |
| Fracture | 0.15 | 0.47 | 0.73 | 0.949 |
| **macro** | | | | **0.914** |

The public teacher ranks the LLM teacher's labels at 0.914 on 100 report-only studies (n small, direction only) and sits
at a *more positive* operating point on every rare label (Fracture 0.47 vs 0.15) — exactly why the targets path
quantile-matches each table onto the LLM blend before mixing (spec §2). **Shard plan:** 4,349 × 5.11 s = **6.2 h** in
one session (guard 8.0 h, margin 1.8 h) or two shards × 3.1 h in the two GPU slots (`--shard k --n-shards 2`, the second
via `--slug tiankljucanin/rsna-knee-teacher-b`); **runs after Saturday's quota reset** (≈ 3.4 h left this week). Then
`merge_teacher.py artifacts/kaggle_out/teacher_s*/raptor_teacher_shard*.npz` → `raptor_teacher.csv` → new version of
Dataset `rsna-knee-teacher-tables` → the production retrain with `TEACHER_TABLES = ("raptor_teacher",)` (Task 12) — or
`("selfdistill_v1", "raptor_teacher")`, both quantile-matched and averaged.

### 2026-09-23 — Submission #17 read **0.941** (fork v8: the S2 8-epoch production members `v09a` 0.8922 / `v08a` 0.8850 at β 0.10) 🔁 INCONCLUSIVE · = #14 (0.941, the 16-epoch members) · −0.001 vs #13 / #15 (0.942)

**What was read.** `rsna-knee-fork` v8 (ref 56489906, sent 11:54, scored by 20:00): #14's graph and blend with the two production
members retrained under the 8-epoch regime (entry "S2 production retrain") — `v08w` + 5-fold `v09h` + `v09a` + `v08a`, one vote
each, β 0.10. Pre-registered rule: ≥ 0.947 ✅ our arm counts / 0.940–0.946 🔁 / ≤ 0.939 ❌ the arm hurts. Public LB **0.941**.

**What it says.** (1) 🔁 by the rule, and −0.001 is 0.2× the 0.005 LB floor — the same number #14 read with the 16-epoch members
(gold-58 0.8768 / 0.8816), so +0.015 gold-58, the 8-epoch budget and the S1 knobs on both members moved the fork by nothing the
public LB can see. (2) Read with #18 (the same `v09a` **alone** = 0.918): at β 0.10 a 0.918-solo member that is correlated with the
stack (ρ ≈ 0.9 to its CoAtNet members) only perturbs the anchor's ranks — the fork is not an instrument for member quality; the
**solo submission is** (floor 0.005 against a 0.918 baseline). (3) P-28's LB question is closed: the production regime is the right
way to *build* a member (0.918 solo), and no member of the current recipe family changes the 0.942 stack at β 0.10. The route to
a public gain stays the one the member-strength programme took: a member trained on a *different teacher* (P-38 ✅ on fold 0,
P-39 the Raptor pass), read solo against 0.918 first (Task 12's rule: ≥ 0.923 ✅ / 0.919–0.922 🔁 / < 0.918 ❌), and only then in
the fork.

**Verdict: 🔁 INCONCLUSIVE (pre-registered band). P-28 → ✅ as a training regime (0.918 solo), 🔁 as a stack contribution.
Final-selection candidates unchanged: #13 / #15 (0.942) and the flat-map hedge #16 (0.940).** Submissions table row 17.

### 2026-09-23 — `v09t`: the production `v09a` recipe on the self-distilled targets (P-38 in production), RunPod 4090, 35 min — gold-58 SWA **0.9009** (S2 `v09a`: 0.8922) · direction only · LB ⏳ (#19, solo vs 0.918)

**Setup.** `ARMS` gains `v09t` = the S2 `v09a` dict under its own version name (final-review item: no collision with Dataset
`rsna-knee-ckpt-v09a`, a `_last.pt` resume or the fork's member slot); the `v09s` guard became `DISTILLED_ARMS = ("v09s", "v09t")`
(either is refused without `TEACHER_TABLES`; probed locally). Kaggle smoke `rsna-knee-train` v28 (`ARM_ONLY = "v09t"`,
`TEACHER_TABLES = ("selfdistill_v1",)`, 4 min) green: `teacher table selfdistill_v1: 4407 studies`, `training targets = (1 - 0.5) * LLM +
0.5 * quantile-matched`. Real run on a RunPod RTX 4090 (Tian chose the pod over Kaggle quota and capped it at one hour): one chained job —
four parallel cache pulls (36 GB in 6 min, ≈ 85 MB/s aggregate), blob verification (71 blobs, 4,407 studies, 0 bad), then
`RSNA_TEACHER_TABLES='("selfdistill_v1",)' bash scripts/runpod_bootstrap.sh train v09t` — `train 4349 / val 58 studies [train_all: val =
gold rows]`, 8 epochs at 4.4 min each (≈ 0.06 s/study), SWA of epochs 5–7, `-> v09t_fold0_best.pt = SWA`; 41 min from job start to
checkpoint. Shipped as Dataset `rsna-knee-ckpt-v09t` (ready 23:22); `rsna-knee-infer` v16 solo (`infer members (1): v09t/fold0 … score
0.9009`, 3 placeholder studies, `constant labels 0`) → **submission #19** (ref 56504077, 23:25). Outputs `artifacts/kaggle_out/pod_v09t/`
(gold-58 `_oof.csv`, train + job logs).

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | SWA 5–7 |
|---|---|---|---|---|---|---|---|---|---|
| `v09t` gold-58 EMA (self-distilled targets) | 0.798 | 0.866 | 0.886 | 0.893 | 0.898 | 0.901 | 0.901 | 0.902 | **0.9009** (CI95 0.867–0.929) |
| `v09a` S2 gold-58 EMA (LLM targets, same recipe) | — | — | — | 0.873 | 0.891 | 0.891 | 0.8925 | 0.8934 | **0.8922** |

Per-label gold-58 (SWA): ACL 0.949 · MCL 0.930 · Medial Meniscus 0.954 · Lateral Meniscus 0.839 · Medial OA 0.977 · Lateral OA 0.803 ·
PF OA 0.799 · Effusion 0.965 · Synovitis 0.754 · Baker's 0.980 · Contusion 0.942 · Fracture 0.919.

**What it says.** (1) +0.0087 on gold-58 is direction only (floor 0.05; under `train_all` the gold rows are training data at weight 8,
so the number is optimistic by construction) — but it has the sign of the fold-0 read (`v09s` +0.0109 vs `v09c`, 12/12 labels) and the
curve is still rising at epoch 7, as `v09a`'s was. (2) The distilled targets change nothing at inference (same architecture, same
cache), so a ✅ read drops straight into `v09a`'s slot in the fork. (3) The measurement that counts is #19 vs #18 (0.918),
pre-registered: **≥ 0.923 ✅ self-distillation transfers to the production member (retrain `v08a` the same way as `v08t`, both
distilled members into the fork at β 0.10) / 0.919–0.922 🔁 / < 0.918 ❌ (the production member keeps the LLM targets; the Raptor
teacher, P-39, is the remaining lever).**
**CORRECTED 2026-09-27:** gold rows are not trained under `train_all` (`split_studies` trains `is_gold == 0` only) — the gold-58
read above is held out, not optimistic by construction (see the same correction in the S2 entry).

**Verdict: ✅ the run (the production regime end to end in 35 min on a 4090, ≈ $0.5 of GPU for the training itself); ⏳ PENDING the
LB (#19).** Submissions table row 19.

**READ 2026-09-24 (00:07): #19 = 0.917 → ❌ by the pre-registered rule** (< 0.918; −0.001 vs #18, 0.2× the LB floor): the self-distilled
production member is indistinguishable from the LLM-target member on the public LB. Entry "Submission #19".

### 2026-09-24 — Submission #19: the self-distilled `v09t` alone reads **0.917** vs the same recipe on the LLM targets (#18, 0.918) ❌ DEAD END for self-distillation in production · ✅ FINDING: the fold-0 OOF-vs-LLM-teacher metric cannot judge a target-source change

**What was read.** `rsna-knee-infer` v16 (ref 56504077, sent 23:25, scored by 00:07): `v09t` = the S2 `v09a` recipe trained on
`0.5 · LLM + 0.5 · quantile-matched selfdistill_v1` (entry "`v09t`"), one member, rank of one model. Pre-registered vs #18 (0.918):
≥ 0.923 ✅ / 0.919–0.922 🔁 / < 0.918 ❌. Public LB **0.917**.

**What it says.** (1) **❌ by the rule** — and honestly a null result: −0.001 is 0.2× the 0.005 floor, so the distilled member is the
LLM-target member with a different seed as far as the public LB can tell. It did not deliver the +0.005 the fold-0 read implied.
(2) **The fold-0 instrument was the problem, not the training.** `v09s` gained +0.0109 OOF *against the LLM targets* (12/12 labels) and
`v09t` gained +0.0087 on gold-58 — but the self-distillation table is the rank-mean of two models that were themselves trained on the
LLM targets, so the student regressed toward a smoothed copy of the LLM signal and agreed with the LLM teacher *better* without moving
toward the truth. An OOF score against the LLM targets rewards exactly that agreement; a **target-source change can only be judged by
held-out truth: gold-58 direction (floor 0.05) and the solo LB (floor 0.005)** — never by OOF vs the teacher it was distilled from
(traps 39). The 12/12 sign consistency, which the noise-floor rule treats as evidence, is consistent with the same artefact.
(3) **Where this leaves the programme.** Every recipe knob (P-29 / P-32 / P-33 / P-34 / P-37) and now self-distillation are
measured on the LB at ±0.001 around 0.918 for a single c02 CoAtNet-1 member; the wall is the *teacher's* information, which our own
OOF cannot add to. The one teacher with more information than ours is the Raptor branch (public LB 0.924 solo, gold-58 0.905–0.917
held out vs our LLM teacher's 0.8948) — **P-39 stays the lever**, judged by gold-58 direction + a solo LB read, exactly as its card
says. (4) Cost of the read: 41 min of a 4090 (≈ $0.5) + one submission — cheaper than the Kaggle T4 route (2.7 h of quota).

**Verdict: ❌ DEAD END — self-distillation from our own OOF does not improve the production member on the public LB (P-38 → ❌ in
production; its fold-0 ✅ is withdrawn as an instrument artefact). ✅ FINDING — target-source changes are read by gold-58 + solo LB
only (traps 39). `v09a` stays the production member; the `v08t` retrain and the distilled-fork rebuild are dropped.** Submissions
table row 19.

### 2026-09-24 — Raptor teacher pass, shards 0/3 and 1/3 (`rsna-knee-teacher` v4, `rsna-knee-teacher-b` v1): **2,900 of 4,349 studies, 0 failed, 5.6–6.2 s/study** ✅ the pass works at scale · Raptor vs the hard LLM teacher macro AUC **0.906** on 2,900 (plausibility, not a verdict)

**Setup.** The 4,349 gold-free training studies split three ways (sorted UIDs; `build_teacher_pass.py --shard k --n-shards 3`; the
two-shard plan needed 6.2 session-hours against 6.3 h of quota and a quota kill keeps few 4-view-complete rows — traps-39-era
arithmetic in the 14:55 handoff). Shards 0 and 1 pushed 14:54 into the two GPU slots, the Raptor branch verbatim (3 public CoAtNet-2
checkpoints, 4 views, 94 windows, both T4s), partial flush every 5 min, `raptor_teacher_shard{k}.npz` (4 × 1,450 × 12 raw probabilities
+ view names / weights + checkpoint sha256) and a receipt each. Outputs `artifacts/kaggle_out/teacher_s0/`, `teacher_s1/`.

| shard | studies | failed | s/study | wall | receipt |
|---|---|---|---|---|---|
| 0/3 (`rsna-knee-teacher` v4) | 1,450 | 0 | 6.20 | 2.50 h | `k_eval 94`, 1,450/1,450 rows finite in all four views |
| 1/3 (`rsna-knee-teacher-b` v1) | 1,450 | 0 | 5.56 | 2.24 h | same |
| spike (v3, 2026-09-23) | 100 | 0 | 5.11 | 0.14 h | — |

The partial-flush curve confirms the guard-stop caveat: at 1.75 h only 627 of shard 1's 1,450 studies had all four views (the runner
covers the views sequentially per GPU), so a session killed at 80 % of its time would have kept ≈ 40 % of its rows.

**Plausibility on the merged 2,900 rows** (`merge_teacher.py … --allow-partial --out artifacts/teacher/raptor_partial_s01.csv`, then
`teacher_plausibility.py`: Raptor probability vs `LLM blend > 0.5`, report-only rows, no gold row in the table):

| label | AUC | ρ | Raptor mean | LLM pos rate | label | AUC | ρ | Raptor mean | LLM pos rate |
|---|---|---|---|---|---|---|---|---|---|
| ACL | 0.912 | 0.64 | 0.43 | 0.21 | PF OA | 0.906 | 0.75 | 0.42 | 0.45 |
| MCL | **0.844** | 0.50 | 0.47 | 0.16 | Effusion | **0.845** | 0.77 | 0.50 | 0.59 |
| Medial Meniscus | 0.945 | 0.82 | 0.48 | 0.40 | Synovitis | **0.847** | 0.73 | 0.52 | 0.12 |
| Lateral Meniscus | 0.947 | 0.62 | 0.47 | 0.15 | Baker's | 0.950 | 0.56 | 0.38 | 0.25 |
| Medial OA | 0.924 | 0.71 | 0.40 | 0.37 | Contusion | 0.903 | 0.67 | 0.48 | 0.17 |
| Lateral OA | 0.905 | 0.64 | 0.40 | 0.26 | Fracture | 0.949 | 0.46 | 0.47 | 0.07 |

**Macro 0.906** (spike: 0.914 on 100 studies; shard 1 alone 0.903).

**What it says.** (1) The pass is sound at scale — no failed study, every row finite in all four views, the view / label order is
right (no label near chance), and the per-study cost is 5.6–6.2 s (the spike's 5.1 s was optimistic by up to 20 %: budget 2.5 h for
shard 2). (2) Raptor and the LLM teacher agree at 0.906 macro — enough to be the same task, far from a copy: on the 2,900 studies they
disagree on real cases, which is exactly what a second teacher must do (the self-distillation table, by contrast, was a smoothed copy of
the LLM signal — traps 39). Agreement is lowest on MCL, Effusion and Synovitis, the three findings where the reports themselves are
least specific; whether Raptor or the LLM is *right* there is what the solo LB of the distilled member will say, not this table.
(3) Raptor's operating point is far more positive than the LLM blend on the rare labels (Fracture mean 0.47 vs a 7 % positive rate,
MCL 0.47 vs 16 %, Synovitis 0.52 vs 12 %) — the kernel's `quantile_match` onto the LLM blend removes that before mixing; a plain
probability average would have flooded the rare labels with positives.

**Verdict: ✅ the pass (2/3 done; shard 2/3 after Saturday); the teacher itself is ⏳ until Task 12's solo read vs 0.918 — judged by
gold-58 direction + solo LB only (traps 39).**

### 2026-09-26 — Raptor teacher pass complete: shard 2/3 (`rsna-knee-teacher` v5) green, **4,349 studies merged, 0 failed**, Raptor vs the hard LLM teacher macro AUC **0.9075** ✅ the table (P-39) · published in `rsna-knee-teacher-tables`

**Setup.** Shard 2/3 = the v4 render with `SHARD = 2` (the only diff; `teacher_pass_test.py` green), the last 1,449 of the sorted
gold-free studies; pushed 17:26, **queued 3 h 10 min** on Kaggle T4 capacity (traps 41), ran 20:35 → 23:39. Output
`artifacts/kaggle_out/teacher_s2/` (receipt `studies 1449`, `failed_uids []`, `k_eval 94`, **7.50 s/study, 3.02 h** — the slowest
session of the three: 6.20 / 5.56 / 7.50 s, so a 1,450-study shard needs up to 3 h, not 2.5). Partial-flush curve as in shard 1:
0 four-view-complete rows until 1.83 h, then ≈ 110 per 5 min.

**Merge** (`merge_teacher.py s0 s1 s2 --expect-n 4349 --out artifacts/teacher/raptor_teacher.csv`): 1,450 + 1,450 + 1,449 studies,
every one complete in all four views, 4,349 rows, **0 gold rows** (the shards are gold-free by construction).

**Plausibility on all 4,349** (`teacher_plausibility.py`: Raptor probability vs `LLM blend > 0.5`):

| label | AUC | ρ | Raptor mean | LLM pos rate | label | AUC | ρ | Raptor mean | LLM pos rate |
|---|---|---|---|---|---|---|---|---|---|
| ACL | 0.913 | 0.63 | 0.42 | 0.21 | PF OA | 0.904 | 0.75 | 0.42 | 0.46 |
| MCL | **0.855** | 0.50 | 0.47 | 0.15 | Effusion | **0.847** | 0.77 | 0.50 | 0.59 |
| Medial Meniscus | 0.943 | 0.82 | 0.48 | 0.40 | Synovitis | **0.847** | 0.73 | 0.52 | 0.12 |
| Lateral Meniscus | 0.947 | 0.62 | 0.48 | 0.15 | Baker's | 0.952 | 0.56 | 0.38 | 0.25 |
| Medial OA | 0.925 | 0.70 | 0.40 | 0.37 | Contusion | 0.902 | 0.66 | 0.48 | 0.17 |
| Lateral OA | 0.903 | 0.63 | 0.40 | 0.26 | Fracture | 0.952 | 0.45 | 0.46 | 0.07 |

**Macro 0.9075** (2,900 rows: 0.906; spike: 0.914). The per-label picture is the 2,900-row one to ±0.011 (MCL 0.844 → 0.855): the
third shard is the same population, and the two readings of "a second opinion that disagrees on MCL / Effusion / Synovitis" stand.

**Published** as a new version of Dataset `tiankljucanin/rsna-knee-teacher-tables` (`raptor_teacher.csv` 1.29 MB beside
`selfdistill_v1.csv`; `datasets files` listed both at 23:40:29). `TEACHER_PATHS["raptor_teacher"]` resolves it on Kaggle and on the pod.

**Verdict: ✅ the table is complete and sound (a plausibility read, never a verdict on the teacher — traps 39); the teacher is ⏳ until
`v09r`'s solo read vs 0.918.**

### 2026-09-26 — `v09r`: the production `v09a` recipe on the Raptor-distilled targets (P-39, plan Task 12) — gold-58 SWA **0.9093** vs 0.8922 (direction only) · solo LB ⏳ #20

**Setup.** Arm `v09r` = the `v09a` dict under its own version name (in `SHIPPED_ARMS`; `DISTILLED_ARMS` pins it to
`("raptor_teacher",)` — traps 40), `TEACHER_TABLES = ("raptor_teacher",)`, `TEACHER_MIX = 0.5`: training target
`yt = 0.5 · LLM blend + 0.5 · quantile_match(Raptor)` on the 4,349 report-only rows, gold rows hard 0/1, evaluation targets unchanged.
Trained on RunPod pod `xwgxw1ai93sjow` (SECURE RTX 4090, EUR-IS-1, MooseFS `/workspace` → cache on `/dev/shm`) with the new
`scripts/runpod_chain.sh v09r` (`TEACHER_WAIT_MIN=40`: the job started 23:33, waited 7 min for the Dataset version, verified 71 blobs /
4,407 studies, trained 23:42 → 00:30 at 5.9 min/epoch, shipped `rsna-knee-ckpt-v09r` 00:30); log lines `teacher table raptor_teacher:
4349 studies`, `training targets = (1 - 0.5) * LLM + 0.5 * quantile-matched ['raptor_teacher']`. Pod 23:30 → 00:32 ≈ 1.03 h ≈ $0.76,
deleted. Logs + gold csv: `artifacts/kaggle_out/pod_v09r/`. The Kaggle smoke was skipped (the T4 queue stood at 3 h; the pod's first
minute shows the same lines, and the local smoke on the 2,900-row partial table had covered the code path).

**Gold-58 per label** (SWA checkpoints; `v09a` = S2 on the LLM targets, `v09t` = self-distilled, both from their shipped csvs):

| label | `v09a` | `v09t` | **`v09r`** | label | `v09a` | `v09t` | **`v09r`** |
|---|---|---|---|---|---|---|---|
| ACL | 0.915 | 0.949 | 0.953 | PF OA | 0.795 | 0.799 | 0.816 |
| MCL | 0.887 | 0.930 | 0.925 | Effusion | 0.969 | 0.965 | 0.974 |
| Medial Meniscus | 0.947 | 0.954 | 0.958 | Synovitis | 0.744 | 0.754 | **0.823** |
| Lateral Meniscus | 0.842 | 0.839 | 0.853 | Baker's | 0.973 | 0.980 | 0.960 |
| Medial OA | 0.986 | 0.977 | 0.967 | Contusion | 0.934 | 0.942 | 0.934 |
| Lateral OA | 0.816 | 0.803 | 0.805 | Fracture | 0.897 | 0.919 | **0.943** |
| **macro** | **0.892** | **0.901** | **0.909** | | | | |

**What it says (direction only — gold floor 0.05, per-label SE ≈ 0.09).** The Raptor-distilled member trains as cleanly as `v09t`
(monotone EMA curve to epoch 5, flat 5–7, SWA ≥ last EMA) and reads +0.017 over `v09a` on the held-out truth, 9/12 labels up; the two
largest moves are on Synovitis (+0.079) and Fracture (+0.046) — Synovitis is one of the three labels where Raptor disagreed most with
the LLM teacher (0.847), so if the gain is real it is where the second opinion was supposed to help. `v09t` read +0.009 here and then
−0.001 on the LB (traps 39), so none of this is a verdict. A second reason for caution: Raptor was trained on this same training set, so its table on
our 4,349 studies is largely in-sample for Raptor (P-39 caveat) — the student may be learning Raptor's fit of *its* training labels.

**Verdict: ⏳ PENDING — solo submission #20 (`rsna-knee-infer` v17, `INFER_MEMBERS = ["v09r"]`) vs #18 0.918: ≥ 0.923 ✅ (then `v08r` and
the fork at β 0.10) / 0.919–0.922 🔁 / < 0.918 ❌.**

**Read 2026-09-27: #20 = 0.927 → ✅ KEEP** (entry "Submission #20").

### 2026-09-27 — Submission #20: the Raptor-distilled `v09r` ALONE reads **0.927** public vs #18 `v09a` 0.918 → **+0.009 ✅ KEEP** · the first change of training targets that transfers to the LB (P-39)

`rsna-knee-infer` v17, `INFER_MEMBERS = ["v09r"]`, Dataset `rsna-knee-ckpt-v09r` (entry "`v09r`"); sent 2026-09-27 00:36, ref 56590282;
**0.927** by the 09:28 read (the laptop slept after the 01:02 poll, so the scoring time is not known).

**The comparison is clean.** #18, #19 and #20 are the same recipe, architecture, data, epochs, SWA window, inference kernel and test
decode — the S2 `v09a` dict under three version names — and differ **only in the training targets**:

| # | member | training targets | gold-58 SWA | public LB | vs #18 |
|---|---|---|---|---|---|
| 18 | `v09a` | LLM blend (3 public label tables) | 0.8922 | 0.918 | — |
| 19 | `v09t` | 0.5 · LLM + 0.5 · quantile-matched self-distillation (our own 5-fold OOF) | 0.9009 | 0.917 | −0.001 🔁 → ❌ by the rule |
| **20** | **`v09r`** | **0.5 · LLM + 0.5 · quantile-matched Raptor** (the public 0.942 notebook's CoAtNet-2 branch, 4 views, over our 4,349 report-only studies) | **0.9093** | **0.927** | **+0.009 ✅** |

**Verdict: ✅ KEEP** — +0.009 is 1.8× the 0.005 public-LB floor and clears the pre-registered 0.923 line (written 2026-09-23, before
the pass ran). One seed; the LB variance of a production member across seeds has never been measured, so the floor is the project's
rule, not a measured seed spread — but the sign also agrees with gold-58 (+0.017, 9/12 labels up; Synovitis +0.079, Fracture +0.046).

**What it says.**
1. **The target source was a binding constraint, and only a teacher with information our labels lack moves it.** Self-distillation
   (the same mechanism, our own OOF as the teacher) read −0.001; the Raptor teacher reads +0.009. The mixing code, the 0.5 mix and the
   quantile matching are identical in #19 and #20 — the difference is what the table knows.
2. **0.927 is at the top of the public stack's single-member range** (0.90–0.928; its best member 0.928, the Raptor branch's best
   single view 0.924 solo). A student trained on half-Raptor targets lands above Raptor's best single view, consistent with it
   combining two teachers (LLM + Raptor) rather than copying one.
3. **The in-sample caveat did not veto it.** The Raptor checkpoints were trained on this same training set (16 epochs, best-gold-epoch +
   top-3 SWA), so the teacher table is largely in-sample for Raptor (P-39 caveat) — whatever it carries (its own training labels,
   image-grounded smoothing), it is information the LLM blend did not have, and it generalised to the hidden test.
4. **Gold-58 direction is now 1 for 2 as a predictor** (`v09t` +0.009 gold → −0.001 LB; `v09r` +0.017 gold → +0.009 LB): traps 39
   stands — the solo LB decided both.
5. **Cost of the lever:** the pass 7.8 Kaggle GPU-h over three sessions (2.50 + 2.24 + 3.02 h) + a 3 h queue; the retrain 48 min on a
   RunPod 4090 (≈ $0.76 incl. setup); one submission.

**Next (pre-registered in P-39 "If it works", card P-40):** `v08r` (the `v08a` DINOv2 recipe on the same table) and the fork at β 0.10
with the Raptor-distilled members vs 0.942 (≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌) — #17 showed the fork at β 0.10 did not register
0.918-level members; a 0.927 member (≈ the best public member) is the first real test of whether one of ours counts in the stack.

### 2026-09-27 — `v08r`: the production `v08a` DINOv2-S recipe on the Raptor-distilled targets (P-40 step B) — gold-58 SWA **0.8981** vs `v08a` 0.8850 (direction only) · shipped, no LB read yet

**Verdict: ✅ the run (complete, shipped); gold-58 +0.013 = 🔁 direction only (floor 0.05); the member's worth is a solo or fork LB
read, not this number (traps 39).**

The S2 `v08a` recipe (DINOv2-S @224, c02 cache, `window_mode="random"` 24 train windows / all at eval, `window_attn`, all 4,349
report-labelled studies + 58 gold as the reported validation, 8 epochs, SWA of the last 3 EMA snapshots, batch 1 × accum 4, `aug
none`) trained on `TEACHER_TABLES=("raptor_teacher",)`, mix 0.5 — the same table and mix as `v09r`. **RunPod A100 SXM 80 GB**
(SECURE, US-MD-1, $1.59/h; every 24/48 GB card read NONE at 12:48 UTC) via `scripts/runpod_chain.sh v08r` with
`CACHE_ROOT=/dev/shm/cache` (MooseFS `/workspace`, 117 GB shm): job 12:52:01 → 13:22:57 UTC — 3 min inputs (4 parallel pulls,
`blobs 71, bad 0, studies 4407`, `raptor_teacher.csv: 4349 rows`), **3.2–3.3 min/epoch (0.04–0.05 s/study, 8 workers)**, ship; pod
12:49:54 → ≈ 13:26 (≈ 36 min ≈ $0.95), deleted, `list-pods` empty.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | SWA 5–7 |
|---|---|---|---|---|---|---|---|---|---|
| gold-58 (EMA) | 0.768 | 0.839 | 0.873 | 0.889 | 0.897 | 0.898 | 0.898 | 0.899 | **0.8981** (CI95 0.865–0.926) |

vs `v08a` (S2, LLM targets): SWA **0.8850** → **+0.013**; the curve plateaus from epoch 4 (`v08a` was still creeping up at epoch 7).
Per label (SWA): ACL 0.939 · MCL 0.880 · MedMen 0.952 · LatMen 0.824 · MedOA 0.972 · LatOA 0.797 · PF OA 0.821 · Effusion 0.924 ·
Synovitis 0.812 · Baker's 0.991 · Contusion 0.947 · Fracture 0.918. The CoAtNet twin moved +0.017 on gold-58 and +0.009 on the LB
(`v09r`); gold direction has predicted the LB 1 time in 2 (traps 39). Dataset **`tiankljucanin/rsna-knee-ckpt-v08r`** (private,
`datasets status` ready: `v08r_fold0_best.pt` 88.7 MB + `v08r_fold0_oof.csv` = the gold-58 predictions); per-epoch gold csvs + the
job log in `artifacts/kaggle_out/pod_v08r/`. No LB read yet — the fork with `v09r` alone (step A) is queued first; `v08r` joins
the fork only if step A reads 🔁 or ✅ (P-40), and a `v08r` solo read (no `v08a` solo baseline exists) is Tian's call.

**Sent as a solo on Tian's go ("submit … too", 15:36 local): submission #21**, `rsna-knee-infer` v19, ref 56610108, 13:41:35 UTC — ⏳ read vs `v09r` 0.927; also the first timed solo for P-41 (Submissions row 21). The fork (step A, v9) was still QUEUED at 13:42 UTC.

**Read 2026-09-27 16:24 UTC: #21 = 0.918** (entry "Submission #21").

### 2026-09-27 — Submission #21: the Raptor-distilled DINOv2-S `v08r` ALONE reads **0.918** public — −0.009 vs `v09r` (0.927), under P-40's ≈ 0.920 bar for a second fork member · P-41's speed read lost to a dead watcher

`rsna-knee-infer` v19, `INFER_MEMBERS = ["v08r"]`, Dataset `rsna-knee-ckpt-v08r` (entry "`v08r`"); sent 13:41:35 UTC, ref 56610108;
**0.918** at the first read after the laptop came back (16:24:19 UTC).

| # | member | backbone | training targets | gold-58 SWA | public LB |
|---|---|---|---|---|---|
| 18 | `v09a` | CoAtNet-1 @224 | LLM blend | 0.8922 | 0.918 |
| 20 | `v09r` | CoAtNet-1 @224 | 0.5 · LLM + 0.5 · Raptor | 0.9093 | **0.927** |
| **21** | **`v08r`** | **DINOv2-S @224** | **0.5 · LLM + 0.5 · Raptor** | **0.8981** | **0.918** |

**Verdict: 🔁 INCONCLUSIVE against the pre-registered ≈ 0.920 bar (−0.002, 0.4× the 0.005 floor) → by the P-40 action rule `v08r` does
not join the fork on solo strength; as a comparison with `v09r` it is a real gap (−0.009, 1.8× the floor).**

**What it says.**
1. **On the same targets the DINOv2-S recipe is ≈ 0.009 behind the CoAtNet-1 recipe** — the order every earlier read gave (fold-0 OOF
   `v08w` 0.8648 < `v09h` 0.8683; gold-58 0.8981 < 0.9093), now on the LB. Different pretraining (SSL ViT vs supervised hybrid) and
   input path (DINOv2 has no `batch_studies` / `aug` — P-35 found light aug worth nothing on it) — not separable from this read.
2. **Whether the Raptor table helped DINOv2 is unmeasured**: no `v08a` solo was ever submitted. Gold-58 says +0.013 (direction only);
   the CoAtNet twin's gold +0.017 became LB +0.009, so a `v08a` solo near 0.910 would be the consistent guess — a guess, not a read.
3. **What it does not say:** whether `v08r` *adds* to `v09r`. A weaker member of a different family can still lift a rank-mean if its
   errors differ; that is a two-member solo read (card P-42), not this one. **Gold-58 hint (hand-computed from the members'
   `_fold0_oof.csv` = gold predictions, 2026-09-27; direction only, floor 0.05; not `blend_check.py` — traps 32):** mean per-label
   Spearman ρ `v09r`/`v08r` **0.924** vs the LLM-target pair `v09a`/`v08a` **0.874**; flat rank-mean `v09r`+`v08r` **0.9085** vs
   `v09r` 0.9093 (−0.001, 5/12 labels up), where `v09a`+`v08a` gave 0.8964 vs `v09a` 0.8922 (+0.004, 7/12 up). Same-family pairs:
   `v09r`/`v09a` ρ 0.932 (rank-mean 0.9057), `v09r`/`v09t` 0.935 (0.9094). **The shared Raptor teacher pulled the two backbones
   together** — 🔁 as evidence (58 studies), but it lowers the prior on P-42.
4. **Gold-58 direction across the four production members:** gold order `v09r` > `v09t` > `v08r` > `v09a`, LB order `v09r` > `v08r` =
   `v09a` > `v09t` — gold-58 got the best member right and the middle wrong (traps 39 stands).

**P-41 (speed) — unread.** `src/watch_submission.py` saw `PENDING` at 13:41:43 UTC and was then killed with the laptop (its log ends
`[killed]`); the session's last manual read was PENDING at 13:43, the next one COMPLETE at 16:24:19 → **scored within ≤ 2 h 43 min of
sending, no tighter** — looser than #19's ≤ 42 min bound, i.e. no information. The speed read rides the next solo, watched with the
laptop awake.

### 2026-09-27 — Submission #23 (P-42): `v09r` + `v08r` rank-mean reads **0.927** = `v09r` alone → 🔁 · the first exactly timed solo: **28.3–29.3 min** (P-41 ✅)

`rsna-knee-infer` v20, `INFER_MEMBERS = ["v09r", "v08r"]` (flat rank-mean, one vote each, one shared c02 decode), sent 16:36:38 UTC,
ref 56614080; `watch_submission.py` (60 s polls) saw PENDING at 16:36:46 and **COMPLETE at 17:05:54 → scored within [28.3, 29.3] min**,
public **0.927**.

| # | members | gold-58 | public LB |
|---|---|---|---|
| 20 | `v09r` (CoAtNet-1, Raptor-distilled) | 0.9093 | 0.927 |
| 21 | `v08r` (DINOv2-S, Raptor-distilled) | 0.8981 | 0.918 |
| **23** | **`v09r` + `v08r`, flat rank-mean** | **0.9085** (hand-computed) | **0.927** |

**Verdict (P-42): 🔁 INCONCLUSIVE — ±0.000 vs `v09r` alone, inside the pre-registered 0.923–0.931 band.** By the card's rule `v08r`
stays out of the fork, and a second Raptor-distilled backbone on the same input is not a demonstrated lever.

**What it says.**
1. **The pair is not additive at 50/50.** On LLM targets every new family lifted the blend (#8 → #9 → #10); on the shared Raptor
   targets the DINOv2-S family adds nothing measurable. Gold-58 predicted exactly this (0.9085 vs 0.9093, ρ 0.924 vs 0.874 for the
   LLM-target pair) — right in size (−0.001 → ±0.000), as gold-58 was for `v08r` vs `v09r` (−0.011 → −0.009); `v09t` (+0.009 →
   −0.001) stays the counter-example, so traps 39 stands: gold-58 is direction only, the solo LB decides.
2. **Nor is `v08r` a near-copy that only dilutes** (a reading, not a verdict): a 50/50 rank-mean with a member 0.009 weaker landed on
   the stronger member, not between them (≈ 0.922–0.923 would be the dilution guess). What `v08r` knows that `v09r` does not roughly
   pays for its weakness — so a DINOv2 member *as strong as* `v09r` might add, and a weighted blend (e.g. 0.75 / 0.25) might read
   above 0.927. Both are sub-floor bets; neither is worth a submission on this evidence.
3. **Where the next gain has to come from** (P-42 "If it fails"): the targets (a Raptor-only mix, a second teacher) or the input
   representation — not a third backbone on the same table and cache.

**P-41 speed: ✅** — the pre-registered read was **≤ 30 min ✅ / 30–42 🔁 / > 42 ❌**; the first exactly timed solo scored in **28.3–29.3
min** with **two** members (the DINOv2-S pass adds ≈ 29 s / 100 studies to CoAtNet-1's ≈ 88 s). Caveats: the baseline (#19, one
member, 2 decode workers) is an upper bound (≤ 42 min), never an exact time, so the size of the speed-up is unknown; Kaggle's own
queue time sits inside both numbers. A solo read is now **≈ 30 min** send-to-score — four member reads fit in two hours.

### 2026-09-27 — Four-reviewer audit of proposals.md: gold-58 diagnostics computed during the review · 🔁 direction only, but they set the next session

Four read-only reviewer agents (auditor / strategist / skeptic / Kaggle feasibility), two rounds with cross-critique, scripts in the
session scratchpad. None of this is an LB read; it re-ranked the backlog (proposals.md rewritten, P-43…P-51) and chose the next
Kaggle session (`v09x` 320 px ‖ `v09u` seed 43). **Verdict: 🔁 INCONCLUSIVE as evidence (gold-58 / proxies) — recorded because the
plan now rests on it.**

**1. Paired study-level bootstrap on gold-58 (reviewer C, 4,000 reps; the members' `_fold0_oof.csv` = gold predictions):**

| pair | Δ gold-58 | bootstrap SD | P(Δ ≤ 0) | what the LB said |
|---|---|---|---|---|
| `v09r` − `v09a` | +0.0172 | 0.0076 | 1.0 % | +0.009 (#20 vs #18) |
| `v09t` − `v09a` | +0.0087 | 0.0061 | 7.6 % | −0.001 (#19 vs #18) |
| `v08r` − `v08a` | +0.0131 | 0.0086 | 6.5 % | never read |

For near-identical members the paired SD is ≈ 0.007, far tighter than the unpaired 0.05 rule — but it covers test sampling only,
not seed noise, and `v09t` shows a 7.6 % tail that the LB then contradicted. The public-LB "0.005 floor" is the top-ten span, not a
noise estimate; with ≈ 400 public studies (unverified) the sampling SD of a paired delta would be ≈ 0.003, and with an unmeasured
seed spread one solo-vs-solo delta is plausibly SD 0.004–0.006 → +0.009 ≈ 1.5–2σ (→ P-44).

**2. Error correlation within each true class on gold-58 (reviewer C; reproduced by A):** `v09r` / `v08r` (Raptor targets, two
families) **0.861** vs `v09a` / `v08a` (LLM targets) **0.777** (+0.084, SD 0.020, higher on 12/12 labels); same family, two teachers
`v09r` / `v09a` 0.883; both differ 0.773 / 0.799. Reading (A's correction of C): the shared teacher raised cross-family agreement,
but architecture still matters (0.883 > 0.861 on 11/12 labels). It explains P-42 (#23 = `v09r`) and predicts #22 ≈ 0.942.

**3. Gold-58 "teacher-mix analogs" (reviewer B; bootstrapped by D).** Rank-space mixes of the LLM blend (0.8948 alone) with a held-out
public CoAt reader (their gold references were pulled from the public Datasets; both readers' epochs were gold-selected, so the
image side is optimistic):

| image teacher | LLM 50 / image 50 | 25 / 75 | image only | mix 1.0 − 0.5 (SD) | mix 0.75 − 0.5 (SD) |
|---|---|---|---|---|---|
| resgated top-3 | 0.9275 | 0.9257 | 0.9095 | −0.018 (0.010) | −0.002 (0.005) |
| D4 | 0.9346 | 0.9383 | 0.9302 | −0.004 (0.008) | +0.004 (0.005) |

A teacher proxy, not a student and not Raptor (Raptor has no gold predictions in our hands — P-49); still, it moved the mix
bracket from 1.0 to 0.75 (P-47). Mix 1.0 is not "Raptor-only" either: `w__` stays LLM-agreement and quantile matching uses the LLM
marginals.

**4. `pilkwang/rsna-knee-weights` ships `oof.npz`** — honest 5-fold DINOv2-S@336 OOF over all 4,407 studies: gold-58 **0.840**; added
at 0.15–0.25 to LLM + CoAt it *lowers* gold 0.003–0.007 (a weaker honest teacher dilutes). Raptor agrees with it far more than
with the reports (ρ 0.819 vs 0.652 with the LLM blend).

**5. Is Raptor copying the dread labels (its likely training labels)?** Overall no: Raptor ~ dread ρ 0.661, ~ LLM 0.652; partial
ρ(Raptor, dread | LLM) 0.285. The Synovitis dispute resolved: positive rates on the 4,349 report-only studies are LLM 12.4 %, dread
24.5 %, raw Raptor 41.0 %; on LLM/dread disagreements Raptor "sides with dread" 78–80 % at a raw 0.5 cut, 52 % at the LLM's
positive rate, 50 % rank vs rank (A: an operating-point effect, which quantile matching removes); within the disagreements Raptor
ranks dread-positive above LLM-positive cases with AUC 0.71 on Synovitis, 0.74 MCL, 0.72 Effusion, 0.50–0.67 elsewhere (C: a lean to
dread on every label); dread's Synovitis column has only 7 distinct values, which caps its global ρ (B). So the in-sample channel
is real but modest and not Synovitis-specific.

**6. Two facts about our own comparisons.** (a) #18 (`v09a`, Kaggle T4, 2 loader workers) vs #20 (`v09r`, RunPod 4090, 8 workers)
— the headline +0.009 — was cross-platform: the window draws are per-worker numpy streams, so worker count changes them, while
init / data order / GPU augmentation share the seed-42 torch stream. (b) Every production member trained on the seed-42 stream
(an arm-dict `seed` was inert — traps 42), and gold rows are held out under `train_all` (the S2 and `v09t` entries' "optimistic by
construction" lines are CORRECTED above).

### 2026-09-27 — Submission #22 (P-40 step A): the public 0.942 graph + `v09r` alone at β 0.10 reads **0.942** → 🔁 · the fork at β 0.10 does not register a 0.927 member either

`rsna-knee-fork` v9 (`build_fork.py --members v09r --member v09r=tiankljucanin/rsna-knee-ckpt-v09r:tiankljucanin/timm-coatnet-rmlp-1-rw-224
--beta 0.10`), sent 2026-09-27 16:36:17 UTC after ≈ 3 h 40 min in the Kaggle GPU queue, ref 56614068; first seen COMPLETE at
2026-09-28 09:12 UTC. The watcher (`watch_submission.py`, task `buiay90kx`) died with the laptop and wrote no row to
`artifacts/submission_timing.csv`, so the scoring time is only bounded (≤ 16 h 36 min) — no information beyond #17's ≤ 8 h 06 min.

| # | our arm in the fork (β) | our member(s)' strength | public LB |
|---|---|---|---|
| 15 | none (the anchor alone) | — | 0.942 |
| 13 | `v08w` + `v09h` (β 0.10) | our 12-member blend read 0.913 (#11) | 0.942 |
| 17 | S2 `v09a` + `v08a` beside `v08w` + `v09h` (β 0.10) | `v09a` 0.918 solo (#18) | 0.941 |
| **22** | **`v09r` alone (β 0.10)** | **0.927 solo (#20)** | **0.942** |

**Verdict (P-40): 🔁 INCONCLUSIVE — ±0.000 vs #13 / #15, inside the pre-registered 0.940–0.946 band** (✅ needed ≥ 0.947). The
reviewers' prior (0.942 ± 0.001) was right.

**What it says.**
1. **At β 0.10 the fork is flat across a 0.014 range of member strength** (0.913-level blend → 0.918 → 0.927 all read 0.941–0.942).
   A 10 % rank-weight on a member 0.015 below the anchor cannot move a 0.942 stack by a readable amount — the same lesson as #17
   ("the fork at β 0.10 does not read member quality, the solo submission does"), now with our best member.
2. **The "Against" point in P-40 stands, not refuted:** `v09r`'s errors track Raptor (within-class ρ 0.861 vs 0.777 for the
   LLM-target pair, 2026-09-27 audit), and the anchor already carries Raptor at 0.40–0.60 weight, so our arm adds little the anchor
   does not have. P-49 (Raptor over gold-58) would measure that redundancy directly without another 8-h fork read.
3. **Consequence for the plan:** member quality is read by solo submissions only (≈ 30 min, P-41); the fork is for final-selection
   builds (P-50). The pre-registered 🔁 branch (β 0.20 once) is an open decision — reviewers B and C recommend dropping it
   (expected ≤ 0.002, and it is weight tuning on the public LB).

### 2026-09-28 — `rsna-knee-train` v30 (P-43 + P-44): `v09x` (the `v09r` recipe at 320 px) gold-58 SWA **0.9094**, `v09u` (`v09r`, seed 43, Kaggle T4) **0.8995** vs `v09r` 0.9093 · 🔁 direction only · shipped, solo reads next

One `PARALLEL_ARMS = ("v09x", "v09u")` session, `TEACHER_TABLES = ("raptor_teacher",)`, mix 0.5, pushed 2026-09-27 18:08:24 UTC,
**COMPLETE in 5.88 h** (no queue). Green on every pre-registered criterion: parent `ok  arm v09x` + `ok  arm v09u` (rc 0,
`_best.pt` written); both children `teacher table raptor_teacher: 4349 studies`, `SWA of last 3 EMA snapshot(s)`, `-> v09*_fold0_best.pt =
SWA`, no `runtime guard`; `v09u.log` `reseeded 43 for arm v09u (base seed 42)`; `v09x.log` `img 320 | batch 1 x accum 4`. Timing:
**`v09x` 0.59 s/study = 43 min/epoch** (P-43 estimated 0.55–0.63; break-even was 0.80), `v09u` 0.27 s/study = 19.5 min/epoch.
Both `_best.pt` 164,789,267 bytes; shipped as Datasets **`rsna-knee-ckpt-v09x`** and **`rsna-knee-ckpt-v09u`** (ready 2026-09-28
09:14 UTC). Outputs: `artifacts/kaggle_out/train_v30/`.

Gold-58 (all 58 held out under `train_all`; `v09r` from `artifacts/kaggle_out/pod_v09r/`, RunPod 4090, seed 42):

| label | `v09r` (224, seed 42, RunPod) | `v09u` (224, seed 43, Kaggle) | `v09x` (320, seed 42, Kaggle) |
|---|---|---|---|
| ACL | 0.953 | 0.949 | 0.961 |
| MCL | 0.925 | 0.907 | 0.934 |
| Medial Meniscus | 0.958 | 0.970 | 0.968 |
| Lateral Meniscus | 0.853 | 0.870 | **0.901** |
| Medial OA | 0.967 | 0.984 | 0.988 |
| Lateral OA | 0.805 | 0.807 | 0.810 |
| PF OA | 0.816 | 0.812 | 0.816 |
| Effusion | 0.974 | 0.944 | 0.937 |
| Synovitis | 0.823 | 0.816 | 0.823 |
| Baker's | 0.960 | 0.933 | 0.960 |
| Contusion | 0.934 | 0.912 | 0.927 |
| Fracture | 0.943 | 0.890 | 0.889 |
| **macro (SWA)** | **0.9093** | **0.8995** | **0.9094** |
| epochs 0–7 EMA | — | 0.772 · 0.854 · 0.890 · 0.899 · 0.899 · 0.898 · 0.900 · 0.898 | 0.759 · 0.853 · 0.881 · 0.906 · 0.908 · 0.909 · 0.911 · 0.911 |

Pairs (mean per-label Spearman ρ on gold-58; flat rank-means, hand-computed): ρ(`v09r`, `v09u`) 0.952, ρ(`v09r`, `v09x`) 0.949,
ρ(`v09u`, `v09x`) 0.947; rank-mean `v09r` + `v09u` **0.9065**, `v09r` + `v09x` **0.9126**, `v09u` + `v09x` 0.9072, all three 0.9110.

**Verdict: 🔁 INCONCLUSIVE on gold-58 (floor 0.05), by design — the LB reads are pre-registered in P-43 / P-44.** Direction only
(traps 39):
1. **The seed/platform twin is 0.010 below its parent on gold-58** (`v09u` − `v09r` = −0.0098, 4 labels up / 8 down, Fracture
   −0.053, Effusion −0.030, Baker's −0.027) — 1.4× the audit's paired-bootstrap SD (≈ 0.007). The same recipe on another seed and
   platform moves gold-58 by about the size of every recent delta we have read, which is the question P-44's solo read answers on
   the LB.
2. **320 px lands where the P-43 hypothesis pointed:** `v09x` = `v09r` on macro (+0.0001) but +0.010 over the same-platform
   `v09u` (which also differs in seed; 9 / 3 labels), and +0.005 over mean(`v09r`, `v09u`) with **10 of 12 labels up**; the
   biggest move is **Lateral Meniscus +0.040** over that mean (0.901, the best any member of ours has read on it), then MCL and ACL
   — small structures, as P-43 argued. A consistent sign across labels is the kind of evidence the per-label floor allows; the
   size is not.
3. **Blend priors:** the seed-twin pair does not help on gold-58 (0.9065 < `v09r`; ρ 0.952 — the twins agree more than any two
   families did), while `v09r` + `v09x` reads 0.9126 (+0.003, sub-floor). #23 showed gold-58 predicts pair blends' LB direction
   well (−0.001 → ±0.000).

**CORRECTED 2026-09-28 (same day, after #24–#26):** points 1 and 3 had the LB direction wrong — `v09u` read 0.927 = `v09r` (not
below it) and the `v09r` + `v09u` pair read 0.930 (not below `v09r`). See the next entry.

### 2026-09-28 — Submissions #24–#26 (P-44 + P-43): `v09u` **0.927** = `v09r` → the retrain spread is ≈ 0 · `v09x` (320 px) **0.929** → 🔁 · `v09r` + `v09u` **0.930** → the production member

Three solos from `rsna-knee-infer` v21 / v22 / v23 (the v30 checkpoints via Datasets `rsna-knee-ckpt-v09u` / `-v09x`), sent 09:26 /
09:28 / 09:32 UTC in the pre-registered order, each watched by `watch_submission.py` (60 s, laptop awake):

| # | members | gold-58 | public LB | sent → scored |
|---|---|---|---|---|
| 18 | `v09a` (LLM targets, Kaggle T4, seed 42) | 0.8922 | 0.918 | — |
| 20 | `v09r` (Raptor mix 0.5, 224, RunPod, seed 42) | 0.9093 | 0.927 | — |
| **24** | **`v09u`** (= `v09r`, Kaggle T4, **seed 43**) | 0.8995 | **0.927** | [20.4, 21.4] min |
| **25** | **`v09x`** (= `v09r` at **320 px**, Kaggle T4, seed 42) | 0.9094 | **0.929** | [29.3, 30.3] min |
| **26** | **`v09r` + `v09u`**, flat rank-mean | 0.9065 | **0.930** | [28.3, 29.3] min |

**Verdicts, by the rules written in P-43 / P-44 before the reads:**
1. **P-44 floor — ✅ KEEP (a measurement).** s = |`v09u` − 0.927| = **0.000** (both rounded to 3 decimals, so the true gap is
   < 0.001). The card's rule for s ≤ 0.002: **one-seed solo deltas need ≥ 0.004** from now on. Caveat, as pre-registered: one draw
   estimates σ poorly (≈ 50 % CV), so 0.004 is a working floor, not a measured SD; keep 0.005 for cross-recipe claims.
2. **P-39 re-confirmed on the platform of #18.** `v09u` > 0.920, and mean(`v09r`, `v09u`) − 0.918 = 0.009 ≥ 0.006. The audit's worry
   that #18 → #20 (+0.009) was partly RunPod-vs-Kaggle (worker count changes the window draws) is answered: a Kaggle T4 / 2-worker
   retrain on the Raptor targets reads exactly what the RunPod one did. The gain was the targets.
3. **P-43 — 🔁 INCONCLUSIVE.** m = mean(0.927, 0.927) = 0.927, threshold max(0.005, 2 · 0) = 0.005 → ✅ needed ≥ 0.932, ❌ ≤ 0.922;
   `v09x` read **+0.002**. By the card, resolution is not a demonstrated lever and 224 stays the production resolution. `v09x` is
   nonetheless our strongest single member (0.929), costs 1.4× the inference (136 vs 98 s / 100 studies in the placeholders) and 2.1×
   the training (43 vs 19.5 min / epoch on a T4).
4. **The 2-seed pair — ✅ KEEP as the production member, 🔁 as a lever.** 0.930 meets the pre-registered line (≥ 0.930) exactly and
   is our best solo, but +0.003 over either member is under the 0.004 floor from point 1 — so seed averaging is adopted (the card
   adopted it "either way": lower private-LB variance, no LB tuning), not proven.

**What it says about gold-58.** Between members of one recipe, gold-58 got the LB direction **wrong twice**: `v09u` −0.010 on gold
→ ±0.000 on the LB; the `v09r` + `v09u` pair −0.003 on gold → +0.003 on the LB. Across families it has been right (#21, #23). So for
differences under ≈ 0.01 between same-recipe members, gold-58 is not even direction (traps 39 extended). The 320-px member's gold
story (10 / 12 labels up over the 224 mean, Lateral Meniscus +0.040) turned into +0.002 on the LB — consistent in sign, too small to
count.

**Next** (proposals.md): P-52 — the three-member rank-mean `v09r` + `v09u` + `v09x` as one solo (no training; 0.1 h placeholder); #27
(the fork at β 0.20, P-40) is scoring.

### 2026-09-28 — Gold-58 per-label diagnosis + c02 slice census (0 GPU): every member trails its OWN LLM labels on ACL / MCL / PF OA / Lat Men · 🔁 direction only, but it sets P-53..P-57

**Gold-58, seed-averaged `v09r` + `v09u` vs the 3-source LLM blend (recomputed locally; `build_targets.load_sources` + `prob_blend`):**
ACL 0.951 vs 0.990 (-0.039), MCL 0.916 vs 0.980 (-0.063, 9 positives), PF OA 0.814 vs 0.903 (-0.089), Lat Men 0.861 vs 0.881 (-0.020);
above the labels on Effusion +0.080, Fracture +0.101, Contusion +0.062, Med OA +0.045, Synovitis +0.032; macro 0.9044 vs 0.8948. The
structural sign holds for all 8 members the workflow checked (224 / 320 px, LLM / Raptor / self-distilled targets). Our gold->LB offset
(+0.021 over six members) equals Scott Willis's (+0.019) and Archit Konde's (+0.020), so the 0.02 single-model gap is visible on gold.

**Census** (`series_meta.csv` + `manifest.csv` pulled from `rsna-knee-cache2-a`, `artifacts/census/`): the fluid series have a median of
30-32 native slices at 3.0 mm (p10 19-22, p90 36-40); c02 stores 18 / 12 / 12 (SAG / COR / AX fluid) -> median stored spacing 5.0 /
7.5 / 9.0 mm; FOV median 160 mm (p10 150, p90 180; < 150 mm in 8 %). By the plan's pre-registered rule the c03 budgets are
min(median native, 24) = **24 / 24 / 24** (SAG_FLUID_NOFS 14, T1 slots 8) and the crop **150 mm** (the plan's guess of a sagittal cap
at 20-22 was wrong: sagittal is also ~30 native). cache_selftest: c03 arrays bit-identical builder vs pipeline, valid windows +56-62 %.

**Verdict: 🔁 direction only (58 studies, labels picked after looking) — the probe (#28) re-reads it on ~400 public studies.** Source
thread synthesis: research.md 2.7.2; plan: docs/superpowers/plans/2026-09-28-single-model-plan.md.

**CORRECTED 2026-09-28 (15:33 UTC, #28):** on the public test the four labels are *not* weak — they average 0.935 vs 0.9275 for the
other eight (entry "Submissions #27–#28"). The gold-58 deficit is gold-specific; the census facts above still stand.

### 2026-09-28 — Submissions #27–#28: the fork at β 0.20 reads **0.941** (🔁, P-40 closes) · the per-label probe reads **0.785** → the "structural deficit" is gold-specific (P-53)

Both first seen scored at 15:33 UTC; the watchers died with the laptop, so the times are bounds (#27 was still pending at 12:12).

**#27 — `rsna-knee-fork` v10, `v09r` alone at β 0.20, sent 09:40 UTC → 0.941, scored within (2 h 32 min, 5 h 53 min].**

| # | our arm | β | public LB |
|---|---|---|---|
| 15 | none (anchor control) | 0 | 0.942 |
| 12 | `v08w` + 5-fold `v09h` (0.87-level) | 0.20 | 0.939 |
| 13 | `v08w` + 5-fold `v09h` | 0.10 | 0.942 |
| 17 | + `v09a` + `v08a` (0.918-level) | 0.10 | 0.941 |
| 22 | `v09r` alone (0.927 solo) | 0.10 | 0.942 |
| **27** | **`v09r` alone** | **0.20** | **0.941** |

**Verdict: 🔁 INCONCLUSIVE** by the P-40 bands (≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌). The pre-registered branch after #22 was "β 0.20
once" — done, so **P-40 closes**: at β 0.10 or 0.20 the fork does not register a member of ours between 0.87 and 0.927 solo, and
P-49 (entry below) gives the reason — `v09r`'s within-class errors track Raptor (ρ 0.835), which the anchor already carries. The fork
is for final-selection builds only (P-50).

**#28 — the P-53 probe, sent 11:56 UTC → 0.785, scored within (15 min, 3 h 36 min].** The #26 pair (0.930) with ACL, MCL, PF OA and
Lateral Meniscus written as a constant 0.5 (each of those AUCs is then exactly 0.5). With S = 12 × 0.930:

- mean4 = 0.5 + 3 × (0.930 − 0.785) = **0.935** (± 0.003 from the 3-decimal rounding of both scores);
- mean8 = (S − 4 · mean4) / 8 = 1.5 × 0.785 − 0.25 = **0.9275** (± 0.001);
- mean8 − mean4 = **−0.0075** (gold-58 for the same pair: 0.914 − 0.886 = **+0.028**).

**Verdict: ✅ KEEP (a measurement), by the rule written before the read:** mean4 ≥ mean8 − 0.01 → **the deficit is gold-specific.** On
≈ 400 public studies our four "structural" labels are, if anything, *stronger* than the other eight. What the probe cannot say: the
gold diagnosis was *relative to the LLM labels* (which do not exist on the test), so "we trail the reports on these labels" is
neither confirmed nor refuted — only "these labels are where we are weak" is refuted. Consequences: the c03 input arm (P-56, session
D, already running) loses its main motivation and runs on the census argument alone (lower prior); the probe is diagnosis only and is
never used to weight labels.

### 2026-09-28 — P-49: Raptor over the 58 gold studies — Raptor 0.9254, the 0.5/0.5 target 0.9268; `v09r` tracks Raptor (within-class ρ 0.835) · 🔁 direction only

`rsna-knee-teacher-gold` v1 (`build_teacher_pass.py --gold`): 58/58 studies, 0 failed, 4.7 s/study (263 s over both T4s). Merged
with `merge_teacher.py --expect-n 58` → `artifacts/teacher/raptor_gold.csv` (a name `TEACHER_PATHS` never reads — gold rows cannot
reach training). The 0.5/0.5 target is computed as training computes it: each gold Raptor value is mapped through the 4,349-study
Raptor ECDF onto the LLM blend's report-only distribution, then averaged with the gold study's LLM value (scratch `p49_gold.py` /
`p49_rho.py`, session scratchpad).

| label | pos | LLM blend | Raptor | 0.5/0.5 target |
|---|---|---|---|---|
| ACL | 24 | 0.990 | 0.980 | 0.995 |
| MCL | 9 | 0.980 | 0.993 | 0.997 |
| Medial Meniscus | 26 | 0.955 | 0.969 | 0.980 |
| Lateral Meniscus | 23 | 0.881 | 0.863 | 0.928 |
| Medial OA | 15 | 0.931 | 0.985 | 0.978 |
| Lateral OA | 11 | 0.808 | 0.836 | 0.819 |
| PF OA | 21 | 0.903 | 0.835 | 0.909 |
| Effusion | 35 | 0.880 | 0.973 | 0.944 |
| Synovitis | 27 | 0.788 | 0.824 | 0.830 |
| Baker's | 12 | 0.947 | 0.978 | 0.957 |
| Contusion | 19 | 0.861 | 0.931 | 0.913 |
| Fracture | 18 | 0.815 | 0.938 | 0.872 |
| **macro** | | **0.8948** | **0.9254** | **0.9268** |

Paired study bootstrap (2,000 reps): Raptor − LLM +0.031 (SD 0.015); target − LLM **+0.032 (SD 0.008), 12/12 labels up**. Raptor alone
is optimistic (its public authors chose epochs / SWA on these 58), the LLM blend is not.

**Within-class ρ on gold-58** (per label, mean of the Spearman ρ inside positives and inside negatives, then macro; the definition
reproduces the audit's `v09r` ~ `v08r` 0.861 as 0.860):

| pair | ρ | pair | ρ |
|---|---|---|---|
| `v09r` ~ `v09u` (seed twins) | 0.916 | `v09a` ~ Raptor (LLM targets) | 0.742 |
| `v09r` ~ `v09a` | 0.888 | LLM ~ Raptor | 0.441 |
| `v09r` ~ Raptor | **0.835** | `v09r` ~ LLM | 0.408 |
| `v09u` / `v09x` / `v08r` ~ Raptor | 0.826 / 0.806 / 0.780 | `v09r` ~ `v08r` | 0.860 |

**Matched-mix analogs on Raptor itself** (yt = (1 − w) · LLM + w · matched Raptor, gold macro): w 0 → 0.8948, 0.25 → 0.9187, **0.5 →
0.9268**, 0.75 → 0.9252 (−0.0017 vs 0.5, SD 0.003), 1.0 → 0.9098 (−0.017, SD 0.009).

**Verdict: 🔁 direction only — a correlation read, as the card pre-registered, never a verdict.** What it says:
1. **Distillation moved our member most of the way onto Raptor:** `v09a` (LLM targets) ~ Raptor 0.742 → `v09r` 0.835 (+0.09), while
   `v09r` ~ LLM is 0.408. The card's "If it works" fires: a high ρ → the fork cannot read `v09r` (consistent with #22 / #27) → P-40
   closed, the fork is for final builds (P-50).
2. **Mix 0.5 is the best matched mix on gold** → P-47 (mix 0.75) is priced at ≈ 0 (−0.002); mix 1.0 is worse (−0.017, 1.9 SD), as the
   D4 / resgated analogs said. Not a reason to change the P-55 student (its 0.75 includes the cross-fit OOF, a different table).
3. **The student trails its own target by 0.018 on gold** (`v09r` 0.9093 vs 0.9268; the seed twin 0.8995). The gap between what the
   targets know and what the student learns is at least as large as any target change we have measured — the P-54 / P-55 cross-fit
   and the P-56 input are both attempts at it.

### 2026-09-28 — P-54 cross-fit sessions A ‖ B: folds 0–3 of the `v09r` recipe (`v09k0` … `v09k3`) trained and scored · ✅ runs green, ⏳ fold 4

`rsna-knee-train` v32 (`v09k0` ‖ `v09k1`, 2.29 h wall) and `rsna-knee-folds` v9 (`v09k2` ‖ `v09k3`, 2.26 h), pushed 12:10 UTC, both
`artifacts/train_xf_{A,B}.py` = `src` + 3 seds (`FORCE_SMOKE = False`, `PARALLEL_ARMS`, `TEACHER_TABLES = ("raptor_teacher",)`).
Every child: `teacher table raptor_teacher: 4349 studies`, `fold k: train 3525–3526 / val 881–882`, 8 epochs with `(held-out eval
deferred to the SWA pass: eval_final_only)`, `SWA of last 3 EMA snapshot(s)`, `-> v09k*_fold*_best.pt = SWA`, no runtime guard; the
parents judged each child by its fold's `_best.pt` (`ok  arm v09kN … (folds [N])` ×4). Pulled to `artifacts/kaggle_out/xf_A|B/`
(4 × 157 MiB checkpoints, 881–882-row OOF csvs incl. 11–12 gold rows each).

| fold | arm | train / val | final loss | OOF vs LLM (`auc_soft`) | gold (n) | val pass |
|---|---|---|---|---|---|---|
| 0 | `v09k0` | 3,525 / 882 | 0.3825 | 0.8816 | 0.9272 (11) | 6.9 min |
| 1 | `v09k1` | 3,525 / 882 | 0.3854 | 0.8728 | 0.9136 (12) | 6.2 min |
| 2 | `v09k2` | 3,526 / 881 | 0.3824 | 0.8707 | 0.8582 (12) | 6.8 min |
| 3 | `v09k3` | 3,526 / 881 | 0.3847 | 0.8679 | 0.8883 (12) | 6.2 min |

**Verdict: ✅ the runs (the cross-fit pipeline works on folds 1–3, which no smoke can reach); no verdict on anything else.** OOF vs the
LLM targets rewards agreement with the teacher (traps 39); per-fold gold on 11–12 studies is noise. Fold 4 (`v09k4`) is session C
(`rsna-knee-train` v33, with `v13a`, pushed 15:35 UTC); then the table (`build_distill_table.py --per-fold-rank`) and the loose gate.

### 2026-09-29 — Sessions C ‖ D: `v09k4` (fold 4) done · c03 `v11a` / `v11b` gold-58 **0.9204 / 0.9167** (pair 0.9207 vs the c02 pair 0.9065) · ResNet-34 `v13a` **0.8306** · ✅ runs green, 🔁 c03 direction only, ❌ `v13a` on gold

**Session C, re-pushed** after v33 died from outside (traps 44) = `rsna-knee-train` v34 (pushed 2026-09-28 18:30 UTC, the same build
`artifacts/train_xf_C.py`), COMPLETE in **2.39 h**: `v09k4` (fold 4, `train 3526 / val 881`, `eval_final_only`, SWA 5–7, 0.29 s/study)
and `v13a` (ResNet-34, all 4,349, 8 epochs in 0.7 h at 0.07 s/study), both `ok  arm`, both `teacher table raptor_teacher: 4349`.
**Session D** = `rsna-knee-folds` v10 (pushed 15:36 UTC, `kernel_sources` = `rsna-knee-cache3-a..d` only), COMPLETE in **3.53 h**:
`v11a` (seed 42) ‖ `v11b` (`reseeded 43`), each on cache `c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20` (4 shards, 4,407
studies), `train 34, eval all` windows, 0.35 s/study ≈ 25 min/epoch, SWA of the last 3, no runtime guard. Pulled to
`artifacts/kaggle_out/xf_C2/` and `c03_D/` (`_best.pt` 164,789,727 / 164,789,267 bytes; `v13a` 85,857,013).

Gold-58 (the `train_all` members' held-out 58; `v09k4` scores fold 4 only — 0.9432 on n = 11, noise), per label, with same-seed
rank-means (scratch `cd_gold.py`):

| label | `v09r` | `v09u` | `v11a` | `v11b` | `v13a` | c02 pair | c03 pair | c03 − c02 (pair means) |
|---|---|---|---|---|---|---|---|---|
| ACL | 0.953 | 0.949 | 0.962 | 0.961 | 0.772 | 0.953 | 0.961 | +0.010 |
| MCL | 0.925 | 0.907 | 0.955 | 0.957 | 0.753 | 0.915 | 0.955 | +0.040 |
| Medial Meniscus | 0.958 | 0.970 | 0.958 | 0.959 | 0.768 | 0.966 | 0.959 | −0.005 |
| Lateral Meniscus | 0.853 | 0.870 | 0.896 | 0.872 | 0.752 | 0.868 | 0.889 | +0.022 |
| Medial OA | 0.967 | 0.984 | 0.986 | 0.984 | 0.984 | 0.977 | 0.988 | +0.009 |
| Lateral OA | 0.805 | 0.807 | 0.822 | 0.812 | 0.797 | 0.805 | 0.818 | +0.012 |
| PF OA | 0.816 | 0.812 | 0.837 | 0.838 | 0.816 | 0.815 | 0.838 | +0.023 |
| Effusion | 0.974 | 0.944 | 0.969 | 0.960 | 0.978 | 0.964 | 0.970 | +0.006 |
| Synovitis | 0.823 | 0.816 | 0.806 | 0.811 | 0.748 | 0.823 | 0.812 | −0.011 |
| Baker's | 0.960 | 0.933 | 0.966 | 0.971 | 0.982 | 0.951 | 0.970 | +0.022 |
| Contusion | 0.934 | 0.912 | 0.947 | 0.937 | 0.823 | 0.928 | 0.941 | +0.019 |
| Fracture | 0.943 | 0.890 | 0.942 | 0.938 | 0.794 | 0.913 | 0.948 | +0.023 |
| **macro** | **0.9093** | **0.8995** | **0.9204** | **0.9167** | **0.8306** | **0.9065** | **0.9207** | **+0.014** |

Paired study bootstrap (2,000 reps) c03 pair − c02 pair **+0.0142 (SD 0.0053)**; the 2-seed mean moves +0.0141 with 10/12 labels
up. Within-class ρ: `v11a` ~ `v11b` 0.934 (the c02 seed twins: 0.916), `v11a` ~ `v09r` 0.908, `v11b` ~ `v09u` 0.894, `v13a` ~ `v09r`
0.667. Rank-means: `v09r` + `v09u` + `v11a` + `v11b` 0.9182; the P-52 trio 0.9110.

**Verdicts.**
- **Runs: ✅** — both sessions green; P-54 now has all five folds (entry below).
- **c03 (P-56): 🔁 direction only, the solo reads decide.** +0.014 is under the 0.05 gold floor, and gold-58 got two same-recipe
  directions wrong (#24, #26 — traps 39); what is new is the *consistency* (both seeds above both c02 seeds, 10/12 labels, 3 SD on the
  paired bootstrap). Note the gains sit on MCL / PF OA / Lat Men — the labels the c03 input was aimed at — although #28 says those are
  not weak on the public test. Pre-registered reads (P-56): m = mean of the two solos vs 0.927 (✅ ≥ 0.9315 / 🔁 0.9285–0.9315 / ❌
  < 0.9285), and the pair vs 0.930.
- **ResNet-34 (P-57): ❌ DEAD END on gold-58** — −0.079 vs `v09r`, beyond the 0.05 floor (paired SD ≈ 0.007, audit), 8/12 labels
  down, and the structure labels collapse to 0.75–0.77 while Medial OA / Effusion / Baker's hold. Scope: the `v09r` optimiser (backbone
  LR 1e-4, 8 epochs, window-attention head) on `resnet34.a1_in1k` — this closes "a ResNet-34 on our recipe", not the thread's tuned
  ResNets. The solo read is deprioritised (Dataset `rsna-knee-ckpt-v13a` shipped for a spare slot).

### 2026-09-29 — P-54 cross-fit table `xfit_v09k` (5 folds, per-fold ranked): gold-58 **0.9028** vs the LLM blend 0.8948 · loose gate **OPEN** → session E pushed · 🔁 direction only

`python src/build_distill_table.py --sets "artifacts/kaggle_out/xf_*/v09k[0-4]_fold[0-9]_oof.csv" --per-fold-rank --out
artifacts/teacher/xfit_v09k.csv` → 4,407 studies from one set of five folds (881–882 rows each; 0 NaN, 0 duplicate UIDs), values =
within-fold ranks in (0, 1). The gate script (scratch `xfit_gate.py`, the P-49 method) on the 58 gold rows — each an honest
out-of-fold prediction:

| label | pos | xfit | LLM | Raptor | 0.5/0.5 target | ρ_w vs Raptor | xfit − LLM |
|---|---|---|---|---|---|---|---|
| ACL | 24 | 0.934 | 0.990 | 0.980 | 0.995 | 0.69 | −0.056 |
| MCL | 9 | 0.910 | 0.980 | 0.993 | 0.997 | 0.71 | −0.069 |
| Medial Meniscus | 26 | 0.956 | 0.955 | 0.969 | 0.980 | 0.81 | +0.001 |
| Lateral Meniscus | 23 | 0.810 | 0.881 | 0.863 | 0.928 | 0.76 | −0.071 |
| Medial OA | 15 | 0.978 | 0.931 | 0.985 | 0.978 | 0.85 | +0.047 |
| Lateral OA | 11 | 0.814 | 0.808 | 0.836 | 0.819 | 0.93 | +0.007 |
| PF OA | 21 | 0.808 | 0.903 | 0.835 | 0.909 | 0.94 | −0.095 |
| Effusion | 35 | 0.960 | 0.880 | 0.973 | 0.944 | 0.84 | +0.081 |
| Synovitis | 27 | 0.816 | 0.788 | 0.824 | 0.830 | 0.83 | +0.028 |
| Baker's | 12 | 0.971 | 0.947 | 0.978 | 0.957 | 0.85 | +0.025 |
| Contusion | 19 | 0.949 | 0.861 | 0.931 | 0.913 | 0.71 | +0.088 |
| Fracture | 18 | 0.926 | 0.815 | 0.938 | 0.872 | 0.89 | +0.111 |
| **macro** | | **0.9028** | **0.8948** | **0.9254** | **0.9268** | **0.818** | **+0.008** |

Paired bootstrap xfit − LLM **+0.0078 (SD 0.0146)**; 4/12 labels below the LLM (the same four "structural" labels as every member).
**Loose gate (Tian, 2026-09-28): E runs unless pooled gold < 0.875 AND ≥ 8/12 below the LLM → 0.9028 ≥ 0.875, 4/12 → OPEN.** Unlike
P-38's teacher (gold 0.873 < 0.895), this teacher is above the labels it is mixed with — the precondition the plan asked for.

**Shipped:** a new version of the PRIVATE Dataset `rsna-knee-teacher-tables` (`raptor_teacher.csv` / `selfdistill_v1.csv` byte-identical
by md5, + `xfit_v09k.csv`); checkpoint Datasets `rsna-knee-ckpt-v09k` (all five `v09k{k}_fold{k}_best.pt` + OOF csvs), `-v11a`,
`-v11b`, `-v13a`, all `ready` 11:36 UTC and added to `kaggle/rsna-knee-infer/kernel-metadata.json`.
**Session E** (P-55) = `rsna-knee-train` v35, `PARALLEL_ARMS = ("v09o", "v09o2")`, `TEACHER_TABLES = ("raptor_teacher", "xfit_v09k")`,
`TEACHER_MIX = 0.75` (`artifacts/train_student_E.py` = `src` + 4 seds), pushed 11:37:33 UTC, RUNNING at once. Local CPU smoke of the
same build (`MODE = "train"`, `FORCE_SMOKE = True`) green first: both tables read (xfit 4,407 rows), `training targets = (1 - 0.75) *
LLM + 0.75 * quantile-matched ['raptor_teacher', 'xfit_v09k']`, `reseeded 43 for arm v09o2`, both `_best.pt` = SWA, exit 0.
**Verdict: 🔁 direction only** — the table's gold is never a verdict (traps 39); the student's solo reads are.

### 2026-09-29 — Submissions #29–#32: P-52 trio **0.931** (🔁) · c03 `v11a` **0.932** / `v11b` **0.929** → m = 0.9305 (🔁) · c03 pair **0.932** vs the c02 pair 0.930 (🔁) — a consistent small c03 edge, under every pre-registered bar

Four solos from `rsna-knee-infer` v25–v28 (each `src` + 3 seds; placeholders green, rows 29–32), sent 11:42–11:51 UTC, each timed by
`watch_submission.py` (60 s polls → `artifacts/submission_timing.csv`):

| # | members | gold-58 | public LB | scored within (min) | pre-registered read |
|---|---|---|---|---|---|
| 29 | `v09r` + `v09u` + `v09x` (P-52) | 0.9110 | **0.931** | [53.5, 54.5] | vs 0.930: ≥ 0.934 ✅ / 0.931–0.933 🔁 / ≤ 0.930 ❌ → **🔁** |
| 30 | `v11a` (c03, seed 42) | 0.9204 | **0.932** | [27.3, 28.3] | with #31 → |
| 31 | `v11b` (c03, seed 43) | 0.9167 | **0.929** | [31.3, 32.3] | m = 0.9305 vs 0.927: ✅ ≥ 0.9315 / 🔁 0.9285–0.9315 / ❌ < 0.9285 → **🔁** |
| 32 | `v11a` + `v11b` (c03 pair) | 0.9207 | **0.932** | [45.4, 46.4] | vs the c02 pair #26 0.930 → +0.002 → **🔁** (under the 0.004 floor) |

**Verdicts.**
- **P-52: 🔁 INCONCLUSIVE, the card closes** — +0.001 over #26 with a third member that costs 1.4× inference and doubled the scoring
  time (54 min vs 28–32 min for a solo). By the card's "If it fails": the pair stays; a third same-input member reads like a seed.
- **P-56 (c03): 🔁 INCONCLUSIVE by the pre-registered bands** — m = 0.9305 is 0.001 under the ✅ bar; the pair is +0.002 over the c02
  pair. What is *not* noise-shaped is the sign pattern: both c03 seeds (0.932, 0.929) read at or above both c02 seeds (0.927, 0.927 —
  the retrain spread measured at 0.000, P-44), the pair is ahead, and gold-58 put both seeds ahead with 10/12 labels up. Four readings,
  one direction, each under its floor — the same shape as P-43 (320 px, +0.002). **#30 `v11a` 0.932 and #32 the c03 pair 0.932 are our
  best solo reads** (previous: #26 0.930). Neither card branch fires: c03 is not the production input by rule, and the input axis does
  not close either. Cost of c03: 0.35 vs 0.29 s/study in training (+20 %), its own decode pass at inference (c03 members cannot share
  the c02 decode).
- Gold-58 this time had the direction right for all three reads (trio > pair, c03 > c02, `v11a` > `v11b`) — still direction only
  (traps 39: it was wrong on #24 / #26).

**Addendum — #33, the P-54 5-fold cross-fit ensemble (`rsna-knee-infer` v29, `INFER_MEMBERS = ["v09k0", …, "v09k4"]`, sent 11:53:42
UTC) reads 0.928, scored within [65.4, 66.4] min.** Pre-registered vs #26 0.930: ≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌ → **❌ DEAD END
as a member**: five fold models, each trained on 80 % of the studies, rank-mean to 0.928 — about one all-data `v09r`-recipe member
(0.927) and under the all-data 2-seed pair (0.930). Fold count does not buy what data volume costs; a production member keeps training
on all 4,349. The cross-fit's purpose is untouched: its OOF table `xfit_v09k` is the P-55 teacher (session E, running), and P-54 card
branch "If it fails: the table still feeds P-55" is the one that fires.

### 2026-09-29 — Session E (P-55): the OOF soft-bootstrapped student `v09o` / `v09o2` gold-58 SWA **0.9121 / 0.9104** (pair 0.9119 vs the c02 pair 0.9065) · ✅ run green, 🔁 direction only · solo reads next

`rsna-knee-train` v35 (pushed 11:37:33 UTC, `artifacts/train_student_E.py`), COMPLETE in **2.94 h**: `PARALLEL_ARMS = ("v09o", "v09o2")`,
the `v09r` recipe on all 4,349 studies (8 epochs, SWA of the last 3), seeds 42 / `reseeded 43`. Both children log `teacher table
raptor_teacher: 4349` **and** `teacher table xfit_v09k: 4407`, then `training targets = (1 - 0.75) * LLM + 0.75 * quantile-matched
['raptor_teacher', 'xfit_v09k']` (= 0.25 LLM + 0.375 Raptor + 0.375 cross-fit OOF); parent `ok  arm` ×2, no `runtime guard`, `_best.pt`
164,789,267 / 164,789,727 bytes. Pulled to `artifacts/kaggle_out/student_E/`; shipped as the PRIVATE Datasets `rsna-knee-ckpt-v09o` /
`-v09o2` (`ready`) and mounted in `kaggle/rsna-knee-infer/kernel-metadata.json`.

Gold-58 per label (scratch `e_gold.py` = `cd_gold.py` + the two students; same-seed rank-means):

| label | `v09r` | `v09u` | `v09o` | `v09o2` | c02 pair | student pair | c03 pair | student − c02 (pair means) |
|---|---|---|---|---|---|---|---|---|
| ACL | 0.953 | 0.949 | 0.968 | 0.929 | 0.953 | 0.950 | 0.961 | −0.002 |
| MCL | 0.925 | 0.907 | 0.937 | 0.925 | 0.915 | 0.931 | 0.955 | +0.015 |
| Medial Meniscus | 0.958 | 0.970 | 0.966 | 0.970 | 0.966 | 0.969 | 0.959 | +0.004 |
| Lateral Meniscus | 0.853 | 0.870 | 0.867 | 0.886 | 0.868 | 0.878 | 0.889 | +0.015 |
| Medial OA | 0.967 | 0.984 | 0.983 | 0.977 | 0.977 | 0.979 | 0.988 | +0.004 |
| Lateral OA | 0.805 | 0.807 | 0.812 | 0.818 | 0.805 | 0.815 | 0.818 | +0.010 |
| PF OA | 0.816 | 0.812 | 0.826 | 0.826 | 0.815 | 0.825 | 0.838 | +0.012 |
| Effusion | 0.974 | 0.944 | 0.954 | 0.960 | 0.964 | 0.963 | 0.970 | −0.002 |
| Synovitis | 0.823 | 0.816 | 0.834 | 0.828 | 0.823 | 0.830 | 0.812 | +0.011 |
| Baker's | 0.960 | 0.933 | 0.960 | 0.962 | 0.951 | 0.964 | 0.970 | +0.014 |
| Contusion | 0.934 | 0.912 | 0.918 | 0.937 | 0.928 | 0.928 | 0.941 | +0.004 |
| Fracture | 0.943 | 0.890 | 0.919 | 0.907 | 0.913 | 0.912 | 0.948 | −0.003 |
| **macro** | **0.9093** | **0.8995** | **0.9121** | **0.9104** | **0.9065** | **0.9119** | **0.9207** | **+0.007** |

Paired study bootstrap (2,000 reps) student pair − c02 pair **+0.0056 (SD 0.0038)**; the 2-seed mean moves +0.007 with 9/12 labels
up. Within-class ρ: `v09o` ~ `v09o2` **0.946** (c02 seed twins 0.916, c03 twins 0.934), `v09o` ~ `v09r` 0.927, `v09o2` ~ `v09u` 0.929,
`v09o` ~ `v11a` 0.914. Epoch curve (EMA, `v09o`): 0.783 · 0.876 · 0.898 · 0.906 · 0.912 · 0.912 · 0.911 · 0.912 — flat from epoch 4.

**Verdicts.**
- **Run: ✅** — the first training session on a two-table target mix; every green criterion of the in-flight table met.
- **Student (P-55): 🔁 direction only, the solo reads decide.** +0.007 is a seventh of the 0.05 gold floor and 1.5 SD on the paired
  bootstrap — half the c03 edge (+0.014, 3 SD), and on the same four structure labels the c03 input moved (MCL / Lat Men / PF OA, plus
  Lateral OA). The two seeds agree with each other more than any pair so far (ρ 0.946), which is what a heavier teacher weight
  should do (0.75 of the target is shared across seeds, not 0.5) — and is the P-38 signature too, so it is not evidence of truth
  (traps 39). Pre-registered reads (P-55): m = mean(`v09o`, `v09o2`) vs 0.927: **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285**;
  the pair ✅ ≥ 0.935.

### 2026-09-29 — P-59 ResNet-34 pipeline control: a CNN learning rate lifts gold-58 **0.8306 → 0.8992** (`v13b`) · frozen BatchNorm **0.9014** (`v13c`) · ✅ the optimiser was the defect, 🔁 BatchNorm

Origin: tonight's research (three agents: noisy-label literature, prior RSNA MRI/CT winners, an audit of our pipeline). The audit
and agent 2 both traced `v13a`'s collapse to the recipe, not the backbone: `v13a` ended with train loss **0.4708** (CoAtNet `v09u`:
0.3829, and below 0.4708 after 2 epochs) because LLRD 0.75 on `lr_backbone` 1e-4 trains the ResNet's stem at 2.4e-5 and layer4 at
7.5e-5; the audit's local probe also found its train-mode BatchNorm batch-partner-dependent (feature change 0.54 relative L2 vs
CoAtNet 0.32). `rsna-knee-train` v37 (`artifacts/train_p59_real.py` = `src` + 3 seds; smoke v36 green), `PARALLEL_ARMS =
("v13b", "v13c")`, c02, Raptor mix 0.5, all 4,349, **1.08 h**: `v13b` = `v13a` + `lr_backbone` 3e-4, `llrd_decay` 1.0, 12 epochs
(log: `backbone LR range 3.00e-04 .. 3.00e-04`); `v13c` = `v13b` + `freeze_bn` (`36 encoder BatchNorm modules held in eval mode`).

| | train loss ep 7 / final | gold-58 by epoch (EMA) | **SWA 9–11** |
|---|---|---|---|
| `v13a` (8 ep, LLRD) | 0.4708 / — | 0.707 … 0.8309 (still rising) | **0.8306** |
| `v13b` | 0.4086 / 0.3906 | 0.745 · 0.804 · 0.852 · 0.877 · 0.885 · 0.889 · 0.895 · 0.899 · 0.898 · 0.899 · 0.899 · 0.901 | **0.8992** |
| `v13c` | 0.4233 / 0.4006 | 0.721 · 0.760 · 0.822 · 0.863 · 0.879 · 0.889 · 0.893 · 0.895 · 0.899 · 0.901 · 0.900 · 0.903 | **0.9014** |

Per label (gold-58): `v13c` ACL 0.950, **MCL 0.844**, Med Men 0.953, Lat Men 0.870, Med OA 0.975, **Lat OA 0.826, PF OA 0.840**,
Effusion 0.954, Synovitis 0.805, **Baker's 0.998**, Contusion 0.906, Fracture 0.896 — the structure labels recovered from 0.75–0.77
to 0.84–0.95; weakest on MCL (9 positives) where every CoAtNet reads 0.91–0.96. Within-class ρ: `v13c` ~ `v09r` **0.860**, ~ `v11a`
0.854, `v13b` ~ `v13c` 0.872 (the CoAtNet seed twins: 0.916; `v13a` ~ `v09r` 0.667) — the most decorrelated member at this
strength we have. Rank-means on gold: `v09r` + `v13c` 0.9105 (vs `v09r` 0.9093), `v11a` + `v13c` 0.9172 (vs `v11a` 0.9204; paired
bootstrap −0.0031, SD 0.0042), `v11a` + `v11b` + `v13c` 0.9189 (vs the c03 pair 0.9207). Speed: 0.12 s/study on a T4 (CoAtNet c02
0.29, c03 0.35).

**Verdicts.**
- **`v13b` − `v13a` = +0.069 on gold-58 → ✅ KEEP (a pipeline finding): beyond the 0.05 floor, 10/12 labels up, and the train loss
  fell from 0.471 to 0.391.** Our recipe's optimiser (LLRD 0.75 on 1e-4) was tuned for ViT / hybrid backbones and under-trains a
  CNN; P-57's ❌ was the recipe, not the ResNet. Any CNN arm uses `llrd_decay` 1.0 and ≈ 3e-4 from now on.
- **`v13c` − `v13b` = +0.002 → 🔁 INCONCLUSIVE**: frozen BatchNorm is not what cost the ResNet, despite the batch-partner probe.
  Not ported to the CoAtNet on this evidence.
- **The ResNet as a member: 🔁 (gold blends flat)** — a 0.90-gold member with ρ 0.86 to the CoAtNets and a quarter of their cost;
  gold-58 cannot resolve a blend (traps 39), a solo / pair LB read can. Shipped as `rsna-knee-ckpt-v13b` / `-v13c`.
- **Open question it raises (card P-61):** the CoAtNet trains under the same LLRD — its top block at 7.5e-5, stem 2.4e-5; only a
  lower LR was ever probed (`v09d` 3e-5 ❌). An upward CoAtNet LR probe is the untested direction.

### 2026-09-30 — Submissions #34–#38: the P-55 student `v09o` / `v09o2` **0.927 / 0.927**, pair **0.928** → ❌ DEAD END (the OOF-target line closes) · ResNet-34 `v13c` **0.921** ❌ · c03 pair + `v13c` **0.932** = the c03 pair ❌

Five solos from `rsna-knee-infer` v30–v34 (placeholders built and checked 2026-09-29, Submissions rows 34–38; the planned 00:03 UTC
send had not happened), sent 11:06 UTC, each timed by `watch_submission.py` (90 s polls → `artifacts/submission_timing.csv`):

| # | members | gold-58 | public LB | scored within (min) | pre-registered read |
|---|---|---|---|---|---|
| 34 | `v09o` (student, seed 42) | 0.9121 | **0.927** | [18.9, 20.4] | with #35 → |
| 35 | `v09o2` (student, seed 43) | 0.9104 | **0.927** | [20.5, 22.0] | m = 0.9270 vs 0.927: ✅ ≥ 0.9315 / 🔁 0.9285–0.9315 / ❌ < 0.9285 → **❌** |
| 36 | `v09o` + `v09o2` (student pair) | 0.9119 | **0.928** | [28.3, 29.8] | vs the c02 pair #26 0.930: ≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌ → **❌** |
| 37 | `v13c` (ResNet-34, CNN LR, frozen BN) | 0.9014 | **0.921** | [15.6, 17.1] | vs 0.927: ≥ 0.932 ✅ / 0.923–0.931 🔁 / ≤ 0.922 ❌ → **❌** |
| 38 | `v11a` + `v11b` + `v13c` | 0.9189 | **0.932** | [48.9, 50.4] | vs the c03 pair #32 0.932: ≥ 0.936 ✅ / 0.933–0.935 🔁 / ≤ 0.932 ❌ → **❌** |

**Verdicts.**
- **P-55 (OOF soft bootstrapping): ❌ DEAD END — the card's "If it fails" fires and the OOF-target line closes.** Both student seeds
  read 0.927, exactly the two c02 seeds of the `v09r` recipe on the 0.5 LLM + 0.5 Raptor targets (`v09r` 0.927, `v09u` 0.927, P-44
  spread 0.000); the pair reads 0.928, −0.002 under the c02 pair. Moving 0.375 of the target from the LLM / Raptor to honest cross-fit
  OOF of the same recipe changes nothing the public test can see — the third null after P-38 (#19, self-distillation) and Nicolai
  Karcher's report in discussion 735304. Where it did move something, it moved the wrong thing: the seeds agree more (gold within-class
  ρ 0.946 vs 0.916 for the c02 twins), so the pair gains +0.001 over its members instead of +0.003. No attribution control, no second
  cross-fit round (plan Step 5). Gold-58 put the student pair +0.006 (paired SD 0.004, 9/12 labels up) over the c02 pair and the LB put
  it −0.002 — a second target change whose sub-0.01 gold gain did not transfer (traps 39, extended today).
- **P-59 / `v13c` as a member: ❌** — 0.921, −0.006 vs 0.927. The CNN learning rate is a real fix (gold 0.831 → 0.899 / 0.901), but it
  lands ResNet-34 a notch under every CoAtNet member of the same targets and input. As a third family on the c03 pair it reads
  0.932 = the pair (#38): a 0.921 member at within-class ρ 0.86 neither adds nor subtracts under a flat rank-mean, the same shape as
  P-42 (#23, DINOv2-S + `v09r` = `v09r`). By P-57's "If it fails", the small-CNN route closes for accuracy. It stays only as an
  Efficiency-track candidate (P-18): 0.921 at 0.12 s/study, scored in ≈ 16 min vs 19–22 min for a c02 CoAtNet solo today — the
  Efficiency formula is still unread. Gold-58 had both directions right this time (`v13c` < `v09r`; trio ≤ the c03 pair).
- **Scoring times:** c02 CoAtNet solos 19–22 min today (27–32 min on 2026-09-28 / 29 — the same kernels, so Kaggle-side load), the
  ResNet 16–17 min, a c02 pair 28–30 min, the c03 + c02 trio 49–50 min.
- **Where it leaves the member line:** best solo 0.932 (#30 `v11a`, #32 the c03 pair, #38); best LB 0.942 (the fork). Of this week's
  reads (P-52 trio, P-53 probe, P-54 fold ensemble, P-55 student, P-56 c03, P-57 / P-59 ResNet), only c03 read a consistent sign,
  every reading of it under its floor. Still live: P-60 (noisy-student regularisation on c03 — blocked on compute until the 2026-10-03
  reset), P-61 (CoAtNet LR upward), P-45 (a second image teacher), P-50 (final selection).

### 2026-09-30 — P-18 step 1: the public Efficiency LB lists us at **rank 2,533 of 4,489** with the 0.942 fork · it seems to read the best-public-score submission, not the fastest · 🔁 the formula is still unread

`kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb` → `artifacts/efficiency_lb/full_leaderboard.csv` (4,489 teams;
columns `EfficiencyRank, TeamName, PublicScore, DateSubmitted` only — no runtime; latest submission in the snapshot 2026-09-28 21:31
UTC). The notebook only re-publishes `leaderboard.csv` from `ryanholbrook/rsna-knee-abnormalities-efficiency-data`, which the CLI
cannot read (`kernels.get` denied — private), and points to the competition's "Efficiency Prize Evaluation" page for the formula.

- **Our row: rank 2,533, public 0.942, dated 2026-09-27 16:36:17** = #22 (the fork, β 0.10 — hours to score). #26 (0.930, scored in
  28 min, sent 2026-09-28 09:32) was already in the snapshot and is not what it lists, so the track appears to read each team's
  best-public-score submission (the latest of our three 0.942s) or its selected ones — inference from one row, not a rule.
- **Runtime weighs heavily:** in 1,996 of 4,488 adjacent pairs the lower-ranked team has the higher public score; the top 100 span
  0.917–0.958 (median 0.940; top 10 median 0.9545), rank 32 = 0.936, rank 39 = 0.931. Scott Willis (0.958) is #1 on both boards.
- **Candidates of ours** (inference cost only; the formula may also count CPU / wall time differently): `v11a` 0.932 (20–28 min to
  score), `v13c` 0.921 (≈ 16 min, 0.12 s/study). Our slowest-possible entry (the fork) is what is listed now.

**Verdict: 🔁 INCONCLUSIVE** — a placement read, not a measurement of ours; step 2 needs the Evaluation page (which submission counts,
the formula) in a browser (brainstorm.md "Efficiency Prize"). Until then the Efficiency track is decided by what we *select* at the end
(P-50), not by what we submit.

### 2026-09-30 — P-60 part 1 (`rsna-knee-folds` v11): `v11n` / `v11n2` guard-stopped in epoch 5 of 12 as planned · gold-58 EMA **0.9166 / 0.9129 at epoch 4** = `v11a` / `v11b` at their epoch 4 (0.9156 / 0.9129) · ✅ run green, ⏳ P-60 waits for part 2

`rsna-knee-folds` v11 (pushed 12:10 UTC, `artifacts/train_p60_kaggle_part1.py` = `src` + 4 seds: `FORCE_SMOKE = False`,
`PARALLEL_ARMS = ("v11n", "v11n2")`, `TEACHER_TABLES = ("raptor_teacher",)`, `runtime_limit_hours` 8.3 → 2.75), COMPLETE in **2.61 h**
(9,398 s; children `RSNA_RUNTIME_H 2.56 h`). Both children: c03 cache (4 shards, 4,407 studies), `teacher table raptor_teacher: 4349`,
`training targets = (1 - 0.5) * LLM + 0.5 * quantile-matched ['raptor_teacher']`, `drop_path_rate 0.1`, `aug heavy`, `epochs 12`,
`swa_last 3`; `v11n2` `reseeded 43`; peak GPU memory 9.44 GiB each (batch 2 × 34 windows). Parent: `ok  arm` ×2, `_last.pt present`,
last lines `stopping: runtime guard`; `all folds complete: False` → no inference. Pulled to `artifacts/kaggle_out/p60_part1/` (logs +
per-epoch OOF csvs only; the checkpoints stay in the kernel output for part 2 to mount). `kaggle quota` after it: **29.42 h used,
0.58 h left**, reset 2026-10-03 00:00 UTC.

Throughput: `v11n` (cuda:0) 0.36 s/study = 26.2 min/epoch, `v11n2` (cuda:1) 0.38 s/study = 27.7 min/epoch, + 0.7–0.8 min held-out
eval each — 3–8 % slower than session D's `v11a` / `v11b` (0.35 s/study, 25.0–25.6 min), the cost of drop-path + heavy GPU aug. The
guard fired at 17.0 of 26.3 min into epoch 5 for `v11n` (≈ 65 % of the epoch) and at 10.2 of 27.7 min for `v11n2` (≈ 37 %).

Gold-58 EMA per epoch (all 58, reported only; epoch 5 is the partial epoch):

| epoch | `v11a` (8 ep) | `v11n` (12 ep + reg) | `v11b` (8 ep) | `v11n2` (12 ep + reg) | c03 pair mean | P-60 pair mean |
|---|---|---|---|---|---|---|
| 0 | 0.8031 | 0.7807 | 0.7838 | 0.7737 | 0.7935 | 0.7772 |
| 1 | 0.8645 | 0.8535 | 0.8573 | 0.8505 | 0.8609 | 0.8520 |
| 2 | 0.8946 | 0.8953 | 0.8923 | 0.8839 | 0.8935 | 0.8896 |
| 3 | 0.9089 | 0.9052 | 0.9047 | 0.9028 | 0.9068 | 0.9040 |
| 4 | 0.9156 | 0.9166 | 0.9129 | 0.9129 | 0.9143 | **0.9148** |
| 5 | 0.9174 | 0.9151 (≈ 65 %) | 0.9141 | 0.9148 (≈ 37 %) | 0.9158 | 0.9150 |
| SWA | 0.9204 | ⏳ | 0.9167 | ⏳ | 0.9186 | ⏳ |

The same two arms on the RunPod A100 (2026-09-29, pod 2, killed at epoch 5 by the balance): `v11n` 0.776 · 0.852 · 0.882 · 0.898 ·
0.911, `v11n2` 0.773 · 0.842 · 0.881 · 0.900 · 0.912 — Kaggle reproduces them within 0.006 at every epoch. (The card's "`v11a` ep 4 ≈
0.906" was wrong; session D's log says 0.9156.)

**Resume mechanics (read in `train_fold`, `src/kaggle_pipeline.py:2690–2832`):** a guard stop saves `_last.pt` with `epoch = 5` as if
complete, so part 2 starts at epoch 6 and the unfinished rest of epoch 5 is never trained — ≈ 0.35 epoch for `v11n`, ≈ 0.63 epoch for
`v11n2` (3 % / 5 % of the 12-epoch schedule). The cosine LR runs on optimiser steps and the scheduler state is restored, so the LR
continues from the exact step; its last ≈ 3 % / 5 % is never reached (final LR ≈ 0.3 % / 0.8 % of peak instead of 0; warm-up 10 %). The partial
epoch never enters the SWA ring (`not guard_hit`), and the ring saved in `_last.pt` (epochs 2–4) is pushed out by 9–11, so the
registered SWA of epochs 9–11 holds. The twins end ≈ 0.3 epoch apart in data seen — well inside seed noise.

**Verdicts.**
- **Run: ✅** — every part-1 criterion met; the first Kaggle session that stops by design and hands `_last.pt` to a resume.
- **P-60: ⏳ PENDING** — nothing is readable before part 2. Direction only: the regularised arms start slower (pair mean −0.016 at
  epoch 0, −0.009 at epoch 1) and have caught up by epoch 4 (+0.0005) while their 12-epoch cosine is still at ≈ 72 % of peak LR vs
  ≈ 37 % for the 8-epoch `v11a` / `v11b` — the shape the card predicts, but a same-recipe gold direction has been wrong before (traps
  39), and ≈ 0.0005 is a hundredth of the gold floor. The epoch-5 `_best.pt` files in the v11 output are mid-schedule EMA snapshots,
  **not members** (traps 47). Part 2 = `rsna-knee-train` resume after 2026-10-03 00:00 UTC, ≈ 3.0 h (6 epochs × 27–28.5 min + the
  SWA pass); then two solo reads against the card's band (m vs 0.9305: ✅ ≥ 0.935 / ❌ ≤ 0.926).

### 2026-09-30 — Silence-aware teacher mix priced on gold-58 (target level, P-49's method): Raptor 0.75 where the report is silent, 0.5 where it speaks → **0.9300 vs 0.9268** (+0.0032, SD 0.0018, 5 up / 1 down); flat mixes at the same mean Raptor weight +0.0004 · 🔁 direction only → card P-62

Question (Tian: "anything we could do in the meantime, the labels?"): the LLM half turns a *silent* report into a confident-looking
negative (label audit 2026-08-28: blend ≈ 0.18, weight 0.69 on UNK cells; 14 of the 41 Synovitis-silent gold studies are positive),
so does the image teacher deserve more weight exactly there? Scratch `silence_mix.py` (session scratchpad), 0 GPU: yt = (1 − w) · LLM
blend + w · matched Raptor on the 58 gold, with w = `w_sil` where pilkwang's verdict is `UNK` and `w_addr` where it is YES / NO; gold
Raptor (`artifacts/teacher/raptor_gold.csv`) mapped through the 4,349-study Raptor ECDF onto the LLM blend's report-only values, as
training does. The flat-0.5 baseline reproduces P-49's **0.9268** exactly. Paired study bootstrap (2,000 reps) against it.

| `w_addr` / `w_sil` | mean Raptor w (gold) | macro | vs flat 0.5 | SD | labels up / down |
|---|---|---|---|---|---|
| 0.5 / 0.5 (= today) | 0.500 | 0.9268 | — | — | — |
| **0.5 / 0.75** | 0.564 | **0.9300** | **+0.0032** | 0.0018 | 5 / 1 |
| 0.5 / 1.0 | 0.627 | 0.9291 | +0.0023 | 0.0037 | 4 / 4 |
| 0.4 / 0.8 | 0.502 | 0.9280 | +0.0012 | 0.0026 | 4 / 6 |
| 0.25 / 0.75 | 0.377 | 0.9258 | −0.0010 | 0.0031 | 4 / 6 |
| 0.25 / 1.0 | 0.441 | 0.9260 | −0.0008 | 0.0045 | 3 / 7 |
| flat 0.6 / 0.65 / 0.7 (controls) | 0.60 / 0.65 / 0.70 | 0.9272 / 0.9263 / 0.9260 | +0.0004 / −0.0005 / −0.0008 | 0.002–0.003 | 5/6 · 5/6 · 5/5 |

Silent cells on gold: Synovitis 41, Baker's 29, Fracture 26, Lateral OA 22, Medial OA 14, PF OA 14, Contusion 10, ≤ 7 elsewhere
(training rows: Synovitis 84 %, Fracture 57 %, Baker's 46 %, Lateral OA 33 %, Medial OA 26 %). Inside the silent cells the LLM
ranks gold positives no better than chance or worse — Fracture **0.170** (4 positives of 26), Baker's 0.482 (1 of 29), Synovitis
0.749 (14 of 41) — and Raptor 0.744 / 0.696 / 0.824. Per label at 0.5 / 1.0: Fracture 0.872 → **0.913**, PF OA 0.909 → 0.920,
Contusion 0.913 → 0.919, Lateral OA 0.819 → **0.798** (22 silent, none positive: more Raptor lifts silent negatives over addressed
positives), Synovitis 0.830 → 0.821.

**Verdict: 🔁 direction only — a target-level analog, never a verdict.** It is the first label-side re-weighting that beats the flat
mix at a matched amount of Raptor (+0.003 vs +0.0004), so the gain is *where* the image teacher speaks, not *how much*; but (1) it is
1.8 SD on 58 studies, (2) Raptor is optimistic on these 58 (its authors chose epochs on them), which flatters any variant that leans
on it, and (3) the student shrinks target gains (P-49: target 0.9268 → `v09r` 0.909) — an LB effect of ≈ +0.001–0.002 would sit under
the 0.004 one-seed floor. What the reading also closes: lowering Raptor on addressed cells (0.25 / x) and raising it everywhere
(flat 0.6–0.7) are both flat-to-negative, and the dread table cannot be priced at all (no gold rows, P-30). Card **P-62** (proposals.md).

### 2026-10-03 — P-45 step 1: D4 as a second image teacher, priced on gold-58 (target level, P-49's method) → 0.5 LLM + 0.25 Raptor + 0.25 D4 **0.9331 vs 0.9268** (+0.0063, SD 0.0041, 8 up / 4 down); D4 ~ Raptor within-class ρ **0.757** · 🔁 direction only

Source: D4's own gold-58 predictions ship inside its public CC0 Dataset `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` as
`d4_gold58_reference.npz` (study_uids, `raw_probabilities` (1, 58, 12), `probability_mean`, `truth`; pulled to `artifacts/d4_ds/`,
gitignored). Its `truth` equals our gold labels column for column (asserted). Its macro is **0.9302**, the card's figure; the
Dataset's `d4_train_serving_equivalence.json` names the checkpoint
`coatnet_rmlp2_…_global96_k12full94_effb24_e24_smart384_full4349_gold58_s42_v1/gold_swa_top2.pt` (CoAtNet-rmlp-2 @384, 24
epochs, SWA of the top-2 epochs **chosen on gold**: optimistic, like Raptor). Scratch `d4_gold.py` (session scratchpad), 0 GPU.

**Matching.** D4 has no 4,349-row table, so its gold values cannot go through its own training ECDF. They are mapped by their mid-rank
among the 58 gold studies onto the LLM blend's report-only quantiles ("gold-rank"). Checked on Raptor, where both methods are possible:
0.5 / 0.5 with gold-rank Raptor reads **0.9262** vs 0.9268 for P-49's train-ECDF matching (−0.0007, SD 0.0035). So gold-rank is a fair
stand-in.

| target (yt on gold) | macro | vs P-49 0.5 / 0.5 (SD) | labels up / down |
|---|---|---|---|
| LLM blend alone | 0.8948 | −0.0320 (0.0081) | 0 / 12 |
| 0.5 LLM + 0.5 Raptor (P-49, today's production target) | **0.9268** | — | — |
| 0.5 LLM + 0.5 D4 | 0.9312 | +0.0043 (0.0057) | 6 / 6 |
| **0.5 LLM + 0.25 Raptor + 0.25 D4** | **0.9331** | **+0.0063 (0.0041)** | **8 / 4** |
| 0.4 LLM + 0.3 Raptor + 0.3 D4 | 0.9347 | +0.0079 (0.0044) | 9 / 3 |
| 0.5 Raptor + 0.5 D4 (no LLM) | 0.9290 | +0.0022 (0.0086) | 6 / 6 |

Per label (LLM / Raptor / D4 / 0.5-0.5 target / 3-way): the 3-way gains on Effusion 0.943 → **0.971**, Baker's 0.957 → 0.978,
Fracture 0.872 → 0.888, PF OA 0.909 → 0.923, Lateral Meniscus 0.928 → 0.933; it loses on Contusion 0.913 → 0.904, ACL 0.995 → 0.991,
Synovitis 0.830 → 0.827, Medial Meniscus 0.980 → 0.977. D4 alone is the weakest of the three sources on ACL (0.958).

**Within-class ρ on gold** (P-49's definition): D4 ~ Raptor **0.757**, D4 ~ LLM 0.409 (Raptor ~ LLM 0.441); our members track Raptor
more than D4: `v11a` ~ Raptor 0.824 vs `v11a` ~ D4 0.781 (`v11b` ~ D4 0.785, `v09r` ~ D4 0.763; the seed twins `v11a` ~ `v11b` 0.934).
So D4 carries image information the student does not get from Raptor today. Reviewer A's worry (D4 is "Raptor's family, little target
diversity") does not hold at ρ 0.757; for comparison our two-family Raptor pair `v09r` ~ `v08r` was 0.860.

**Verdict: 🔁 direction only. This is a target-level analog, not a verdict.** The three-source target is +1.5 SD over today's on 58
studies, the largest target-level gain priced since P-49 (silence-aware mixing: +0.0032; flat mixes: ≤ +0.0004). But both image
teachers chose their epochs on these same 58 studies, which flatters any mix that leans on them. The student also shrinks target
gains (P-49: target 0.9268 → `v09r` 0.909 / `v11a` 0.920 on gold). By the one precedent, Raptor (target +0.032 on gold → +0.009
solo LB), a +0.006 target gain is worth ≈ +0.002 on the LB, under the 0.004 one-seed floor for one seed and readable as m over two.
The reading justifies the P-45 builder and a 58-study gold spike, which must reproduce 0.9302. The full pass and the arm need
Tian's go (card P-45).

### 2026-10-03 — P-45 gold spike (`rsna-knee-teacher-d4` v1, 0.1 h on 2×T4): D4 run by our builder reproduces its own gold reference on its original grid — macro **0.9301 vs 0.9302**, max |Δp| 0.0083, mean 0.0010 · ✅ wiring verified; the original grid is the pass grid

Tian's go for P-45 (2026-10-03, after P-62 was stopped). `src/build_d4_teacher_pass.py --gold --sub-chunk 30`: the 58 gold studies
under both input grids, 0 failed, scored against `d4_gold58_reference.npz` (computed by D4's authors on their own hardware).

| grid | macro AUC (ours) | vs reference 0.9302 | max / mean \|Δp\| | authors' T4 tolerance (max 0.03, mean 0.003, AUC −0.003) |
|---|---|---|---|---|
| **original** (D4's own Global96 stack) | **0.9301** | −0.0001 | 0.0083 / 0.0010 | **PASS** |
| notebook (the 0.942 graph's capacity-aware dense grid) | 0.9290 | −0.0012 | 0.1227 / 0.0104 | FAIL |

The strict 1e-3 line fails for both, as expected: the reference was computed on different hardware and a different OpenCV build.
**The rule set before the read: the original grid unless it fails to run.** It is also the grid the reference and the gold
analog (0.9331) describe. So the 0.942 notebook feeds D4 a grid it was not trained on, at a −0.001 cost on gold.

**Timing:** the original grid runs ≈ 64–68 s per 29-study child (2 workers, one per T4) incl. ≈ 22 s of per-child overhead → ≈ 1.5
s/study; peak 1.76 GiB per T4. The full pass (4,349 studies, sub-chunks of 400) is projected at **≈ 2 GPU-h in one 2×T4 session**,
pushed as `rsna-knee-teacher-d4` v2 at 09:58 UTC ⏳.

Caveat carried to the student: D4's checkpoint name reads `full4349_gold58`, so its predictions on the 4,349 training studies are
**in-sample** (it was fitted on them, with its own unknown "DualSharp" LLM targets). Raptor was in the same position and transferred
(+0.009 LB), but after the pass compare D4-train ~ LLM against Raptor-train ~ LLM on the 4,349. If D4 tracks the LLM labels much
more closely than Raptor does, it is replaying labels rather than adding image information.

### 2026-10-03 — Quantile matching flattens the teacher: matched Raptor keeps 14–32 distinct values per label on the 58 gold; raw-probability mixing reads 0.9308 vs 0.9268 matched (+0.004, SD 0.0044, 4 up / 7 down) · 🔁 direction only, not adopted

Question: the LLM blend is a discrete, low-valued distribution on under-reported labels (report-only positive rate at a 0.5 cut:
Synovitis 0.12 vs raw Raptor 0.41 vs gold 0.47; Fracture 0.07 / 0.21 / 0.31; Lateral Meniscus 0.15 / 0.37 / 0.40). Mapping the
teacher onto those quantiles (`quantile_match`, training's method) may collapse the teacher's ranking where the LLM says nothing.
Scratch `match_vs_raw.py`, 0 GPU, P-49's method, paired bootstrap 1,000 reps.

| target on gold | macro | vs today (SD) | labels up / down |
|---|---|---|---|
| 0.5 LLM + 0.5 matched Raptor (today, P-49) | 0.9268 | — | — |
| 0.5 LLM + 0.5 raw Raptor | 0.9308 | +0.0040 (0.0044) | 4 / 7 |
| rank mix 0.5 / 0.5 (scale-free reference) | 0.9320 | +0.0052 (0.0040) | 7 / 5 |
| 0.4 LLM + 0.6 raw Raptor | 0.9337 | +0.0069 (0.0043) | 7 / 5 |
| 0.5 LLM + 0.25 raw Raptor + 0.25 raw D4 | 0.9345 | +0.0077 (0.0053) | 7 / 5 |
| rank mix 0.5 LLM + 0.25 R + 0.25 D4 | 0.9380 | +0.0112 (0.0044) | 7 / 4 |

Matched vs raw at 0.5 / 0.5, per label: Fracture 0.872 → 0.907, Effusion 0.943 → 0.971, Baker's 0.957 → 0.976, Lateral OA +0.011;
Contusion −0.016, PF OA −0.010, ACL −0.006, Medial / Lateral Meniscus −0.005. Among the 58 gold studies, matched Raptor has only
14 (Baker's) to 32 (Contusion) distinct values; the LLM blend's distinct values on the 4,349 rows run from 85 (Baker's) to 234.

**Verdict: 🔁 direction only, not adopted.** Matching does flatten the teacher (the ties are real), but the matched → raw gain is
1 SD, and the labels split 4 up / 7 down. For the three-source target the raw form is only +0.0014 over matched (0.9345 vs 0.9331).
A rank mix is not a training target: P-00 showed rank-percentile targets put confident negatives at ≈ 0.3. Raw teacher
probabilities also move the target's operating point: MCL would be 0.32 positive vs gold 0.16. So the P-45 student keeps matched
mixing, the form that transferred for Raptor (+0.009 LB), and adds D4 as one clean change. Raw mixing is a candidate to bundle into a
final recipe, not a session of its own.

### 2026-10-03 — P-60 part 2 (`rsna-knee-train` v39, 2.84 h): the first Kaggle resume works; `v11n` / `v11n2` gold-58 SWA **0.9152 / 0.9166** (m 0.9159 vs `v11a` / `v11b` m 0.9186) · ✅ run green, ⏳ solos #39 / #40

`artifacts/train_p60_kaggle_part2.py` (= part 1's build + the 8.3 h guard) with `artifacts/p60_part2_kernel-metadata.json` (c03 ×4 +
`rsna-knee-folds`), pushed 09:08 UTC, COMPLETE 12:16 UTC. Each child log has `resume: copied v11n*_fold0_last.pt into WORK` and
`resumed fold 0 at epoch 6 (best 0.915x at epoch 5)`, epochs 6–11, `SWA of last 3`, no runtime guard. This is the first time a
guard-stopped Kaggle run resumed across sessions (traps 31). Gold-58 EMA by epoch (6 → 11): `v11n` 0.9162 · 0.9163 · 0.9158 · 0.9146
· 0.9158 · 0.9155, SWA **0.9152** (CI 0.885–0.941); `v11n2` 0.9163 · 0.9195 · 0.9175 · 0.9159 · 0.9159 · 0.9162, SWA **0.9166**. Gold
plateaus from epoch ≈ 5 and does not rise over the longer, more regularised tail. Direction only (gold has inverted the LB order in
≥ 4 forum reports; research.md 2.7.3). Shipped as `rsna-knee-ckpt-v11n` / `-v11n2`; solos #39 / #40 sent 12:26 / 12:30 UTC
(pre-registered: m vs 0.9305 — ✅ ≥ 0.935 / 🔁 0.926–0.935 / ❌ ≤ 0.926).

**Read (2026-10-03, 13:00 UTC): #39 `v11n` 0.932, #40 `v11n2` 0.932 → m = 0.932 vs 0.9305 → 🔁 INCONCLUSIVE (+0.0015, inside
the band).** The noisy-student bundle (drop-path 0.1, heavy augmentation, 12 epochs) does not move the c03 CoAtNet on the LB.
Gold called it flat too (m 0.9159 vs 0.9186). The seeds agree more (0.932 / 0.932 vs `v11a` / `v11b` 0.932 / 0.929): +0.003 on
seed 43, = on seed 42. Not adopted as the production recipe (1.5× training time for a gain under every floor). `v11n` / `v11n2` are
interchangeable 0.932 members. **What it closes:** for the CoAtNet, regularisation and a longer schedule are not where the forum's
0.94–0.95 single models come from. The small-CNN version (`v13h`, P-64) is the other half of that question and is running.

### 2026-10-03 — P-45 D4 pass complete (`rsna-knee-teacher-d4` v2, 2.37 h): 4,349 / 4,349 studies, 0 failed; the in-sample check passes (ρ D4 ~ LLM **0.695** vs Raptor 0.652, bar 0.80) · ✅ table published

`--grid original --sub-chunk 400`: 11 child runs of 395–396 studies, 731–806 s each, 1.85 s/study + 45 s per run; peak 1.49 GiB per T4.
`merge_teacher.py --teacher d4` → `artifacts/teacher/d4_teacher.csv` (4,349 rows, md5 `2990af2c`). New version of the private Dataset
`rsna-knee-teacher-tables` (the other three tables md5-identical).

**The in-sample check** (D4 was fitted on these studies, like Raptor). Rule set before the read: a macro ρ(D4, LLM) ≥ 0.80 against
Raptor's ≈ 0.65 would mean it replays the labels, and the student would stop. Scratch `d4_insample.py`, report-only rows:

| macro over 12 labels | D4 | Raptor |
|---|---|---|
| Spearman ρ with the LLM blend | **0.695** | 0.652 |
| AUC vs the hard LLM label (> 0.5) | 0.945 | 0.907 |
| positive rate at a 0.5 cut | 0.249 | 0.326 (LLM 0.267) |
| ρ D4 ~ Raptor, all cells / report-silent cells | 0.859 / 0.724 | — |

D4 sits slightly closer to the reports than Raptor does (+0.04 ρ, as an in-sample fit would), far from replay. It disagrees with
Raptor most where the report is silent (ρ 0.724), which is where the image half matters. Per label, D4 is closest to the LLM on
Medial Meniscus 0.853, Effusion 0.838 and PF OA 0.802, and least close on Fracture 0.443 and MCL 0.485. Its operating point follows
the reports: Synovitis 0.256 positive vs Raptor 0.410, LLM 0.124 and gold 0.47. Quantile matching removes that difference in
training.

**Verdict: ✅ table published; the student pair runs.** Session B = `rsna-knee-train` v41 (12:35 UTC):
`PARALLEL_ARMS = ("v11d", "v11dl")`, `TEACHER_TABLES = ("raptor_teacher", "d4_teacher")`, mix 0.5 → 0.5 LLM + 0.25 matched Raptor +
0.25 matched D4. `v11d` is the `v11a` recipe and reads P-45; `v11dl` adds lr 2e-4 / LLRD 0.85 and reads P-61 against `v11d` in
the same session. Kaggle smoke v40 was green (both tables read; LR ranges 2.4e-5 → 1e-4 vs 8.9e-5 → 2e-4).

### 2026-10-03 — Sessions A (`rsna-knee-train-b` v4, 4.40 h) ‖ B (`rsna-knee-train` v41, 3.82 h): four c03 arms, gold-58 SWA `v11p` **0.9185** · `v13h` **0.9001** · `v11d` **0.9184** · `v11dl` **0.9132** vs `v11a` 0.9204 / `v13c` 0.9014 · ✅ runs green, 🔁 gold flat (direction only) · solos next

Both pushed ≈ 12:21 / 12:35 UTC, COMPLETE before 17:20 UTC. Every child log has the right teacher table(s) (`raptor_teacher: 4349`,
plus `d4_teacher: 4349` in B), cache `c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20` (= c03), `SWA of last 3`,
`-> <arm>_fold0_best.pt = SWA`, no runtime guard (traps 47 checked). `v13h`: `freeze_bn: 36 encoder BatchNorm`, drop-path 0.1, LR
3e-4 uniform, epochs 0–29. `v11dl`: LR 8.87e-05 .. 2.00e-04 (decay 0.85). One seed (42) per idea.

Gold-58 per label (SWA; `v11a` = c03 production recipe, `v13c` = 12-epoch light-aug ResNet-34):

| label | `v11a` | `v11p` (P-63) | `v11d` (P-45) | `v11dl` (P-61) | `v13c` | `v13h` (P-64) |
|---|---|---|---|---|---|---|
| ACL | 0.962 | 0.966 | 0.939 | 0.947 | 0.950 | 0.972 |
| MCL | 0.955 | 0.964 | 0.952 | 0.934 | 0.844 | 0.898 |
| Medial Meniscus | 0.958 | 0.950 | 0.965 | 0.972 | 0.953 | 0.940 |
| Lateral Meniscus | 0.896 | 0.916 | 0.882 | 0.872 | 0.870 | 0.827 |
| Medial OA | 0.986 | 0.988 | 0.986 | 0.992 | 0.975 | 0.988 |
| Lateral OA | 0.822 | 0.828 | 0.838 | 0.818 | 0.826 | 0.836 |
| PF OA | 0.837 | 0.844 | 0.844 | 0.813 | 0.840 | 0.865 |
| Effusion | 0.969 | 0.954 | 0.966 | 0.978 | 0.954 | 0.906 |
| Synovitis | 0.806 | 0.784 | 0.806 | 0.799 | 0.805 | 0.779 |
| Baker's | 0.966 | 0.989 | 0.973 | 0.980 | 0.998 | 0.975 |
| Contusion | 0.947 | 0.928 | 0.931 | 0.934 | 0.906 | 0.945 |
| Fracture | 0.942 | 0.912 | 0.938 | 0.918 | 0.896 | 0.872 |
| **macro** | **0.9204** | **0.9185** | **0.9184** | **0.9132** | **0.9014** | **0.9001** |

- `v11p` vs `v11a` −0.002 (7 up / 5 down); `v11d` vs `v11a` −0.002 (4 up / 6 down); `v11dl` vs `v11d` −0.005 (6 / 6); `v13h` vs
  `v13c` −0.001 (6 / 6). All far inside the 0.05 gold floor; signs scatter like seeds.
- **`v13h`'s curve plateaus at ≈ 0.900 from epoch 12** (0.9000 at 12, 0.9023 peak at 18, 0.9000 at 29): on gold, 30 epochs + heavy
  aug buy nothing over `v13c`'s 12. Gold has inverted the LB order in ≥ 4 forum reports (research.md 2.7.3), so the solo decides.
- **`v11dl` (P-61) learns slower early and ends lower** (epoch 2 0.876 vs `v11d` 0.896; final EMA 0.9147 vs 0.9178).
- Within-class ρ to `v11a` on gold: `v11p` 0.965, `v11d` 0.964 (seed-like); `v13h` 0.920 (= `v13c` 0.922). Gold rank-means:
  `v11a` + `v11d` 0.9222, `v11a` + `v11p` 0.9213, `v11a` + `v13h` 0.9170 (direction only).

Kaggle GPU 14.10 / 30 h after both (resets 2026-10-10). **Verdict: ✅ the runs are green and shippable; 🔁 gold is flat for all four
ideas (none clears its floor either way).** Read rules pre-registered in handoff 2026-10-03: `v11p` / `v11d` vs `v11a` 0.932 — ✅
≥ 0.936 / 🔁 0.929–0.935 / ❌ ≤ 0.928; `v13h` vs `v13c` 0.921 — ✅ ≥ 0.925, matters as a member ≥ 0.930, CNN line becomes the main
bet ≥ 0.935; `v11dl` vs `v11d` ± 0.004. Solos `v11p`, `v13h`, `v11d` tonight, `v11dl` 2026-10-04.

### 2026-10-03 — Submissions #41–#43: ResNet-34 `v13h` **0.931** (✅ +0.010 vs `v13c` 0.921) · `v11p` (P-63 reader) **0.929** (🔁 −0.003) · `v11d` (P-45 D4 targets) **0.930** (🔁 −0.002) — the long heavy-aug CNN is the one change that moved, and gold called it flat

Placeholders `rsna-knee-infer` v37 / v38 / v39 (built from current `src` + 3 seds; each green: `smoke False`, `infer members (1):
<arm>/fold0`, c03 decode-once verified, `constant labels 0`; `v11p` loads strictly, so its reader came from the checkpoint config).
Sent 17:32 / 17:39 / 17:43 UTC, read by 18:22 UTC, timed by `src/watch_submission.py`.

| # | member | gold-58 SWA | LB | scored within | pre-registered read | verdict |
|---|---|---|---|---|---|---|
| 41 | `v11p` = `v11a` + spatial reader + slot-count norm | 0.9185 | **0.929** | [30.5, 32.0] min | vs `v11a` 0.932: ✅ ≥ 0.936 / 🔁 0.929–0.935 / ❌ ≤ 0.928 | 🔁 (bottom edge, −0.003) |
| 42 | `v13h` = ResNet-34, c03, CNN LR, frozen BN, heavy aug, drop-path 0.1, 30 ep | 0.9001 | **0.931** | [17.0, 18.5] min | vs `v13c` 0.921: ✅ ≥ 0.925; member ≥ 0.930; main bet ≥ 0.935 | **✅ KEEP (+0.010)**; member yes; main bet no |
| 43 | `v11d` = `v11a` recipe on 0.5 LLM + 0.25 Raptor + 0.25 D4 | 0.9184 | **0.930** | [36.4, 37.9] min | vs `v11a` 0.932: same bands as #41 | 🔁 (−0.002) |

**What this says.**

1. **P-64 ✅: the forum's CNN recipe transfers.** The same ResNet-34 goes 0.921 → 0.931. That is +0.010, beyond both the one-seed
   floor (0.004, P-44) and the LB floor (0.005). It changes four things at once versus `v13c`: c03 input, heavy aug, drop-path 0.1,
   and 30 epochs instead of 12. Gold read it as −0.001 and its curve was flat from epoch 12, so gold-58 has again missed an LB
   change it should have seen (traps 39 extended).
2. **`v13h` matches our best CoAtNet (0.932) at about half the cost.** It scores in ≈ 18 min vs 30–38 min for a c03 CoAtNet solo,
   and the checkpoint is 86 MB vs 165 MB. Its within-class ρ to `v11a` on gold is 0.920 vs ≈ 0.965 for a CoAtNet variant, so it is
   the different family the ensemble needs (research.md 2.7.4: different families pay +0.005–0.007). It is also an Efficiency-prize
   candidate (P-18).
3. **P-63 🔁, not adopted.** D4's per-finding reader on our CoAtNet reads −0.003 at one seed; the forum prior was weak.
4. **P-45 🔁, not adopted for training targets.** The D4 teacher's +0.006 at gold target level did not reach the LB (0.930 vs
   0.932), the same pattern as P-55 (traps 39). Later members stay on 0.5 LLM + 0.5 Raptor. `v11d` remains a 0.930 member on a
   different teacher mix.
5. The CoAtNet band is now six reads, 0.929–0.932: `v11a`, `v11b`, `v11n`, `v11n2`, `v11p`, `v11d`.

**Verdict:** ✅ KEEP the P-64 recipe for CNN members (`v13h` = production CNN member). 🔁 P-63 and P-45 (as a training target) are
closed, not adopted. `v11dl` (P-61) is read 2026-10-04 (placeholder `rsna-knee-infer` v40 green). The next tests are two: a
two-family blend `v11a` + `v13h` (keep only if ≥ 0.936, the best member + 0.004), and the staged session C (ResNet-50 ‖
EfficientNet-B0 on the `v13h` recipe, needs Tian's go).

### 2026-10-04 — Session C (`rsna-knee-train` v43, 5.87 h): the `v13h` recipe on ResNet-50 (`v13r`) and EfficientNet-B0 (`v13e`) · gold-58 SWA **0.9111 / 0.9126** vs `v13h` 0.9001 · ✅ runs green, 🔁 direction only · solos next

Pushed 18:35 UTC 2026-10-03 on Tian's go (Kaggle smoke v42 green), COMPLETE overnight. Both child logs: `teacher table
raptor_teacher: 4349`, c03 cache, LR 3e-4 uniform, `freeze_bn` 53 / 49 encoder BatchNorm modules, drop-path 0.1, epochs 0–29,
`SWA of last 3`, `-> <arm>_fold0_best.pt = SWA`, no runtime guard (traps 47 checked). EfficientNet's `0 stages` / `over 0 blocks`
line is cosmetic: at `llrd_decay` 1.0 every encoder parameter takes the uniform LR (`param_groups`' fall-through branch). Training
speed 0.30–0.31 s/study (≈ 22 min/epoch, both arms), vs `v13h` 0.09 s/study; the session is 5.87 h vs 4.40 h for A. Checkpoints:
`v13r` 92 MB, `v13e` 17 MB. Kaggle GPU 20.14 / 30 h after it. Shipped as `rsna-knee-ckpt-v13r` / `-v13e`, mounted in `rsna-knee-infer`.

| label | `v11a` | `v13h` | `v13r` (ResNet-50) | `v13e` (EffNet-B0) |
|---|---|---|---|---|
| ACL | 0.962 | 0.972 | 0.980 | 0.957 |
| MCL | 0.955 | 0.898 | 0.880 | 0.952 |
| Medial Meniscus | 0.958 | 0.940 | 0.966 | 0.946 |
| Lateral Meniscus | 0.896 | 0.827 | 0.886 | 0.853 |
| Medial OA | 0.986 | 0.988 | 0.986 | 0.984 |
| Lateral OA | 0.822 | 0.836 | 0.801 | 0.832 |
| PF OA | 0.837 | 0.865 | 0.867 | 0.875 |
| Effusion | 0.969 | 0.906 | 0.963 | 0.916 |
| Synovitis | 0.806 | 0.779 | 0.781 | 0.826 |
| Baker's | 0.966 | 0.975 | 0.987 | 0.978 |
| Contusion | 0.947 | 0.945 | 0.942 | 0.970 |
| Fracture | 0.942 | 0.872 | 0.893 | 0.861 |
| **macro** | **0.9204** | **0.9001** | **0.9111** | **0.9126** |

- vs `v13h`: `v13r` +0.011 (8 up / 4 down), `v13e` +0.0125 (8 / 4). Gold has missed the LB direction for CNN recipe changes
  before (traps 39, #42), so direction only.
- Within-class ρ on gold: CNN ~ `v11a` 0.916–0.921, CNN ~ CNN 0.936–0.948 (CoAtNet variant ~ `v11a` ≈ 0.965). The three CNNs
  are closer to each other than to the CoAtNet, but less alike than two CoAtNet variants.
- Gold rank-means: `v11a` + `v13r` + `v13e` 0.9233, `v11a` + `v13e` 0.9224, `v11a` + `v13r` 0.9210, `v11a` + `v13h` 0.9170, all
  four 0.9186 (direction only).

**Verdict: ✅ runs green; 🔁 gold +0.011 / +0.0125 over `v13h`, under the 0.05 floor.** The solos `v13r` / `v13e` and the
`v11a` + `v13h` blend are read 2026-10-04 (pre-registered: each CNN vs `v13h` 0.931 — ✅ ≥ 0.935 / 🔁 0.928–0.934 / ❌ ≤ 0.927;
a blend is kept only if ≥ its best member's solo + 0.004).

### 2026-10-04 — Submissions #44–#47: EfficientNet-B0 `v13e` **0.935 = our best solo** (✅ +0.004 vs `v13h`) · ResNet-50 `v13r` **0.934** (🔁 +0.003) · `v11a` + `v13h` **0.934** (🔁 +0.002 over its best member — the first blend above both members) · `v11dl` **0.927** (🔁 −0.003, P-61 closed)

Placeholders `rsna-knee-infer` v40 / v41 / v42 / v43 (current `src` + 3 seds; each green: `smoke False`, the right `infer members`,
c03 decode-once verified, `constant labels 0`; the blend decodes once for both members). Sent 07:17–07:27 UTC on Tian's go
("Continue, submit the work"), timed by `src/watch_submission.py`.

| # | what | gold-58 | LB | scored within | pre-registered read | verdict |
|---|---|---|---|---|---|---|
| 44 | `v11dl` = `v11d` + lr 2e-4 / LLRD 0.85 (P-61) | 0.9132 | **0.927** | [28.9, 30.4] min | vs `v11d` 0.930: ✅ ≥ 0.934 / ❌ ≤ 0.926 | 🔁 (−0.003; gold −0.005 the same way) → P-61 closed, not adopted |
| 45 | `v13r` = ResNet-50 a1, `v13h` recipe | 0.9111 | **0.934** | [18.4, 19.9] min | vs `v13h` 0.931: ✅ ≥ 0.935 / 🔁 0.928–0.934 / ❌ ≤ 0.927 | 🔁 (+0.003, top edge) |
| 46 | `v13e` = EfficientNet-B0 ra, `v13h` recipe | 0.9126 | **0.935** | [13.8, 15.3] min | same bands | **✅ KEEP (+0.004) — best solo of ours** |
| 47 | `v11a` + `v13h`, flat rank-mean | 0.9170 | **0.934** | [27.3, 28.8] min | keep if ≥ 0.936 (best member 0.932 + 0.004) | 🔁 (+0.002 over the best member) |

**What this says.**

1. **The CNN line is now our strongest.** On the `v13h` recipe the three CNNs read EfficientNet-B0 0.935 > ResNet-50 0.934 >
   ResNet-34 0.931, which is the forum's ranking. All three sit at or above the six-read CoAtNet band (0.929–0.932).
   - `v13e` is our best solo (+0.003 over `v11a`), from a 17 MB checkpoint, and scores in ≈ 15 min, the fastest of ours.
   - Gold ordered the CNNs the same way (+0.011 / +0.0125 over `v13h`) but put all three under `v11a` (0.9204). That is the
     cross-family gold bias seen in #42.
2. **Different families add a little; same-family blends never did.** `v11a` + `v13h` reads 0.934, above both members (0.932 /
   0.931): +0.002 over the best, +0.0025 over their mean. Every earlier blend of same-teacher, same-family members read flat
   (P-42, P-52, #32, #38). The size of this gain is under the 0.004 bar, so it is direction only. The forum's +0.005–0.007 came
   from blends of 0.94 members.
3. **P-61 closes.** A higher, flatter CoAtNet LR reads −0.003 on both LB and gold. The CoAtNet line ends at the `v11a` recipe.

**Verdict:** ✅ KEEP `v13e` (production solo, best Efficiency candidate); 🔁 `v13r`; 🔁 the two-family blend (keep the direction,
not the blend); P-61 closed. The trio `v11a` + `v13r` + `v13e` is #48 (sent 07:56 UTC; keep only if ≥ 0.939).

### 2026-10-04 — Submission #48: three-family blend `v11a` + `v13r` + `v13e` **0.938 = our best own-model score** (+0.003 over its best member `v13e` 0.935, +0.004 over the members' mean) · 🔁 by the pre-registered bar (≥ 0.939), but the second cross-family blend in a row above every member

`rsna-knee-infer` v44: `INFER_MEMBERS = ["v11a", "v13r", "v13e"]`, flat rank-mean, one c03 decode pass (placeholder green: `infer members
(3)`, `3 members in 1 geometry group(s)`, decode-once verified, `constant labels 0`). Sent 07:56:41 UTC, ref 56818172, read 08:40:31
UTC, scored within [42.3, 43.8] min. Gold-58 0.9233 (`v11a` alone 0.9204).

| blend | members (solo LB) | best member | mean | LB | over best | over mean |
|---|---|---|---|---|---|---|
| #47 `v11a` + `v13h` | 0.932 / 0.931 | 0.932 | 0.9315 | 0.934 | +0.002 | +0.0025 |
| #48 `v11a` + `v13r` + `v13e` | 0.932 / 0.934 / 0.935 | 0.935 | 0.9337 | **0.938** | +0.003 | +0.0043 |
| (earlier, same family) #32 `v11a` + `v11b`, #38 + `v13c`, P-42, P-52 | — | — | — | flat | ≈ 0 | — |

**Reading.** Each gain is under the 0.004 bar on its own. But two cross-family blends in a row read above every member, and four
same-family blends read flat, so the sign is consistent. The gain also grows with the number of families (2 → 3). This is the
forum's pattern (different families pay +0.005–0.007, research.md 2.7.4) at a smaller size, because our members are weaker
(0.93 vs 0.94). 0.938 is 0.004 under the public-stack fork (0.942), from three models we trained, in ≈ 44 min.

**Verdict: 🔁 by rule (0.001 under the bar); adopted as our best own ensemble, the candidate own leg for P-50.** Next levers, both
members not blends: a stronger CNN (a second `v13e` seed, a bigger EfficientNet) or more CNN families on the `v13h` recipe.

### 2026-10-04 — P-66 on RunPod: EfficientNet-B3 @ 288 `v13b3` trained in 2.2 h on one RTX 4090 · gold-58 SWA **0.9222** vs `v13e` 0.9126 (+0.0096, 6 up / 5 down / 1 tie) · ✅ run green, 🔁 direction only · `v13e2` (seed twin) training · solos 10-06

**Setup (Tian's go, critic-reviewed: proposals.md P-66).**
- **Pod:** `kqgjkh0329ieqq`, secure RTX 4090, EUR-IS-1, $0.74/h, created 12:17 UTC.
  - 16 vCPU and 93 GB allotted (`nproc` reports 64; the host has 377 GB).
  - `/workspace` is MooseFS (network) and `/dev/shm` is 43 GB, so the 51 GB c03 cache went on the 80 GB container disk.
  - `/kaggle` → `/workspace/kaggle`, so `_last.pt` survives a stop (traps 46).
- **Speeds:** bandwidth from a public GCS object 29 MB/s; the four parallel c03 pulls ran at ≈ 70–93 MB/s in total, 51 GB in ≈ 11 min.
- **The job:** `scripts/runpod_chain.sh v13b3 v13e2` with:
  - `CACHE_PREFIX=rsna-knee-cache3` and `RSNA_TEACHER_TABLES=("raptor_teacher",)` (mix 0.5, the `v13e` target);
  - `SEQ_ARMS=1` (the arms train one after the other) and `AUTO_STOP=1`.
- **Relaunch:** the first launch died in the blob check, and the relaunch started 12:33 (traps 52; ≈ 12 min lost).

**`v13b3`** = the `v13h` recipe on EfficientNet-B3 (`timm efficientnet_b3.ra2`) at 288 px:
- **Recipe:** c03, CNN LR 3e-4 uniform, frozen BN (78 modules), heavy aug, drop-path 0.1, 30 epochs, SWA of epochs 27–29, all 4,349
  report-only studies.
- **Timing:** trained 12:34 → 14:46 UTC (2.2 h) at 0.06 s/study, ≈ 4.4 min/epoch.
- **Hardware:** 10.7 GB VRAM, GPU 83–92 % busy.
- **Run checks:** the log has `SWA of last 3`, `-> v13b3_fold0_best.pt = SWA` and no runtime guard. The checkpoint is 45 MB.
- **Kaggle comparison:** on a T4 the P-66 card estimated 12–15 h, two sessions with a resume, with the memory at risk.

Gold-58 EMA by epoch (all 58, reported only):

| epoch | 0 | 4 | 9 | 14 | 19 | 24 | 29 | SWA |
|---|---|---|---|---|---|---|---|---|
| `v13b3` (B3 @ 288, RunPod) | 0.7789 | 0.9122 | 0.9218 | 0.9224 | 0.9256 | 0.9221 | 0.9221 | **0.9222** (CI95 0.895–0.946) |
| `v13e` (B0 @ 224, Kaggle) | 0.7757 | 0.9032 | 0.9079 | 0.9104 | 0.9115 | 0.9136 | 0.9126 | 0.9126 |

Per label, gold-58 SWA (`v13b3` from the log's SWA table; the others from the "Session C" entry above):

| label | `v11a` | `v13r` | `v13e` | `v13b3` | `v13b3` − `v13e` |
|---|---|---|---|---|---|
| ACL | 0.962 | 0.980 | 0.957 | 0.977 | +0.020 |
| MCL | 0.955 | 0.880 | 0.952 | 0.948 | −0.004 |
| Medial Meniscus | 0.958 | 0.966 | 0.946 | 0.982 | +0.036 |
| Lateral Meniscus | 0.896 | 0.886 | 0.853 | 0.906 | +0.053 |
| Medial OA | 0.986 | 0.986 | 0.984 | 0.984 | 0.000 |
| Lateral OA | 0.822 | 0.801 | 0.832 | 0.841 | +0.009 |
| PF OA | 0.837 | 0.867 | 0.875 | 0.867 | −0.008 |
| Effusion | 0.969 | 0.963 | 0.916 | 0.911 | −0.005 |
| Synovitis | 0.806 | 0.781 | 0.826 | 0.805 | −0.021 |
| Baker's | 0.966 | 0.987 | 0.978 | 0.984 | +0.006 |
| Contusion | 0.947 | 0.942 | 0.970 | 0.969 | −0.001 |
| Fracture | 0.942 | 0.893 | 0.861 | 0.893 | +0.032 |
| **macro** | **0.9204** | **0.9111** | **0.9126** | **0.9222** | **+0.0096** |

**How B3 relates to the members (OOF csvs, gold rows; direction only).**
- **Within-class ρ** (Spearman inside each label's 0 and 1 classes, averaged):
  - `v13b3` ~ `v13e` 0.888, `v13b3` ~ `v13r` 0.879, `v13b3` ~ `v11a` 0.853;
  - `v13e` ~ `v13r` 0.886, measured the same way.
  - So B3 is as far from B0 as ResNet-50 is, and is the least like the CoAtNet. The session C entry's ρ figures use another method;
    compare within this list only.
- **Gold rank-means:**
  - `v11a` + `v13b3` 0.9267;
  - `v11a` + `v13e` + `v13b3` 0.9255 and `v11a` + `v13r` + `v13b3` 0.9254, vs the #48 trio's 0.9233;
  - all four 0.9250; `v13e` + `v13b3` 0.9211.

**Reading.**
- Most of the gain sits in the menisci, ACL and Fracture, where B0 was our weakest CNN on gold. Capacity and resolution look like
  they help structure-level findings.
- Gold has inverted CNN recipe reads before: `v13h` −0.001 on gold vs +0.010 on the LB (traps 39). +0.0096 is 0.2× the 0.05 floor.
- The LB solo decides, by P-66's pre-registered rule against `v13e` 0.935: ✅ ≥ 0.939 / 🔁 0.931–0.938 / ❌ ≤ 0.930. It is a member
  candidate if ≥ 0.933.

**`v13e2`** (`v13e` at seed 43; `reseeded 43 for arm v13e2` in the log) started on the same pod at 14:46. Both ship after it as
`rsna-knee-ckpt-v13b3` / `-v13e2`, each with its training log in the Dataset. Then the pod stops itself.

**Verdict: ✅ the run (green, ≈ 2.4 pod-h ≈ $1.8 so far); 🔁 gold +0.0096 vs `v13e`, under the 0.05 floor.** The solo is read
2026-10-06 (10-05's five slots are taken).

**CORRECTED 2026-10-04 (16:00):** the seed twin `v13e2` reads within-class ρ 0.888 against `v13e` (entry below). That is the same
as `v13b3` ~ `v13e`, so "B3 is as far from B0 as ResNet-50 is" shows no family diversity: on 58 studies this ρ cannot separate a
backbone change from a seed change.

### 2026-10-04 — P-66 complete: the seed twin `v13e2` (`v13e` at seed 43) trained in 65 min · gold-58 SWA **0.9151** vs `v13e` 0.9126 · both arms shipped, pod stopped itself and was deleted · ≈ $2.6 in total · solos 10-06

**`v13e2`** trained on the same pod right after `v13b3`, 14:46 → ≈ 15:49 UTC:
- **Setup:** `v13e` exactly at seed 43 (`reseeded 43 for arm v13e2 (base seed 42)`), Raptor 0.5, all 4,349 studies, 30 epochs.
- **Speed:** 0.03 s/study, ≈ 2.1 min train + 0.1 min val per epoch.
- **Run checks:** `SWA of last 3`, `-> v13e2_fold0_best.pt = SWA`, no runtime guard. The checkpoint is 17.7 MB.

**What happened after training:**
- **Ship:** the chain shipped it as `rsna-knee-ckpt-v13e2` (15:50:08 UTC: best.pt, OOF, log; confirmed with `kaggle datasets files`).
- **Stop:** the chain stopped the pod through the GraphQL `podStop` mutation; `get-pod` showed `EXITED`, which proves the traps 51 fix.
- **Local backup:** my watcher missed the window between SWA and the stop (under a minute), so the backup is the Dataset download.
  Both arms are in `artifacts/kaggle_out/pod_<arm>/`. The pod was deleted at 15:53 (`list-pods` empty).
- **Cost:** pod 12:17 → 15:50 ≈ 3.55 h × $0.74 ≈ $2.6, plus volume storage. RunPod billing read $1.99 at 15:53 and lags. The plan was
  ≈ $4.5 with a $7 cap.

Gold-58 EMA by epoch (all 58, reported only):

| epoch | 0 | 4 | 9 | 14 | 19 | 24 | 29 | SWA |
|---|---|---|---|---|---|---|---|---|
| `v13e2` (seed 43, RunPod) | 0.7796 | 0.9102 | 0.9188 | 0.9201 | 0.9178 | 0.9150 | 0.9158 | **0.9151** (CI95 0.886–0.940) |
| `v13e` (seed 42, Kaggle) | 0.7757 | 0.9032 | 0.9079 | 0.9104 | 0.9115 | 0.9136 | 0.9126 | 0.9126 |

**Per label, `v13e2` vs `v13e`:**
- ACL 0.969 / 0.957, MCL 0.952 / 0.952, Medial Meniscus 0.946 / 0.946, Lateral Meniscus 0.846 / 0.853;
- Medial OA 0.984 / 0.984, Lateral OA 0.841 / 0.832, PF OA 0.873 / 0.875, Effusion 0.932 / 0.916;
- Synovitis 0.806 / 0.826, Baker's 0.987 / 0.978, Contusion 0.972 / 0.970, Fracture 0.872 / 0.861.
- The largest per-label seed moves are 0.012–0.020: ACL, Effusion, Synovitis.

**Within-class ρ on gold** (Spearman inside each label's 0 and 1 classes, averaged; the method of the `v13b3` entry):
- `v13e2` ~ `v13e` **0.888** (same recipe, seed only);
- `v13b3` ~ `v13e` 0.888, `v13e` ~ `v13r` 0.886;
- `v13e2` ~ `v13b3` 0.900, ~ `v13r` 0.864, ~ `v11a` 0.859.

**So on 58 studies a seed change scatters the within-class ranking as much as a backbone change.** This measure cannot separate
family diversity from seed noise. The `v13b3` entry's "as far from B0 as ResNet-50 is" is corrected below it.

**Gold rank-means (direction only):**
- `v13e` + `v13e2` 0.9160;
- the #48 trio 0.9233 → + `v13e2` 0.9238 → + `v13b3` 0.9250 → all five 0.9256;
- `v11a` + `v13b3` + `v13e` + `v13e2` 0.9256.

**Reading.** The seed moves gold-58 by +0.0025 macro. That is 0.05× the gold floor, as expected. The LB solo is the real measurement
(P-66's rule):
- s = |`v13e2` − 0.935|;
- s ≤ 0.003: the CNN bands stand;
- s ≥ 0.005: one-seed CNN deltas need ≥ s, and the D / E reads are re-read with ±s.

**Verdict: ✅ the run (green, shipped, pod stopped and deleted; P-66 cost ≈ $2.6 of the $7 cap); 🔁 gold +0.0025, a seed-level
difference. The solo is read 2026-10-06.**

### 2026-10-04 — Session D (`rsna-knee-train-b` v6, 5.94 h): P-62 on the CNNs, `v13es` / `v13rs` = `v13e` / `v13r` + Raptor 0.75 on report-silent cells · gold-58 SWA **0.9107 / 0.9160** vs 0.9126 / 0.9111 (pair mean +0.0015) · ✅ runs green, 🔁 direction only · solos 10-05

Pushed 10:15 UTC on Tian's go (Kaggle smoke `rsna-knee-train-b` v5 green), COMPLETE 16:12 UTC.

**Run checks:**
- The parent logged `ok  arm` ×2.
- Each child logged:
  - `teacher table raptor_teacher: 4349`;
  - the `P-62: report-silent cells (pilkwang UNK) mix at 0.75` line with the silent shares (ACL 8 %, MCL 10 %, Medial Meniscus 6 %,
    Lateral Meniscus 10 %, Medial OA 26 %, Lateral OA 33 %, PF OA 18 %, Effusion 10 %, Synovitis 84 %);
  - `freeze_bn` 49 / 53 modules;
  - epochs 0–29, `SWA of last 3`, `= SWA`;
  - no runtime guard (traps 47).
- Shipped as `rsna-knee-ckpt-v13es` / `-v13rs` with their logs (both `ready`; files confirmed).
- Kaggle GPU after it: 26.67 / 30 h (3.33 h left until the 2026-10-10 reset).

| epoch | 0 | 4 | 9 | 14 | 19 | 24 | 29 | SWA |
|---|---|---|---|---|---|---|---|---|
| `v13es` (B0 + silent mix) | 0.7744 | 0.8970 | 0.9129 | 0.9141 | 0.9108 | 0.9113 | 0.9108 | **0.9107** |
| `v13e` (B0, flat 0.5) | 0.7757 | 0.9032 | 0.9079 | 0.9104 | 0.9115 | 0.9136 | 0.9126 | 0.9126 |
| `v13rs` (R50 + silent mix) | 0.7362 | 0.8657 | 0.9071 | 0.9119 | 0.9148 | 0.9167 | 0.9160 | **0.9160** |
| `v13r` (R50, flat 0.5) | — | — | — | — | — | — | — | 0.9111 |

**Per label, silent mix vs flat** (gold-58 SWA; `v13es` / `v13e`, then `v13rs` / `v13r`):

| label | `v13es` / `v13e` | `v13rs` / `v13r` |
|---|---|---|
| ACL | 0.966 / 0.957 | 0.985 / 0.980 |
| MCL | 0.925 / 0.952 | 0.909 / 0.880 |
| Medial Meniscus | 0.960 / 0.946 | 0.968 / 0.966 |
| Lateral Meniscus | 0.870 / 0.853 | 0.880 / 0.886 |
| Medial OA | 0.984 / 0.984 | 0.988 / 0.986 |
| Lateral OA | 0.822 / 0.832 | 0.818 / 0.801 |
| PF OA | 0.862 / 0.875 | 0.875 / 0.867 |
| Effusion | 0.937 / 0.916 | 0.935 / 0.963 |
| Synovitis | 0.799 / 0.826 | 0.817 / 0.781 |
| Baker's | 0.978 / 0.978 | 0.986 / 0.987 |
| Contusion | 0.960 / 0.970 | 0.945 / 0.942 |
| Fracture | 0.865 / 0.861 | 0.886 / 0.893 |

**How much the silent mix changes the model** (gold, the within-class ρ method of the P-66 entries):
- `v13es` ~ `v13e` 0.932 and `v13rs` ~ `v13r` 0.948, at the same seed.
- The seed twin `v13e2` ~ `v13e` is 0.888.
- **So the silent-mix target moves the predictions less than a seed change does.** That points to an LB delta inside seed noise.
- Synovitis, the label with 84 % silent cells, moves in opposite directions in the two families (−0.027 / +0.036).

**Gold rank-means:**
- `v13es` + `v13rs` 0.9170 vs `v13e` + `v13r` 0.9161;
- `v11a` + `v13rs` + `v13es` 0.9224 vs the #48 trio 0.9233;
- all five 0.9208.

**The LB read decides**, per P-62's pre-registered rule: m(`v13es`, `v13rs`) vs 0.9345 → ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299.
Placeholders (all green: `smoke False`, 1 member, `decode-once verified`, `constant labels 0`):
- `rsna-knee-infer` **v49** = `v13es` solo (the checkpoint scores 0.9107);
- `rsna-knee-infer` **v50** = `v13rs` solo (0.916).

The 10-06 solos of P-66 are ready as well: **v47** = `v13b3`, **v48** = `v13e2`.

**Verdict: ✅ the runs; 🔁 gold pair mean +0.0015 (0.03× the floor), labels split.** Solos sent after 00:00 UTC 2026-10-05.

### 2026-10-05 — Submissions #49–#53 (the 10-05 five) · #50 EfficientNet-B3 `v13b3` **0.940 = our best solo** (✅ +0.005 vs `v13e`) · #51 seed twin `v13e2` **0.938**: s = 0.003, the CNN one-seed bands stand · #53 B3 swap **0.940** (🔁, = B3 alone) · **#52 all five (B6) 0.942 = the public-stack fork, from our own models (✅ +0.004 vs #48)**; a flat blend ≈ the members' mean + a gain that grows with member count · **#49 fork + #48 trio at β 0.45 = 0.943 (🔁 +0.001; rank 373, the 0.942 plateau is ≈ 1,000 teams wide)**

Sent 00:41–00:45 UTC by the `auto_submit.py` fallback run. The scheduled 00:00:30 run sent nothing (traps 20 addendum). Placeholders
were green on 10-04 (candidates.md); rows #49–#53 are in the Submissions table. Each message carries its pre-registered read.

| # | what | gold-58 | LB | scored within | pre-registered read | verdict |
|---|---|---|---|---|---|---|
| 50 | `v13b3` = EfficientNet-B3 ra2 @ 288, `v13h` recipe (RunPod, P-66) | 0.9222 | **0.940** | [18.5, 20.0] min | vs `v13e` 0.935: ✅ ≥ 0.939 / 🔁 0.931–0.938 / ❌ ≤ 0.930 | **✅ KEEP (+0.005) — our best solo** |
| 49 | `rsna-knee-fork` v11: the public 0.942 stack + the #48 trio as our leg at β 0.45 | — | **0.943** | [318.1, 319.6] min | vs #13 0.942: ✅ ≥ 0.945 / 🔁 0.941–0.944 / ❌ ≤ 0.940 | 🔁 (+0.001) — our best public number, rank 373 |
| 52 | `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`, flat rank-mean (B6) | 0.9256 | **0.942** | [48.6, 50.1] min | vs #48 0.938: ✅ ≥ 0.941 / 🔁 0.936–0.940 / ❌ ≤ 0.935 | **✅ KEEP (+0.004) — our best own, = the fork** |
| 53 | `v11a` + `v13r` + `v13b3`, flat rank-mean (B3 swap) | 0.9254 | **0.940** | [45.6, 47.1] min | vs #48 0.938: ✅ ≥ 0.941 / 🔁 0.936–0.940 / ❌ ≤ 0.935 | 🔁 (+0.002; = `v13b3` alone) |
| 51 | `v13e2` = `v13e` exactly at seed 43 (RunPod, P-66) | 0.9151 | **0.938** | [12.5, 14.0] min | s = \|LB − 0.935\|: ≤ 0.003 the bands stand / ≥ 0.005 widen them | **✅ measured: s = 0.003 → the bands stand** |

**#51, what it says.**
1. **The CNN seed spread on the LB is 0.003.** That is one pair, and the twin also changed platform (RunPod, 8 loader workers vs
   Kaggle's 2). It is under the 0.004 one-seed bar (P-44), so the bands stand. P-44's bar now rests on a CNN pair as well as one
   CoAtNet draw.
2. **0.003 is also the size of every gain we read on 10-04:** `v13r` +0.003 over `v13h` (#45), and #48 +0.003 over its best member.
   One reseed moved a solo as far as adding two families did.
3. **The B0 recipe is worth ≈ 0.9365 (the mean of two seeds), not 0.935.** Single-seed solos are draws.
4. **One 17 MB model ties our best ensemble (#48 0.938).** So #48's gain over its best member cannot be told apart from seed luck.
   - B6 (#52) holds both B0 seeds.
   - B5 (#48 + `v13e2`) is the clean test of a second seed inside the ensemble.
5. **Gold-58 agreed this time:** `v13e2` +0.0025 over `v13e` there. That is the same direction, but gold is not a judge of member
   order (traps 39).
6. **For #50 (`v13b3`) the pre-registered bar stays at 0.935.** A B3 read under 0.938 is no better than a B0 reseed, so it is also
   read against the B0 seed mean, 0.9365.

**Verdict #51: ✅ measurement.** s = 0.003, so the CNN one-seed bands stand (≥ 0.004, P-44). `v13e2` = our best solo (0.938, tied
with #48) and a final-ensemble member.

**#50, what it says.**
1. **EfficientNet-B3 @ 288 reads 0.940, ✅ by the pre-registered rule:** +0.005 vs `v13e` 0.935, bar ≥ 0.939.
   - Against the B0 seed mean (0.9365, #51) the margin is +0.0035, and against `v13e2` it is +0.002. Both are under the 0.004
     one-seed bar.
   - So "B3 beats B0" is ✅ by the rule but only just. The real claim is that B3 is at least as good as B0, plausibly better by a few
     thousandths.
2. **Capacity has paid at every step on the `v13h` recipe:**
   - ResNet-34 0.931 → ResNet-50 0.934;
   - EfficientNet-B0 0.935 / 0.938 → B3 0.940.
   - Gold-58 had B3 +0.0096 over B0, the same direction.
3. **For the first time one of our own models beats our best ensemble.** 0.940 > #48 0.938, and it is 0.002 under the public-stack
   fork (0.942).
4. **Cost:** 20 min to score (B0: 14), a 2.2 h RunPod train.
5. **What it unlocks (candidates.md):**
   - T2, a B3 seed twin (≈ $1.8 RunPod), meets its gate; it needs a critic-checked case and Tian's go;
   - B3 joins the final-member recipes (P-50).
   - T5 (a bigger CNN) was gated on "A1 ✅ by a clear margin". +0.0035 over the B0 seed mean is not clear, so T5 stays gated.

**Verdict #50: ✅ KEEP** (+0.005 vs 0.935, bar 0.004; +0.0035 vs the B0 seed mean, below the bar). `v13b3` = our best solo, 0.940.

**#53, what it says.**
1. **The B3 swap reads 0.940: 🔁 by its rule.** It is +0.002 vs #48 0.938, inside the 0.936–0.940 band. It also equals `v13b3` alone
   (#50, 0.940): `v11a` (0.932) and `v13r` (0.934) add nothing on top of B3.
2. **Every flat rank-mean we have sent lands at about the members' mean LB + 0.002–0.005.** This is the rule that explains all our
   blends. Gain over the members' mean, then over the best member:

   | blend | members (solo LB) | mean | LB | over mean | over best |
   |---|---|---|---|---|---|
   | #23 `v09r` + `v08r` (CoAtNet + DINOv2) | 0.927 / 0.918 | 0.9225 | 0.927 | +0.0045 | 0 |
   | #26 `v09r` + `v09u` (seed pair) | 0.927 / 0.927 | 0.927 | 0.930 | +0.003 | +0.003 |
   | #32 `v11a` + `v11b` (seed pair) | 0.932 / 0.929 | 0.9305 | 0.932 | +0.0015 | 0 |
   | #38 `v11a` + `v11b` + `v13c` | 0.932 / 0.929 / 0.921 | 0.9273 | 0.932 | +0.0047 | 0 |
   | #47 `v11a` + `v13h` | 0.932 / 0.931 | 0.9315 | 0.934 | +0.0025 | +0.002 |
   | #48 `v11a` + `v13r` + `v13e` | 0.932 / 0.934 / 0.935 | 0.9337 | 0.938 | +0.0043 | +0.003 |
   | #53 `v11a` + `v13r` + `v13b3` | 0.932 / 0.934 / 0.940 | 0.9353 | 0.940 | +0.0047 | 0 |

   - Three-member blends gain +0.0043–0.0047 over the mean; pairs gain +0.0015–0.0045.
   - **A blend beats its best member only when the members are within ≈ 0.004 of each other** (#26, #47, #48). With one member
     clearly ahead (#23, #38, #53) the blend only matches it.
   - n = 7 and LB-rounded to 0.001, so this is a working rule, not a law. It fits every blend we have.
3. **What it changes:**
   - **Weak members are now a cost.** `v11a` (0.932), `v13h` (0.931) and `v13r` (0.934) pull the mean down by more than diversity
     gives back once B3 (0.940) is in. B1 (`v11a` + `v13h` + `v13r` + `v13e`, mean 0.933) and B2 (the CNN trio, mean 0.9333) both
     predict ≈ 0.937. That is under B3 alone, so neither is worth a slot.
   - **The strong-only blends are the ones to build next:**
     - `v13b3` + `v13e2` (mean 0.939) predicts ≈ 0.941–0.944;
     - `v13b3` + `v13e2` + `v13e` (mean 0.9377) predicts ≈ 0.940–0.942.
     - B0 and B3 are within-class ρ 0.888 apart, the same as a seed pair, and #26's seed pair still gained +0.003.
   - **The lever is member strength, not member count.** That favours T2 (a B3 seed twin: two ≈ 0.940 members).
4. **Prediction for #52 (B6, all five), written before it scored:** members 0.932 / 0.934 / 0.935 / 0.940 / 0.938, mean 0.9358, so
   the rule predicts ≈ 0.938–0.941, centre 0.940.

**Verdict #53: 🔁** (+0.002 vs #48; = `v13b3` alone). It is not our best own pick over B3 solo: equal score, three times the members.

**CORRECTED 2026-10-05 (01:35), after #52:** the rule in #53's point 2 holds for pairs and triples only. #52's five members gained
+0.0062 over their mean and read 0.942, above the rule's 0.938–0.941. So the gain over the mean grows with member count, and point 3's
"weak members are now a cost" is wrong as a general claim. At five members, `v11a` / `v13r` / `v13e` beside B3 + the second B0 seed
lifted the blend 0.002 *above* B3 alone. The corrected rule and the B rows' predictions are in #52's block below and in candidates.md.

**#52, what it says.**
1. **B6 (all five) reads 0.942: ✅ KEEP by its rule** (+0.004 vs #48, bar ≥ 0.941). It is +0.002 over its best member (`v13b3` 0.940)
   and +0.0062 over the members' mean (0.9358).
2. **Our own models now equal the public-stack fork (#13 / #15, 0.942).** That is five models we trained, flat weights, nothing tuned
   on the public LB, scored in 50 min (the fork takes hours).
3. **The blend rule, corrected:** a flat rank-mean ≈ the members' mean solo LB + g(n), with n the member count.
   - g(2) ≈ +0.0015–0.0045 (#23, #26, #32, #47);
   - g(3) ≈ +0.0043–0.0047 (#38, #48, #53);
   - g(5) ≈ +0.006 (#52, n = 1).
   - Both member strength and member count pay. More decorrelated members means more variance cancels, the textbook ensemble
     result. At three members the count gain cannot outweigh one member that is 0.005 ahead (#53); at five it did.
   - My prediction for #52, written before it scored, was 0.938–0.941. It read 0.942, 0.001 above the top of that range.
4. **What it changes:**
   - **B6 is our own final-pick candidate (P-50), replacing #48.**
   - It is also the leg for C2 (the fork + B6 at β 0.45), if C1 (#49, the fork with the #48 trio) reads ✅.
   - **Strong members added to B6 should pay:** T2 (a B3 seed twin) would make a six-member B6 with a higher mean. That raises
     T2's value.
   - Pruning ablations (B4 = B6 − `v13e2`, B5 = B6 − `v13b3`) explain B6 but cannot beat it under this rule.
   - B10 (`v13b3` + `v13e2`, pred. ≈ 0.941–0.943) and B11 (+ `v13e`, ≈ 0.942) test strength against count. B1 / B2 (≈ 0.937–0.939)
     have lost their purpose.
5. **Gold-58 agreed on the order:** B6 0.9256 > B3 swap 0.9254 > #48 0.9233. The same direction as the LB, by margins gold cannot
   resolve.

**Verdict #52: ✅ KEEP** (+0.004 vs #48, bar ≥ 0.941). Our best own score, **0.942 = the public-stack fork**, and the own final-pick
candidate.

**CORRECTED 2026-10-05 (02:10), from the RunPod critic's review:** the blend table above missed two flat blends whose members all
have solo reads:
- #29 `v09r` + `v09u` + `v09x`: 0.927 / 0.927 / 0.929, mean 0.9277 → 0.931 (+0.0033);
- #36 `v09o` + `v09o2`: 0.927 / 0.927 → 0.928 (+0.001).

With all ten, **the gain over the members' mean splits by family, not only by count:**
- **same recipe** (seed or small variants): #26 +0.003, #32 +0.0015, #36 +0.001, #29 +0.0033, so ≈ +0.001–0.0033;
- **cross-family:** #47 +0.0025, #23 +0.0045, #38 +0.0047, #48 +0.0043, #53 +0.0047, and #52 +0.0062 at five members.

So g(2) spans 0.001–0.0045 and g(3) 0.0033–0.0047. The low ends are the same-recipe blends. LB rounding (0.001) makes the
third-decimal spans false precision; read the rule as "cross-family ≈ +0.004–0.006, same-recipe ≈ +0.001–0.003". This changes the
predictions for the EfficientNet-only blends:
- B10 (`v13b3` + `v13e2`) ≈ 0.940–0.942;
- B11 (+ `v13e`) ≈ 0.941–0.942.

B0 and B3 sit at within-class ρ 0.888, the same as a seed pair, so these blends are priced as same-recipe. Neither is predicted to
beat B6. **What beats B6 is most likely a new family at ≥ 0.935, not more EfficientNets.**

**#49, what it says (read 06:01 UTC, scored within [318.1, 319.6] min ≈ 5.3 h).**
1. **The fork with our #48 trio at β 0.45 reads 0.943: 🔁 by its rule** (+0.001 vs #13 / #15 0.942; ✅ needed ≥ 0.945). It is our
   highest number on the board.
2. **+0.001 is worth ≈ 1,000 ranks here.** The full leaderboard (06:02 UTC, 5,187 teams): 0.942 = rank 1,381, 0.943 = rank 373,
   0.945 = rank 239, 0.950 = rank 102. About 1,000 teams sit on the forked public stack at exactly 0.942, so any own leg that adds
   anything at all clears the plateau. The 0.946–0.947 teams (research.md 2.7.4) have own legs of ≈ 0.94–0.95 members.
3. **Our leg at β 0.45 added +0.001 with a 0.938 leg.** The rule from the own blends (a flat blend ≈ the members' mean + a gain) says
   the leg's level is what matters: B6 (0.942) as the leg should add more than the trio (0.938) did. That is C2.
4. **Earlier legs at β 0.10–0.20 added 0.000 / −0.001** (#13, #14, #22, #27: Raptor-distilled CoAtNets that correlate with the
   stack). The cross-family trio at β 0.45 is the first own leg to move the fork up. Both the weight and the leg changed, so this
   read cannot separate them; it does not need to.
5. **Scoring time: 5.3 h for a three-member leg** (the fork alone took up to 8 h at #17). A C2 leg has five members; the two extra
   are B3 (20 min solo) and a B0 (14 min). Expect ≈ 6 h, inside the ≤ 9 h limit, but send C2 first in its day.

**Verdict #49: 🔁 by the pre-registered rule (+0.001, band 0.941–0.944); our best public number (0.943, rank 373).** The fork +
own leg stays the second final-pick candidate, and C2 (B6 as the leg) is the next fork build.

### 2026-10-05 — Session E on RunPod, chain 1: `v13ecp` (B0 on 0.5 Raptor + 0.5 Claude, no LLM-blend share) trained in 71 min · gold-58 SWA **0.9063** vs the B0 seed pair 0.9126 / 0.9151 · ✅ run green, shipped; 🔁 direction only · solo 10-06

**Setup** (P-65 step 2, Tian's go "Go E"; candidates.md T1). Pod `j4obvfotdbdudo`, one RTX 4090, US-NC-1, $0.74/h, created 09:09 UTC.
- **Arm:** `v13e` exactly (EfficientNet-B0 @ 224 on c03, CNN LR 3e-4 uniform, frozen BN, heavy aug, drop-path 0.1, 30 epochs, SWA
  27–29, seed 42, all 4,349 report-only studies).
- **Only the target changes:** `TEACHER_TABLES = ("raptor_teacher", "claude_v1")`, `TEACHER_MIX = 1.0`. The log confirms `training
  targets = (1 - 1.0) * LLM + 1.0 * quantile-matched ['raptor_teacher', 'claude_v1']`, both tables 4,349 studies. So the LLM blend's
  half of `v13e`'s target is replaced by the Claude (Opus 5.5) relabel. Evaluation targets are unchanged.
- **Speed:** 0.03 s/study, 2.2 min train + 0.1 min val per epoch, the same as `v13e2` on the EUR-IS-1 pod. The network volume did not
  slow it. The 48 GB c03 cache pulled and verified in 3 min (71 blobs, 0 bad).
- **Run checks:** `SWA of last 3`, `-> v13ecp_fold0_best.pt = SWA`, no runtime guard, `completed: true`.
- **Ship:** `rsna-knee-ckpt-v13ecp` at 10:24 UTC (best.pt 17.7 MB, OOF, log; `kaggle datasets files` confirmed). Local backup
  `artifacts/kaggle_out/pod_v13ecp/`, md5 identical to the pod's file.

Gold-58 EMA by epoch (all 58, reported only):

| epoch | 0 | 4 | 9 | 14 | 19 | 24 | 29 | SWA |
|---|---|---|---|---|---|---|---|---|
| `v13ecp` (Raptor + Claude) | 0.7619 | 0.9038 | 0.9113 | **0.9185** | 0.9148 | 0.9063 | 0.9063 | **0.9063** (CI95 0.876–0.932) |
| `v13e2` (seed 43) | 0.7796 | 0.9102 | 0.9188 | 0.9201 | 0.9178 | 0.9150 | 0.9158 | 0.9151 |
| `v13e` (seed 42) | 0.7757 | 0.9032 | 0.9079 | 0.9104 | 0.9115 | 0.9136 | 0.9126 | 0.9126 |

The late drift from the epoch-14 peak is −0.012, vs −0.005 for `v13e2`. That is direction only, like everything on gold here.

**Per label vs the B0 seed mean** (`v13ecp` / mean of `v13e`, `v13e2` / Δ), next to what the target itself did on gold in the pilot
(0.5 Opus + 0.5 Raptor minus 0.5 LLM + 0.5 Raptor, entry "P-65 gold-58 BLIND pilot"):

| label | `v13ecp` | B0 mean | Δ student | Δ target (pilot) |
|---|---|---|---|---|
| ACL | 0.956 | 0.963 | −0.007 | +0.003 |
| MCL | 0.912 | 0.952 | **−0.041** | 0.000 |
| Medial Meniscus | 0.957 | 0.946 | +0.011 | +0.008 |
| Lateral Meniscus | 0.878 | 0.850 | **+0.029** | +0.010 |
| Medial OA | 0.986 | 0.984 | +0.002 | −0.014 |
| Lateral OA | 0.857 | 0.837 | **+0.020** | +0.042 |
| PF OA | 0.879 | 0.874 | +0.005 | +0.024 |
| Effusion | 0.892 | 0.924 | **−0.032** | −0.009 |
| Synovitis | 0.809 | 0.816 | −0.007 | +0.033 |
| Baker's | 0.937 | 0.983 | **−0.046** | −0.043 |
| Contusion | 0.941 | 0.971 | **−0.030** | +0.006 |
| Fracture | 0.874 | 0.867 | +0.007 | +0.029 |
| **macro** | **0.9063** | **0.9139** | **−0.0076** | **+0.0073** |

- **The student follows the Claude labels where they moved most.** Baker's falls as the target did, the size thresholds the pilot
  flagged. Lateral OA and Lateral Meniscus rise with it. Signs agree on 7 of the 11 labels the target moved.
- **MCL and Contusion fall where the target was flat.** That is the seed-sized scatter of 58 studies (per-label seed moves reach 0.02)
  or a real cost; gold cannot tell.
- **On gold the target rose +0.007 and the student fell −0.008.** Gold has inverted the LB direction of target changes before (traps
  39), so this predicts nothing about the solo.

**Diversity on gold** (within-class Spearman, the method of the P-66 entries): `v13ecp` ~ `v13e` 0.883, ~ `v13e2` 0.873, ~ `v13es`
0.888, ~ `v13b3` 0.860, ~ `v13r` 0.861, ~ `v11a` 0.834. The seed pair `v13e` ~ `v13e2` is 0.888. So the target swap moves the ranking
about as much as a seed does, and gold cannot separate the two. Gold rank-means: `v13ecp` + `v13e` 0.9131, + `v13e2` 0.9157; B6 + `v13ecp`
0.9244 vs B6 0.9256.

**The read is the solo LB** (candidates.md T1, pre-registered): vs the B0 seed mean 0.9365, ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932.
Placeholder `rsna-knee-infer` **v53** pushed 10:28 UTC.

**Verdict: ✅ the run (green, shipped, backed up); 🔁 gold −0.008, a seed-level difference against a 0.05 floor.** The solo is read
2026-10-06. Chain 2 (`v13ec`, 0.25 LLM + 0.5 Raptor + 0.25 Claude) started on the same pod at 10:24.

**Read 2026-10-06: #57 = 0.935 → 🔁** (−0.0015 vs the B0 seed mean; = `v13e` at the same seed). Entry "Submissions #54–#58".

### 2026-10-05 — Session E chain 2: `v13ec` (B0 on 0.25 LLM + 0.5 Raptor + 0.25 Claude) gold-58 SWA **0.9116** vs the B0 seed pair 0.9126 / 0.9151 · pod stopped itself and was deleted, E ≈ $1.9 · epoch selection on gold, tested fairly on both E arms: **+0.007 / −0.001** · ✅ run green, shipped; 🔁 direction only · solo 10-06

**Setup:** `v13e` exactly (B0, seed 42), `TEACHER_TABLES = ("claude_rap_v1",)` at `TEACHER_MIX = 0.75`. The log reads `training targets
= (1 - 0.75) * LLM + 0.75 * quantile-matched ['claude_rap_v1']`, which is 0.25 LLM + 0.5 Raptor + 0.25 Claude, the design P-65
registered. Trained 10:25 → 11:36 UTC at the same 2.3 min/epoch; `SWA of last 3`, `-> v13ec_fold0_best.pt = SWA`, no guard.

**What happened after training:**
- **Ship:** `rsna-knee-ckpt-v13ec` at 11:36 UTC (best.pt 17.7 MB, OOF, log; `kaggle datasets files` confirmed). The local backup is the
  Dataset download, `artifacts/kaggle_out/pod_v13ec/`; the pod was already stopped, so there is no md5 against it.
- **Stop:** the job's final line fired `/workspace/stopper.sh` (traps 51 addendum). `get-pod` read `EXITED` at 11:47; the pod was
  deleted at 11:48 and `list-pods` is empty.
- **Cost:** pod 09:09 → ≈ 11:37 ≈ 2.5 h × $0.74 ≈ $1.9 (billing read $1.26 for the day at 11:48 and lags). ≈ $5 of RunPod credit is left.

Gold-58 EMA by epoch (all 58, reported only):

| epoch | 0 | 4 | 9 | 14 | 19 | 24 | 29 | SWA |
|---|---|---|---|---|---|---|---|---|
| `v13ec` (0.25 Claude) | 0.7613 | 0.8973 | 0.9080 | 0.9108 | **0.9169** | 0.9137 | 0.9117 | **0.9116** (CI95 0.880–0.939) |
| `v13ecp` (0.5 Claude) | 0.7619 | 0.9038 | 0.9113 | **0.9185** | 0.9148 | 0.9063 | 0.9063 | 0.9063 |
| `v13e` (no Claude, same seed) | 0.7757 | 0.9032 | 0.9079 | 0.9104 | 0.9115 | 0.9136 | 0.9126 | 0.9126 |

**Per label, the two doses vs the B0 seed mean** (`v13e`, `v13e2`):

| label | `v13ec` | `v13ecp` | B0 mean | Δ 0.25 | Δ 0.5 |
|---|---|---|---|---|---|
| ACL | 0.952 | 0.956 | 0.963 | −0.011 | −0.007 |
| MCL | 0.921 | 0.912 | 0.952 | **−0.032** | **−0.041** |
| Medial Meniscus | 0.966 | 0.957 | 0.946 | +0.020 | +0.011 |
| Lateral Meniscus | 0.871 | 0.878 | 0.850 | +0.021 | +0.029 |
| Medial OA | 0.989 | 0.986 | 0.984 | +0.005 | +0.002 |
| Lateral OA | 0.847 | 0.857 | 0.837 | +0.011 | +0.020 |
| PF OA | 0.874 | 0.879 | 0.874 | 0.000 | +0.005 |
| Effusion | 0.919 | 0.892 | 0.924 | −0.004 | **−0.032** |
| Synovitis | 0.822 | 0.809 | 0.816 | +0.006 | −0.007 |
| Baker's | 0.946 | 0.937 | 0.983 | **−0.037** | **−0.046** |
| Contusion | 0.938 | 0.941 | 0.971 | **−0.033** | **−0.030** |
| Fracture | 0.894 | 0.874 | 0.867 | +0.028 | +0.007 |
| **macro** | **0.9116** | **0.9063** | **0.9139** | **−0.0023** | **−0.0076** |

- **Baker's, MCL and Contusion fall at both doses** (−0.03 to −0.05). Both E arms and `v13e` share seed 42, so `v13e` alone is the
  same-seed control, and it has MCL 0.952 and Contusion 0.970. Baker's is the pilot's size-threshold loss. MCL and Contusion were
  flat in the pilot's target, so their fall has no explanation on the label side.
- **Effusion recovers at the half dose** (−0.004 vs −0.032). The menisci and the lateral compartment rise at both doses.
- **None of this clears the 0.05 gold floor.** The solos decide (traps 39).

**Diversity on gold** (within-class ρ): `v13ec` ~ `v13e` 0.886, ~ `v13ecp` 0.881, ~ `v13es` 0.907, ~ `v13b3` 0.885, ~ `v13r` 0.870,
~ `v11a` 0.837. As with `v13ecp`, the ranking moves about as much as a seed does (seed pair 0.888). Gold rank-means: B6 0.9256, + `v13ec`
0.9240, + both E arms 0.9234.

**Epoch selection on gold, tested fairly (Tian asked whether shipping the last epochs loses performance).** Production arms ship the
SWA of the last three EMA snapshots (`ckpt_policy="last"`); per-epoch weights are not saved, only per-epoch gold predictions. The test
picks the best epoch on a random half of the 58 gold studies and scores it on the other half against the shipped SWA. 500 splits, both
directions, every label with ≥ 2 positives and ≥ 2 negatives in each half.

| arm | in-sample peak − SWA | fair gain of the picked epoch vs SWA | splits > 0 | picked epoch, median (IQR) |
|---|---|---|---|---|
| `v13ecp` | +0.0121 (epoch 14) | **+0.0066** (sd 0.0084) | 82 % | 14 (13–14) |
| `v13ec` | +0.0053 (epoch 19) | **−0.0006** (sd 0.0081) | 58 % | 18 (15–19) |

For `v13ec` the candidates are epochs 0–24 only: epochs 25–29's per-epoch files were on the deleted pod, and those epochs sit next to
the SWA. About half of `v13ecp`'s in-sample peak survives the fair test, and none of `v13ec`'s does. Both are far below the gold floor.
The question belongs to the P-67 five-fold ruler, where the epoch budget is already a planned variable (the 50-epoch arm); saving an EMA
snapshot every few epochs (≈ 18 MB each for B0) would make an earlier epoch submittable. Earlier reads of the same question: P-22 (the
concat head decayed, +0.013 split-half on 882 OOF studies), the 8-epoch CoAtNets (−0.0001 / −0.0002), P-29 (16 epochs over-trained by
0.012 on OOF; fixed by the epoch budget, not by selection).

**The read is the solo LB** (candidates.md A5): vs the B0 seed mean 0.9365, ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932. With A4 it is a
three-dose read on one backbone and seed (0 / 0.25 / 0.5 Claude). Placeholder `rsna-knee-infer` **v54**.

**Verdict: ✅ the run (green, shipped, backed up; E ≈ $1.9, pod deleted); 🔁 gold −0.002, a seed-level difference.** The solo is read
2026-10-06. Epoch selection on gold: 🔁, mixed sign on the two arms, under the floor.

**Read 2026-10-06: #58 = 0.932 → ❌** by the pre-registered rule (−0.0045 vs the B0 seed mean). With #57, P-65 closes ❌. Entry
"Submissions #54–#58".

### 2026-10-05 — Per-label model routing on gold-58, tested fairly: **−0.006 vs B6's flat rank-mean** (in-sample it looks like +0.012) · ❌ DEAD END on gold

Tian asked whether an ensemble that routes each finding to the member best at it would beat the flat blend. No GPU: the gold-58
predictions of 11 members (`v11a`, `v13r`, `v13e`, `v13b3`, `v13e2`, `v11n`, `v11n2`, `v13es`, `v13rs`, `v13ecp`, `v13ec`) plus B6 itself
as a twelfth candidate.

| label | B6 | best candidate (in-sample) |
|---|---|---|
| ACL | 0.974 | 0.985 `v13rs` |
| MCL | 0.959 | 0.959 B6 |
| Medial Meniscus | 0.968 | 0.982 `v13b3` |
| Lateral Meniscus | 0.885 | 0.906 `v13b3` |
| Medial OA | 0.988 | 0.989 `v11n2` |
| Lateral OA | 0.837 | 0.857 `v13ecp` |
| PF OA | 0.869 | 0.879 `v13ecp` |
| Effusion | 0.961 | 0.969 `v11a` |
| Synovitis | 0.817 | 0.826 `v13e` |
| Baker's | 0.986 | 0.987 `v13r` |
| Contusion | 0.966 | 0.972 `v13e2` |
| Fracture | 0.899 | 0.942 `v11a` |
| **macro** | **0.9256** | **0.9377** (+0.0121) |

- **The fair test:** per label, pick the best candidate on a random half of the 58 studies, score the routing on the other half,
  and compare with B6 on that half. Over 300 splits × both directions (every label with ≥ 2 positives and ≥ 2 negatives per half),
  the routing reads **−0.0062** (sd 0.0066) and is worse than B6 in **83 %** of the 600 evaluations.
- **Why:** the per-label gaps between members are inside the per-label SE (≈ 0.09 on 9–35 positives). The in-sample "best" member is
  mostly the luckiest one, and the flat blend's averaging is worth more than the routing's selection.
- **On the LB** the same idea is per-label weight tuning, which the public stack's author expects to give back (P-50, #16: its
  LB-tuned per-label map is worth ≈ +0.002 publicly).

**Verdict: ❌ DEAD END for routing chosen on gold-58** (proposals.md Dropped directions). A per-label weighting could only be chosen
on a ruler of thousands of studies: the P-67 five-fold OOF (4,349 studies, report-label targets, valid for image-side members), with
shrinkage toward equal weights. Even there, members of similar quality leave little for routing to gain over a flat rank-mean.

### 2026-10-06 — Submissions #54–#58 (the 10-06 five) · **#54 the fork + B6 at β 0.45 = 0.944 = our best public number** (🔁 +0.001 vs #49; rank 337 of 5,293) · #55 B13 (B6 + two CoAtNets) **0.940** and #56 B11 (the EfficientNet triple) **0.941**: both 🔁 under B6 0.942, B6 stays the own pick · **P-65 closed ❌**: the Claude relabel reads #57 `v13ecp` **0.935** (🔁) and #58 `v13ec` **0.932** (❌), no lift at either dose

Sent 00:03–00:06 UTC by `auto_submit.py` (pid 23960, `artifacts/submit_plan_1006.json`), in Tian's 10-05 order (the best-expected
performers first). The placeholders were green on 10-05 (candidates.md). The laptop had been asleep since 18:50 UTC and was woken by
hand at 00:02:32, so the sends left 2.5 min late (traps 53). The watchers of #54 and #55 ended after their first poll, so those two
scoring times are not recorded; both took > 40 min (the submitter's summary at 00:44 still read them PENDING).

| # | what | gold-58 | LB | scored within | pre-registered read | verdict |
|---|---|---|---|---|---|---|
| 54 | `rsna-knee-fork` v12: the public 0.942 stack + B6 as our leg at β 0.45 (C2) | — | **0.944** | > 41 min, not recorded | vs #49 0.943: ✅ ≥ 0.946 / 🔁 0.942–0.945 / ❌ ≤ 0.941 | 🔁 (+0.001) — our best public number, rank 337 |
| 55 | B6 + `v11n` + `v11n2`, flat rank-mean (B13) | 0.9256 | **0.940** | > 40 min, not recorded | vs B6 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938; ≥ 0.943 = more CoAtNet weight helps | 🔁 (−0.002); the CoAtNet branch does not fire |
| 56 | `v13b3` + `v13e2` + `v13e`, flat rank-mean (B11) | 0.9196 | **0.941** | [23.0, 24.5] min | as #55; ≥ 0.943 = strength over diversity | 🔁 (−0.001) |
| 57 | `v13ecp` = `v13e` (B0, seed 42) on 0.5 Raptor + 0.5 Claude (P-65) | 0.9063 | **0.935** | [14.0, 15.5] min | vs the B0 seed mean 0.9365: ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932 | 🔁 (−0.0015; = `v13e` at the same seed) |
| 58 | `v13ec` = `v13e` on 0.25 LLM + 0.5 Raptor + 0.25 Claude (P-65) | 0.9116 | **0.932** | [35.0, 36.5] min | as #57 | ❌ (−0.0045, at the harmful line) |

**#57 / #58, what they say (P-65).**
1. **The Claude relabel does not lift the CNN at either dose.** Same backbone, same seed (42); only the target changes:

   | Claude share of the training target | arm | LB |
   |---|---|---|
   | 0 (0.5 LLM + 0.5 Raptor) | `v13e` / `v13e2` (seeds 42 / 43) | 0.935 / 0.938 (mean 0.9365) |
   | 0.25 (0.25 LLM + 0.5 Raptor + 0.25 Claude) | `v13ec` | 0.932 |
   | 0.5 (0.5 Raptor + 0.5 Claude, no LLM blend) | `v13ecp` | 0.935 |

   - The pair mean is 0.9335, −0.003 under the seed mean: inside a two-arm band of ± 0.0045, on the wrong side. Neither arm came near
     ✅ (≥ 0.9405).
   - The dose order is not monotone (the half dose is the worst). That is no effect plus seed-sized scatter (s = 0.003, #51), not a
     dose-response.
   - Against `v13e` at the same seed: 0.000 at the full dose, −0.003 at the half dose.
2. **Gold-58 had both arms under the B0 seed pair** (0.9063 / 0.9116 vs 0.9126 / 0.9151), the same direction as the LB. Within E it had
   the order reversed (`v13ec` above `v13ecp` on gold, below it on the LB). Direction only (traps 39).
3. **What closes:**
   - P-65's own rule: "If it fails: the label line closes; image-side (recipe, families, P-62 silent-cell teacher) only."
   - P-46 step 1 (dread as a fourth LLM vote) waited on this read under "if the Claude vote does not move the LB, the LLM half is not
     binding", so it closes too.
   - Every training-target change except the Raptor table has now read flat or worse on the LB: P-38 self-distillation (#19 −0.001),
     P-55 the OOF student (#34–#36), P-45 D4 (#43 −0.002), and P-65 the Claude relabel (pair −0.003). Only the image-grounded Raptor
     table transferred (P-39, +0.009). The forum said the same of report extraction (Tucker / Yann / tennogh, research.md 2.7.3).
4. **What it changes:**
   - The week-2 retrains train on 0.5 LLM + 0.5 Raptor (± the P-62 silent weight, read 10-07). `claude_v1` / `claude_rap_v1` stay in
     the private Dataset, unused.
   - `v13ecp` (0.935) is member-grade, but at seed distance from `v13e` on gold (ρ 0.883): a same-recipe vote at most.
   - B12 (B6 + every new solo ≥ 0.936) gets no member from E; it waits for the A3 solos.

**Verdicts #57 / #58: 🔁 (`v13ecp`, −0.0015) and ❌ (`v13ec`, −0.0045, by the pre-registered rule). P-65 ❌ DEAD END as a
training-target lever (no lift at either dose); P-46 closes with it.**

**#55 / #56, what they say (B13, B11).**
1. **Neither beats B6 (0.942). B6 stays the own final pick (P-50 pick 1).** Both are 🔁 by the band.
2. **The blend rule held, with one refinement:**

   | blend | members (solo LB) | mean | LB | over mean | rule's prediction |
   |---|---|---|---|---|---|
   | #52 B6 (3 families) | 0.932 / 0.934 / 0.935 / 0.940 / 0.938 | 0.9358 | 0.942 | +0.0062 | 0.938–0.941 (made before it scored) |
   | #56 B11 (1 family) | 0.940 / 0.938 / 0.935 | 0.9377 | 0.941 | +0.0033 | 0.941–0.942 |
   | #55 B13 (3 families, 7 members) | B6's five + 0.932 / 0.932 | 0.9347 | 0.940 | +0.0053 | 0.941–0.942 |

   - **B11** gained exactly the same-recipe gain (+0.001–0.0033). Three EfficientNets at a higher mean lost to B6, whose weaker CoAtNet
     and ResNet-50 bring two more families. At this spread of member quality, diversity beats strength.
   - **B13:** member count alone does not pay. `v11n` / `v11n2` are `v11a`-recipe variants (0.932 each). They lowered the mean, added
     no family, and the gain over the mean fell from +0.0062 to +0.0053; the blend lost 0.002.
   - **The refinement:** the gain grows with the number of *families*, not of members. A member that is under the mean and of a family
     already present costs. (12 flat blends with solo-read members now; LB rounded to 0.001.)
3. **Gold-58 saw neither:** it had B13 = B6 (0.9256 both) and B11 0.006 under B6. The LB says −0.002 and −0.001.
4. **What it changes:**
   - The week-2 retrains add no CoAtNet seed (B13's branch). Whether the one CoAtNet stays at all is B14 (B6 − `v11a`, 10-07).
   - A sixth *family* stays the predicted lever: P-69 (our ConvNeXt-T) and the public ConvNeXt-T reader (candidates.md B16).
   - C3 (the fork with a better own leg) stays closed: its gate needed an own blend ≥ 0.943.

**Verdicts: #55 🔁 (−0.002 vs B6), #56 🔁 (−0.001 vs B6).** B6 stays pick 1.

**#54, what it says (C2).**
1. **The public 0.942 stack + B6 at β 0.45 reads 0.944: 🔁 by its rule** (+0.001 vs #49 0.943; ✅ needed ≥ 0.946). It is +0.002 over
   the anchor alone (#13 / #15, 0.942) and our best public number: **rank 337 of 5,293** (leaderboard, 09:25 UTC).
2. **A stronger leg moved the fork one more tick:** a 0.938 leg (#48) gave +0.001, a 0.942 leg (B6) +0.002. That is the direction the
   blend rule predicts, but it is the size of the fork's own run-to-run spread (one tick, entry below), so the two reads cannot be told
   apart.
3. **The plateau moved under us.** By 10-06 the public community stack reads 0.943 (984 teams at exactly 0.943), and a public fork of
   it with a ConvNeXt-T leg reads 0.944 (194 teams). #54 is level with that public notebook, not above it, and #49 is now on the
   plateau (entry below).
4. **Scoring time not recorded** (traps 53); > 41 min. #49 took 5.3 h with a three-member leg.
5. **What it changes:** the fork pick (P-50 pick 2) is `rsna-knee-fork` v12 (C2), replacing v11 (#49): equal or better on every read,
   built the same way, β chosen before sending.

**Verdict #54: 🔁 by the pre-registered rule (+0.001, band 0.942–0.945); our best public number (0.944, rank 337) and the new fork
pick.**

### 2026-10-06 — The public frontier moved: the community stack now reads 0.943 (984 teams at exactly 0.943), and a public fork with its own ConvNeXt-T leg reads 0.944 (194 teams) · its author measures the stack's run-to-run spread at one tick · ✅ a read (no GPU), it changes how we read fork deltas

Read at 09:25 UTC from the full leaderboard CSV (`kaggle competitions leaderboard -d`) and from the notebook
`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` (112 votes, last run 10-06 01:29 UTC; pulled read-only, never run).

- **The leaderboard (5,293 teams):** top 0.964; 10th 0.960; 60 ≥ 0.955; 123 ≥ 0.950; 319 ≥ 0.945. **984 teams sit at exactly
  0.943** and 194 at 0.944. On 10-05 at 06:02 the plateau was 0.942 (≈ 1,000 teams; 0.942 = rank 1,381); now 0.942 has 190. We are
  rank 337 at 0.944 (#54).
- **What the 0.943 is:** "the community's 0.943 inference stack", reproduced cell for cell by skarin ("Reproducing the 0.943 Public
  Stack"). Its stage list (DINO ensemble, A5 attention pooling, RadImageNet heads, Raptor + four CoAt readers including Global96 and
  Repair-v1) is the "Speedy Raptors" build we decomposed on 09-27: our anchor's cells plus two CoAt readers (Infrastructure
  2026-09-27). It was ≤ 0.001 better then and is 0.001 better now.
- **What the 0.944 adds:** a 2.5D ConvNeXt-Tiny reader its author trained.
  - Input: canonical volumes at 0.4 mm/px and a 154 mm field of view; 12 windows × 3 slices per series, up to 6 series; a 2-layer
    study transformer and one attention pooling per finding.
  - Training: 5 folds grouped by report text, 14 epochs at 256 px, plain BCE on the mean of four public label tables, no image teacher.
  - **Solo 0.929 (3 folds). 70 % stack + 30 % reader = 0.944**, the weight chosen before sending. Their sweep: 15 % and 30 % → 0.944,
    **45 % → 0.942**, under the stack alone.
- **Run-to-run spread:** "The stack is not fully deterministic. The same pipeline has scored 0.944, 0.944 and 0.943 for us." So a
  fork read carries about one tick of its own noise. Our two anchor-only reads agreed (#13 / #15, 0.942), which is one pair.
- **Their other notes match ours:**
  - OOF self-distillation lifted their single reader (0.922 → 0.927) but lowered the five-fold ensemble (0.930 → 0.929) while OOF
    rose 0.888 → 0.903: "use teachers of a different architecture, and judge it on the leaderboard". That is our P-38 / P-55, traps 39,
    and P-68's cross-family design.
  - The stack's blend constants were tuned on the public LB and gold-58, and they expect it to behave differently on private (P-50's
    case for an own pick).
  - The stack + reader takes 6.5–8 h to score.

**What it changes for us:**
1. **Fork deltas of +0.001 are inside the fork's run-to-run noise.** #49 (+0.001) and #54 (+0.002 over our anchor) rank the legs only
   weakly. The fork pick is chosen on construction (a flat β, the best own blend as the leg), not on these ticks.
2. **Re-anchoring on the 0.943 stack stays dropped** (proposals.md Dropped directions): the 0.943 stack + a 0.929 leg = 0.944 = our
   0.942 anchor + B6. Re-anchoring could gain at most the 0.001 between the anchors, needs a `build_fork.py` rewrite for a different
   cell layout, and scores in 6.5–8 h.
3. **Their reader is a sixth family we could mount without training** (public Dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader`,
   inference code in the notebook). It shares neither our input (c03) nor our targets (Raptor). As a sixth B6 member the blend rule
   predicts ≈ 0.941–0.943 (a cross-family member under the mean). That is candidates.md B16; it needs a leg runner in the infer kernel
   and a licence read. P-69 (our own ConvNeXt-T on the `v13h` recipe) is unchanged; theirs shows a ConvNeXt-T family reaches 0.929
   without an image teacher.
4. **Where we stand:** our own models (0.942) are one tick under the public stack. The fork with our leg is level with the best public
   notebook. The top ten are 0.016 above that.

**Verdict: ✅ a read (no GPU, no submission).** Fork reads carry about one tick of noise; the fork pick stays C2 (#54).

### 2026-10-06 — Gold-58: what a blend gains by pair type — cross-family pairs **+0.0053** over their members' mean, same-recipe pairs **+0.0025**, r(ρ, gain) = −0.89; a fitted model puts a fourth family on B6 at **≈ +0.0008 LB** · 🔁 direction only (gold), it corroborates the LB blend rule

Tian asked whether ConvNeXt helped anyone and whether to spend the remaining time on new families or on the ones we have
(research.md 2.7.8 holds the forum and literature halves). This is our own half. No GPU: `src/blend_diversity_gold.py` reads the 58
gold predictions (SWA) of 15 members (CoAtNet `v11a` / `v11b` / `v11n` / `v11n2` / `v11d` / `v11p`; ResNet `v13h` / `v13r` /
`v13rs`; EfficientNet `v13e` / `v13e2` / `v13b3` / `v13es` / `v13ecp` / `v13ec`). For every pair it scores the flat rank-mean
against the pair's mean macro-AUC.

| pair type | pairs | mean Spearman ρ | gain over the pair's mean | SD |
|---|---|---|---|---|
| cross-family | 72 | 0.922 | **+0.0053** | 0.0012 |
| same family, other architecture (B0 / B3, R34 / R50) | 7 | 0.944 | +0.0035 | 0.0007 |
| same recipe (seed twins, target or regularisation variants) | 26 | 0.955 | **+0.0025** | 0.0009 |

- **Correlation drives the gain:** r(ρ, gain) = −0.89 over all 105 pairs. The same ordering as the LB (same recipe +0.001–0.003,
  cross-family +0.004–0.006, entries "Submissions #49–#53" and "#54–#58"). Gold's gains are ≈ 1.5× the LB's (B6: +0.0093 on gold
  vs +0.0062 on the LB).
- **A model, fitted on 1,500 random subsets of 2–7 members:** gain over the members' mean ≈ −0.0057 + 0.118·(1 − ρ̄) +
  0.0037·ln(n), R² 0.80. It reads B6's own gain at +0.0079 (observed +0.0093).
- **The marginal value of a sixth member X added to B6**, on gold, by X's solo quality relative to B6's members' mean (0.9163) and
  X's ρ to the five:

  | X's solo vs B6's mean | ρ 0.955 (a seed) | ρ 0.944 (a sibling) | ρ 0.922 (a new family) | ρ 0.90 |
  |---|---|---|---|---|
  | −0.010 | −0.0018 | −0.0014 | −0.0005 | +0.0004 |
  | −0.005 | −0.0010 | −0.0005 | +0.0003 | +0.0012 |
  | ± 0 | −0.0001 | +0.0003 | **+0.0012** | +0.0020 |
  | +0.005 | +0.0007 | +0.0011 | +0.0020 | +0.0029 |

  A new family of average quality adds +0.0012 on gold, ≈ **+0.0008 on the LB** (× 0.67): one tick. Raising every member by +0.003
  raises the blend by ≈ +0.003.
- **The LB says the same by family count** (12 flat blends with solo-read members): gain over the mean +0.0024 with one family,
  +0.0039 with two, +0.0051 with three. Each extra family added ≈ +0.0012–0.0015, and the slope is falling.

**Verdict: 🔁 direction only (gold-58, traps 39), but it agrees with the LB in every ordering.** At our level a fourth family is worth
≈ 0–2 LB ticks; per-member quality moves the blend one for one. The recommendation built on it is research.md 2.7.8 (improve the
families we have; at most one hedged ConvNeXt-T arm; drop NFNet; demote B16), pending Tian.

## Infrastructure

### 2026-09-27 — The "0.943 Speedy Raptors CoAtNet D4" notebook is our anchor **plus two CoAt readers**, not a faster graph; its "< 30 min" is a 3-study commit run · P-41 (threaded scan + 8 decode workers) smoke-green and byte-identical

**Verdict: ✅ FINDING (read-only) for the notebook; 🔧 P-41 shipped, output identity ✅ (binary), effect on scoring time ⏳ PENDING
(the next solo submission, timed by `src/watch_submission.py`).**

**The notebook** (`haideptry/rsna-knee-speedy-raptors-coatnet-d4-0943`, last run 2026-09-24; local copy in the repo root, untracked),
diffed cell by cell against `notebook_score_0.942.ipynb`: its cells 2–29 are our anchor's cells 12–49 **byte-identical** (to the last
newline) except (a) cell 27 ↔ our 45: **two more CoAt readers** — Global96 top-3 (`mattiaangeli/rsna-knee-coatnet-global96-top3`, epochs
16/23/18) and Repair-v1 top-3 (epochs 12/7/11) — beside resgated top-3 and D4, the family reduced as the rank of the four readers'
*probability mean* (ours: rank mix of two), `private_alpha 0.4` and the per-label outer map hard-coded instead of read from `RUN`;
(b) cell 9 ↔ 23: the DINO stage's ordering pass also stores full-header `(path, spacing)` lists (`RAPTOR_HEADER_CACHE`) that the Raptor
stage reuses instead of re-reading every header; (c) A5 0.52 / Rad 0.55 · 0.20 / calibrator 0.40 hard-coded (= our `speedy` preset).
Every speed device it advertises is already in our anchor: the 6 frozen DINOv2 blocks computed once for 20 members
(`SHARED_DINO_PREFIX_LAYERS`, hash-checked), CPU model templates deep-copied instead of 20 `from_pretrained`, A5 folds replicated on
both T4s with alternating 8-study micro-batches, one 48-study CPU decode block kept ahead of the GPU, the MaxSpan forward / reverse views
from one decode, the Raptor order + 192 MB pixel LRU caches, the memoised asset catalogue.

**The "sub-30 minute" claim is its commit run.** Its own `diagnostics/phase_events.jsonl` (pulled with `kernels output`): **3 studies,
238 s** end to end (DINO 57 s, A5 16 s, Rad 8 s, Raptor 29 s, D4 ‖ residual 57 s, Repair-v1 ‖ Global96 67 s; our anchor's placeholder:
184–204 s). The CoAt readers run in parallel pairs **only when the cohort is ≤ 48 studies**
(`RSNA_PARALLEL_COAT_READERS` default `'1' if len(ids) <= 48`); on the hidden test all four run one after another, so it does
*more* serial work than our anchor. The "1322 test studies" in its log is its cache-sizing assumption (0.3 × train), not a count. Our
own measurement of the anchor's Raptor branch alone — 5.1–7.5 s/study on T4 ×2 over 4,349 studies (P-39 pass) — rules out any
graph of this shape scoring a hidden test of hundreds of studies in 30 min. Kaggle's score-sorted listing (09:55) places it **below**
`romantamrazov/rsna-knee-dinosaur-v5` (our anchor's source), above which sit `jiweiliu/rsna-knee-fast-2xt4-inference` (the origin of
the 2×T4 harness), `evgendvorkin/rsna-versia-5` and `pjmathematician/rsna-knee-d4-blend` / `-d4-lite`. Re-anchoring on it would buy
≤ +0.001 (its own claim, 0.2× the floor) for two more serial CoAtNet-2 @384 readers (3 checkpoints × ≤ 94 windows each).

**What our scoring actually costs (bounds from the submissions API `date` vs the time each score was read; no exact scoring time has
ever been measured):** solo #19 (`rsna-knee-infer` v16, one c02 CoAtNet-1 member) sent 21:25 UTC, read 22:07 UTC → **≤ 42 min**;
fork #17 sent 09:54 UTC, read 18:00 UTC → **≤ 8 h 06 min**; #12 / #16 "≈ 10 / 12 h" are morning reads. The fork's hours are the anchor
graph (untouchable without changing its score); the solo's minutes are mostly our own DICOM I/O, which ran on **one** header thread
(`scan_series`) and **two** decode workers.

**P-41, shipped the same morning (one change, output-neutral by construction):** `scan_series` reads its per-series header on 16
threads (`RSNA_SCAN_THREADS`; `pool.map` keeps order); the decode-once pass uses `Config.infer_workers` = `min(8, 2 × usable CPUs)`
(`RSNA_INFER_WORKERS`; 0 on Windows, where spawn cannot pickle the loader's local Dataset); a Kaggle smoke now uses that real worker
count (traps 12d — before, smoke decoded in-process and could not exercise the pool). Checks: local `v09r` solo render on the 3 sample
studies before / after (`RSNA_WORKERS=0`) → `submission.csv`, `manifest_test.csv`, `series_scan_test.csv` **byte-identical**; the scan
frame identical at 1 / 16 / 64 threads; `src/cache_selftest.py` PASSED; local train smoke green. **Kaggle `rsna-knee-infer` v18**
(`FORCE_SMOKE=True`, `MODE="infer"`, `INFER_MEMBERS=["v09r"]`, 10:08 → 10:10): `cpus 4, usable 4` → **8 decode workers**, `scanned 15
series … (16 threads)`, `decode-once verified … 3 studies rebuilt, identical` (main-process rebuild vs worker output), **`submission.csv`
byte-identical to v17's (#20 placeholder)**. Outputs `artifacts/kaggle_out/infer_v18_p41_smoke/`. The speed read (P-41 card: ≤ 30 min
✅ / 30–42 🔁 / > 42 ❌) rides the next solo submission; v18 itself is a smoke render (0.4 h guard) and is **not** submittable.

**Also shipped:** `src/watch_submission.py` — polls one submission, prints status changes, appends sent → first-seen-scored
(± the poll interval) to `artifacts/submission_timing.csv`; exits 0 / 2 / 3 / 4 (COMPLETE / ERROR / timeout / unknown ref); API errors
(traps 20 token window, 429s) are retried. Tested on #20 (already scored → bound only) and on a simulated PENDING → COMPLETE sequence.

**First use, #21 (read 2026-09-27 16:24 UTC): no timing.** The watcher died with the laptop 8 s after its first poll, so the sent →
scored time is only the manual bound **≤ 2 h 43 min** (PENDING 13:43, COMPLETE first seen 16:24:19) — no information vs #19's ≤ 42 min.
The watcher needs the machine awake for the whole ≈ 40 min; P-41's speed stays ⏳ (entry "Submission #21").

**Read, #23 (2026-09-27): scored within [28.3, 29.3] min of sending → P-41 ✅** by the pre-registered ≤ 30 min line (two members,
8 decode workers; `artifacts/submission_timing.csv`). The #19 baseline is only a ≤ 42 min bound (entry "Submission #23").

### 2026-09-21 — P-27 fork builder + P-28 production regime shipped; local checks ✅ KEEP the code · Kaggle ⏳

- `src/build_fork.py` → `kaggle/rsna-knee-fork/` (54 cells: 50 anchor + 3 fork + credits; 20 sources; 66 KB
  payload; `--check` byte-identical rebuild; `selftest_blend` proves β=0 / β=1 rank identities, monotonicity
  in β, and that the validator rejects a bad schema / UID set / NaN frame). The embedded payload decodes to
  the sed'd pipeline (`FORCE_SMOKE = False`, `MODE = "infer"`, `INFER_MEMBERS = ["v08w", "v09h"]`, 2,953 lines,
  sha-checked at run time).
- `src/kaggle_pipeline.py`: `train_all`, `swa_last`, `split_studies`, `average_state_dicts`, SWA ring in
  `_last.pt`, `_lastema.pt`, `ARM_ONLY`, per-arm resume copy (traps 31), `oof_eval` gold-only guard (traps 32);
  `PROD` arms `v09a` / `v08a`; `SHIPPED_ARMS` hold `v08w` / `v09h`. `window_head_test.py` 37/37 green; local
  `MODE="train"` smoke green for both arms (`SWA of last 1 EMA snapshot(s)`, `_best.pt` = SWA,
  `_lastema.pt`, `_last.pt` with the ring, decode-once verified, inference on `v09a`).
- `kaggle/rsna-knee-folds/kernel-metadata.json` now mounts the four c02 shards + the CoAtNet-1 weights, so
  either training kernel can host either arm (and the other's resume).
- **Kaggle smokes green** (`rsna-knee-train` v18 = `v09a`, `rsna-knee-folds` v5 = `v08a`, ~1.5 min each after
  boot): `ARM_ONLY: running only …`, both caches indexed (4,407 studies), `fold 0: train 4 / val 8 studies
  [train_all: val = gold rows]`, `auc_gold` on 8 rows with its CI, `SWA of last 1 EMA snapshot(s)`,
  decode-once verified, inference on the arm, `constant labels 0`, no `!!`. Real runs: `rsna-knee-train`
  v19 (`v09a`, pushed 2026-09-21 evening), `rsna-knee-folds` v6 (`v08a`, pushed after the fork's placeholder
  run freed the second GPU slot). Fork `rsna-knee-fork` v1 = placeholder run (20 sources accepted by the push).
- **Fork v1 placeholder run (3 studies): the whole graph ran, then the arm cell's `finally` crashed.** Kaggle
  status `ERROR` with an **empty kernel log** (0 bytes — the outputs folder is the evidence, not the log:
  `diagnostics/current_phase.json` holds the error). `btkd_v559_complete.json`: status COMPLETE, 20 DINO
  members, 5 A5 folds, 4 Raptor views, both CoAt children `rc=0`, **224 s** for the anchor graph; our
  subprocess `rc=0` in **115 s** (6 members, one geometry group, decode-once verified, `blend: by_version ->
  v08w (1 fold), v09h (5 folds)`), `fork_diagnostics.json` status `beta0.20`, Spearman(ours, anchor) written
  per finding (meaningless at n=3). The crash: `rsna_phase('fork_ours', 'COMPLETE', status=…)` — the anchor's
  logger takes `status` positionally → `TypeError: multiple values for argument 'status'`, raised *after*
  `submission.csv` was written. Fixed in `build_fork.py` (`outcome=`) with a builder guard; v2 was pushed by
  mistake before the rebuild (the guard fired on its own comment), v3 carries the fix. With 3 studies the
  β = 0.2 blend is identical to the anchor (a 1/3 rank step cannot be overturned by 0.2 × 2/3), as expected.

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

### 2026-08-30 — Rerun cost of the blend: #9 ≈ 2× #8 per hidden-test study, `v10c` is the expensive member ✅ measured

Read off the infer kernels' placeholder runs (3 studies on a T4, so warm-up inflates the small members) and the
scoring wall-clock: #8 (infer v10) scored in ≤ 65 min; **#9 (infer v12) took ≈ 1 h 50 min** (17:08 → ≈ 19:00).

| per 100 hidden-test studies | #8 infer v10 | #9 infer v12 | #10 infer v13 |
|---|---|---|---|
| DICOM decode passes | 1 (c01, 224 px; 2 s/study) | **2** (c01 + c02 at 336 px) | 2 |
| 8 c01 checkpoints (v05a 60 s, v05b 21 s, v05g 5×22 s, v06c 96 s) | 287 s | 287 s | 287 s |
| `v08w` (DINOv2-S, all windows @224) | — | 30 s | 30 s |
| **`v10c` (CoAtNet-2 @384, 42 windows)** | — | **174 s** | 166 s |
| `v09h` (CoAtNet-1 @224, all windows) | — | — | 44 s |
| model total | 287 s | 491 s | 535 s |
| ≈ total with decode | ≈ 8 min | ≈ 15 min | ≈ 16 min |

Readings: (1) **`v10c` alone costs 60 % of the eight old checkpoints together** for an OOF at parity with `v09h`
(0.8641 vs 0.8683) — if the LB confirms 384 px buys nothing, dropping it from the default blend cuts the rerun by a
third; (2) the second decode pass is the other half of the increase — a c02-only blend (every c01 member is
dominated by its c02 counterpart) would need one pass again; (3) the 9 h rerun cap is far away (≈ 2 h for ~750
studies implied by #8's ≤ 65 min at 8 min/100), so cost is a budget question, not a risk yet. The `v09h` 5-fold
adds 4 × 44 s = ~3 min/100 (one vote, five checkpoints).
**#10 (infer v13) scored in ≈ 2 h 10 min** (17:37 → ≈ 19:45; both submissions may have queued behind each other),
consistent with 16 min/100 × ~750 studies. A 5-fold `v09h` in the same kernel adds ≈ 15 min to that.

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


> **EXTENDED 2026-09-03 — eighth point, submission #11, and the first one where the offset rule is
> *expected* to break.** The #10 blend with `v09h` as five folds: fold-0 proxy **0.8820** → LB **0.913**,
> offset **+0.031** — just outside the +0.020…+0.030 band that held for seven straight submissions. This is
> the #6 mechanism, not a new one: the proxy OOF holds each study out once and therefore cannot see the
> variance reduction of rank-meaning five folds, so **the offset rule does not apply to a blend whose member
> count changed without its OOF changing** (`v05g` alone showed +0.039 for the same reason). Read the other
> way round: the OOF was *unchanged* at 0.8820 while the LB moved +0.001, i.e. the folds bought a real but
> sub-floor amount that the proxy could not price. Nothing to adopt or drop — the seven-version blend with
> five `v09h` folds is the default, and the +0.02–0.03 offset stays usable only for blends compared at
> equal fold counts.

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
| 9 | 2026-08-30 | rsna-knee-infer v12 (Datasets `rsna-knee-ckpt-v05` = `v05a`/`v05b`, `-v05g`, `-v06` = `v06c`, `-v10c`, `timm-coatnet-rmlp-2-rw-384`, `convnext-tiny-224-hf`; `rsna-knee-train` v17 = `v08w`, pinned later that evening as `rsna-knee-ckpt-v08w`) | **P-23 #2 + P-25 + P-26**: #8's four versions + **`v08w`** (DINOv2-S @224 on the **c02** 2–98 % cache with the **window-attention head**) + **`v10c`** (CoAtNet-2 @384, c02, 42 eval windows); 10 checkpoints, 6 votes; the first mixed-geometry submission — two decode-once groups (c01 + c02) in one kernel | fold-0 proxy 0.8795 (+0.0073 vs #8) | **0.909** | **+0.009 vs 0.900 → ✅ KEEP (1.8× the 0.005 LB floor)**; offset +0.0295 (0.8795 + 0.028 predicted 0.9075). First LB evidence that the 0.936 notebook's mechanism — wide band + per-label window attention + hybrid backbone — carries to the hidden test. Rerun cost ≈ 1 h 50 min vs ≤ 65 min for #8 (Infrastructure 2026-08-30 "Rerun cost of the blend") |
| 10 | 2026-08-30 | rsna-knee-infer v13 (#9's mounts + Dataset `rsna-knee-ckpt-v09h` = `v09h` fold 0, `timm-coatnet-rmlp-1-rw-224`) | **P-23 #2a**: #9's six versions + **`v09h`** (CoAtNet-1 @224 on c02, window-attention head, all windows at eval — the best single model, fold-0 OOF 0.8683); 11 checkpoints, 7 votes, three c02 members in one decode group | fold-0 proxy 0.8820 (+0.0024 vs #9) | **0.912** | **+0.003 vs #9 → 🔁 INCONCLUSIVE as an increment** (0.6× the 0.005 floor; the OOF predicted exactly this), **+0.012 vs #8 → ✅ for the c02 lane (2.4× the floor)**. Best number on the board → **default blend = these seven** (pre-registered tie rule: more members, more robust privately). Offset +0.030; seven calibration points now sit in +0.020…+0.030. Scoring took ≈ 2 h 10 min (17:37 → ≈ 19:45), as the cost table predicted. Next LB move needs a member with ρ < 0.80 vs this blend, i.e. a new input representation (P-23 #3/#4) or P-17 |
| 11 | 2026-08-30 (read 2026-09-03) | rsna-knee-infer v14 (#10's mounts; `rsna-knee-ckpt-v09h` re-versioned to 5 folds + `rsna-knee-ckpt-v08w` pin) | **the #10 blend with `v09h` as 5 folds instead of 1** (15 checkpoints, still 7 votes — `INFER_BLEND="by_version"`); no config change to any other member, two decode groups as in #10 | fold-0 proxy 0.8820 (unchanged — pooled `v09h` OOF 0.8625) | **0.913** | **+0.001 vs #10 → 🔁 INCONCLUSIVE, and pre-registered as such** (the rule set before sending: 0.912–0.916 = 🔁 because folds are replicates; ≥ 0.917 = a real fold-ensemble gain). Consistent with #6/#7: `v05g`'s five folds bought +0.009 as a *lone* version and +0.000 inside a blend — here, inside one vote of seven, they buy +0.001. **Best number on the board, so it stays the default blend**; the value is robustness on private, not public LB. Offset vs the fold-0 proxy is **+0.031** (n=8), a touch above the +0.020–0.030 band — expected, since a pooled/fold-0 proxy cannot see the variance reduction of averaging five models (the #6 lesson). Ref 55899052; scored ≈ 2 h 40 min as the rerun-cost table predicted |
| 12 | 2026-09-21 (read 2026-09-22) | rsna-knee-fork v3 (17 public sources of the 0.942 notebook + Datasets `rsna-knee-ckpt-v08w`, `-v09h` 5-fold, `timm-coatnet-rmlp-1-rw-224`) | **P-27**: the public 0.942 notebook's cells 0–49 byte-identical (20 DINOv2 + A5 + RadImageNet/calibrator + Raptor ×4 @94 windows + CoAt family, its LB-tuned weights untouched), FineSpacing stage dropped, **our c02 arm (`v08w` 1 fold + `v09h` 5 folds, one vote each) rank-blended at β = 0.20** on top; fail-soft to the exact anchor | none for the fork (our arm's fold-0 proxy ≈ 0.87; no OOF for the anchor) | **0.939** | **−0.003 vs the author's 0.942 → the pre-registered action rule (< 0.940) fires, but as evidence it is 🔁 INCONCLUSIVE** (0.6× the 0.005 floor). Two confounds neither the log nor the outputs can resolve for a hidden-test rerun: (a) the anchor has never been scored from our account (all 17 sources are unpinned Datasets — a re-versioned source moves the anchor), and (b) the arm is fail-soft, so a skipped arm would have submitted the anchor unchanged. The offset rule does not apply (no OOF). Next: β 0.10 with the same members (fork v4), the anchor alone as the control (β 0), then β 0.10 with the P-28 members. Ref 56442019; scored ≈ 10 h (21:24 → before 07:50) |
| 13 | 2026-09-22 | rsna-knee-fork v4 (same 20 sources as v3) | **P-27, β 0.10**: identical to #12 except the blend weight of our arm (0.20 → 0.10); the pre-registered action for a < 0.940 read | none (fork) | **0.942** | sent 08:30, ref 56458837; placeholder: `fork_diagnostics.json` status `beta0.10`, subprocess rc 0 in 98 s, anchor sha = submission sha on the 3 placeholder studies (expected — a 1/3 rank step cannot be overturned at β 0.10). Read-out in the Scoreboard row |
| 14 | 2026-09-22 | rsna-knee-fork v5 (v4's 20 sources + Datasets `rsna-knee-ckpt-v09a`, `rsna-knee-ckpt-v08a`; 22 sources) | **P-27 + P-28**: #13's blend with the two production members added to our arm — `INFER_MEMBERS = [v08w, v09h, v09a, v08a]`, one vote per version (`v09h` = 5 folds inside its vote), β 0.10 | none (fork; the new members have no OOF — traps 32) | **0.941** | sent 08:41, ref 56459131; placeholder: `blend: by_version -> v08w (1 fold), v09h (5 folds), v09a (1 fold), v08a (1 fold)`, rc 0 in 82 s, anchor graph 186 s with 20/20 DINO + 5 A5 folds. Hidden-test arm cost ≈ #12's + two single-fold members (≈ +20 min). Read-out vs #13 in the Scoreboard row |
| 15 | 2026-09-22 | rsna-knee-fork v6 (the 20 sources of v3/v4) | **Anchor-only control (P-27)**: `build_fork.py --beta 0.0 --members v08w v09h` — the fork's arm cell raises `_ForkControl` before the runtime gate, our subprocess is never launched, `submission.csv` is the anchor's own file (`fork_diagnostics.json`: `status anchor_control`, `subprocess null`, anchor sha = submission sha; anchor graph 204 s, 20/20 DINO + 5 A5 folds on the placeholder) | none | **0.942** | sent 10:08, ref 56461317. Purpose: fix the anchor's score from *our* account — the number #12 (0.939, β 0.20), #13 (β 0.10) and #14 (β 0.10 + P-28 members) are compared against, instead of the author's stated 0.942. Expected ≈ 6 h to score (no arm) |
| 16 | 2026-09-22 | rsna-knee-fork v7 (the 20 sources of v3/v4/v6) | **Final-selection hedge**: `build_fork.py --anchor-preset parent --beta 0.0` — the anchor's own `PRESET` default patched `speedy` → `parent` (one token, asserted once), which flattens the per-label outer CoAtNet map (LatMen 1.00, ACL / LatOA / Fracture 0.75, MedMen 0.80) to 0.60; our arm not run. Placeholder green 20:29 (fork log `preset=parent … outer CoAtNet weight per finding: flat 0.60`; `btkd_v559_complete.json` diff vs v6: every per-label weight 0.6; `status: anchor_control`, submission sha = anchor sha) | none (fork) | **0.940** | sent 20:37, ref 56471784, read 2026-09-23 08:40 (≈ 12 h to score, no arm). **−0.002 vs #15 (anchor 0.942) → 🔁 (0.4× the floor), inside the expected 0.939–0.941 band**: the public-tuned per-label map buys ≈ +0.002 on the public split. Validated flat-weights hedge for the second final slot (Scoreboard row; entry 2026-09-23) |
| 17 | 2026-09-23 | rsna-knee-fork v8 (v5's 22 sources: the 20 public ones + Datasets `rsna-knee-ckpt-v09a` / `-v08a`, both re-versioned 11:44 to the S2 checkpoints) | **P-27 + P-28 S2**: #14's blend with the two production members retrained under the 8-epoch regime — `INFER_MEMBERS = [v08w, v09h, v09a, v08a]`, one vote per version (`v09h` = 5 folds inside its vote), β 0.10; `v09a` = CoAtNet-1 + `batch_studies 2, grad_accum 2, aug light` (gold-58 SWA 0.8922), `v08a` = DINOv2-S (0.8850) | none (fork; production members have no OOF — traps 32) | **0.941** | **read 20:00 → 🔁** (−0.001 vs 0.942, = #14 with the 16-epoch members; entry "Submission #17 read 0.941"). sent 11:54, ref 56489906. Placeholder green 11:45 → 11:53 (0.12 h): `status beta0.10`, `members [v08w, v09h, v09a, v08a]`, subprocess rc 0 in 190 s, 8 checkpoints in one geometry group, `v09a/fold0 … [epoch 7, score 0.8922]` / `v08a/fold0 … [score 0.885]` = the S2 checkpoints, anchor sha = submission sha on the 3 placeholder studies (expected at β 0.10). **Read vs #13 / #15 (0.942): ≥ 0.947 = our arm counts (a correctly trained production member helps), 0.940–0.946 = 🔁, ≤ 0.939 = the arm hurts**; vs #14 (0.941, the 16-epoch members) isolates the epoch budget + S1 knobs. ≈ 10 h to score (≈ 22:00); honest expectation 0.942 ± 0.001. 4 submissions left today |
| 18 | 2026-09-23 | rsna-knee-infer v15 (v14's ckpt Datasets + `rsna-knee-ckpt-v09a`) | **Solo baseline (member-strength plan Task 11)**: `MODE="infer"`, `INFER_MEMBERS = ["v09a"]` — the S2 production `v09a` alone (CoAtNet-1 c02 window_attn, all 4,349 studies, 8 ep SWA, `batch_studies 2, grad_accum 2, aug light`) | none (production member, gold-58 0.8922 — traps 32) | **0.918** | sent 14:36, ref 56493264; placeholder 2.2 min (`infer members (1): v09a/fold0 … score 0.8922`, `constant labels 0`). **The reference for the Raptor-distilled retrain (P-39)**: ≥ 0.923 ✅ / ≤ 0.922 🔁 / < 0.918 ❌. A single model of ours above our own 12-member blend (#11, 0.913) — the "our members are ≈ 0.89–0.90 solo" estimate in the 2026-09-23 "#16 … why nothing of ours moves 0.942" entry was ≈ 0.02 too low for the all-data SWA member |
| 19 | 2026-09-23 | rsna-knee-infer v16 (v15's mounts + Dataset `rsna-knee-ckpt-v09t`) | **Self-distilled production member, solo (P-38 in production)**: `INFER_MEMBERS = ["v09t"]` — the `v09a` recipe trained on 0.5 · LLM + 0.5 · quantile-matched `selfdistill_v1` (RunPod 4090, 35 min), gold-58 SWA 0.9009 (`v09a`: 0.8922) | none (production member — traps 32) | **0.917** | **read 2026-09-24 00:07 → ❌ by the pre-registered rule (< 0.918): −0.001 vs #18, sub-floor — no transfer** (entry "Submission #19"). sent 23:25, ref 56504077; placeholder 2 min (`infer members (1): v09t/fold0 … [epoch 7, score 0.9009]`, `constant labels 0`). **Read vs #18 (0.918): ≥ 0.923 ✅ self-distillation transfers to the production member → retrain `v08a` the same way and put both into the fork; 0.919–0.922 🔁; < 0.918 ❌.** 2 submissions left today |
| 20 | 2026-09-26 | rsna-knee-infer v17 (v16's mounts + Dataset `rsna-knee-ckpt-v09r`) | **Raptor-distilled production member, solo (P-39 / Task 12)**: `INFER_MEMBERS = ["v09r"]` — the `v09a` recipe trained on 0.5 · LLM + 0.5 · quantile-matched `raptor_teacher` (RunPod 4090, 48 min), gold-58 SWA 0.9093 (`v09a`: 0.8922, `v09t`: 0.9009) | none (production member — traps 32) | **0.927** | **read 2026-09-27 09:28 → ✅ KEEP: +0.009 vs #18 (0.918), above the 0.923 line** (entry "Submission #20"). sent 2026-09-27 00:36, ref 56590282; placeholder 4 min after an immediate start (`infer members (1): v09r/fold0 … [epoch 7, score 0.9093, ema True]`, `decode-once verified`, `constant labels 0`; outputs `artifacts/kaggle_out/infer_solo_v09r/`). **Read vs #18 (0.918): ≥ 0.923 ✅ the Raptor teacher transfers → `v08r` + the fork at β 0.10; 0.919–0.922 🔁; < 0.918 ❌.** 4 submissions left today |
| 21 | 2026-09-27 | rsna-knee-infer v19 (v17's mounts + Dataset `rsna-knee-ckpt-v08r`) | **Raptor-distilled DINOv2-S member, solo (P-40 step B)**: `INFER_MEMBERS = ["v08r"]` — the S2 `v08a` recipe (DINOv2-S @224, c02, window_attn, 8 ep SWA 5–7) trained on 0.5 · LLM + 0.5 · quantile-matched `raptor_teacher` (RunPod A100, 31 min), gold-58 SWA 0.8981 (`v08a`: 0.8850); **first solo with P-41** (16-thread scan, 8 decode workers) | none (production member — traps 32) | **0.918** | **read 2026-09-27 16:24 UTC → −0.009 vs `v09r` (#20, 0.927), 0.002 under the ≈ 0.920 fork-member bar; P-41 speed unread (the watcher died with the laptop: PENDING at 13:43, first seen COMPLETE 14:24 → ≤ 2 h 43 min, no information vs #19's ≤ 42)** (entry "Submission #21"). sent 13:41:35 UTC, ref 56610108; placeholder green (`smoke False`, `infer members (1): v08r/fold0 … [epoch 7, score 0.8981, ema True]`, `decode-once workers: 8 (cpus 4, usable 4)`, `decode-once verified`, `constant labels 0`; outputs `artifacts/kaggle_out/infer_solo_v08r/`); scoring timed by `src/watch_submission.py` (60 s polls → `artifacts/submission_timing.csv`). **Reads:** LB vs `v09r` 0.927 (no `v08a` solo baseline exists); P-41 speed **≤ 30 min ✅ / 30–42 🔁 / > 42 ❌** vs #19's ≤ 42 min bound. 4 submissions left today |
| 22 | 2026-09-27 | rsna-knee-fork v9 (the anchor's 16 public sources — 13 Datasets, 1 Model, 2 kernels — + Datasets `rsna-knee-ckpt-v09r`, `timm-coatnet-rmlp-1-rw-224`) | **P-40 step A**: the public 0.942 graph (cells 0–49 verbatim) + our arm = **`v09r` alone** (Raptor-distilled CoAtNet-1, solo 0.927 #20) at **β 0.10** — `build_fork.py --members v09r --member v09r=… --beta 0.10` | none (production member — traps 32) | **0.942** | **read 2026-09-28 09:12 UTC (first look) → 🔁 ±0.000 vs 0.942 (P-40)**; scoring time unknown — the watcher died with the laptop and wrote no row, so the only bound is ≤ 16 h 36 min after sending (entry "Submission #22"). sent 16:36:17 UTC, ref 56614068, after **≈ 3 h 40 min QUEUED** (pushed 12:48; traps 41 second instance); placeholder green (`fork_diagnostics.json`: `status beta0.10`, `members [v09r]`, subprocess rc 0 in 104 s, `v09r/fold0 … [epoch 7, score 0.9093, ema True]`, `decode-once workers: 8`, anchor sha = submission sha `7c6dfe8b…` as expected at β 0.10 on 3 studies, elapsed 0.10 h; outputs `artifacts/kaggle_out/fork_v9/`); watched by `watch_submission.py` (120 s). **Read vs #13 / #15 (0.942): ≥ 0.947 ✅ our member counts / 0.940–0.946 🔁 / ≤ 0.939 ❌** (P-40) |
| 23 | 2026-09-27 | rsna-knee-infer v20 (v19's mounts) | **P-42**: `INFER_MEMBERS = ["v09r", "v08r"]` — flat rank-mean of the two Raptor-distilled families (CoAtNet-1 solo 0.927 + DINOv2-S solo 0.918), one vote each, one shared decode (`artifacts/infer_pair_v09r_v08r.py` = `src` + 3 seds) | none (production members — traps 32); gold-58 hand-computed: ρ 0.924, rank-mean 0.9085 vs `v09r` 0.9093 | **0.927** | **read 17:05:54 UTC → 🔁 ±0.000 vs `v09r` (P-42); scored within [28.3, 29.3] min of sending → P-41 ✅** (`artifacts/submission_timing.csv`; entry "Submission #23"). sent 16:36:38 UTC, ref 56614080; placeholder green (`smoke False`, `infer members (2): v09r/fold0, v08r/fold0`, 1 geometry group, `decode-once workers: 8 (cpus 4, usable 4)`, `decode-once verified`, scores 0.9093 / 0.8981, `blend: by_version -> v09r (1 fold), v08r (1 fold)`, `constant labels 0`, inference 0.6 min; outputs `artifacts/kaggle_out/infer_pair_v09r_v08r/`); **the P-41 timed solo** — `watch_submission.py` 60 s polls, laptop awake. **Reads:** LB vs `v09r` 0.927: **≥ 0.932 ✅ / 0.923–0.931 🔁 / ≤ 0.922 ❌** (P-42); P-41 speed **≤ 30 min ✅ / 30–42 🔁 / > 42 ❌**. 2 submissions left today |
| 24 | 2026-09-28 | rsna-knee-infer v21 (v20's mounts + Datasets `rsna-knee-ckpt-v09x`, `rsna-knee-ckpt-v09u`) | **P-44, the seed/platform twin solo**: `INFER_MEMBERS = ["v09u"]` — the `v09r` recipe exactly, **seed 43**, trained on a Kaggle T4 (`rsna-knee-train` v30), gold-58 SWA 0.8995 (`v09r`: 0.9093) (`artifacts/infer_solo_v09u.py` = `src` + 3 seds) | none (production member — traps 32) | **0.927** | **read 09:48:00 UTC → s = \|0.927 − 0.927\| = 0.000 (P-44 ✅: one-seed deltas now need ≥ 0.004; P-39 re-confirmed); scored within [20.4, 21.4] min of sending** (entry "Submissions #24–#26"). sent 09:26:38 UTC, ref 56636408; placeholder green (`smoke False`, `infer members (1): v09u/fold0`, `img_size 224`, `decode-once workers: 8`, `decode-once verified`, `[epoch 7, score 0.8995, ema True]`, `constant labels 0`, 98 s / 100 studies; outputs `artifacts/kaggle_out/infer_solo_v09u/`); watched by `watch_submission.py` (60 s). **Read (P-44):** s = \|`v09u` − 0.927\| = one draw of the Kaggle-retrain spread; **`v09u` ≤ 0.920 → re-open P-39** |
| 25 | 2026-09-28 | rsna-knee-infer v22 (v21's mounts) | **P-43, the 320-px student solo**: `INFER_MEMBERS = ["v09x"]` — the `v09r` recipe at **img_size 320** (batch 1 × accum 4), seed 42, Kaggle T4 (`rsna-knee-train` v30), gold-58 SWA 0.9094 (`artifacts/infer_solo_v09x.py`) | none (production member — traps 32) | **0.929** | **read 09:59:07 UTC → 🔁 +0.002 vs m = 0.927 (needed ≥ 0.932, P-43); the best solo single member of ours; scored within [29.3, 30.3] min** (entry "Submissions #24–#26"). sent 09:28:50 UTC, ref 56636467; placeholder green (`infer members (1): v09x/fold0`, timm `img_size 320` (445 tensors), `decode-once verified`, `[epoch 7, score 0.9094, ema True]`, `constant labels 0`, **136 s / 100 studies** = 1.4× `v09u`; outputs `artifacts/kaggle_out/infer_solo_v09x/`); watched (60 s). **Read (P-43):** vs m = mean(0.927, #24), s from #24: **✅ ≥ m + max(0.005, 2s) / ❌ ≤ m − max(0.005, 2s) / 🔁 otherwise** |
| 26 | 2026-09-28 | rsna-knee-infer v23 (v21's mounts) | **P-44, the 2-seed production member**: `INFER_MEMBERS = ["v09r", "v09u"]` — flat rank-mean of the seed-42 (RunPod) and seed-43 (Kaggle) copies of one recipe (`artifacts/infer_pair_v09r_v09u.py`) | none (production members — traps 32); gold-58 hand-computed: ρ 0.952, rank-mean 0.9065 vs `v09r` 0.9093 | **0.930** | **read 10:01:41 UTC → the pre-registered ≥ 0.930 line met exactly (+0.003 vs either member, under the 0.004 floor #24 sets) → the production member; scored within [28.3, 29.3] min** (entry "Submissions #24–#26"). sent 09:32:25 UTC, ref 56636549; placeholder green (`infer members (2): v09r/fold0, v09u/fold0`, 1 geometry group, `decode-once verified`, scores 0.9093 / 0.8995, `blend: by_version -> v09r (1 fold), v09u (1 fold)`, `constant labels 0`; outputs `artifacts/kaggle_out/infer_pair_v09r_v09u/`); watched (60 s). **Read (P-44):** **≥ 0.930 ✅ seed ensembling is a lever / otherwise 🔁** — it becomes the production member either way. 2 submissions left today (1 after the fork) |
| 27 | 2026-09-28 | rsna-knee-fork v10 (v9's 19 sources) | **P-40 β 0.20 retry** (Tian's go after #22 🔁): the public 0.942 graph (cells 0–49 verbatim) + **`v09r` alone** at **β 0.20** — `build_fork.py --members v09r --member v09r=… --beta 0.20` (only the β and the pipeline payload, now carrying the P-43 code, differ from v9) | none (production member — traps 32) | **0.941** | **read (first seen) 15:33 UTC → 🔁 −0.001 vs #13 / #15 0.942, inside the pre-registered 0.940–0.946 band (P-40 closes); +0.002 over #12 (β 0.20, 0.87-level members, 0.939) is sub-floor; the watcher died with the laptop → scored within (2 h 32 min, 5 h 53 min] of sending** (entry "Submissions #27–#28"). sent 09:40:09 UTC, ref 56636712; pushed 09:27 and RUNNING at once (no queue); placeholder green (`fork_diagnostics.json`: `status beta0.20`, `members [v09r]`, subprocess rc 0 in 286 s, `[epoch 7, score 0.9093, ema True]`, `decode-once verified`, anchor sha = submission sha `7c6dfe8b…` on 3 studies as at #12 (β 0.20), elapsed 0.18 h; outputs `artifacts/kaggle_out/fork_v10/`); watched (120 s). **Read vs #13 / #15 (0.942): ≥ 0.947 ✅ / 0.940–0.946 🔁 / ≤ 0.939 ❌** (P-40); compare #12 (β 0.20, 0.87-level members) 0.939. 1 submission left today |
| 28 | 2026-09-28 | rsna-knee-infer v24 (v23 + `PROBE_CONST_LABELS`) | **P-53 DIAGNOSTIC probe, not a candidate**: the #26 pair (`v09r` + `v09u`) with ACL, MCL, PF OA, Lateral Meniscus written as a constant 0.5 | none | **0.785** | **read (first seen) 15:33 UTC → mean4 = 0.5 + 3 × (0.930 − 0.785) = 0.935 (± 0.003), mean8 = 1.5 × 0.785 − 0.25 = 0.9275 (± 0.001): mean8 − mean4 = −0.0075 → the gold-58 structural deficit (+0.028) does NOT replicate on the public test — gold-specific by the pre-registered rule (mean4 ≥ mean8 − 0.01); c03 (P-56) runs with a lower prior; scored within (15 min, 3 h 36 min]** (entry "Submissions #27–#28"). sent 11:56:44 UTC, ref 56639910; placeholder verified (the 4 columns = 0.5, the other 8 identical to v23, `constant labels 4`, the guard allows <= 6). **Read:** mean4 = 0.5 + 3 x (0.930 - probe) (+-0.003), mean8 = (12 x 0.930 - 4 x mean4) / 8; structural weakness confirmed on the test if mean8 - mean4 >= 0.03 (gold-58 says 0.028); if mean4 >= mean8 - 0.01 the gold deficit is gold-specific. 0 submissions left today |
| 29 | 2026-09-29 | rsna-knee-infer v25 (v24's mounts; `artifacts/infer_p52_trio.py` = `src` + 3 seds) | **P-52, the three-member production blend**: `INFER_MEMBERS = ["v09r", "v09u", "v09x"]` — flat rank-mean, `by_version` | none (production members — traps 32); gold-58 trio 0.9110 vs pair 0.9065 | **0.931** | **read 12:36:34 UTC → 🔁 (+0.001 vs #26; the pair stays, P-52 closes); scored within [53.5, 54.5] min** (entry "Submissions #29–#32"). sent 11:42:07 UTC, ref 56673976; placeholder green (`smoke False`, `infer members (3): v09r/fold0, v09u/fold0, v09x/fold0`, img 224 / 224 / 320, `decode-once verified`, `constant labels 0`, inference 0.5 min; outputs `artifacts/kaggle_out/infer_p52_trio/`); watched by `watch_submission.py`. **Read (P-52, pre-registered):** vs #26 0.930: **≥ 0.934 ✅ / 0.931–0.933 🔁 / ≤ 0.930 ❌** |
| 30 | 2026-09-29 | rsna-knee-infer v26 (+ Datasets `rsna-knee-ckpt-v11a`, `-v11b`, `-v09k`, `-v13a`) | **P-56, c03 solo #1**: `INFER_MEMBERS = ["v11a"]` — the `v09r` recipe on the c03 input (24/24/24/14/8/8, 150 mm), seed 42, Kaggle T4 (`rsna-knee-folds` v10) (`artifacts/infer_solo_v11a.py`) | none; gold-58 SWA 0.9204 | **0.932** | **read 12:13:36 UTC — our best single member (was `v09x` 0.929); scored within [27.3, 28.3] min.** sent 11:45:18 UTC, ref 56674064; placeholder green (`infer members (1): v11a/fold0`, **`decode-once verified [c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20]`** — the first c03 decode at inference, `[epoch 7, score 0.9204, ema True]`, `constant labels 0`). **Read with #31 (P-56):** m = mean(#30, #31) vs 0.927: **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285** |
| 31 | 2026-09-29 | rsna-knee-infer v27 (v26's mounts) | **P-56, c03 solo #2**: `INFER_MEMBERS = ["v11b"]` — as #30, seed 43 (`artifacts/infer_solo_v11b.py`) | none; gold-58 SWA 0.9167 | **0.929** | **read 12:20:18 UTC → m = 0.9305 → 🔁 (P-56 bands); scored within [31.3, 32.3] min.** sent 11:48:00 UTC, ref 56674132; placeholder green (`infer members (1): v11b/fold0`, c03 decode verified, `[epoch 7, score 0.9167, ema True]`, `constant labels 0`, CSV differs from v26's). Read with #30 |
| 32 | 2026-09-29 | rsna-knee-infer v28 (v26's mounts) | **P-56, the c03 pair**: `INFER_MEMBERS = ["v11a", "v11b"]` (`artifacts/infer_pair_v11a_v11b.py`) | none; gold-58 pair 0.9207 vs the c02 pair 0.9065 | **0.932** | **read 12:37:04 UTC → +0.002 vs the c02 pair → 🔁; = the best solo read (with #30); scored within [45.4, 46.4] min.** sent 11:50:40 UTC, ref 56674211; placeholder green (`infer members (2): v11a/fold0, v11b/fold0`, c03 decode verified, `constant labels 0`). **Read (P-56):** vs #26 (the c02 pair) 0.930 — the like-for-like input comparison |
| 33 | 2026-09-29 | rsna-knee-infer v29 (v26's mounts) | **P-54, the 5-fold cross-fit ensemble**: `INFER_MEMBERS = ["v09k0", …, "v09k4"]` — five fold models of the `v09r` recipe, each on 4/5 of the data, one vote each (`artifacts/infer_xf5_v09k.py`) | none (each fold's OOF is vs the LLM — traps 39) | **0.928** | **read 13:00:08 UTC → ❌ by the P-54 bands (≤ 0.930): −0.002 vs #26; scored within [65.4, 66.4] min** (entry "Submissions #29–#32" addendum). sent 11:53:42 UTC, ref 56674282; placeholder green (`infer members (5): v09k0/fold0 … v09k4/fold4`, c02 decode verified, `blend: by_version`, `constant labels 0`); five members ≈ 1–1.5 h to score. **Read (P-54):** vs #26 0.930: **≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌** |
| 34 | 2026-09-30 | rsna-knee-infer v30 (+ Datasets `rsna-knee-ckpt-v09o`, `-v09o2`) | **P-55, student solo #1**: `INFER_MEMBERS = ["v09o"]` — the `v09r` recipe (CoAtNet-1, c02, window_attn, all 4,349, 8 ep, SWA 5–7) on 0.25 LLM + 0.375 Raptor + 0.375 cross-fit OOF (`xfit_v09k`, mix 0.75), seed 42, Kaggle T4 (`rsna-knee-train` v35) (`artifacts/infer_solo_v09o.py`) | none; gold-58 SWA 0.9121 | **0.927** | **read 11:26:32 UTC → with #35 m = 0.927 → ❌ (P-55); scored within [18.9, 20.4] min** (entry "Submissions #34–#38"). sent 11:06:07 UTC, ref 56705590 (the three student placeholders were built 2026-09-29, the planned 00:03 UTC send did not happen); placeholder green (`smoke False`, `infer members (1): v09o/fold0`, `decode-once verified [c02_…crop130…]`, `[epoch 7, score 0.9121, ema True]`, `blend: by_version`, `constant labels 0`; outputs `artifacts/kaggle_out/infer_solo_v09o/`); watched by `watch_submission.py` (90 s). **Read with #35 (P-55, pre-registered):** m = mean(#34, #35) vs 0.927: **✅ m ≥ 0.9315 / 🔁 0.9285 ≤ m < 0.9315 / ❌ m < 0.9285** |
| 35 | 2026-09-30 | rsna-knee-infer v31 (v30's mounts) | **P-55, student solo #2**: `INFER_MEMBERS = ["v09o2"]` — as #34, seed 43 (`artifacts/infer_solo_v09o2.py`) | none; gold-58 SWA 0.9104 | **0.927** | **read 11:28:09 UTC → m = 0.9270 < 0.9285 → ❌ DEAD END (P-55); scored within [20.5, 22.0] min.** sent 11:06:11 UTC, ref 56705594; placeholder green (`infer members (1): v09o2/fold0`, c02 decode verified, `[epoch 7, score 0.9104, ema True]`, `constant labels 0`, CSV differs from v30's). Read with #34 |
| 36 | 2026-09-30 | rsna-knee-infer v32 (v30's mounts) | **P-55, the student pair**: `INFER_MEMBERS = ["v09o", "v09o2"]` (`artifacts/infer_pair_v09o_v09o2.py`) | none; gold-58 pair 0.9119 vs the c02 pair 0.9065 | **0.928** | **read 11:36:06 UTC → −0.002 vs the c02 pair → ❌ (≤ 0.930); scored within [28.3, 29.8] min.** sent 11:06:16 UTC, ref 56705596; placeholder green (`infer members (2): v09o/fold0, v09o2/fold0`, c02 decode verified, `constant labels 0`). **Read (P-55):** vs #26 (the c02 pair) 0.930: **≥ 0.935 ✅ / 0.931–0.934 🔁 / ≤ 0.930 ❌** |
| 37 | 2026-09-30 | rsna-knee-infer v33 (+ Datasets `rsna-knee-ckpt-v13b`, `-v13c`) | **P-59, ResNet-34 solo**: `INFER_MEMBERS = ["v13c"]` — ResNet-34 (c02, Raptor mix 0.5, all 4,349) at a CNN LR (uniform `lr_backbone` 3e-4, `llrd_decay` 1.0, 12 ep, SWA 9–11) + frozen encoder BatchNorm (`rsna-knee-train` v37) (`artifacts/infer_solo_v13c.py`) | none; gold-58 SWA 0.9014 (ρ to `v09r` 0.86) | **0.921** | **read 11:23:32 UTC → −0.006 vs 0.927 → ❌ (≤ 0.922); scored within [15.6, 17.1] min.** sent 11:06:26 UTC, ref 56705605; placeholder green (`infer members (1): v13c/fold0`, c02 decode verified, `[epoch 11, score 0.9014, ema True]`, `constant labels 0`). **Read (the P-57 bands, written before sending):** vs 0.927: **≥ 0.932 ✅ / 0.923–0.931 🔁 / ≤ 0.922 ❌** — a ✅/🔁 makes the ResNet a candidate family for a blend and for the Efficiency track (P-18; 0.12 s/study vs CoAtNet c03 0.35) |
| 38 | 2026-09-30 | rsna-knee-infer v34 (v33's mounts) | **P-59, c03 pair + ResNet**: `INFER_MEMBERS = ["v11a", "v11b", "v13c"]` — flat rank-mean, `by_version`, two decode-once passes (c03 for the CoAtNets, c02 for the ResNet) (`artifacts/infer_trio_v11a_v11b_v13c.py`) | none; gold-58 trio 0.9189 vs the c03 pair 0.9207 | **0.932** | **read 11:56:54 UTC → = the c03 pair → ❌ (≤ 0.932); scored within [48.9, 50.4] min.** sent 11:06:31 UTC, ref 56705607; placeholder green (`infer members (3): v11a/fold0, v11b/fold0, v13c/fold0`, `2 geometry group(s)`, both decodes verified, `constant labels 0`). **Read (written before sending):** vs #32 (the c03 pair) 0.932: **≥ 0.936 ✅ / 0.933–0.935 🔁 / ≤ 0.932 ❌** — the first CNN family in a solo blend (#23's second family was DINOv2-S, 0.927 = its first member alone). 0 submissions left today |
| 39 | 2026-10-03 | rsna-knee-infer v35 (+ Datasets `rsna-knee-ckpt-v11n`, `-v11n2`) | **P-60, noisy-student solo #1**: `INFER_MEMBERS = ["v11n"]` — the `v11a` recipe (c03 CoAtNet-1, Raptor 0.5, all 4,349) + drop-path 0.1 + aug heavy + 12 epochs (SWA 9–11), seed 42 (`rsna-knee-folds` v11 → `rsna-knee-train` v39) (`artifacts/infer_solo_v11n_1003.py`) | none; gold-58 SWA 0.9152 | **0.932** | **read 12:57:00 UTC → = `v11a` (#30); with #40, m = 0.932 → 🔁 (0.926 < m < 0.935); scored within [28.9, 30.4] min.** sent 12:26:37 UTC, ref 56797717; placeholder green (`infer members (1): v11n/fold0`, c03 decode verified, `[epoch 11, score 0.9152]`, `constant labels 0`). **Read (P-60, written before sending):** m = mean(#39, #40) vs m(`v11a`, `v11b`) = 0.9305: **✅ m ≥ 0.935 / 🔁 0.926 < m < 0.935 / ❌ m ≤ 0.926** |
| 40 | 2026-10-03 | rsna-knee-infer v36 (v35's mounts) | **P-60, noisy-student solo #2**: `INFER_MEMBERS = ["v11n2"]` — as #39, seed 43 (`artifacts/infer_solo_v11n2_1003.py`) | none; gold-58 SWA 0.9166 | **0.932** | **read by 13:00 UTC → +0.003 vs `v11b` (#31 0.929); m(#39, #40) = 0.932 vs 0.9305 → 🔁 P-60.** sent 12:30:48 UTC, ref 56797813; placeholder green (`infer members (1): v11n2/fold0`, `constant labels 0`). Read with #39 (above) |
| 41 | 2026-10-03 | rsna-knee-infer v37 (+ Datasets `rsna-knee-ckpt-v11p`, `-v13h`, `-v11d`, `-v11dl`) | **P-63, reader solo**: `INFER_MEMBERS = ["v11p"]` — the `v11a` recipe (c03 CoAtNet-1, Raptor 0.5) + per-finding spatial reader + slot-count norm, seed 42 (`rsna-knee-train-b` v4) (`artifacts/infer_solo_v11p.py`) | none; gold-58 SWA 0.9185 | **0.929** | **read 18:04:48 UTC → −0.003 vs `v11a` 0.932 → 🔁 (band 0.929–0.935), not adopted; scored within [30.5, 32.0] min.** sent 17:32:50 UTC, ref 56803390; placeholder green (`infer members (1): v11p/fold0`, strict load, `constant labels 0`) |
| 42 | 2026-10-03 | rsna-knee-infer v38 (v37's mounts) | **P-64, long heavy-aug CNN solo**: `INFER_MEMBERS = ["v13h"]` — ResNet-34 on c03, CNN LR 3e-4 uniform, frozen BN, aug heavy, drop-path 0.1, 30 epochs (SWA 27–29), Raptor 0.5 (`artifacts/infer_solo_v13h.py`) | none; gold-58 SWA 0.9001 | **0.931** | **read 17:57:38 UTC → +0.010 vs `v13c` 0.921 → ✅ KEEP (bar ≥ 0.925; member ≥ 0.930 yes; main bet ≥ 0.935 no); scored within [17.0, 18.5] min — our fastest solo.** sent 17:39:09 UTC, ref 56803492; placeholder green (`v13h/fold0: timm:resnet34`, `constant labels 0`) |
| 43 | 2026-10-03 | rsna-knee-infer v39 (v37's mounts) | **P-45, D4-target student solo**: `INFER_MEMBERS = ["v11d"]` — the `v11a` recipe on 0.5 LLM + 0.25 matched Raptor + 0.25 matched D4, seed 42 (`rsna-knee-train` v41) (`artifacts/infer_solo_v11d.py`) | none; gold-58 SWA 0.9184 | **0.930** | **read 18:21:49 UTC → −0.002 vs `v11a` 0.932 → 🔁 (band 0.929–0.935); the gold target-level +0.006 did not transfer; scored within [36.4, 37.9] min.** sent 17:43:54 UTC, ref 56803573; placeholder green (`infer members (1): v11d/fold0`, `constant labels 0`) |
| 44 | 2026-10-04 | rsna-knee-infer v40 (v37's mounts) | **P-61, higher CoAtNet LR solo**: `INFER_MEMBERS = ["v11dl"]` — `v11d` + lr 2e-4 / LLRD 0.85 (`rsna-knee-train` v41) (`artifacts/infer_solo_v11dl.py`) | none; gold-58 SWA 0.9132 | **0.927** | **read 07:48:21 UTC → −0.003 vs `v11d` 0.930 → 🔁 (bands ✅ ≥ 0.934 / ❌ ≤ 0.926); gold −0.005 the same way → P-61 closed; scored within [28.9, 30.4] min.** sent 07:17:59 UTC, ref 56817302 |
| 45 | 2026-10-04 | rsna-knee-infer v41 (+ Datasets `rsna-knee-ckpt-v13r`, `-v13e`) | **P-64 follow-up, ResNet-50 solo**: `INFER_MEMBERS = ["v13r"]` — ResNet-50 a1 on the `v13h` recipe (`rsna-knee-train` v43) (`artifacts/infer_solo_v13r.py`) | none; gold-58 SWA 0.9111 | **0.934** | **read 07:41:36 UTC → +0.003 vs `v13h` 0.931 → 🔁 (top of 0.928–0.934); scored within [18.4, 19.9] min.** sent 07:21:44 UTC, ref 56817383 |
| 46 | 2026-10-04 | rsna-knee-infer v42 (v41's mounts) | **P-64 follow-up, EfficientNet-B0 solo**: `INFER_MEMBERS = ["v13e"]` — EfficientNet-B0 ra on the `v13h` recipe (`rsna-knee-train` v43) (`artifacts/infer_solo_v13e.py`) | none; gold-58 SWA 0.9126 | **0.935** | **read 07:39:27 UTC → +0.004 vs `v13h` 0.931 → ✅ KEEP (bar ≥ 0.935) — our best solo (was 0.932); scored within [13.8, 15.3] min, our fastest.** sent 07:24:08 UTC, ref 56817439 |
| 47 | 2026-10-04 | rsna-knee-infer v43 (v41's mounts) | **Two-family blend**: `INFER_MEMBERS = ["v11a", "v13h"]` — flat rank-mean, one c03 decode pass (`artifacts/infer_pair_v11a_v13h.py`) | none; gold-58 0.9170 | **0.934** | **read 07:56:24 UTC → +0.002 over the best member (`v11a` 0.932) → 🔁 (keep bar ≥ 0.936); the first blend of ours above both members; scored within [27.3, 28.8] min.** sent 07:27:35 UTC, ref 56817516 |
| 48 | 2026-10-04 | rsna-knee-infer v44 (v41's mounts) | **Three-family blend**: `INFER_MEMBERS = ["v11a", "v13r", "v13e"]` — flat rank-mean of CoAtNet-1 + ResNet-50 + EfficientNet-B0, one c03 decode pass (`artifacts/infer_trio_v11a_v13r_v13e.py`) | none; gold-58 0.9233 | **0.938** | **read 08:40:31 UTC → +0.003 over the best member (`v13e` 0.935), +0.004 over the members' mean → 🔁 by rule (keep bar ≥ 0.939); our best own-model score; scored within [42.3, 43.8] min.** sent 07:56:41 UTC, ref 56818172 |
| 49 | 2026-10-05 | rsna-knee-fork v11 | **C1 / P-50**: the public 0.942 stack + our #48 trio (`v11a` + `v13r` + `v13e`) as the leg at **β 0.45** (`src/build_fork.py`) | none (fork) | **0.943** | **read 06:01:18 UTC → +0.001 vs #13 0.942 → 🔁 (band 0.941–0.944); our best public number, rank 373 of 5,187 (the 0.942 plateau ≈ 1,000 teams); scored within [318.1, 319.6] min ≈ 5.3 h.** sent 00:41:45 UTC, ref 56838006, by the `auto_submit.py` fallback run (the 00:00:30 scheduled run sent nothing: traps 20 addendum) |
| 50 | 2026-10-05 | rsna-knee-infer v47 (+ Dataset `rsna-knee-ckpt-v13b3`) | **A1 / P-66, EfficientNet-B3 solo**: `INFER_MEMBERS = ["v13b3"]`, B3 ra2 @ 288 on the `v13h` recipe, trained on RunPod (`artifacts/infer_solo_v13b3.py`) | none; gold-58 0.9222 | **0.940** | **read 01:02:27 UTC → +0.005 vs `v13e` 0.935 → ✅ KEEP (bar ≥ 0.939); our best solo, above #48; scored within [18.5, 20.0] min.** sent 00:42:30 UTC, ref 56838023. Read vs `v13e` 0.935: ✅ ≥ 0.939 / 🔁 0.931–0.938 / ❌ ≤ 0.930; a member if ≥ 0.933 |
| 51 | 2026-10-05 | rsna-knee-infer v48 (+ Dataset `rsna-knee-ckpt-v13e2`) | **A2 / P-66, seed twin**: `INFER_MEMBERS = ["v13e2"]`, `v13e` at seed 43, RunPod (`artifacts/infer_solo_v13e2.py`) | none; gold-58 0.9151 | **0.938** | **read 00:57:11 UTC → s = 0.003 → ✅ measured, the CNN bands stand; = #48, our best solo; scored within [12.5, 14.0] min.** sent 00:43:16 UTC, ref 56838038. s = \|score − 0.935\|: s ≤ 0.003 the bands stand / s ≥ 0.005 one-seed CNN deltas need ≥ s |
| 52 | 2026-10-05 | rsna-knee-infer v51 | **B6, five-member blend**: flat rank-mean of `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` (`artifacts/infer_B6_five.py`) | none; gold-58 0.9256 | **0.942** | **read 01:34:03 UTC → +0.004 vs #48 → ✅ KEEP (bar ≥ 0.941); our best own score, = the public-stack fork (#13 / #15); +0.002 over its best member, +0.0062 over the members' mean; scored within [48.6, 50.1] min.** sent 00:44:02 UTC, ref 56838060. Read vs #48 0.938: ✅ ≥ 0.941 / 🔁 0.936–0.940 / ❌ ≤ 0.935 |
| 53 | 2026-10-05 | rsna-knee-infer v52 | **B3, the B3 swap**: flat rank-mean of `v11a` + `v13r` + `v13b3` (B3 replaces B0 in #48) (`artifacts/infer_B3_swap.py`) | none; gold-58 0.9254 | **0.940** | **read 01:31:48 UTC → +0.002 vs #48 → 🔁; = `v13b3` alone (a blend ≈ the members' mean + 0.004, entry "Submissions #49–#53"); scored within [45.6, 47.1] min.** sent 00:44:48 UTC, ref 56838082. Read vs #48 0.938: ✅ ≥ 0.941 / 🔁 0.936–0.940 / ❌ ≤ 0.935 |
| 54 | 2026-10-06 | rsna-knee-fork v12 | **C2 / P-50**: the public 0.942 stack + B6 (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`) as the leg at **β 0.45** (`src/build_fork.py`) | none (fork) | **0.944** | **read by 09:25 UTC → +0.001 vs #49 0.943 → 🔁 (band 0.942–0.945); our best public number, rank 337 of 5,293 (the public stack now reads 0.943 and a public ConvNeXt-T fork 0.944); scoring time not recorded (watcher died, traps 53).** sent 00:03:18 UTC, ref 56864525, by `auto_submit.py` (pid 23960) |
| 55 | 2026-10-06 | rsna-knee-infer v55 | **B13**: flat rank-mean of B6 + `v11n` + `v11n2`, three CoAtNet + four CNN votes (`artifacts/infer_B13.py`) | none; gold-58 0.9256 | **0.940** | **→ −0.002 vs B6 0.942 → 🔁 (band 0.939–0.945); the ≥ 0.943 CoAtNet branch does not fire; scoring time not recorded (watcher died).** sent 00:04:04 UTC, ref 56864597 |
| 56 | 2026-10-06 | rsna-knee-infer v56 | **B11**: flat rank-mean of `v13b3` + `v13e2` + `v13e`, the EfficientNet triple (`artifacts/infer_B11.py`) | none; gold-58 0.9196 | **0.941** | **read 00:29:21 UTC → −0.001 vs B6 → 🔁; the same-recipe gain (+0.0033 over the members' mean); scored within [23.0, 24.5] min.** sent 00:04:51 UTC, ref 56864648 |
| 57 | 2026-10-06 | rsna-knee-infer v53 | **A4 / P-65, `v13ecp` solo**: `v13e` (B0, seed 42) on 0.5 Raptor + 0.5 Claude, no LLM-blend share (`artifacts/infer_solo_v13ecp.py`) | none; gold-58 0.9063 | **0.935** | **read 00:21:06 UTC → −0.0015 vs the B0 seed mean 0.9365 → 🔁 (band 0.933–0.940); = `v13e` at the same seed; scored within [14.0, 15.5] min.** sent 00:05:38 UTC, ref 56864717 |
| 58 | 2026-10-06 | rsna-knee-infer v54 | **A5 / P-65, `v13ec` solo**: `v13e` on 0.25 LLM + 0.5 Raptor + 0.25 Claude (`claude_rap_v1` at mix 0.75) (`artifacts/infer_solo_v13ec.py`) | none; gold-58 0.9116 | **0.932** | **read 00:42:55 UTC → −0.0045 vs the B0 seed mean → ❌ (≤ 0.932); with #57, P-65 closes ❌; scored within [35.0, 36.5] min.** sent 00:06:24 UTC, ref 56864754 |

## Closed cards index (moved here from proposals.md on 2026-10-05)

proposals.md holds live cards only (Tian, 2026-10-05). One line per measured or retired card, pointing at its entry above; the full card text before each prune is in git (`git show 8304c96:docs/proposals.md`, `git show c2b0e32:docs/proposals.md`, `git show fd96082:docs/proposals.md`). New rows are appended when a live card is measured (`/update`).

| id | title | final verdict | where the result lives |
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
| P-65 | Grading-aware Claude (Opus) relabel of the reports as a training target (session E on RunPod, ≈ $1.9: `v13ecp` 0.5 Claude, `v13ec` 0.25 Claude; B0, seed 42) | ❌ DEAD END as a target lever: #57 `v13ecp` **0.935** (🔁) / #58 `v13ec` **0.932** (❌) vs the B0 seed mean 0.9365 — pair −0.003, not monotone in dose; gold-58 the same direction. The gold pilot (+0.007 with Raptor, target level) did not transfer, like D4 and `xfit_v09k`. `claude_v1` / `claude_rap_v1` stay private and unused | experiments.md 2026-10-04 "P-65 gold-58 BLIND pilot", "P-65 full pass"; 2026-10-05 "Session E on RunPod, chain 1", "chain 2"; 2026-10-06 "Submissions #54–#58" |
| P-46 | Upgrade the LLM half of the targets (absorbs P-16, P-30) | retired 2026-10-06 by its own rule: step 2 (the re-label) ran as P-65 ❌, and step 1 (dread as a 4th vote) was to close if the Claude vote did not move the LB — the LLM half is not binding | experiments.md 2026-10-06 "Submissions #54–#58"; the P-65 row above |
