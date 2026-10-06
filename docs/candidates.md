# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-06 (09:40 UTC). All five 10-06 reads are in (#54–#58, experiments.md "Submissions #54–#58"): **C2 0.944** (the new
fork pick), B13 0.940 and B11 0.941 (B6 stays the own pick), `v13ecp` 0.935 and `v13ec` 0.932 (P-65 closed ❌). Their branches: C3's
gate is closed, B12 gets no member from E, and the week-2 retrains add no CoAtNet seed and no Claude target. **Tian, 10-05 ≈ 12:00 UTC:
no training on 10-05 or 10-06; the next days only submit work we already have.** Training (section D) resumes at the 10-10 reset.
5 slots on every UTC day to the 10-22 deadline. **10-07 is set up (10:28 UTC 10-06):** placeholders B14 = infer v57, B4 = v58, B5 =
v59 green; `artifacts/submit_plan_1007.json`; `auto_submit.py` pid 25032 waits for 2026-10-07 00:00:30 UTC. **The laptop must stay
awake: on AC power, lid open** (it was on battery at 10:30; traps 53). **Tian, 10-06: focus on single models and a new family
member for the blend, not on labels** (section D, proposals.md).

## Priority — every candidate, ranked, with its day

**P1** is the next day's five: the 10-07 sends, the pre-registered fallbacks of the 10-06 reads (built and scheduled).
**P2** decides the own final pick or the fork leg, if its gate opens. **P3** is contingent or explains a P1 / P2 read. **P5** is held:
the blend rule prices it under B6, or a read closed it. "Build" means a `rsna-knee-infer` placeholder (≈ 5–10 GPU-min each; 2.84 h
of Kaggle GPU is left until 10-10, enough for ≈ 15).

| Prio | Day | Candidate | What it decides | Gate | Status |
|---|---|---|---|---|---|
| **P1** | 10-07 | **A3** `v13es` + `v13rs`: the P-62 pair, two sends | The silent mix for the week-2 retrains and the P-68 arms (the last open target question) | — | **v49** + **v50** ✅; queued (sends 2, 3) |
| **P1** | 10-07 | **B14**: B6 − `v11a` (the four CNNs) | Is the CoAtNet family needed at all? B13 (#55 0.940) already said more CoAtNet weight does not help | — | infer **v57** ✅; queued (send 5) |
| **P1** | 10-07 | **B4**: B6 − `v13e2` | With B5: does B3 or the second B0 seed carry B6? Decides B3 × 2 seeds (RunPod) vs B0 × 2 (Kaggle) in week 2. Takes C3's slot (its gate closed on 10-06) | — | infer **v58** ✅; queued (send 1) |
| **P1** | 10-07 | **B5**: B6 − `v13b3` | the second half of B4's read. Takes B12's slot (no E solo reached 0.936) | — | infer **v59** ✅; queued (send 4) |
| **P2** | 10-08 | **B16**: B6 + the public ConvNeXt-T reader (goodpjw2008, Apache-2.0) | A sixth family without training; the families, not the member count, carry the blend gain (10-06). Tian's 10-06 focus | the external-member hook (P-71) | needs code (P-71), then a placeholder |
| **P2** | 10-08 | **B12**: B6 + every A3 solo that reads ≥ 0.936 | The own final pick's successor (pick 1) | ≥ 1 of `v13es` / `v13rs` ≥ 0.936 (the E solos read 0.935 / 0.932, so neither joins) | build after the 10-07 solos |
| **P2** | 10-08 | **C3**: the public stack + the best own blend as the leg, β 0.45 | Fork pick 2 with a better leg | an own blend ≥ 0.943 (B12 or B16). **Not open after 10-06** (B13 0.940, B11 0.941) | build with `src/build_fork.py` only if gated in; send 1 of its day |
| **P3** | 10-08 | **B9**: `v11a` + `v13rs` + `v13es` | #48 with the silent-mix members | A3 ✅ only | build |
| P5 | hold | B1, B2, B7, B8, B10, B15 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6; **B8 and B15 closed by B13** (same-family members under the mean cost) | hold |
| — | 10-09 | **Freeze the shortlist for P-50** | pick 1 = B6 (or a B row ≥ 0.943), pick 2 = C2 (#54, fork v12) | — | Tian, by 10-22 |

**What the blend rule expects** (section B): every P1–P3 row predicts 0.939–0.943, inside B6's 🔁 band. These sends are worth their
slots for what they decide (the week-2 retrain composition, the silent mix), not as likely new bests. A slot left empty costs
nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Public-stack fork | **0.944** | **#54 = C2**: our anchor (0.942 alone, #13 / #15) + B6 at β 0.45, rank 337 of 5,293; 0.943 with the #48 trio (#49). The public community stack now reads 0.943 alone and a public ConvNeXt-T fork of it 0.944; its run-to-run spread is one tick (experiments.md 2026-10-06 "The public frontier moved") |
| Our best own | **0.942** | #52 = B6: flat rank-mean of `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`. B13 (+ two CoAtNets) 0.940, B11 (the EfficientNet triple) 0.941. Best solo #50 `v13b3` 0.940 |
| Solos | **0.940** / 0.938 / 0.935 / 0.935 / 0.934 / 0.932 / 0.932 / 0.932 / 0.932 / 0.931 | `v13b3` / `v13e2` / `v13e` / `v13ecp` / `v13r` / `v11a` / `v11n` / `v11n2` / `v13ec` / `v13h` |
| CNN seed spread | s = 0.003 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

A4 / A5 (the P-65 Claude solos) were read on 10-06: 0.935 / 0.932, P-65 closed ❌ (experiments.md "Submissions #54–#58"). A3 is the
last solo read before training resumes. Delete each row once its score is logged.

| # | Candidate | What it tests / contributes | Decides | Read rule | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| A3 | `v13es` + `v13rs` (P-62: Raptor 0.75 on report-silent cells) — **one read, two submissions** | Does weighting the image teacher on report-silent cells lift the CNNs? | Whether the final members train with `TEACHER_SILENT_MIX` | mean of the two vs 0.9345: ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299. On gold, both move less than a seed change, so 🔁 is likely | **v49** + **v50** ✅; 10-07 | 0.9107 / 0.9160 |

## B. Submission candidates — ensembles of members we have

Since 10-05 every row here is read against **#52 (B6) 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938**, unless the row says otherwise
(B6 is the bar to beat for our own final pick). Inside 🔁, the pick-1 tie rule is the one used since #10: an equal score with more
members wins. "Build" means a new `rsna-knee-infer` placeholder (recipe at the bottom).

**Blend rule (10-05, all 10 flat blends with solo-read members, experiments.md "Submissions #49–#53" + its CORRECTED note):**
a flat rank-mean reads ≈ its members' mean solo LB + a gain that depends on family diversity and count:
- same recipe (seeds, small variants): + 0.001–0.0033 (#26, #29, #32, #36);
- cross-family: + 0.0025–0.0047 at 2–3 members (#23, #38, #47, #48, #53), + 0.006 at 5 (#52).
B0 and B3 count as same-recipe (within-class ρ 0.888 = a seed pair). The target variants (`v13es`, `v13rs`, `v13ecp`, `v13ec`)
also sit at seed distance from their parents on gold (ρ 0.88–0.95). "Pred." below is that rule's range, LB rounded to 0.001. Member
solos: `v13b3` 0.940, `v13e2` 0.938, `v13e` / `v13ecp` 0.935, `v13r` 0.934, `v11a` / `v11n` / `v11n2` / `v13ec` 0.932, `v13h` 0.931.

**10-06 refinement (B13 #55, B11 #56):** the gain grows with the number of *families*, not of members. B11 (one family) gained
+0.0033 over its mean, the same-recipe value. B13 = B6 + two more CoAtNet-recipe votes under the mean gained +0.0053 (B6: +0.0062)
and lost 0.002 to B6. A member that is under the blend's mean *and* of a family already present costs; a new family is what pays.

| # | Prio | Members | What it tests / contributes | Read rule (beyond the header bands) | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| B12 | P2 | B6 + every A3 solo ≥ 0.936 (of `v13es`, `v13rs`; the E solos `v13ecp` 0.935 / `v13ec` 0.932 did not qualify) | The pick-1 successor. A qualifying member sits at or above B6's members' mean (0.9358). But both A3 arms are B6 families' variants at seed distance, so by the 10-06 refinement pred. ≈ B6 − 0.001 to + 0.001; more only if a solo reads ≥ 0.940 | ≥ 0.943 → the new pick 1 and the C3 leg | build after the 10-07 solos | 0.9221 (with all four) |
| B14 | P1 | B6 − `v11a` = `v13r` + `v13e` + `v13b3` + `v13e2` | The CoAtNet ablation of B6. Pred. ≈ 0.940–0.941 (mean 0.9368, two families) | ≥ 0.942 → the CoAtNet is not needed: week 2 drops the CoAtNet retrain. ≤ 0.940 → it stays. (B13 already read: no *second* CoAtNet seed) | infer **v57** ✅; 10-07 send 5 | 0.9211 |
| B4 | P1 | `v11a` + `v13r` + `v13e` + `v13b3` (B6 − `v13e2`) | Pred. ≈ 0.939–0.941 (mean 0.9353, n 4). With B5: does B3 or the second B0 seed carry B6? | B4 − B5 ≥ 0.002 → B3 carries it: week 2 trains B3 × 2 (RunPod). ≤ −0.002 → B0 seeds suffice | infer **v58** ✅; 10-07 send 1 | 0.9250 |
| B5 | P1 | `v11a` + `v13r` + `v13e` + `v13e2` (B6 − `v13b3`) | Pred. ≈ 0.939–0.940 (mean 0.9348, n 4) | with B4 | infer **v59** ✅; 10-07 send 4 | 0.9238 |
| B9 | P3 | `v11a` + `v13rs` + `v13es` | #48 with the P-62 members swapped in | A3 ✅ only; vs #48 0.938: ≥ 0.941 → the silent mix also helps inside a blend | build | 0.9224 |
| B16 | P2 | B6 + the public 2.5D ConvNeXt-T reader (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, 3 checkpoints, 0.929 solo) | A sixth family at no training cost; shares neither our input (c03) nor our targets (Raptor). Pred. ≈ 0.941–0.943 (mean of six 0.9347 + a cross-family gain; more if its independence is worth more than our families' ρ ≈ 0.83–0.89) | ≥ 0.943 → a final-pick member and the C3 leg; also a reason to drop P-69's own training arm | **needs code (P-71)**: run the reader's `infer.py` on `test_series` in a subprocess and rank it in as one member; ≈ 20 min extra scoring; licence Apache-2.0 (read 10-06) | — (no gold predictions) |
| B15 | P5 | B12 + `v11n` + `v11n2` | B12 and B13 combined | **closed 10-06**: B13 read 0.940 (< 0.943) | — | — |
| B8 | P5 | every member with a solo ≥ 0.930: B6 + `v11n` + `v11n2` + `v13h` + `v11d`, plus the new solos ≥ 0.930 | Does "everything" beat a curated 3–5? (Tian prefers 3–5) | **held 10-06**: B13 (B6 + two same-family members) lost 0.002, and B8 adds four such members | — | 0.9248 (the nine without the 10-06 arms) |
| B1 | P5 | `v11a` + `v13h` + `v13r` + `v13e` | A fourth member (ResNet-34) on the trio. Pred. ≈ 0.937–0.939: under B6 | hold | **v45** ✅ | 0.9186 |
| B2 | P5 | `v13h` + `v13r` + `v13e` (CNN trio) | Is the CoAtNet needed? **Superseded by B14** (the same question on B6's own members) | hold | **v46** ✅ | 0.9129 |
| B7 | P5 | `v11a` + `v13b3` | The two best families as a pair. Pred. ≈ 0.938–0.940 | hold | build | 0.9267 |
| B10 | P5 | `v13b3` + `v13e2` | The strongest pair. Pred. ≈ 0.940–0.942 (same-recipe gain); B11 (+ `v13e`) read 0.941 | hold | build | — |

## C. Submission candidates — the public stack plus our models (P-50)

**C2 was read on 10-06: #54 = 0.944** (🔁 +0.001 vs #49; rank 337), the fork pick (P-50 pick 2). The public community stack itself
now reads 0.943, a public fork of it with a 0.929 ConvNeXt-T leg 0.944, and that author saw the stack score 0.944 / 0.944 / 0.943 on
one pipeline: fork deltas of one tick are noise (experiments.md 2026-10-06 "The public frontier moved").

| # | Prio | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|---|
| C3 | P2 | Public stack + **the best own blend after 10-07** (B12 or B16) as our leg at β 0.45 | Whether a better own leg lifts the fork further | Gate: the leg read ≥ 0.943 (C2 ≥ 0.942 is met). **Not open after 10-06** (B13 0.940, B11 0.941). vs C2 0.944: ✅ ≥ 0.947 / 🔁 0.942–0.946 / ❌ ≤ 0.941 | build with `src/build_fork.py --members …` only if gated in; send 1 of its day |
| — | — | Re-anchoring on the 0.943 public stack | **Not planned:** a dropped direction (proposals.md), re-confirmed 10-06 — at most the 0.001 between the anchors, a `build_fork.py` rewrite, 6.5–8 h scoring | — | — |
| — | — | Fork at other β | **Not planned:** it tunes a weight to the public LB | — | — |

## Order

- **10-06 (sent 00:03–00:06 UTC, all read):** C2 0.944 → B13 0.940 → B11 0.941 → A4 0.935 → A5 0.932 (experiments.md "Submissions
  #54–#58"). C3's gate stayed closed and no E solo qualified for B12, so both pre-registered fallbacks fire.
- **10-07 (set up 10-06, 10:28 UTC):** B4 (v58) → A3a (v49) → A3b (v50) → B5 (v59) → B14 (v57), `artifacts/submit_plan_1007.json`,
  `auto_submit.py` pid 25032, log `artifacts/auto_submit_1007.log`. **Lid open, on AC power (traps 53).** No fork on 10-07, so every
  read lands within ≈ 1 h; read the scores from `kaggle competitions submissions`, not from the summary.
- **10-07, after the reads:** B16 (P-71, Tian's 10-06 focus; if its code is done), B12 (if an A3 solo ≥ 0.936), B9 (if A3 ✅),
  C3 (only if a leg reads ≥ 0.943) for `submit_plan_1008.json`.
- **10-09:** only rows that still decide something; otherwise leave the slots empty. Freeze the P-50 shortlist.
- **10-10:** training resumes (section D). The submissions after that are the new arms' solos.

## D. Training candidates (GPU) — paused until the 10-10 reset

**Paused (Tian, 2026-10-05 ≈ 12:00 UTC):** no training on 10-05 or 10-06; the days until 10-10 only submit. Kaggle GPU minutes go to
placeholders only.

**What the 10-06 reads settled for week 2:** the targets stay 0.5 LLM + 0.5 Raptor (P-65 ❌: no Claude share), ± the P-62 silent
weight (A3, 10-07); no second CoAtNet seed (B13); B14 / B4 / B5 (10-07) settle the CoAtNet retrain and B3 × 2 vs B0 × 2. A new
family (P-69, or the public reader as B16) is the predicted lever.

**Budget (2026-10-06, 10:40 UTC).** Kaggle: **2.84 h** left until 10-10 (`kaggle quota`), then 30 h on 10-10 and 30 h on 10-17; the quota
counts *session* hours and every session has two T4s (`PARALLEL_ARMS`), so ≈ 60 T4-GPU-h per week. RunPod: **≈ $5** ≈ 6.5 h on a 4090
($0.74/h); every pod needs a written case checked by a critic subagent, then Tian's go. Measured speeds: a 30-ep CNN arm ≈ 5.9 h on a
T4 (B0 / R50 / R34), B0 ≈ 71 min and B3 @ 288 ≈ 2.2 h on a 4090; B3 @ 288 is RunPod-only (12–15 h and a memory risk on a T4).
Deadline 10-22; final picks can be changed until then (10-15 is the entry / merger deadline).

**Focus (Tian, 2026-10-06): single-model strength and new family members for the blend; labels de-emphasised.** This replaces the
10-05 order (the P-68 teacher first). Label-side arms (P-68) run only on an otherwise idle GPU slot. **RunPod on 10-07 was priced and
rejected** by the critic (≈ $3.8, no decision unlocked before the 10-17 retrains); everything runs on Kaggle.

**Before 10-10 (no training; placeholders and smokes only, 2.84 GPU-h left):**
- P-71 / B16: the external-member hook for the public ConvNeXt-T reader, a placeholder, a 10-08 send.
- P-69 / P-72: the weight Datasets (`timm-convnext-tiny`, `timm-eca-nfnet-l0`), CPU loader checks, one PARALLEL Kaggle smoke of both
  arms (≈ 0.1 h), so session B starts at 00:00 on 10-10 without a code change.
- P-74 (a): `channels_last` on the CNN encoders, checked in the same smoke (s/study, equal outputs).
- P-73: the token-mixer head as `v14tx`, unit-checked, ready for the proxy loop.

**The 10-10 plan (revised 2026-10-06; two Kaggle sessions at a time, two T4s each; ≈ 30 session-h this week):**
1. **00:00, session A:** the P-67 floor pair `v14p` ‖ `v14p2` (≈ 3.75 h) → the ruler's floor (from the per-fold paired
   differences), the B0 proxy OOF that P-72's Δ_div needs, and the `cnnoof_v1` table.
2. **00:00, session B: two new families as production arms:** P-69 ConvNeXt-T ‖ P-72 `eca_nfnet_l0` (fallback `regnety_040`),
   ≈ 6 h. Each solo is read on 10-10 / 10-11 (a member if ≥ 0.933), then B6 + the qualifying ones.
3. **≈ 04:00, session C (after A):** the first two single-model variables, `v14lr` (lowres) ‖ `v14gd` (grid).
4. **Then, two proxy variants per session in the evidence order:** `v14mx`, `v14r288`, `v14tx` (P-73), `v14ep20`, `v14prog`
   (P-74 b), `v14bl` / `v14ns` / `v14sh`, `v14th`, `v14db`. A variant that clears 1.5 × the floor gets ONE production transfer arm
   before the week-2 retrains.
5. **Label-side, idle slots only:** `v13ex` (P-68). A silent mix only if A3 reads ✅ on 10-07.

| # | Arm(s) | What it tests / adds | Gate | Est. cost | Card |
|---|---|---|---|---|---|
| **T8** | **P-69** ConvNeXt-T on the `v13h` recipe | a sixth family for B6 (+ 0.002–0.003 by the blend rule) | 10-10 quota; weight Dataset + loader check + smoke before | 1 arm ≈ 6 h (session B) | P-69 |
| **T10** | **P-72** a seventh family: `eca_nfnet_l0` (fallback `regnety_040`) on the `v13h` recipe | a family that is neither ResNet, EfficientNet, CoAtNet nor ConvNeXt; a member if ≥ 0.933 | as T8 | 1 arm ≈ 6 h (session B) | P-72 |
| **T7** | **P-67 loop**: proxy baseline × 2 seeds (the floor), then one variable per 5-fold run: `v14lr` / `v14th` / `v14gd` / `v14bl` / `v14ns` / `v14sh` (augmentation components), `v14mx` (mixup), `v14r288` (B0 @ 288), `v14tx` (P-73 token mixer), `v14prog` (P-74 progressive resize), `v14db` (blank windows), `v14ep20` (longer schedule); later drop-path / EMA, slot layout | the single-model recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method. The epoch-budget variant answers the epoch-selection question too | the 10-10 reset | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67, P-73, P-74 |
| T9 | **P-68** (demoted 10-06, label-side): `v13ex` (B0 on Raptor + `xfit_v09k`); `v11o` / `v13eo` paused | the forum's multi-source pseudo-label gain, cross-family form | an otherwise idle GPU slot | ≈ 6 T4-h per arm | P-68 |
| T6 | Final members (week of 10-17): B3 × 2 seeds (RunPod, or Kaggle if P-74 b works), B0 × 2, R50, the T8 / T10 families that read ≥ 0.933, one CoAtNet unless B14 drops it (B13: no second CoAtNet seed) — all on the winning stack and target (0.5 LLM + 0.5 Raptor ± the silent weight; no Claude, P-65 ❌) | the two final picks (B6-successor; the fork leg) | T7 / T8 / T10 reads; A3; B14 / B4 / B5 | ≈ 30 Kaggle session-h + ≈ $4 RunPod for the B3s | P-50 |
| T2 | B3 at seed 43 **now** | de-biases the single 0.940 draw; the critic's case is written | **paused** with the rest of D; folded into T6 (the final B3 retrain on the winning recipe *is* the second draw) | ≈ $1.9 | P-50 |
| T3 / T4 | R50 seed 2; B3 on the winning target | — | = T6 | — | P-50 |
| ~~T5~~ | a bigger CNN (B4 @ 288 / 336) | **dropped 10-05:** the forum's ≥ 0.949 singles are R50-class at 224–288, "bigger is null", no gain above 288; the critic priced B4 @ 336 at ≈ 25 GB VRAM and ≈ $3.7 | — | — | → P-69 (a new family instead) |

**Dropped directions (research.md 2.10, item 5):** resolution > 288, B4-class capacity, 3D, MIL / bags, DINOv3 / RadImageNet / medical
foundation backbones, fork β or blend-weight tuning, geometric TTA, co-teaching, external datasets, multimodal-LLM image labelling.

## How to build and send a candidate

**Ensemble or solo placeholder** (one at a time; it needs a free GPU slot, traps 50). Add any new member's
`tiankljucanin/rsna-knee-ckpt-<arm>` to `kaggle/rsna-knee-infer/kernel-metadata.json` first (all members above are mounted as of v54):
```bash
sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "infer"/' \
    -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = ["v11a", "v13r", "v13b3"]/' src/kaggle_pipeline.py > artifacts/infer_<name>.py
python src/nbgen.py artifacts/infer_<name>.py kaggle/rsna-knee-infer/rsna-knee-infer.ipynb
kaggle kernels push -p kaggle/rsna-knee-infer
```

**Green check** (the log is JSON):
- run `python src/kaggle_log.py <log> '"smoke"' "infer members" "decode-once verified" "constant labels"`;
- green = `"smoke": "False"`, the right members, `decode-once verified`, `constant labels 0`.

**Send:**
- one at a time: `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <N> -f submission.csv -m "<what + its read rule>"`,
  then `python src/watch_submission.py --ref <ref> --every 90`;
- a whole day: a plan file `artifacts/submit_plan_<mmdd>.json` (a list of `{slug, version, message}`) and
  `src/auto_submit.py --plan <file> --at <day>T00:00:30Z`, started detached (the command is in handoff.md, entry 2026-10-05 00:40 → 09:20, next action 4). One submitter per UTC day.
- After the read: `/update` (experiments.md row, card status), and delete the row here.
