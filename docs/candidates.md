# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-08 (09:45 UTC). **The 10-08 three are read** (experiments.md "Submissions #64–#66"; sent 06:16–06:18 UTC by the
fallback after the armed submitter died with the laptop): **A6 `v15c` (ConvNeXt-T) 0.942 = our best solo, level with B6**; **B17 = B6 +
`v15c` 0.943 = our best own blend** (🔁 +0.001; pick 1 by convention, C3's gate opens, week 2 trains more ConvNeXt: P-80); **B12 0.941**
(closed). B17 is only +0.001 over `v15c` alone, so the next own-blend question is a *small* blend of the strongest members (B18). **The
public plateau jumped to 0.950 on 10-08** (338 teams ≥ 0.950; we are rank 720 at 0.944), which reprices C3 and re-opens the fork's
anchor (brainstorm.md). Two 10-08 slots are still free (until 00:00 UTC). Earlier, the 10-07 reads (#59–#63, experiments.md "Submissions #59–#63"): **B14 0.942 = B6**
(the CoAtNet is not needed: week 2 drops its retrain), **B4 0.941 / B5 0.940** (B4 − B5 = +0.001: neither week-2 branch fires, and
P-69 runs one arm), **`v13es` 0.936 / `v13rs` 0.937** (P-62 closed 🔁, not adopted; both qualify for B12). The 10-06 reads (#54–#58)
made C2 the fork pick (0.944), kept B6 as the own pick and closed P-65. 5 slots on every UTC day to the 10-22 deadline; an unattended
send needs the laptop on AC power with the lid open (traps 53). **Tian, 10-06: focus on single models and a new family member for the
blend, not on labels** (section D, proposals.md). **Tian, 10-06 night: ONE ConvNeXt-T arm with its own recipe (P-69 `v15c`), on
RunPod after the 10-07 sends (cap $2.5), read as A6 (solo) and B17 (B6 + it)** — the plan is
`docs/superpowers/plans/2026-10-06-p69-convnext-plan.md`. Kaggle training (section D) resumes at the 10-10 reset.

## Priority — every candidate, ranked, with its day

**P1** is the next sends. **P2** decides the own final pick or the fork leg. **P3** is contingent or explains a P1 / P2 read. **P5** is
held: the blend rule prices it under the current pick, or a read closed it. "Build" means a `rsna-knee-infer` placeholder (≈ 5–10
GPU-min each; 2.60 h of Kaggle GPU is left until 10-10).

| Prio | Day | Candidate | What it decides | Gate | Status |
|---|---|---|---|---|---|
| **P1** | 10-08 | **B18**: `v15c` + `v13b3`, the two strongest members, two families | Does a small blend of the strongest beat the wide flat one, now that the top member is a new family? Pred. 0.943–0.946 | — (both shipped and mounted) | infer **v63** ✅ (green 10:09 UTC); **sent #67 10:09 UTC 10-08** (ref 56948421, Tian's go), ⏳ scoring |
| **P1** | 10-08 | **C4**: the public 0.949 checkpoint + B17 as our leg at β 0.35 (section C) | What our models add on top of the best public single model | **Tian's go 10-08: build it and accept the OAI risk to read it; whether it is ever a final pick is his separate call** | `rsna-knee-fork949` **v1** pushed (`src/build_fork949.py`), placeholder ⏳; send = the day's 5th slot |
| **P2** | 10-09 | **C3**: the 0.942 public stack + **B17** (or B18 if it reads higher) as the leg, β 0.45 | Fork pick 2 with a better leg | **open since 10-08** (B17 0.943). **But** the public plateau is now 0.950 (10-08): C3 predicts ≈ 0.944–0.945 (C2 0.944 + B17's +0.001 over B6), 0.005 under it. Tian decides between C3 and re-anchoring first (brainstorm.md) | build with `src/build_fork.py --member v15c=rsna-knee-ckpt-v15c:timm-convnext-tiny-in12k` (B17's members); send 1 of its day |
| **P2** | after 10-10 or on RunPod | **`v15c2`** (P-80): the ConvNeXt seed twin, then `v15c` + `v15c2` as one vote | The ConvNeXt family's seed spread; the final ConvNeXt vote | Tian's go (where it trains) | arm built and smoke-green (v47); `INFER_VOTE_GROUPS` not written |
| P5 | hold | **B16** (P-71) | — | **recommended DROP 10-08:** `v15c` 0.942 makes the 0.929 public reader a second, weaker ConvNeXt | hold |
| P5 | hold | B1, B2, B7, B8, B9, B10, B12, B15 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6; **B8 and B15 closed by B13**; **B9 closed 10-07**; **B12 read 0.941 on 10-08 (closed)** | hold |
| — | 10-09 | **Freeze the shortlist for P-50** | pick 1 = **B17 (#65, 0.943)** (or B18 if it reads higher), pick 2 = C2 (#54, fork v12) or C3, or a re-anchored fork | — | Tian, by 10-22 |

**What the blend rule expects** (section B): B18 0.943–0.946 (a cross-family pair: members' mean 0.941 + 0.0025–0.0047). It is the
only own row with a real chance above B17. A slot left empty costs nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Public-stack fork | **0.944** | **#54 = C2**: our anchor (0.942 alone, #13 / #15) + B6 at β 0.45, rank 337 of 5,293; 0.943 with the #48 trio (#49). The public community stack now reads 0.943 alone and a public ConvNeXt-T fork of it 0.944; its run-to-run spread is one tick (experiments.md 2026-10-06 "The public frontier moved") |
| Our best own | **0.943** | **#65 = B17: flat rank-mean of B6 + `v15c`** (10-08). B6 #52 = `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` 0.942; B12 (B6 + `v13es` + `v13rs`) 0.941; B13 (+ two CoAtNets) 0.940, B11 (the EfficientNet triple) 0.941; drop-one (10-07): B14 (no CoAtNet) 0.942, B4 (no `v13e2`) 0.941, B5 (no `v13b3`) 0.940 |
| Solos | **0.942** / 0.940 / 0.938 / 0.937 / 0.936 / 0.935 / 0.935 / 0.934 / 0.932 / 0.932 / 0.932 / 0.932 / 0.931 | **`v15c` (ConvNeXt-T, #64)** / `v13b3` / `v13e2` / `v13rs` / `v13es` / `v13e` / `v13ecp` / `v13r` / `v11a` / `v11n` / `v11n2` / `v13ec` / `v13h` |
| CNN seed spread | s = 0.003 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

A6 (`v15c`, P-69) was read on 10-08: **0.942**, our best solo (experiments.md "Submissions #64–#66"). A3 (the P-62 pair) was read on
10-07, A4 / A5 (P-65) on 10-06. Delete each row once its score is logged.

| # | Candidate | What it tests / contributes | Decides | Read rule | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| A7 | `v15c2` (P-80): `v15c` at seed 43 | the ConvNeXt seed spread s_c | whether #64's 0.942 is the family's level; the final ConvNeXt vote | s_c = \|`v15c2` − 0.942\|: ≤ 0.003 the bands stand / ≥ 0.005 #64 was a draw | **not trained** (Tian's go: RunPod ≈ $1.46 or Kaggle after 10-10) | — |

## B. Submission candidates — ensembles of members we have

**Since 10-08 every row here is read against #65 (B17) 0.943: ✅ ≥ 0.947 / 🔁 0.940–0.946 / ❌ ≤ 0.939**, unless the row says otherwise
(B17 is the bar to beat for our own final pick; from 10-05 to 10-07 it was B6 0.942 with the bands one tick lower). Inside 🔁, the pick-1 tie rule is the one used since #10: an equal score with more
members wins. "Build" means a new `rsna-knee-infer` placeholder (recipe at the bottom).

**Blend rule (10-05, all 10 flat blends with solo-read members, experiments.md "Submissions #49–#53" + its CORRECTED note):**
a flat rank-mean reads ≈ its members' mean solo LB + a gain that depends on family diversity and count:
- same recipe (seeds, small variants): + 0.001–0.0033 (#26, #29, #32, #36);
- cross-family: + 0.0025–0.0047 at 2–3 members (#23, #38, #47, #48, #53), + 0.006 at 5 (#52).
B0 and B3 count as same-recipe (within-class ρ 0.888 = a seed pair). The target variants (`v13es`, `v13rs`, `v13ecp`, `v13ec`)
also sit at seed distance from their parents on gold (ρ 0.88–0.95). "Pred." below is that rule's range, LB rounded to 0.001. Member
solos: `v13b3` 0.940, `v13e2` 0.938, `v13rs` 0.937, `v13es` 0.936, `v13e` / `v13ecp` 0.935, `v13r` 0.934, `v11a` / `v11n` / `v11n2` / `v13ec` 0.932, `v13h` 0.931.

**10-06 refinement (B13 #55, B11 #56):** the gain grows with the number of *families*, not of members. B11 (one family) gained
+0.0033 over its mean, the same-recipe value. B13 = B6 + two more CoAtNet-recipe votes under the mean gained +0.0053 (B6: +0.0062)
and lost 0.002 to B6. A member that is under the blend's mean *and* of a family already present costs; a new family is what pays.

**10-07 refinement (B14 #63, B4 #59, B5 #62):** B6 without its CoAtNet reads 0.942 = B6. The two-family B14 gained +0.0053 over its
mean, as much as the three-family B5: a family member ≈ 0.004 under the others' mean adds no diversity gain. The drop-one costs
follow the members' solos (the B3 −0.002, the B0 seed twin −0.001, the CoAtNet 0.000).

**10-08 refinement (B17 #65, B12 #66, A6 #64):** the rule held a fourth time. B17 (four families) gained +0.0062 over its mean, the
same as B6, so the fourth family moved it by its mean shift only (+0.001). B12 (two more votes of present families) gained +0.0050
and lost 0.001. **New:** the best member now reads 0.942 alone, and B17 is only +0.001 over it: a flat six-vote mean gives the
strongest model 1/6 of the weight. The own-blend question becomes "which few strong members", not "how many families".

| # | Prio | Members | What it tests / contributes | Read rule (beyond the header bands) | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| B18 | P1 (sent #67) | `v15c` + `v13b3` (ConvNeXt-T 0.942 + EfficientNet-B3 0.940; two families, two votes) | Strength over width, now with a cross-family pair at the top (B11 tested it inside one family and lost 0.001). Pred. 0.943–0.946 (mean 0.941 + the cross-family pair gain 0.0025–0.0047) | vs B17 0.943: ≥ 0.944 → pick 1 and the C3 / fork leg; ≤ 0.942 → width wins, B17 stays | infer **v63** ✅, sent #67 | — |
| B12 | — | B6 + `v13es` + `v13rs` | **read 10-08: 0.941 (#66), closed** | — | infer v60 | 0.9242 |
| B16 | P5 (recommended DROP 10-08) | B6 + the public 2.5D ConvNeXt-T reader (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, 3 checkpoints, 0.929 solo) | A sixth family at no training cost; shares neither our input (c03) nor our targets (Raptor). Pred. ≈ 0.941–0.943 (mean of six 0.9347 + a cross-family gain; more if its independence is worth more than our families' ρ ≈ 0.83–0.89). **If `v15c` becomes a member, B16 adds a second ConvNeXt and its value falls further** | ≥ 0.943 → a final-pick member and the C3 leg | **needs code (P-71)**: run the reader's `infer.py` on `test_series` in a subprocess and rank it in as one member; ≈ 20 min extra scoring; licence Apache-2.0 (read 10-06) | — (no gold predictions) |
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
| C3 | P2 | Public 0.942 stack + **B17** (or B18 if it reads higher) as our leg at β 0.45 | Whether a better own leg lifts the fork further | Gate: the leg read ≥ 0.943 — **open since 10-08 (B17 0.943)**. vs C2 0.944: ✅ ≥ 0.947 / 🔁 0.942–0.946 / ❌ ≤ 0.941. Pred. 0.944–0.945, under the new 0.950 public plateau | build with `src/build_fork.py --members …` plus `--member v15c=rsna-knee-ckpt-v15c:timm-convnext-tiny-in12k`; send 1 of its day; **Tian picks C3 vs re-anchoring first** |
| C4 | **P1 (Tian's go 10-08)** | **The public 0.949 checkpoint** (`nartaa`, one CoAtNet-2, read 10-08: experiments.md "The 0.949 / 0.950 public notebooks, read") **+ B17 (or B18) as our leg** | A fork on a 0.949 anchor instead of 0.942: likely ≈ 0.949–0.952 (C2 read +0.002 over both its parts; the 0.950 author got +0.001 from a 0.929 leg) | Gate: Tian accepts the OAI rule risk (the checkpoint was trained on 2,399 OAI knees; the host allows OAI only if generally accessible, a participant in China reports it is not, no ruling since). vs the anchor alone (0.949): ✅ ≥ 0.952 / 🔁 0.947–0.951 / ❌ ≤ 0.946 | **built 10-08:** `src/build_fork949.py` (the anchor's five cells byte-identical + a clock cell + payload + arm; β 0.35 a priori because the leg reads 0.006 under the anchor, C2's 0.45 was for an equal leg; diagnostic CSVs at β 0.20 / 0.50, unscored); local dry runs green (blend path and fail-soft path); `rsna-knee-fork949` v1 placeholder ⏳ |
| — | — | Fork at other β | **Not planned:** it tunes a weight to the public LB | — | — |

## Order

- **10-06 (sent 00:03–00:06 UTC, all read):** C2 0.944 → B13 0.940 → B11 0.941 → A4 0.935 → A5 0.932 (experiments.md "Submissions
  #54–#58"). C3's gate stayed closed and no E solo qualified for B12, so both pre-registered fallbacks fire.
- **10-07 (sent 00:00–00:04 UTC, all read by 01:02):** B4 0.941 → A3a 0.936 → A3b 0.937 → B5 0.940 → B14 0.942 (experiments.md
  "Submissions #59–#63"). P-62 closed 🔁, B12's gate met, B9 closed, the CoAtNet retrain dropped, P-69 = one arm.
- **10-07 06:26–08:24 UTC:** the P-69 RunPod run (`v15c`, one arm), 1.97 pod-h ≈ $1.46, shipped and backed up, pod deleted
  (experiments.md 2026-10-07 "P-69 on RunPod"). Placeholders B12 = v60, A6 = v61, B17 = v62, all green.
- **10-08 (sent 06:16–06:18 UTC by the fallback; the armed pid 25568 died with the laptop on 10-07, traps 53):** A6 **0.942** →
  B17 **0.943** → B12 **0.941** (experiments.md "Submissions #64–#66"). P-69 closed ✅, B17 = pick 1, C3's gate open, B12 closed.
  Two slots are still free until 00:00 UTC: B18 (P1) can take one if Tian approves its placeholder and send today.
- **10-09:** C3 or a re-anchored fork (Tian's call, brainstorm.md), sent first in its day; B18 if not sent on 10-08. Freeze the P-50
  shortlist.
- **10-10:** training resumes (section D). The submissions after that are the new arms' solos.

## D. Training candidates (GPU) — paused until the 10-10 reset

**Paused (Tian, 2026-10-05 ≈ 12:00 UTC):** no training on 10-05 or 10-06; the days until 10-10 only submit. Kaggle GPU minutes go to
placeholders only.

**What the 10-06 and 10-07 reads settled for week 2:** the targets stay 0.5 LLM + 0.5 Raptor (P-65 ❌: no Claude share; P-62 🔁: no
silent weight), unless the P-68 pair below reads ✅; no CoAtNet retrain at all (B14 #63 = B6 without it; B13: no second seed);
B3 × 2 vs B0 × 2 stays open (B4 − B5 = +0.001, inside ± 0.002; the direction favours the B3). **10-08: ConvNeXt joins the finals**
(B17 ≥ 0.943 fired P-69's week-2 rule): the seed twin `v15c2` first (P-80, T8), and the ConvNeXt pair is one vote in both picks.

**Budget (2026-10-06, 10:40 UTC).** Kaggle: **2.84 h** left until 10-10 (`kaggle quota`), then 30 h on 10-10 and 30 h on 10-17; the quota
counts *session* hours and every session has two T4s (`PARALLEL_ARMS`), so ≈ 60 T4-GPU-h per week. RunPod: **≈ $5** ≈ 6.5 h on a 4090
($0.74/h); every pod needs a written case checked by a critic subagent, then Tian's go. Measured speeds: a 30-ep CNN arm ≈ 5.9 h on a
T4 (B0 / R50 / R34), B0 ≈ 71 min and B3 @ 288 ≈ 2.2 h on a 4090; B3 @ 288 is RunPod-only (12–15 h and a memory risk on a T4).
Deadline 10-22; final picks can be changed until then (10-15 is the entry / merger deadline).

**Tian's decisions, 2026-10-06 night (they supersede the evening's "no ConvNeXt training"):**
- **One ConvNeXt-T arm with its own recipe** (P-69 `v15c`, research.md 2.7.9): RunPod, one 4090, cap $2.5, 288 px; a second seed only
  if B4 − B5 ≤ −0.002 on 10-07 and both fit the cap (it read +0.001: one arm). Approving the plan was Tian's go for that one pod (critic: GO-WITH-CHANGES).
- **Single-model research → cards only** (P-75 … P-79); Tian picks later.

**Tian's decisions, 2026-10-06 evening** (after research.md 2.7.8: a fourth family ≈ +0.0008 LB on B6; no forum read shows ConvNeXt
helping beyond one tick; the forum's biggest single-model jumps came from different-source pseudo-labels):
- **Run the pseudo-label pair:** P-68 `v13ex` ‖ `v13ex2`, one session at the 10-10 reset.
- ~~No ConvNeXt training~~ (superseded by the night's decision above).
- **Everything else is decided later:** NFNet (P-72, recommended drop), B16 (P-71, recommended demote), P-73, P-74. None of them is
  scheduled.
- The P-67 loop stands as approved on 10-05.

**RunPod on 10-07 for P-68 was priced and rejected** by the critic (≈ $3.8, no decision unlocked before the 10-17 retrains); the P-68
pair runs on Kaggle. P-69's RunPod run happened on 10-07; the next candidate for a pod is P-80's `v15c2` (not approved yet).

**Before 10-10 (no Kaggle training; placeholders and smokes only; 2.75 GPU-h left at 15:10 UTC 10-06):**
- ✅ **`v13ex2` done (10-06):** in SHIPPED_ARMS and DISTILLED_ARMS, unit check, local smoke, and the pair's Kaggle smoke
  `rsna-knee-train-b` v8 green (both tables read, `reseeded 43`, SWA `_best.pt`). Session B at 10-10 is ready.
- ✅ **P-69 smoke done (10-06):** `rsna-knee-train` v47 green; the RunPod run follows the 10-07 sends.
- ✅ **Placeholders done (10-07):** B12 = infer v60, A6 = v61, B17 = v62 (≈ 0.2 GPU-h together); 2.60 h of Kaggle GPU left until 10-10.
- Pending Tian, optional: P-73 (`v14tx`), P-74 (a) `channels_last`, the P-71 hook for B16.

**The 10-10 plan (revised 2026-10-06 evening; two Kaggle sessions at a time, two T4s each; ≈ 30 session-h this week):**
1. **00:00, session A** (`rsna-knee-train`): the P-67 floor pair `v14p` ‖ `v14p2` (≈ 3.75 h, Raptor only). It gives the ruler's floor
   (from the per-fold paired differences), the B0 proxy OOF, and the `cnnoof_v1` table.
2. **00:00, session B** (`rsna-knee-train-b`): **the P-68 pseudo-label pair `v13ex` ‖ `v13ex2`** (≈ 6 h; tables Raptor + `xfit_v09k`,
   so it cannot share a session with a Raptor-only arm). Read the two solos on 10-10 / 10-11: mean vs the B0 seed mean 0.9365, ✅ ≥ 0.9410 /
   🔁 0.9320–0.9409 / ❌ ≤ 0.9319. If ✅, the final members train on the mix and `v11o` (the CoAtNet student on `cnnoof_v1`) is next.
3. **≈ 04:00, session C (after A):** the first two single-model variables, `v14lr` (lowres) ‖ `v14gd` (grid).
4. **Then, two proxy variants per session in the evidence order:** `v14mx`, `v14r288`, `v14ep20`, `v14bl` / `v14ns` / `v14sh`, `v14th`,
   `v14db`; plus `v14tx` / `v14prog` if Tian approves P-73 / P-74. A variant that clears 1.5 × the floor gets ONE production
   transfer arm before the week-2 retrains.
5. ~~The silent mix~~: not adopted (A3 read 🔁 on 10-07, P-62 closed).

| # | Arm(s) | What it tests / adds | Gate | Est. cost | Card |
|---|---|---|---|---|---|
| **T9** | **P-68** pseudo-label pair `v13ex` ‖ `v13ex2` (B0, seeds 42 / 43, on 0.5 LLM + 0.25 Raptor + 0.25 CoAtNet cross-fit OOF `xfit_v09k`); `v11o` / `v13eo` after the floor run, only if the pair reads ✅ | the forum's largest single-model lever (different-source pseudo-labels, Raymond / SpeedSci +0.011), the one target change we have not tested | **approved by Tian 10-06 evening**; `v13ex2` added and the pair smoke-green (`rsna-knee-train-b` v8) | ≈ 6 h, one session | P-68 |
| **T7** | **P-67 loop**: proxy baseline × 2 seeds (the floor), then one variable per 5-fold run: `v14lr` / `v14th` / `v14gd` / `v14bl` / `v14ns` / `v14sh` (augmentation components), `v14mx` (mixup), `v14r288` (B0 @ 288), `v14db` (blank windows), `v14ep20` (longer schedule); `v14tx` / `v14prog` if P-73 / P-74 are approved; later drop-path / EMA, slot layout | the single-model recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method. The epoch-budget variant answers the epoch-selection question too | the 10-10 reset | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67 |
| **T8** | **P-80 ConvNeXt seed twin `v15c2`** (= `v15c` at seed 43; arm built, smoke-green v47), then `v15c` + `v15c2` as one vote (`INFER_VOTE_GROUPS`, not written) | the ConvNeXt family's seed spread; the final ConvNeXt vote. `v15c` (P-69, closed ✅) read **0.942 solo** and B17 **0.943** on 10-08 | **P-69's week-2 rule fired (B17 ≥ 0.943); Tian's go for where it trains** | RunPod ≈ $1.46 (1.5–2 pod-h, critic-checked case first; ≈ $3.5 left) or ≈ 6.5–10 T4-h on Kaggle after 10-10 | P-80 |
| T10 | P-72 a seventh family (`eca_nfnet_l0`) | **pending Tian** (recommended drop: no read on this task, in1k only) | — | 1 arm ≈ 6 h | P-72 |
| T6 | Final members (week of 10-17): **the ConvNeXt pair (`v15c` + `v15c2`, one vote; T8)**, B3 × 2 seeds (RunPod, or Kaggle if P-74 b works), B0 × 2, R50; no CoAtNet retrain (B14 #63 = B6 without it) — all on the winning recipe and target (0.5 LLM + 0.5 Raptor, or the P-68 mix if it reads ✅; no Claude, P-65 ❌; no silent weight, P-62 🔁) | the two final picks (the B17 / B18 successor; the fork leg) | T7 / T9 reads (A3 / B14 / B4 / B5 read 10-07; B3 × 2 vs B0 × 2 still open) | ≈ 30 Kaggle session-h + ≈ $4 RunPod for the B3s | P-50 |
| T2 | B3 at seed 43 **now** | de-biases the single 0.940 draw; the critic's case is written | **paused** with the rest of D; folded into T6 (the final B3 retrain on the winning recipe *is* the second draw) | ≈ $1.9 | P-50 |
| T3 / T4 | R50 seed 2; B3 on the winning target | — | = T6 | — | P-50 |
| ~~T5~~ | a bigger CNN (B4 @ 288 / 336) | **dropped 10-05:** the forum's ≥ 0.949 singles are R50-class at 224–288, "bigger is null", no gain above 288; the critic priced B4 @ 336 at ≈ 25 GB VRAM and ≈ $3.7 | — | — | (a new family instead: P-69, T8) |

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
