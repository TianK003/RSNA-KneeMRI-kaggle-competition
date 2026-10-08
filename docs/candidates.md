# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-08 (12:20 UTC). **C4 (the public OAI-trained 0.949 model + B17 at β 0.35) read 0.950 (#68; 🔁 +0.001 vs the anchor):
our best public number, rank 157 of 5,487 (was 720), inside the 247-team 0.950 tie.** Almost all of it is the public checkpoint: on a
0.949 anchor our leg is worth one tick, and a fork's lift is capped by the gap between the leg and the anchor (experiments.md "Submission
#68"). Whether an OAI-trained pick 2 is acceptable is Tian's call (brainstorm.md). **B18 (`v15c` + `v13b3`) read 0.944 = our best own
number (#67; pick 1, OAI-free).** Earlier today (#64–#66, experiments.md "Submissions #64–#66"): **A6 `v15c` (ConvNeXt-T) 0.942 = our
best solo**; **B17 = B6 + `v15c` 0.943**; **B12 0.941** (closed). All five 10-08 slots are used. Earlier, the 10-07 reads (#59–#63, experiments.md "Submissions #59–#63"): **B14 0.942 = B6**
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
GPU-min each; ≈ 2.3 h of Kaggle GPU is left until 10-10).

| Prio | Day | Candidate | What it decides | Gate | Status |
|---|---|---|---|---|---|
| **P1** | 10-09 | **C4b**: the public 0.949 checkpoint + **B18** (#67, 0.944) as our leg at β 0.35 (section C) | The OAI-carrying fork pick 2 on our best own leg (cheaper than C4: two members) | **Tian: is an OAI-trained pick 2 acceptable (brainstorm.md)?** Then a go for the placeholder and the send | build: `src/build_fork949.py --members v15c v13b3` (one command, ≈ 10 GPU-min); send 1 of its day |
| **P2** | 10-08 evening (RunPod) | **`v15c2`** (P-80, T8): the ConvNeXt seed twin, then `v15c` + `v15c2` as one vote | The ConvNeXt family's seed spread; the final ConvNeXt vote; the leg of both picks | **approved + critic GO; on hold until the evening of 10-08 (Tian)** | relaunch recipe in T8; `INFER_VOTE_GROUPS` not written |
| **P2** | 10-09 | **A8**: `v15c` scored at 320 px at inference (section A) | Free read of the test-time resolution effect; whether T11 is worth training | a go for one placeholder + one send | build (one placeholder) |
| P3 | 10-09+ | **C3**: the 0.942 public stack + B18 as the leg, β 0.45 | The OAI-free fork pick 2. Predicted ≈ 0.944–0.945 = B18's own level, so it is worth a slot only if Tian rules C4 / C4b out as pick 2 (C2 0.944 already holds that slot until then) | Tian's OAI call | build with `src/build_fork.py --members v15c v13b3 --member v15c=rsna-knee-ckpt-v15c:timm-convnext-tiny-in12k` |
| P5 | hold | **B16** (P-71) | — | **recommended DROP 10-08:** `v15c` 0.942 makes the 0.929 public reader a second, weaker ConvNeXt | hold |
| P5 | hold | B1, B2, B7, B8, B9, B10, B12, B15 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6; **B8 and B15 closed by B13**; **B9 closed 10-07**; **B12 read 0.941 on 10-08 (closed)** | hold |
| — | 10-09 | **Freeze the shortlist for P-50** | pick 1 = **B18 (#67, 0.944; OAI-free)**, pick 2 = **C4 (#68, 0.950) / C4b if OAI is acceptable**, else C2 (#54, 0.944) or C3 | — | Tian, by 10-22 |

**What the blend rule expects:** B18 read 0.944 inside its predicted 0.943–0.946 (#67). The next own blends are the two-family top with
a second seed of each (P-80 `v15c2`, the B3 seeds in week 2). A slot left empty costs nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Fork on the 0.949 anchor (OAI-trained) | **0.950** | **#68 = C4**: the public `nartaa` CoAtNet-2 (0.949 alone, its author's score; not re-scored from our account) + B17 at β 0.35, rank 157 of 5,487 (10-08 12:12 UTC). Read rule for a C4 successor: vs 0.950, ✅ ≥ 0.953 / 🔁 0.948–0.952 / ❌ ≤ 0.947 (experiments.md "Submission #68") |
| Public-stack fork | **0.944** | **#54 = C2**: our anchor (0.942 alone, #13 / #15) + B6 at β 0.45, rank 337 of 5,293; 0.943 with the #48 trio (#49). The public community stack now reads 0.943 alone and a public ConvNeXt-T fork of it 0.944; its run-to-run spread is one tick (experiments.md 2026-10-06 "The public frontier moved") |
| Our best own | **0.944** | **#67 = B18: flat rank-mean of `v15c` + `v13b3`** (10-08); #65 B17 (B6 + `v15c`) 0.943. B6 #52 = `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` 0.942; B12 (B6 + `v13es` + `v13rs`) 0.941; B13 (+ two CoAtNets) 0.940, B11 (the EfficientNet triple) 0.941; drop-one (10-07): B14 (no CoAtNet) 0.942, B4 (no `v13e2`) 0.941, B5 (no `v13b3`) 0.940 |
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
| A7 | `v15c2` (P-80): `v15c` at seed 43 | the ConvNeXt seed spread s_c | whether #64's 0.942 is the family's level; the final ConvNeXt vote | s_c = \|`v15c2` − 0.942\|: ≤ 0.003 the bands stand / ≥ 0.005 #64 was a draw | **not trained**; approved for RunPod (critic GO), on hold until the evening of 10-08 (T8) | — |
| A8 | `v15c` scored at 320 px at inference (`INFER_OVERRIDES = {"v15c": {"img_size": 320}}`; `img_size` is an `INFER_MEMBER_KEYS` key) | the test-time resolution effect on the same weights, no seed noise (the critic's free version of T11) | whether T11 is worth training | vs `v15c` 0.942: ≥ 0.946 ✅ (T11 then worth $1.4) / 0.939–0.945 🔁 / ≤ 0.938 ❌ | build (one placeholder) | — |
| A9 | **`v15co`** (P-81): `v15c` + 2,399 OAI knees, masked soft targets for Synovitis / PF OA / Lateral OA (the 0.949 author's three; Lateral Meniscus masked, critic 10-08) | whether external OAI supervision lifts our strongest single model (the 0.949 author: +0.005, one seed, confounded) | the OAI-carrying leg of the C4 fork pick; never pick 1 | vs `v15c` 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938 (one seed, s = 0.003); then as the C4b leg | not trained (T12) | — |

## B. Submission candidates — ensembles of members we have

**Since 10-08 (10:47 UTC) every row here is read against #67 (B18) 0.944: ✅ ≥ 0.948 / 🔁 0.941–0.947 / ❌ ≤ 0.940**, unless the row says otherwise
(B18 is the bar to beat for our own final pick; B17 0.943 held it for the morning of 10-08, B6 0.942 from 10-05 to 10-07). Inside 🔁, the pick-1 tie rule is the one used since #10: an equal score with more
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
| B18 | — | `v15c` + `v13b3` | **read 10-08: 0.944 (#67) = pick 1** (experiments.md "Submission #67") | — | infer v63 | — |
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

**C4 was read on 10-08: #68 = 0.950** (🔁 +0.001 vs the 0.949 anchor; rank 157). On a 0.949 anchor, our 0.943 leg is worth the same one
tick as the public 0.929 reader. A fork's lift ≈ a diversity gain of +0.002–0.003, minus β × (anchor − leg). So a C4 successor gains
≈ +0.0004 per +0.001 on our leg, and β cannot buy a tick (experiments.md "Submission #68"). Every C4-family pick carries the OAI rule
risk; C2 / C3 do not.

| # | Prio | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|---|
| C4b | **P1 (needs Tian's OAI call)** | **The public 0.949 checkpoint + B18** (#67, 0.944) as our leg at β 0.35 | The C4 pick on our best own leg: one tick closer to the anchor, and more independent of it (B17 carries `v11a`, a CoAtNet like the anchor; B18 is two CNNs); two members score in ≈ 38 min against B17's six. Pred. 0.950–0.951 | vs C4 0.950: ✅ ≥ 0.953 / 🔁 0.948–0.952 / ❌ ≤ 0.947. Tie rule: an equal read takes C4b (the stronger leg on paper, the faster rerun) | build: `src/build_fork949.py --members v15c v13b3`, then the `rsna-knee-fork949` placeholder (≈ 10 GPU-min) |
| C3 | P3 | Public 0.942 stack + **B18** (#67, 0.944) as our leg at β 0.45 | The OAI-free fork pick 2 with a better leg | Gate: the leg read ≥ 0.943 (open since 10-08). vs C2 0.944: ✅ ≥ 0.947 / 🔁 0.942–0.946 / ❌ ≤ 0.941. Pred. 0.944–0.945 = B18's own level, so **worth a slot only if Tian rules the C4 family out as pick 2** | build with `src/build_fork.py --members …` plus `--member v15c=rsna-knee-ckpt-v15c:timm-convnext-tiny-in12k`; send 1 of its day |
| — | — | Fork at other β | **Not planned:** it tunes a weight to the public LB, and on #68's arithmetic it is worth < 0.0002 | — | — |

## Order

- **10-06 (sent 00:03–00:06 UTC, all read):** C2 0.944 → B13 0.940 → B11 0.941 → A4 0.935 → A5 0.932 (experiments.md "Submissions
  #54–#58"). C3's gate stayed closed and no E solo qualified for B12, so both pre-registered fallbacks fire.
- **10-07 (sent 00:00–00:04 UTC, all read by 01:02):** B4 0.941 → A3a 0.936 → A3b 0.937 → B5 0.940 → B14 0.942 (experiments.md
  "Submissions #59–#63"). P-62 closed 🔁, B12's gate met, B9 closed, the CoAtNet retrain dropped, P-69 = one arm.
- **10-07 06:26–08:24 UTC:** the P-69 RunPod run (`v15c`, one arm), 1.97 pod-h ≈ $1.46, shipped and backed up, pod deleted
  (experiments.md 2026-10-07 "P-69 on RunPod"). Placeholders B12 = v60, A6 = v61, B17 = v62, all green.
- **10-08 (sent 06:16–06:18 UTC by the fallback; the armed pid 25568 died with the laptop on 10-07, traps 53):** A6 **0.942** →
  B17 **0.943** → B12 **0.941** (experiments.md "Submissions #64–#66"). P-69 closed ✅, B17 = pick 1, C3's gate open, B12 closed.
  Then, on Tian's go: B18 #67 **0.944** (pick 1) and C4 #68 **0.950** (the public 0.949 model + B17, OAI risk accepted for the read;
  rank 157). All 5 slots used.
- **10-08 evening:** `v15c2` on RunPod (T8; approved, on hold until then).
- **10-09:** C4b if Tian accepts an OAI-carrying pick 2 (else C3), sent first in its day; A8 (free). Freeze the P-50 shortlist.
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
| **T8** | **P-80 ConvNeXt seed twin `v15c2`** (= `v15c` at seed 43), then `v15c` + `v15c2` as one vote (`INFER_VOTE_GROUPS`, not written) | the ConvNeXt family's seed spread; the final ConvNeXt vote | **APPROVED (Tian's go + critic GO), ON HOLD until the evening of 10-08 (Tian, 11:33 UTC: "stop the runpod run, we will run it in the evening").** Pod `88cdnd5fmxjqn6` was created 11:23 and deleted 11:33 during the cache pull (≈ $0.13, nothing trained). **Relaunch recipe:** create-pod (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, SECURE, RTX 4090, disk 40, persistent 100 GB at /workspace, ports 22/tcp + 8888/http, startSsh); check `stat -f -c %T /workspace` (10-08: local NVMe xfs → `CACHE_ROOT=/workspace/cache`; on MooseFS split per traps 55); `apt-get install bc git`, clone the repo into /workspace/repo, scp `~/.kaggle/credentials.json` (check its expiry ≥ 3 h away, traps 20), `ln -sfn /workspace/kaggle /kaggle`; stopper `runpod_stopper.sh /workspace/job_v15c2.log $((POD_T0 + 9360))`; export the pod key from /proc/1/environ, then `CACHE_PREFIX=rsna-knee-cache3 RSNA_TEACHER_TABLES='("raptor_teacher",)' AUTO_STOP=1 SEQ_ARMS=1 SHIP_TRIES=3 SHIP_WAIT_S=120 MAX_POD_H=2.6 POD_T0=<creation epoch> runpod_chain.sh v15c2` | ≈ 2.05 pod-h ≈ $1.5 (cap ≈ $1.9) | P-80 |
| T11 | **`v15c320`** (arm built and pushed 10-08, `4b241bc`: `v15c` at 320 px; local smoke green) | the resolution probe | **HELD 10-08 by the critic:** 288 → 320 is only 1.11× finer and P-43's 1.43× step read +0.002, so it would read inside the floor and is really a third ConvNeXt vote for ≈ $1.4; the money is better kept for a ConvNeXt on the P-68 mix if that pair reads ✅ (10-10/11). The free read first: A8 | ≈ $1.4 | P-80 |
| **T12** | **P-81 `v15co`** (built and smoke-green 10-08, `8a0c1b2`): one RunPod 4090 job with `OAI=1` (download 7,196 OAI series ≈ 58 GB with nda-tools + build shard 90 on the pod, then train) | OAI as our own training data | **critic GO WITH CHANGES (10-08)**, applied: 3 labels, `.env` shredded after the download, OAI_MAX_PREP_H 1.2 abort, a US datacentre, stopper cap ≈ 4.3 h; the label-split infer read (`v15co` on its 3 labels + `v15c` on the other 9) is to build. **Waits on a RunPod top-up + Tian's go**; scp `.env`, `image03.txt`, `oai_targets.csv` to the pod by hand | ≈ 3.3–3.8 pod-h ≈ $2.5–2.8 (cap 4.3 h ≈ $3.2); /workspace ≥ 200 GB | P-81 |
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
