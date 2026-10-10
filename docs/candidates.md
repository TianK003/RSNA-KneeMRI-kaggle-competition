# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-10 (17:50 UTC). **P-81 `v15co` is trained** (RunPod, 13:46–15:57 UTC, $2.64, ≈ $9.5 of the last $12.15 left):
gold-58 0.9211 (`v15c` 0.9234), its three OAI labels 0.849 vs 0.841 (direction only); shipped `rsna-knee-ckpt-v15co`, backed up, pod
deleted. Its sends (A9 solo, the label split, a C4b leg) wait on Tian's go and a free GPU slot for the placeholder. **Two Kaggle sessions
run on Tian's go:** P-82 (A12 `v16d1` ‖ A13 `v16d3`, a DINOv2-S LR bracket at 280 px, T13; `rsna-knee-train-b` v13, queued 50 min,
≈ 22:00–22:15 UTC) and P-67 session C (`v14lr` ‖ `v14gd`, T7; `rsna-knee-train` v51, ≈ 19:00–20:30 UTC). Earlier, 09:00 UTC. **Sessions A and B are done (experiments.md 2026-10-10 "Sessions A ‖ B read"): the P-67 floor is 0.0014 (ablation bar ≈ 0.002; `v14p2` fold 0 went NaN, traps 63), and the P-68 pair `v13ex` / `v13ex2` trained green and is shipped. **Its solos read 0.937 (#74) / 0.936 (#75): pair mean 0.9365 = the B0 seed mean → 🔁, P-68 closed, not adopted** (experiments.md "Submissions #74–#75"); Raptor 0.5 stays the target and the label side is closed. The other three 10-10 slots stay empty: no queued candidate is worth one. **The live lever is the P-67 loop (T7, on Tian's go).** Kaggle GPU 9.75 h used of 30. Earlier, 00:55 UTC. **The two forks on B18 are read (experiments.md "Submissions #69 and #73"): C4b 0.951 (#73; 🔁 +0.001
vs C4) → pick 2 by the rule; our best public number, rank 218 of 5,626. C3 0.946 (#69; 🔁 +0.002 vs C2) → the OAI-free fork, our best
OAI-free number.** Pick 1 stays B19 (#71, 0.944) unless Tian moves it to C3 (brainstorm.md, new open question). The fork rule held on
both: a fork ≈ the weighted mean of its parts + 0.002–0.004. The queue is now training (section D): the Kaggle quota reset at 10-10 00:00
UTC (30 h, next reset 10-17), and **sessions A (P-67 floor) and B (P-68 pair) run since 01:21 UTC**; 10-10 has 5 free slots and nothing
staged. **Tian (10-10): improve the models first; the picks wait until the end.** Earlier, 2026-10-09 (09:15 UTC). **The 10-09 reads (Tian's lineup, sent 06:51–06:54 UTC; experiments.md "Submissions #70–#72"):
A7 `v15c2` 0.941 (#70: ConvNeXt seed spread s_c = 0.001, the bands stand, P-80 closed ✅); B19 0.944 (#71) = B18 → pick 1 by the tie
rule; A8 (`v15c` scored at 320 px) 0.942 (#72) = at 288 → T11 not trained. C3 (#69, the fork) read 0.946 later that day. One 10-09 slot is
free. Tian, ≈ 09:30 UTC: an OAI-trained pick 2 is accepted (pick 2 = the C4 family), and no training at 336 px. The C4b placeholder
(`rsna-knee-fork949` v2) was built and, on Tian's go, sent at 10:05 UTC as #73 (read 0.951). All 5 slots of 10-09 are used.** Earlier, 10-08 12:20 UTC: **C4 (the public OAI-trained 0.949 model + B17 at β 0.35) read 0.950 (#68; 🔁 +0.001 vs the anchor):
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
| **P1** | 10-10 → | **The P-67 loop** (T7: session C = `v14lr` ‖ `v14gd`, then the evidence order; P-68 read 🔁 on 10-10, so this is one of the two OAI-free levers left) | the single-model recipe on a measured ruler (bar ≈ 0.002) | **Tian's go 10-10**; 20.25 h of Kaggle GPU left this week | smoke v50 green (`aug_extra` lowres / grid, both `ok arm`); **real session `rsna-knee-train` v51 pushed 14:43 UTC** |
| **P1** | 10-11 | **A12 `v16d1` / A13 `v16d3`** (P-82, a DINOv2-S retrain on our recipe, T13) | whether a third family (a ViT) reaches member grade; then B20 = B19 + its vote | **Tian's go 10-10** | smokes v11 / v12 green (≈ 19 min / epoch → ≈ 6.5 h); **real session `rsna-knee-train-b` v13 pushed 14:43 UTC, queued 50 min, RUNNING from ≈ 15:30 → ≈ 22:00–22:15 UTC** |
| P2 | 10-10 → 10-11 | **A9 `v15co`** (P-81, section A) | An OAI leg can only enter C4b: ≈ +0.001–0.002 on pick 2 at +0.005 on the leg | the send: Tian's go | **trained, shipped 15:57 UTC (`rsna-knee-ckpt-v15co`), pod deleted ($2.64)**; gold-58 0.9211; the placeholder needs a free GPU slot (both busy until P-67 C ends) and one dead mount dropped (traps 64) |
| P5 | hold | **B16** (P-71) | — | **recommended DROP 10-08:** `v15c` 0.942 makes the 0.929 public reader a second, weaker ConvNeXt | hold |
| P5 | hold | B1, B2, B7, B8, B9, B10, B12, B15 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6; **B8 and B15 closed by B13**; **B9 closed 10-07**; **B12 read 0.941 on 10-08 (closed)** | hold |
| — | 10-10 | **The P-50 shortlist** | pick 1 = **B19 (#71, 0.944; OAI-free; = B18 #67, pick 1 by the tie rule)**, or C3 if Tian moves it (brainstorm.md); pick 2 = **C4b (#73, 0.951; OAI-trained anchor, accepted 10-09)**; **C3 (#69, 0.946)** replaces it only if the host rules OAI out. A better own leg later is rebuilt into all three | — | Tian, by 10-22 |

**What the blend rule expects:** B18 read 0.944 inside its predicted 0.943–0.946 (#67), and B19 (the ConvNeXt twins as one vote +
`v13b3`) read the same 0.944 (#71): a second seed of a family already in the blend adds nothing readable on the public LB; it hedges the
private one. The B3 seed in week 2 is that kind of hedge too. A slot left empty costs nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Fork on the 0.949 anchor (OAI-trained) | **0.951** | **#73 = C4b**: the public `nartaa` CoAtNet-2 (0.949 alone, its author's score; not re-scored from our account) + B18 at β 0.35, rank 218 of 5,626 (10-10 00:50 UTC); #68 C4 (+ B17) 0.950. Read rule for a successor: vs 0.951, ✅ ≥ 0.954 / 🔁 0.949–0.953 / ❌ ≤ 0.948. A leg at the anchor's level (≈ 0.949) predicts ≈ 0.953; a ✅ needs a leg above the anchor (≈ 0.950–0.952) (experiments.md "Submissions #69 and #73") |
| Public-stack fork | **0.946** | **#69 = C3**: our anchor (0.942 alone, #13 / #15) + B18 at β 0.45 (10-09); #54 C2 (+ B6) 0.944, rank 337 of 5,293; 0.943 with the #48 trio (#49). The public community stack now reads 0.943 alone and a public ConvNeXt-T fork of it 0.944; its run-to-run spread is one tick (experiments.md 2026-10-06 "The public frontier moved") |
| Our best own | **0.944** | **#71 = B19: the ConvNeXt twins (`v15c` + `v15c2`) as one vote + `v13b3`** (10-09; pick 1 by the tie rule) = **#67 B18: flat rank-mean of `v15c` + `v13b3`** (10-08); #65 B17 (B6 + `v15c`) 0.943. B6 #52 = `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` 0.942; B12 (B6 + `v13es` + `v13rs`) 0.941; B13 (+ two CoAtNets) 0.940, B11 (the EfficientNet triple) 0.941; drop-one (10-07): B14 (no CoAtNet) 0.942, B4 (no `v13e2`) 0.941, B5 (no `v13b3`) 0.940 |
| Solos | **0.942** / 0.941 / 0.940 / 0.938 / 0.937 / 0.937 / 0.936 / 0.936 / 0.935 / 0.935 / 0.934 / 0.932 / 0.932 / 0.932 / 0.932 / 0.931 | **`v15c` (ConvNeXt-T, #64; 0.942 also scored at 320 px, #72)** / `v15c2` (its seed twin, #70) / `v13b3` / `v13e2` / `v13rs` / `v13ex` (#74) / `v13es` / `v13ex2` (#75) / `v13e` / `v13ecp` / `v13r` / `v11a` / `v11n` / `v11n2` / `v13ec` / `v13h` |
| CNN seed spread | s = 0.003; ConvNeXt s_c = 0.001 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365. ConvNeXt (10-09): #70 `v15c2` 0.941 vs #64 `v15c` 0.942, two-seed level ≈ 0.9415 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

A10 / A11 (the P-68 solos `v13ex` 0.937 / `v13ex2` 0.936, #74 / #75: pair = the B0 seed mean, 🔁) were read on 10-10 (experiments.md "Submissions #74–#75"). A7 (`v15c2`, 0.941: s_c = 0.001) and A8 (`v15c` at 320 px, 0.942 = at 288) were read on 10-09 (experiments.md "Submissions
#70–#72"); A6 (`v15c` 0.942) on 10-08, A3 on 10-07, A4 / A5 on 10-06. Delete each row once its score is logged.

| # | Candidate | What it tests / contributes | Decides | Read rule | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| A9 | **`v15co`** (P-81): `v15c` + 2,399 OAI knees, masked soft targets for Synovitis / PF OA / Lateral OA (the 0.949 author's three; Lateral Meniscus masked, critic 10-08) | whether external OAI supervision lifts our strongest single model (the 0.949 author: +0.005, one seed, confounded) | the OAI-carrying leg of the C4 fork pick; never pick 1 | vs `v15c` 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938 (one seed, s = 0.003); then as the C4b leg | **trained and shipped 10-10 (T12; gold-58 0.9211, its 3 OAI labels 0.849 vs `v15c` 0.841, direction only)**; placeholder: drop `rsna-knee-ckpt-v13ex2` (P-68 closed) for `-v15co` (49 sources, traps 64), then a free GPU slot | — |
| A12 | **`v16d1`** (P-82): DINOv2-S/14 at 280 px on `v15c`'s data side, top block 1e-4 / LLRD 0.8, wd 0.05, drop path 0 → 0.1, 20 epochs | whether the ViT family reaches member grade on our recipe (`v08r` read 0.918 on the old one) | B20 = B19 + the better DINOv2 arm | the better arm vs `v15c` 0.942: ✅ ≥ 0.940 → B20 / 🔁 0.935–0.939 → B20 on a free slot / ❌ ≤ 0.934 → DINOv2 stays dropped; the arms ≥ 0.004 apart = an LR direction | not trained (T13) | — |
| A13 | **`v16d3`** (P-82): = A12 with top block 3e-4 / LLRD 0.75 | the high end of the LR bracket | as A12 | as A12 | not trained (T13) | — |

## B. Submission candidates — ensembles of members we have

**Since 10-08 (10:47 UTC) every row here is read against #67 (B18) 0.944: ✅ ≥ 0.948 / 🔁 0.941–0.947 / ❌ ≤ 0.940**, unless the row says otherwise
(B18 / B19 are the bar to beat for our own final pick; B17 0.943 held it for the morning of 10-08, B6 0.942 from 10-05 to 10-07). Inside 🔁, the pick-1 tie rule is the one used since #10: an equal score with more
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
| B19 | — | `v15c` + `v15c2` as ONE vote (`INFER_VOTE_GROUPS = {"convnext": ("v15c", "v15c2")}`) + `v13b3` | **read 10-09: 0.944 (#71) = B18 → pick 1 by the tie rule** (experiments.md "Submissions #70–#72") | — | infer v66 | 0.9252 |
| B18 | — | `v15c` + `v13b3` | **read 10-08: 0.944 (#67)**; pick 1 until B19 tied it; still the leg of C3 / C4b | — | infer v63 | 0.9260 |
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
risk; C2 / C3 do not. **Tian accepted that risk for pick 2 on 10-09** (brainstorm.md), so pick 2 is a C4-family fork.

**C4b and C3 were read on 10-09 (experiments.md "Submissions #69 and #73"): C4b #73 = 0.951 → pick 2; C3 #69 = 0.946 → the OAI-free
fork.** Both carry B18 as the leg. Their rebuild commands, for a new leg: C4b = `src/build_fork949.py --members <arms> --beta 0.35`
(`rsna-knee-fork949`); C3 = `src/build_fork.py --members <arms> --member <arm>=<ckpt Dataset>:<weights Dataset> … --beta 0.45`
(`rsna-knee-fork`; `v13b3` has no stored mapping: `--member v13b3=tiankljucanin/rsna-knee-ckpt-v13b3:tiankljucanin/timm-efficientnet-b3-ra2`).
A C3-type fork scores in ≈ 6.7 h, a C4b-type in ≈ 1.75 h.

| # | Prio | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|---|
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
- **10-08 evening:** `v15c2` on RunPod (T8): the 19:38 community attempt died on its host (traps 61); relaunched on a secure 4090,
  trained 20:14–21:41 UTC, shipped (gold 0.9180), pod deleted.
- **10-09 (Tian's lineup; sent 06:51–06:54 UTC by `auto_submit.py`):** C3 #69 **0.946** → A7 `v15c2` #70 **0.941** → B19 #71 **0.944**
  (pick 1 by the tie rule) → A8 #72 **0.942** (experiments.md "Submissions #70–#72"). P-80 closed ✅, T11 not trained. One slot free.
  Then Tian: an OAI-trained pick 2 is accepted, no 336 px training; the C4b placeholder (`rsna-knee-fork949` v2) was built and sent
  at 10:05 UTC as #73 **0.951** → pick 2 (experiments.md "Submissions #69 and #73"). All 5 slots used.
- **10-10:** training resumes (section D). On Tian's go, sessions A (`v14p` ‖ `v14p2`, v49) and B (`v13ex` ‖ `v13ex2`, v10) were pushed
  at 01:21 UTC after green smokes. Both COMPLETE (A 5.63 h, B 4.02 h): the P-67 floor 0.0014 (bar ≈ 0.002); the P-68 pair shipped,
  A10 sent 08:49 UTC as #74 (infer v68) → 0.937, A11 08:55 UTC as #75 (infer v69) → 0.936: pair = the B0 seed mean, P-68 closed 🔁. The other three slots stay empty. **Tian (10-10): focus on improving the models until the picks are due;** the RunPod top-up for
  P-81 is his call in the morning. The submissions after that are the new arms' solos, then the rebuilt forks if a leg improves.
- **10-10 afternoon (Tian's go on all three):** RunPod top-up ($12.15 to the end) → P-81 `v15co` on a secure 4090 (13:08 UTC pod,
  training 13:46); after the thread review (743374 / 746792 / 735304) and the DINOv2 research (research.md 2.7.10), P-82's LR bracket
  at 280 px and P-67 session C on Kaggle, smokes first (v11 green, v12 timing, v50).

## D. Training candidates (GPU) — open since the 10-10 reset

**10-10: sessions A and B are done** (experiments.md 2026-10-10 "Sessions A ‖ B read"): A = `rsna-knee-train` v49 (`v14p` ‖ `v14p2`,
5.63 h: floor 0.0014, bar ≈ 0.002, `v14p2` fold 0 NaN, traps 63), B = `rsna-knee-train-b` v10 (`v13ex` ‖ `v13ex2`, 4.02 h: green,
shipped). Both slots are free; `kaggle quota` at 08:48 UTC: 9.75 h used of 30, next reset 10-17. **What the 10-09 forks add (experiments.md "Submissions #69 and #73"):** the own leg is
the one lever left, and it feeds pick 1 in full, C3 at β 0.45 and C4b at β 0.35. A leg at ≈ 0.949 (+0.005 over B18) lifts C4b to
≈ 0.953 only. An OAI-free gain lands in full on pick 1 and at 0.45 on C3; an OAI-trained one (T12) can only enter C4b. So T9 / T7 come
before T12.

**Paused (Tian, 2026-10-05 ≈ 12:00 UTC):** no training on 10-05 or 10-06; the days until 10-10 only submit. Kaggle GPU minutes go to
placeholders only.

**What the 10-06 and 10-07 reads settled for week 2:** the targets stay 0.5 LLM + 0.5 Raptor (P-65 ❌: no Claude share; P-62 🔁: no
silent weight), and the P-68 pair read 🔁 on 10-10 (#74 / #75: = the B0 seed mean), so that is final; no CoAtNet retrain at all (B14 #63 = B6 without it; B13: no second seed);
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
| **T9** | **P-68** pseudo-label pair `v13ex` ‖ `v13ex2` (B0, seeds 42 / 43, on 0.5 LLM + 0.25 Raptor + 0.25 CoAtNet cross-fit OOF `xfit_v09k`); `v11o` / `v13eo` after the floor run, only if the pair reads ✅ | the forum's largest single-model lever (different-source pseudo-labels, Raymond / SpeedSci +0.011), the one target change we have not tested | **READ 10-10: #74 `v13ex` 0.937 / #75 `v13ex2` 0.936 → pair mean 0.9365 = the B0 seed mean → 🔁, P-68 closed, not adopted** (experiments.md "Submissions #74–#75"); `v11o` / `v13eo` / `cnnoof_v1` not run. Trained 10-10 (`rsna-knee-train-b` v10, 4.02 h, green), shipped `rsna-knee-ckpt-v13ex` / `-v13ex2` | ≈ 6 h, one session | P-68 |
| **T7** | **P-67 loop** (**the floor pair ✅ read 10-10: floor 0.0014, bar ≈ 0.002, `v14p` pooled 0.8628; `v14p2` fold 0 NaN, traps 63**): proxy baseline × 2 seeds (the floor), then one variable per 5-fold run: `v14lr` / `v14th` / `v14gd` / `v14bl` / `v14ns` / `v14sh` (augmentation components), `v14mx` (mixup), `v14r288` (B0 @ 288), `v14db` (blank windows), `v14ep20` (longer schedule); `v14tx` / `v14prog` if P-73 / P-74 are approved; later drop-path / EMA, slot layout | the single-model recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method. The epoch-budget variant answers the epoch-selection question too | **Tian's go 10-10 for session C** (`v14lr` ‖ `v14gd`, `rsna-knee-train`, the floor run's slot and image); traps 63's guard coded (`4faecd0`); smoke v50 green; **real session v51 pushed 14:43 UTC** (watcher `artifacts/watch_sC_1010.log`) | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67 |
| **T8** | **P-80 ConvNeXt seed twin `v15c2`** (= `v15c` at seed 43), then `v15c` + `v15c2` as one vote (`INFER_VOTE_GROUPS`, built 10-08) | the ConvNeXt family's seed spread; the final ConvNeXt vote | **READ 10-09: #70 `v15c2` 0.941 → s_c = 0.001, the bands stand; #71 B19 (the twins as one vote + `v13b3`) 0.944 = B18 → pick 1; P-80 closed ✅** (experiments.md "Submissions #70–#72"). **✅ DONE 10-08: trained 20:14–21:41 UTC, gold-58 SWA 0.9180 (`v15c` 0.9234; the pair's rank-mean 0.9234), shipped `rsna-knee-ckpt-v15c2` 21:42, backed up (`artifacts/kaggle_out/pod_v15c2/`), pod deleted, 1.59 pod-h ≈ $1.41** (experiments.md "P-80 on RunPod"). Ran on secure pod `nvjvi86joqceu9` (RTX 4090, $0.89/h, 100 GB local NVMe `/workspace`, `ssh root@69.145.85.70 -p 10179`; created 20:07:05 UTC, `POD_T0` 1791490025; repo `30c8a8d`; stopper 22:43 UTC = 2.6 h ≈ $2.3 cap; AUTO_STOP, `MAX_POD_H=2.6`; the chain's first step run in the foreground first, traps 61; cache verify 71 blobs / 0 bad / 4,407 studies; the T8 config confirmed in the training log: ConvNeXt-T @ 288, lr 1e-4, decay 0.9, 20 epochs, SWA 3, `train_all`, seed 43, Raptor mix 0.5). `_lastema.pt` was lost to AUTO_STOP (traps 56; unused). The first 10-08 attempt **died 40–80 s into the chain at 19:38 UTC** on community pod `kygt8m42pscb6y` (its AUTO_STOP trap fired in pip / the CUDA check; that host's fault, the package set re-tested clean; its log was removed with the container, traps 61; deleted 20:05). Was: launched on community pod `kygt8m42pscb6y` (RTX 4090, $0.34/h: no 4090 for 40 min, then Tian chose "retry community, then secure"), repo at `30c8a8d`, stopper deadline 22:28 UTC (3.0 h ≈ $1.0), AUTO_STOP; no public IP, so the local backup comes from the shipped Dataset (traps 60). Was: approved (Tian's go + critic GO), on hold until the evening of 10-08. Pod `88cdnd5fmxjqn6` was created 11:23 and deleted 11:33 during the cache pull (≈ $0.13, nothing trained). **Relaunch recipe:** create-pod (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, SECURE, RTX 4090, disk 40, persistent 100 GB at /workspace, ports 22/tcp + 8888/http, startSsh); check `stat -f -c %T /workspace` (10-08: local NVMe xfs → `CACHE_ROOT=/workspace/cache`; on MooseFS split per traps 55); `apt-get install bc git`, clone the repo into /workspace/repo, scp `~/.kaggle/credentials.json` (check its expiry ≥ 3 h away, traps 20), `ln -sfn /workspace/kaggle /kaggle`; stopper `runpod_stopper.sh /workspace/job_v15c2.log $((POD_T0 + 9360))`; export the pod key from /proc/1/environ, run the chain's first step in the foreground (traps 61), then `CACHE_PREFIX=rsna-knee-cache3 RSNA_TEACHER_TABLES='("raptor_teacher",)' AUTO_STOP=1 SEQ_ARMS=1 SHIP_TRIES=3 SHIP_WAIT_S=120 MAX_POD_H=2.6 POD_T0=<creation epoch> runpod_chain.sh v15c2` on a line of its own | ≈ 2.1 pod-h ≈ $1.9 at $0.89/h (cap 2.6 h ≈ $2.3) | P-80 |
| T11 | **`v15c320`** (arm built and pushed 10-08, `4b241bc`: `v15c` at 320 px; local smoke green) | the resolution probe | **NOT TRAINED (10-09): A8 #72 read `v15c` at 320 px 0.942 = at 288, under its ≥ 0.946 gate** (experiments.md "Submissions #70–#72"). Was: **HELD 10-08 by the critic:** 288 → 320 is only 1.11× finer and P-43's 1.43× step read +0.002, so it would read inside the floor and is really a third ConvNeXt vote for ≈ $1.4; the money is better kept for a ConvNeXt on the P-68 mix if that pair reads ✅ (10-10/11). The free read first: A8 | ≈ $1.4 | P-80 |
| **T12** | **P-81 `v15co`** (built and smoke-green 10-08, `8a0c1b2`): one RunPod 4090 job with `OAI=1` (download 7,196 OAI series ≈ 58 GB with nda-tools + build shard 90 on the pod, then train) | OAI as our own training data | **critic GO WITH CHANGES (10-08)**, applied: 3 labels, `.env` shredded after the download, OAI_MAX_PREP_H 1.2 abort, a US datacentre, stopper cap ≈ 4.3 h; the label-split infer read (`v15co` on its 3 labels + `v15c` on the other 9) is built: `INFER_MEMBER_LABELS` (unit-checked; {} = the old blend). **✅ DONE 10-10 (Tian's top-up + go): trained 13:46–15:57 UTC on secure 4090 `3lutummotrslpb` (US, CUDA 12.8, cgroup 10.2 CPUs, 200 GB NVMe, proxy-only SSH; created 13:08:27 after three broken community 4090s, traps 65), 6,748 studies (2,399 OAI: Lateral OA / PF OA 2,399, Synovitis 1,659 supervised), gold-58 SWA 0.9211; shipped `rsna-knee-ckpt-v15co` 15:57, backed up (`artifacts/kaggle_out/pod_v15co/`), pod deleted ≈ 17:45, $2.64** (experiments.md "P-81 on RunPod"). Recipe notes: inputs went as a 0.31 MB base64 bundle over the proxy (`image03.txt` trimmed 91 → 7 MB), repo `3c199ba` (traps 66), OAI ready 37 min after creation | 2.2 h trained, 2.8 pod-h billed: $2.64 | P-81 |
| **T13** | **P-82 DINOv2-S pair `v16d1` ‖ `v16d3`** (`0df06f3`; an LR bracket, Tian): DINOv2-S at 280 px on `v15c`'s data side, top block 1e-4 / LLRD 0.8 vs 3e-4 / 0.75, wd 0.05, drop path 0 → 0.1, 20 epochs | a third family (a ViT) for pick 1 and C3 | **Tian's go 10-10**; smoke `rsna-knee-train-b` v11 green (5.79 GiB peak at 280 px × 34 windows × 2 studies); timing smoke v12 (240 studies, both arms at once): 0.25–0.26 s / study ≈ 19 min / epoch → 20 epochs ≈ 6.5 h under the 8.3-h guard, so 20 stays; **real session v13 pushed 14:43 UTC** (watcher `artifacts/watch_sD_1010.log`) | ≈ 6.5 h, one session | P-82 |
| T10 | P-72 a seventh family (`eca_nfnet_l0`) | **pending Tian** (recommended drop: no read on this task, in1k only) | — | 1 arm ≈ 6 h | P-72 |
| T6 | Final members (week of 10-17): **the ConvNeXt pair (`v15c` + `v15c2`, one vote; T8; B19 #71 0.944 = B18, so a second seed is a private-LB hedge, not a public-LB lever)**, B3 × 2 seeds (RunPod, or Kaggle if P-74 b works), B0 × 2, R50; no CoAtNet retrain (B14 #63 = B6 without it) — all on the winning recipe and target (0.5 LLM + 0.5 Raptor: the P-68 mix read 🔁 on 10-10; no Claude, P-65 ❌; no silent weight, P-62 🔁) | the two final picks (the B17 / B18 successor; the fork leg) | T7 / T9 reads (A3 / B14 / B4 / B5 read 10-07; B3 × 2 vs B0 × 2 still open) | ≈ 30 Kaggle session-h + ≈ $4 RunPod for the B3s | P-50 |
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
