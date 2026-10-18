# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-18 (11:15 UTC). **Tian, 10-18: C13-V2 is pick 1 and C12 pick 2; boost V2 to finish top 10 % on private. C14 (V2 + s6fall in the R6 half) is built and its placeholder runs; C15 (mirror TTA on the 224 px CoAtNet) and C16 (public weights) follow one read at a time. Retraining the public models is ruled out (experiments.md "C14 built"); P-67 stops.**

Earlier, 10:10 UTC. **#96–#100 are read (experiments.md 2026-10-18 "Submissions #96–#100"): `v15cj` 0.936 (❌, P-86 closed), B23 0.943 (❌), C12 0.954 (🔁 +0.001 vs C8: pick 1 if fastMRI passes), C13 0.958 (= the anchor) and C13-V2 0.958 (our best entry, above the anchor inside the tie: pick 2's strongest public ticket). Rank 883 of 6,166. Nothing is queued to send: the shortlist is complete, and the one open lever for pick 1 is the fastMRI check (brainstorm.md), worth 0.002 (C12 vs C11). 10-18 has 5 slots; Kaggle GPU 29.49 h, the last weekly budget.**

Earlier, 2026-10-17 (23:25 UTC). **Sent on 10-17's last slots (Tian: "submit our latest models"): #96 `v15cj` solo, #97 C13 (the 0.958 anchor verbatim), #98 B23 (`v15cj` + `v13b3`), #99 C12 (R7 + s6fall + B18), #100 C13-V2 (heliosli's "0.9586" verbatim), all ⏳; 10-17's five slots are used.** Earlier, 2026-10-17 (13:15 UTC). **C10w-b05 is read (experiments.md 2026-10-17 "Submission #95"): #95 = 0.957 (❌ by its rule), one tick under the 0.958 anchor even at β 0.05. It ranks above C10 / C10w inside the 0.957 tie, so the cost shrinks with β. We are rank 960 of 6,137 (8th of 231 at 0.957); 815 teams sit at exactly 0.958. New: C13 (section C), the 0.958 anchor (or heliosli's newer V2) submitted verbatim as a pick-2 option, Tian's go. heliosli's V2 ("0.9586") reads 0.958 by the LB. 10-17 has 5 slots; Kaggle GPU 30 h, the last weekly budget (next refresh 10-24, after the deadline).**

Earlier, 2026-10-16 (20:15 UTC). **Tian's go: C10w-b05 (the 0.958 `blend-mine-in` + B21 at β 0.05) built and sent as #95 (20:13 UTC, ref 57282094; placeholder green). It is read by its position in the 0.958 tie against the anchor's verbatim copies (section C). 10-16 has 1 slot left.**

Earlier, 18:40 UTC. **#92–#94 are read (experiments.md 2026-10-16 "Submissions #92–#94"):**
- **C11 = 0.952 (🔁):** the fastMRI-free pick 1, in place of C6.
- **C10 = 0.957 (🔁 = its anchor):** pick 2's candidate, in place of C9.
- **C10w = 0.957 (❌):** one tick under its 0.958 anchor, with the arm run.

**On a ≥ 0.957 anchor, B21 at β 0.20 is break-even to a cost, so own-leg work for pick 2 is closed.** The leg still pays in pick 1
(≈ +0.00045 per +0.001). We are rank 617 of 6,103 at 0.957. **New: C12 (R7 + s6fall + B18), pick 1 with fastMRI; P3, Tian's go.**
10-16 has 2 slots left and nothing worth one; Kaggle GPU 3.53 h until the 10-17 00:00 UTC reset.

Earlier, 12:30 UTC. **Tian's go: C11, C10 and C10w are built and sent (#92 12:19, #93 12:23, #94 12:32 UTC; placeholders green, all three anchors reproduce their authors' own outputs exactly). 10-16 has 2 slots left; Kaggle GPU 3.53 h until 10-17 00:00 UTC. No pick decisions until 10-21 (Tian).** Earlier, 12:00 UTC. **C9 is read (experiments.md 2026-10-16 "Submission #91 and the 0.957 / 0.958 frontier"): #91 = 0.956, ahead of C7 inside the tie (C9 > C7 by sign) → pick 2 = C9 by its rule. The public frontier moved to 0.958: heliosli's R6 + flip TTA notebook reads 0.957, and with his per-finding Raptor weights (a public-LB fit) 0.958. We are rank 774 of 6,079. nartaa's `s6fall` (0.950 alone, no OAI, no fastMRI) is public. New: C10 (the R6 + flip notebook + B21, pick 2) and C11 (s6fall + B18, an OAI- and fastMRI-free pick 1), both P1 on Tian's go and both needing a builder spec.** 10-16 has 5 slots; Kaggle GPU ≈ 3.9 h until the 10-17 reset. Earlier, 2026-10-14 (09:10 UTC). **C9 is built and sent (#91 ⏳, 09:02 UTC, Tian's go; placeholder green): experiments.md "C9 built and sent". Tian, 10-14: the Fable arm of P-87 runs after the weekly Claude quota resets on 10-16; no team merger.** Earlier, 08:40 UTC. **C8′ and C7′ are read (experiments.md "Submissions #89 / #90"): C8′ = 0.953 (#89, 🔁 = C8) and C7′ = 0.956 (#90) with no move inside the 0.956 tie, so C7′ ≤ C7. `v15cr` is closed as a leg member; the picks keep their legs (C7 = B21, C8 = B18) and C6′ is dropped. We are rank 226 of 5,958 (23rd of 761 at 0.956). New: C9 (section C) = varshithakolupuri's public notebook (the 0.956 blend + nartaa's unused 224 px checkpoint, OAI-trained) + B21 at β 0.20, for pick 2: that notebook's lift ranks under ours in the same tie, and the two lifts have different sources.** 10-14 has 5 slots. Earlier, 2026-10-13 (17:05 UTC). **C7 and C8 are read (experiments.md "Submissions #87 / #88"): C7 = 0.956 (#87, 🔁 = its anchor, 14th of 547 in the tie: the arm ran; rank 159 of 5,930) → pick 2 by its rule, in place of C5. C8 = 0.953 (#88, ✅ +0.003 vs C6) → pick 1 by its rule if Tian accepts fastMRI; R7 alone ≈ 0.948–0.951.** The external-data check: C8 has no OAI, and its fastMRI (inside R7) is click-through, so it likely passes the host's 08-27 test (not ruled by name). C7 carries OAI twice, which likely fails the host's 09-21 test (the NIH country bar; not ruled). C7 and C8 share R7. Whether fastMRI may enter pick 1 is Tian's call (brainstorm.md). 10-13 has 2 slots left. Earlier, 10:25 UTC. **C7 and C8 are sent** (Tian's go; experiments.md "C7 and C8 built and sent"): `src/build_fork954.py --anchor 0956` → `rsna-knee-fork956` v1 = C7 (#87, 10:06 UTC) and v2 = C8 (#88, 10:19 UTC), both placeholders green. **C6′ is held** (Tian). 10-13 has 2 slots left; Kaggle GPU 4.35 h until 10-17. Earlier, 09:45 UTC. **B22 is read (experiments.md "Submission #86"): 0.945 (#86), 🔁 +0.001 vs B18 and above both predicted bands: a member on different targets counts like a new family in our own blend; our best own-model number (OAI-free); the C6′ gate fires. The public frontier moved to 0.956 (experiments.md "The public frontier moved to 0.956"): heliosli's nartaa 0.5 + R7 0.5 (R7 = S6's recipe + a fastMRI auxiliary loss); 357 teams at exactly 0.956, we are rank 571 at 0.954. Pick 2 → **C7** (the 0.956 notebook + B21 at β 0.20, P1); pick 1 → **C8** (R7 + B18 at β 0.45) if Tian accepts fastMRI in pick 1 (brainstorm.md), and C6′ either way; C5′ dropped. 10-13 has 4 slots; Kaggle GPU 4.59 h until 10-17; RunPod ≈ $5.8.** Earlier, 2026-10-12 (17:45 UTC). **P-85 is read (experiments.md "Submissions #84–#85"): `v15cr` solo 0.932 (#85, −0.010 vs `v15c`); nartaa's 0.949 + it at β 0.35 = 0.950 (#84) → k ≈ 0.026–0.035 = S6's range (B18 0.012–0.021). Without Raptor our ConvNeXt is as complementary to nartaa as S6 is, but as a leg it buys nothing (0.950 = C4b − 0.001); P-85 closed. Three cheap follow-ups on existing checkpoints are queued, none with an expected tick: B22 (P2), then C6′ (gated on B22) and C5′ (P3). 10-13 has 5 slots; Kaggle GPU 4.59 h until 10-17; RunPod ≈ $5.8.** Earlier, 16:10 UTC. **P-85 `v15cr` is trained (gold-58 0.9121) and its two reads are going out (the solo + the 0.949-anchor fork, 10-12's last slots). The labels survey is read (experiments.md "The public label sets re-surveyed"): no public set beats our blend; one JEV arm is card P-86 (T17).** Earlier, 12:00 UTC. **C6 is read (experiments.md "Submission #82"): S6 + B18 at β 0.45 = 0.950 (#82), ✅ +0.006 vs A14 / B19 → the OAI-free pick-1 candidate (it shares S6 with pick 2, brainstorm.md). The P-85 gate fired: `v15cr` training on a RunPod A40 (P1 below).** Earlier, 11:45 UTC. **P-84 is read (experiments.md 2026-10-12 "P-84 on RunPod + Submission #83"): `v15cb` (`v15c` + the ≈ 6–94 % slice band) = 0.943 (#83) vs `v15c` 0.942 → 🔁 +0.001; P-84 closed, `trim_ends` stays off. The pod is deleted ($1.75; RunPod ≈ $7.6 left).** C6 (#82) is still scoring; P-85 `v15cr` waits on it (≥ 0.950). 10-12 has 2 slots left. Earlier, 07:40 UTC. **A14 is read (experiments.md 2026-10-12 "Submission #81"): S6 alone = 0.944 (#81) = B19, so by its rule S6 is no pick-1 lever by itself, and the C6 (≥ 0.948) and T14 / P-83 (≥ 0.950) gates fail; P-83 is dropped. New: the 0.954 blend's diversity gain is +0.0055–0.0075, against +0.0027–0.0047 for our B18 on the same anchor, so C6 (S6 + B18 at β 0.45, OAI-free) is re-priced to 0.946–0.953. It is the one candidate with a pick-1 upside: its send is Tian's call (P1 below).** 10-12 has 4 slots left; Kaggle GPU 4.88 h until 10-17; RunPod ≈ $9.5. Earlier, 2026-10-11 (17:55 UTC). **C5 is read (experiments.md 2026-10-11 "Submission #80"): #80 = 0.954 → 🔁 Δ 0.000 vs its anchor at three decimals, but 9th of 425 in the 0.954 tie, above the blend's verbatim copies (the arm ran: a sub-tick lift); our best public number, rank 147; pick 2's candidate by its rule (C4b 0.951 before). The fork rule's diversity gain sat at its low end (+0.0014–0.0026), so C6 is re-priced to ≈ A14 (its use is pick-1 content, not a tick). The A14 placeholder (`rsna-knee-fork954` v2) is green; pid 15852 sends it at 00:00:30 UTC 10-12.** Earlier, 11:55 UTC. **The public frontier moved to 0.954 (experiments.md 2026-10-11 "The public frontier moved to 0.954"): heliosli's public blend = nartaa's 0.949 CoAtNet 0.7 + S6 (ConvNeXt-S, 4 folds, joint-anchored views) 0.3; we fell to rank 504 at 0.951. C4c is dominated and dropped; C5 (the 0.954 blend + B21) and A14 (S6 alone) are the new P1, both needing a builder and Tian's go.** **B21 is read (experiments.md 2026-10-11 "Submission #79"): `v15co` + `v13b3` 0.946 (#79) vs B18 0.944 → 🔁 +0.002, inside its prediction (0.945–0.946); the blend rule held a sixth time. Its pre-registered gate (≥ 0.946 → build C4c) fires: C4c (the 0.949 anchor + B21 at β 0.35, pred. 0.951–0.952, ≈ 1.75 h to score) is the one send it unlocks, on Tian's go. B21 is OAI-carrying, so it is the C4-family leg only.** 10-11 has 1 slot left; Kaggle GPU 4.95 h until 10-17. Earlier, 10:10 UTC. **A9 is read (experiments.md 2026-10-11 "Submission #78"): `v15co` 0.945 (#78) vs `v15c` 0.942 →
🔁 +0.003, the top of the band; our best single model, OAI-trained → pick 2 only; P-81 closed.** Its one follow-up is B21 (`v15co` +
`v13b3`, pred. 0.945–0.946) as the cheap read of a C4b leg, then C4c only if B21 ≥ 0.946: worth ≈ +0.0005–0.0007 on pick 2, a tick at
best (Tian's go). The label split is held. 10-11 has 2 slots left; Kaggle GPU 4.98 h until 10-17. Earlier, 08:30 UTC. **Both overnight sessions are read (experiments.md 2026-10-11 "P-82 read", "P-67 session C read"):
P-82 ❌: A12 `v16d1` 0.932 (#76) / A13 `v16d3` 0.893 (#77), the better DINOv2 arm 0.010 under `v15c` → no B20, DINOv2 stays dropped.
P-67 session C: `v14lr` −0.0019 🔁 / `v14gd` −0.0021 ❌ on the proxy → no transfer arm.** 10-11 has 3 slots left; Kaggle GPU 5.0 h
until the 10-17 reset (a two-variant P-67 session took 7.56 h); RunPod ≈ $9.5. The live candidates: A9 (`v15co`, Tian's go; its mount is
in place since infer v70) and the next P-67 variants. Earlier, 2026-10-10 (17:50 UTC). **P-81 `v15co` is trained** (RunPod, 13:46–15:57 UTC, $2.64, ≈ $9.5 of the last $12.15 left):
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
| **P1** | 10-18 (Tian: "do it and submit as soon as possible") | **C14** (section C): V2 + nartaa's s6fall inside the R6 half (0.6 R6 / 0.4 s6fall), self-checked against #100 | pick 1: does s6fall lift V2 (tie position vs #100)? | Tian's go | **sent as #101** (11:54 UTC, ref 57353067, ⏳; placeholder green: SELF_CHECK 0.0, V2 part = #100's output, s6fall = C11's; experiments.md "C14 built") |
| P2 | after C14 reads | **C15** (section C): the mirror TTA on V2's 224 px CoAtNet, on top of whichever of V2 / C14 reads higher | a small genuine gain (≈ +10 min) | C14 read | not built |
| P3 | after C15, Tian | **C16** (section C): V2's four Raptor-half weights pushed a further 0.05 | public rank only: never a pick (a public fit) | Tian ("we could try to make it look better publicly") | not built |
| P5 | stopped 10-18 (Tian) | **P-67 session D** = `v14mx` ‖ `v14sh` (`artifacts/train_v14d_{smoke,real}.py`) | two variables on the proxy, bar +0.002 | — | **stopped 10-18** (Tian: own-pipeline work ends, since no own model can pass 0.95 by 10-22); never started. Was: built, not started; Kaggle smoke, then the real run (≈ 7.5 h). Feeds pick 1's leg only, ≈ +0.00045 per +0.001 on it: a variant that clears the proxy bar cannot reach the LB through a transfer arm and a rebuilt fork before ≈ 10-21. `v14r288` swapped out for `v14sh` (experiments.md "P-67 session D") |
| P2 | 10-11 → | **The P-67 loop** (T7: next `v14mx`, `v14r288`, `v14ep20`, …; session C read 10-11: lowres 🔁 −0.0019, grid ❌ −0.0021; with P-68 and P-82 closed this is the one OAI-free lever left) | the single-model recipe on a measured ruler (bar ≈ 0.002) | **Tian's go per session**; Kaggle GPU 30 h from the 10-17 reset, the last weekly budget (a two-variant session took 7.56 h), or RunPod ≈ $0.6 per proxy (critic + go) | session D's slot and image are free; no session queued |
| P5 | dropped 10-14 | ~~**C6′**~~ (section C; S6 + B22 at β 0.45) | — | **dominated:** C8 (R7 + B18) reads 0.953 over its 0.948–0.951 prediction, and C8′ (#89) showed `v15cr` adds nothing readable to a leg on a heliosli anchor | never built |
| P5 | after C8 | **A15** (section A): R7 alone (the 0.956 notebook with stage-3 weights 0 / 1, as A14 was for S6) | **priced by C8 (10-13): R7 ≈ 0.948–0.951 with C6's k;** no decision waits on the exact number | — | the builder refuses `--s6-only` with `--anchor 0956`: needs its own patch |
| P5 | dropped 10-13 | ~~**C5′**~~ (section C; a `v15cr` leg on the 0.954 blend) | — | **dominated:** the public anchor moved to 0.956 the same morning; the leg question moves to C7 if ever | never built |
| P5 | dropped 10-12 | ~~**T14 = P-83 `v15s`**~~ (ConvNeXt-S on the `v15c` recipe at 288 px; section D) | — | **gate A14 ≥ 0.950 failed** (A14 = 0.944: a public ConvNeXt-S at 384 reads our ConvNeXt-T level) | never built; proposals.md Dropped directions |
| P5 | dropped 10-11 | **C4c** (section C; B21 read 0.946 on 10-11, the gate fired; **dominated by the public 0.954 blend the same day**) | a better pick-2 leg: ≈ +0.0007 on pick 2 (pred. 0.951–0.952 vs C4b 0.951), a tick at best; replaces C4b only at ≥ 0.952 | **Tian's go** (1 slot left on 10-11; send it first in a day otherwise) | build: `src/build_fork949.py --members v15co v13b3 --beta 0.35` → `rsna-knee-fork949` placeholder (the C4b v2 build is the template), then send |
| P5 | hold | **B16** (P-71) | — | **recommended DROP 10-08:** `v15c` 0.942 makes the 0.929 public reader a second, weaker ConvNeXt | hold |
| P5 | hold | B1, B2, B7, B8, B9, B10, B12, B15 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6; **B8 and B15 closed by B13**; **B9 closed 10-07**; **B12 read 0.941 on 10-08 (closed)** | hold |
| — | 10-21 (Tian) | **The P-50 shortlist** | **Tian, 10-18: pick 1 = C13-V2** (#100, 0.958; heliosli's "0.9586" notebook verbatim, our best entry; OAI-trained, accepted 10-09), **or C14** (V2 + s6fall) if it reads above V2. **Pick 2 = C12** (#99, 0.954; R7 + s6fall + B18, OAI-free, fastMRI inside R7), the hedge if OAI-trained models are ruled out. Note (not Tian's decision): if the fastMRI check fails, C11 (#92, 0.952) is the pick-2 hedge free of both. C13 (#97, 0.958) and the 0.957 forks (C10 / C10w / C10w-b05) are the other pick-1 alternatives. **Select both by hand** | — | Tian, 10-21 (final by 10-22 23:59 UTC) |

**What the blend rule expects:** B18 read 0.944 inside its predicted 0.943–0.946 (#67), and B19 (the ConvNeXt twins as one vote +
`v13b3`) read the same 0.944 (#71): a second seed of a family already in the blend adds nothing readable on the public LB; it hedges the
private one. The B3 seed in week 2 is that kind of hedge too. A slot left empty costs nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| The 0.958 anchors verbatim (OAI-trained) | **0.958** | **#100 = C13-V2** (10-18): heliosli's "0.9586" notebook verbatim (R6 from `-w1`), our best entry: rank 883 of 6,166, 719th of 1,041 at 0.958, ≈ 150 stable teams above the 0.958 anchor's verbatim score. **#97 = C13**: the 0.958 `blend-mine-in` verbatim (= the anchor). **Pick 2** (experiments.md "Submissions #96–#100") |
| OAI-free fork on R7 + s6fall | **0.954** | **#99 = C12** (10-18): R7 + s6fall (per finding 0.5 / 0.5 rank) + B18 at β 0.45; +0.001 over C8, the low end of its 0.953–0.956 prediction; fastMRI inside R7. Pick 1 if fastMRI passes |
| Fork on the R6 + flip notebook (OAI-trained) | **0.957** | **#93 = C10** (10-16): heliosli's R6 + flip TTA notebook (0.957) + B21 at β 0.20, = the anchor at three decimals. **Pick 2.** **#94 C10w** (the 0.958 `blend-mine-in` + B21) also reads 0.957, one tick under its own anchor: on these anchors B21 at β 0.20 is break-even to a cost (break-even leg ≈ 0.947–0.951; experiments.md "Submissions #92–#94"). **#95 C10w-b05** (the same at β 0.05) reads 0.957 too, above C10 / C10w inside the tie (experiments.md "Submission #95"). Rank 960 of 6,137 (8th of 231 at 0.957) on 10-17; pick 2 until C13-V2 (#100, 0.958) |
| OAI- and fastMRI-free fork on s6fall | **0.952** | **#92 = C11** (10-16): nartaa's s6fall (0.950 alone, S6's architecture on JEV labels) + B18 at β 0.45; k ≈ 0.015–0.023 (S6's 0.020–0.028 or under). Pick 1 if fastMRI is refused. Each +0.001 on our leg ≈ +0.00045 |
| Fork on the 0.956 variant (OAI-trained) | **0.956** | **#91 = C9** (10-14, read 10-16): varshithakolupuri's 0.956 variant (nartaa's 384 + 224 CoAtNets, R7) + B21 at β 0.20, ahead of C7 inside the tie (C9 > C7 by sign). Pick 2 until C10 (#93, 0.957) |
| Fork on the 0.956 notebook (OAI-trained) | **0.956** | **#87 = C7**: heliosli's public `0-957-latest-dataset-score-in-one-click` (nartaa's 0.949 CoAtNet 0.5 + R7 0.5; 547 teams at exactly 0.956 on 10-13 17:02 UTC) + B21 at β 0.20; = the anchor at three decimals, 14th in the tie (above the verbatim copies), rank 159 of 5,930. Pick 2 until C9 (10-16). Each +0.001 on our leg ≈ +0.0002 (experiments.md "Submissions #87 / #88") |
| OAI-free fork on R7 | **0.953** | **#88 = C8**: heliosli's R7 alone (≈ 0.948–0.951 by C6's k; fastMRI auxiliary loss, SKM-TEA views) + B18 at β 0.45 (10-13); +0.003 over C6 with the identical leg. Pick 1 if Tian accepts fastMRI, until C12 (#99, 0.954). Each +0.001 on our leg ≈ +0.00045 |
| Fork on the 0.954 blend (OAI-trained) | **0.954** | **#80 = C5**: heliosli's public 0.954 blend (nartaa's 0.949 CoAtNet 0.7 + S6 0.3) + B21 at β 0.20, rank 147 of 5,768 (10-11 17:49 UTC), rank 200 of 5,808 (10-12 07:27 UTC, 18th of 620 at 0.954); = the anchor at three decimals but above its verbatim copies in the tie. **S6 alone = 0.944 (#81, A14)**, so the blend's own diversity gain is +0.0055–0.0075 (experiments.md "Submission #81"). On this anchor each +0.001 on our leg is ≈ +0.0002 and β buys ≤ +0.001 (experiments.md "Submission #80") |
| OAI-free fork on S6 | **0.950** | **#82 = C6**: heliosli's S6 alone (0.944, #81) + B18 at β 0.45 (10-12); the fastMRI-free pick 1 until C11 (#92, 0.952) on 10-16. All diversity gain (k ≈ 0.020–0.028); each +0.001 on our leg ≈ +0.00045 (experiments.md "Submission #82") |
| Fork on the 0.949 anchor (OAI-trained) | **0.951** | **#73 = C4b**: the public `nartaa` CoAtNet-2 (0.949 alone, its author's score; not re-scored from our account) + B18 at β 0.35, rank 218 of 5,626 (10-10 00:50 UTC); #68 C4 (+ B17) 0.950; #84 (+ `v15cr`, 0.932 solo, no Raptor) 0.950, k ≈ 0.026–0.035 vs B18's 0.012–0.021. Read rule for a successor: vs 0.951, ✅ ≥ 0.954 / 🔁 0.949–0.953 / ❌ ≤ 0.948. A leg at the anchor's level (≈ 0.949) predicts ≈ 0.953; a ✅ needs a leg above the anchor (≈ 0.950–0.952) (experiments.md "Submissions #69 and #73") |
| Public-stack fork | **0.946** | **#69 = C3**: our anchor (0.942 alone, #13 / #15) + B18 at β 0.45 (10-09); #54 C2 (+ B6) 0.944, rank 337 of 5,293; 0.943 with the #48 trio (#49). The public community stack now reads 0.943 alone and a public ConvNeXt-T fork of it 0.944; its run-to-run spread is one tick (experiments.md 2026-10-06 "The public frontier moved") |
| Our best own | **0.945** (0.946 with OAI) | **#86 = B22: `v15c` + `v13b3` + `v15cr`** (10-13; +0.001 over B18: a member on different targets counts like a family, experiments.md "Submission #86") / (**#79 = B21: `v15co` + `v13b3`**, 10-11; OAI-carrying, the C4-family leg only) / **#71 = B19: the ConvNeXt twins (`v15c` + `v15c2`) as one vote + `v13b3`** (10-09; pick 1 by the tie rule) = **#67 B18: flat rank-mean of `v15c` + `v13b3`** (10-08); #65 B17 (B6 + `v15c`) 0.943. B6 #52 = `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` 0.942; B12 (B6 + `v13es` + `v13rs`) 0.941; B13 (+ two CoAtNets) 0.940, B11 (the EfficientNet triple) 0.941; drop-one (10-07): B14 (no CoAtNet) 0.942, B4 (no `v13e2`) 0.941, B5 (no `v13b3`) 0.940 |
| Solos | (0.945 OAI) / 0.943 / **0.942** / 0.941 / 0.940 / 0.938 / 0.937 / 0.937 / 0.936 / 0.936 / 0.936 / 0.935 / 0.935 / 0.934 / 0.932 / 0.932 / 0.932 / 0.932 / 0.932 / 0.932 / 0.931 | (**`v15co`** = `v15c` + OAI, #78: OAI-trained, pick 2 only) / `v15cb` (= `v15c` + the ≈ 6–94 % slice band, #83: 🔁 +0.001, P-84 closed) / **`v15c` (ConvNeXt-T, #64; 0.942 also scored at 320 px, #72)** / `v15c2` (its seed twin, #70) / `v13b3` / `v13e2` / `v13rs` / `v13ex` (#74) / `v13es` / `v13ex2` (#75) / `v15cj` (an OpenJev share, #96: −0.006, P-86 closed) / `v13e` / `v13ecp` / `v13r` / `v11a` / `v11n` / `v11n2` / `v13ec` / `v15cr` (Opus labels only, #85) / `v16d1` (DINOv2-S, #76; `v16d3` 0.893, #77) / `v13h` |
| CNN seed spread | s = 0.003; ConvNeXt s_c = 0.001 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365. ConvNeXt (10-09): #70 `v15c2` 0.941 vs #64 `v15c` 0.942, two-seed level ≈ 0.9415 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

The P-86 solo (`v15cj` = `v15c` on an OpenJev share of the targets, infer v79, #96: **0.936**, −0.006 vs `v15c`; its blend B23 =
`v15cj` + `v13b3`, infer v80, #98: **0.943** vs B18 0.944; P-86 closed ❌) was read on 10-18 (experiments.md "Submissions #96–#100").
The P-85 solo (`v15cr` = `v15c` on Opus report labels alone, infer v77, #85: **0.932**, −0.010 vs `v15c`; its fork #84 0.950, k ≈ 0.026–0.035;
P-85 closed) was read on 10-12 (experiments.md "Submissions #84–#85").
The P-84 solo (`v15cb` = `v15c` + the ≈ 6–94 % slice band, infer v76, #83: **0.943**, 🔁 +0.001 vs `v15c`; P-84 closed) was read on 10-12
(experiments.md "P-84 on RunPod + Submission #83").
A14 (heliosli's S6 alone, `rsna-knee-fork954` v2, #81: **0.944** = B19; by its rule S6 is no pick-1 lever by itself, the C6 / P-83 gates
fail; C6 re-priced to 0.946–0.953 on S6's diversity) was read on 10-12 (experiments.md "Submission #81").
A9 (the P-81 OAI ConvNeXt `v15co`, #78: **0.945**, 🔁 +0.003 vs `v15c`, pick 2 only) was read on 10-11 (experiments.md "Submission #78");
its follow-up B21 read 0.946 (#79); C4c was dropped for C5 (section C). A12 / A13 (the P-82 DINOv2-S solos `v16d1` 0.932 / `v16d3` 0.893, #76 / #77: ❌, no B20) were read on 10-11 (experiments.md "P-82 read").
A10 / A11 (the P-68 solos `v13ex` 0.937 / `v13ex2` 0.936, #74 / #75: pair = the B0 seed mean, 🔁) were read on 10-10 (experiments.md "Submissions #74–#75"). A7 (`v15c2`, 0.941: s_c = 0.001) and A8 (`v15c` at 320 px, 0.942 = at 288) were read on 10-09 (experiments.md "Submissions
#70–#72"); A6 (`v15c` 0.942) on 10-08, A3 on 10-07, A4 / A5 on 10-06. Delete each row once its score is logged.

A15 (R7 alone, the 0.956 notebook with stage-3 weights 0 / 1, as A14 was for S6) is priced by C8 since 10-13 (R7 ≈ 0.948–0.951 with C6's k) and dropped to P5: no decision waits on it. No single-model candidate is queued: the next one comes from the P-67 loop (section D).

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
| B22 | — | `v15c` + `v13b3` + `v15cr` | **read 10-13: 0.945 (#86) → 🔁 +0.001 vs B18; above both predicted bands (+0.007 over the members' mean, as much as our four-family blends): a same-family member on different targets counts like a new family; our best own-model number, OAI-free; the C6′ gate fired** (experiments.md "Submission #86") | — | infer v78 | 0.9273 |
| B21 | — | `v15co` + `v13b3` | **read 10-11: 0.946 (#79) → 🔁 +0.002 vs B18; the C4c gate fired; OAI-carrying, the C4-family leg only** (experiments.md "Submission #79") | — | infer v75 | not computed |
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

**C5 was read on 10-11 (experiments.md "Submission #80"): #80 = 0.954** (🔁 = its anchor at three decimals; 9th of 425 in the 0.954
tie, above the blend's verbatim copies, so the arm ran and added a sub-tick lift) → pick 2's candidate by its rule, in place of C4b.
Rebuild for a new leg: `src/build_fork954.py --members <arms> --beta 0.20` (`rsna-knee-fork954`); it scored within ≤ 5.6 h. The fork
rule's diversity gain sat at its low end there (+0.0014–0.0026), plausibly because the anchor already carries a ConvNeXt (S6).

**C6 was read on 10-12 (experiments.md "Submission #82"): #82 = 0.950** (✅ +0.006 vs A14 / B19 0.944; OAI-free) → the pick-1
candidate by its rule. All of the +0.006 is diversity gain (S6 and B18 both read 0.944; k ≈ 0.020–0.028). Rebuild for a new
OAI-free leg: `src/build_fork954.py --s6-anchor --members <arms> --beta 0.45` (`rsna-knee-fork954`); it scored in 3.8 h. On it each
+0.001 on our leg is ≈ +0.00045, and a more different leg can add more than a stronger one. **C5 and C6 share S6** (brainstorm.md).

**The P-85 fork was read on 10-12 (experiments.md "Submissions #84–#85"): #84 = nartaa's 0.949 + `v15cr` at β 0.35 = 0.950**
(k ≈ 0.026–0.035, S6's range; B18 on the same anchor 0.012–0.021). A Raptor-free leg is as different from nartaa's CoAtNet as S6, but
at 0.932 it only matches a 0.944 Raptor-lineage leg (C4b 0.951). `rsna-knee-fork949` v3 is that fork; C4b rebuilds with `--members v15c
v13b3 --beta 0.35`.

**10-13: the public frontier moved to 0.956** (experiments.md "The public frontier moved to 0.956"): heliosli's public `0-957-latest-dataset-score-in-one-click` = nartaa's 0.949 CoAtNet 0.5 + **R7** 0.5. R7 = S6's ConvNeXt-S recipe + a fastMRI auxiliary loss, newer Opus targets and fluid / state stackers; it has no OAI. C7 and C8 are the C5 and C6 moves on it. Both need `src/build_fork954.py` taught the 0.956 notebook (same seven-cell layout, same pinned image, same stage files; R7's Dataset `heliosli/rsna-knee-compact-r7cnxfsss`, bundle `8c7b666e27dc0e8f`) and a green placeholder. **B22 read 0.945 (#86)**, so C6′'s gate fired.

**C7 and C8 were read on 10-13 (experiments.md "Submissions #87 / #88"): #87 C7 = 0.956** (🔁 = the anchor at three decimals; 14th of 547 in the tie, above the verbatim copies, so the arm ran: a sub-tick lift, g ≈ +0.002–0.003) → pick 2 by its rule, in place of C5. **#88 C8 = 0.953** (✅ +0.003 vs C6 with the identical leg) → pick 1 by its rule if Tian accepts fastMRI; R7 alone ≈ 0.948–0.951. Rebuild for a new leg: C7 = `build_fork954.py --anchor 0956 --members <arms> --beta 0.20`; C8 = `--anchor 0956 --compact-anchor --members <arms> --beta 0.45` (`rsna-knee-fork956`). **The external-data check:** C8 carries no OAI. Its fastMRI is click-through, so it likely passes the host's 08-27 test (not ruled by name). C7 carries OAI twice, which likely fails the host's 09-21 test (not ruled). C7 and C8 share R7 (brainstorm.md).

**C8′ and C7′ were read on 10-14 (experiments.md "Submissions #89 / #90"): #89 C8′ = 0.953 (= C8), #90 C7′ = 0.956 with no move in
the tie (≤ C7).** `v15cr` adds nothing readable to a leg on a heliosli anchor (R7 and S6 already train on Opus report targets), so the
picks keep B21 / B18 and C6′ is dropped. On these anchors a leg-level change is worth ≤ +0.0005, under what the LB can show: the next
fork move is a change on the anchor side, **C9** (nartaa's unused 224 px checkpoint in the Raptor half, plus our B21).

**C9 was read on 10-16 (experiments.md "Submission #91 and the 0.957 / 0.958 frontier"): #91 = 0.956, ahead of C7 inside the tie → pick 2 = C9.** The same day the public frontier stood at 0.958: heliosli's R6 + flip TTA notebook (0.957) and his per-finding Raptor weights on top (0.958). nartaa's `s6fall` (0.950 alone, OAI- and fastMRI-free) became public. The fork moves on them are C10 (pick 2) and C11 (pick 1).

**C10, C10w and C11 were read on 10-16 (experiments.md "Submissions #92–#94"):**
- **#92 C11 = 0.952** (🔁): the fastMRI-free pick 1, in place of C6. k ≈ 0.015–0.023 vs B18, at or under S6's.
- **#93 C10 = 0.957** (🔁 = its anchor): pick 2, in place of C9.
- **#94 C10w = 0.957** (❌): one tick under its 0.958 anchor, with the arm run.

On a ≥ 0.957 anchor the leg's weighted-mean cost (0.2 × (anchor − 0.946) ≈ 0.0023) is as large as its diversity gain. No leg we can
train by 10-22 moves pick 2; pick 2 follows the public anchor.

Rebuild commands:
- C10 = `build_fork954.py --anchor 0957 --members <arms> --beta 0.20 --gate-hours 7.5 --sec-per-study 3.0` (`rsna-knee-fork957`);
- C10w = the same with `--anchor 0958` (`rsna-knee-fork958`);
- C11 = `--anchor 0950s6f --members <arms> --beta 0.45` (`rsna-knee-forks6f`), built only from the local copy of nartaa's script in
  `artifacts/anchors/` (traps 73).

A new own leg is worth rebuilding into the pick-1 forks only (C11 / C12). **C12** was the one pick-1 anchor move (read below).

**C10w-b05 was read on 10-17 (experiments.md "Submission #95"): #95 = 0.957** (❌ by its rule): B21 at β 0.05 still costs the 0.958
anchor its tick. It ranks above C10 / C10w inside the 0.957 tie, so the cost shrinks with β. The anchor sits in the bottom ≈ 0.0006 of
its tick, and k on it is < 0.012. Every fork of ours on this anchor reads under the anchor itself, which makes the anchor verbatim
(**C13**, read below) the one pick-2 ticket that could carry its 0.958.

**C12, C13 and C13-V2 were read on 10-18 (experiments.md "Submissions #96–#100"):**
- **#99 C12 = 0.954** (🔁 +0.001 vs C8): pick 1 if fastMRI passes. The s6fall stage most likely ran (a fallback is C8's notebook
  exactly), and the run fit the time limit on the full hidden test. Rebuild: `build_fork954.py --anchor c12 --members <arms> --beta 0.45
  --gate-hours 7.5 --sec-per-study 3.0` (`rsna-knee-forkc12`).
- **#97 C13 = 0.958** (= the anchor) and **#100 C13-V2 = 0.958**, our best entry, above the anchor's verbatim score inside the tie
  (V2 > C13 by sign): pick 2. Rebuild: `src/build_anchor_copy.py --anchor 0958 | 09586` (`rsna-knee-anchor958` / `-anchor9586`).

The shortlist is complete: no fork candidate is queued.

| # | Prio | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|---|
| C14 | P1 (new 10-18; Tian's go) | **heliosli's V2 (C13-V2, #100) + nartaa's s6fall inside the R6 half**: ours = 0.6 rank(R6) + 0.4 rank(s6fall) per finding, then V2's stage-3 and Part B weights unchanged; fallback = V2's own file | pick 1. s6fall (0.950, JEV labels, nartaa's trainer) is the one strong public model V2 does not use; ≈ 6.5 h in total. Pred. −0.001 to +0.002 (est.) | vs #100 by tie position: ✅ our best rises above #100's place (stable neighbours 879 / 884 on 10-18 10:04) or reads 0.959 / no move = C14 ≤ V2 | **built 10-18:** `src/build_anchor_copy.py --anchor 09586s6f` → `rsna-knee-anchor9586s6f`; SELF_CHECK rebuilds #100's file before s6fall starts; s6fall starts ≤ 6.0 h, killed at 8.4 h; mock test 7 / 7 (experiments.md "C14 built") |
| C15 | P2 (after C14) | the mirror TTA (plain + anatomical mirror, ported from V2's 384 px stage) on V2's 224 px CoAtNet, on the better of V2 / C14 | a small genuine gain; check first that nartaa's trainer used the mirror at 224 (`training_code/coatnet`) | vs its base by tie position | not built (≈ +10 min runtime) |
| C16 | P3 (Tian's call) | V2's four Raptor-half weights pushed a further 0.05 (Lateral OA 0.10, PF OA 0.70, Synovitis 0.90, Baker's 0.25) | the public rank only: a public fit, so never a pick | vs its base by tie position | not built |
| ~~C6′~~ | P5 (dropped 10-14: dominated by C8; C8′ showed `v15cr` adds nothing readable to a leg on a heliosli anchor) | S6 (heliosli, alone) + B22 (`v15c` + `v13b3` + `v15cr`) at β 0.45, OAI-free | — | — | never built |
| ~~C5′~~ | P5 (dropped 10-13: the public anchor moved to 0.956) | the public 0.954 blend + `v15co` + `v13b3` + `v15cr` at β 0.20 | pick 2 with a Raptor-free member in the leg; sub-tick on the old anchor, and C7 supersedes the anchor | — | never built |
| ~~C4c~~ | P5 (dropped 10-11: dominated by the public 0.954) | the public OAI-trained 0.949 model + B21 (`v15co` + `v13b3`) at β 0.35 (`build_fork949.py --members v15co v13b3 --beta 0.35`) | a better pick-2 leg: on the fork rule ≈ +0.00035 per +0.001 on the leg → pred. ≈ 0.951–0.952. The anchor learned from the same OAI knees, so the leg–anchor diversity gain may shrink | vs C4b 0.951: ✅ ≥ 0.954 / 🔁 0.949–0.953 / ❌ ≤ 0.948; replaces C4b as pick 2 only at ≥ 0.952 | **build on Tian's go** (B21 read 0.946 ≥ 0.946); ≈ 1.75 h to score |
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
- **10-11 (Tian: "pull the results and submit these notebooks"):** both sessions COMPLETE (D 7.35 h, C 7.56 h). P-82 shipped
  (`rsna-knee-ckpt-v16d1` / `-v16d3`), infer smoke v71 (v70 died on a build bug, traps 67), placeholders v72 / v73; A12 sent 07:34 UTC as
  #76 → **0.932**, A13 07:37 UTC as #77 → **0.893**: P-82 ❌, no B20. Session C: `v14lr` −0.0019 / `v14gd` −0.0021 → no transfer arm.
  Then, on Tian's go ("Send it"): A9 placeholder infer v74 (green, 2.2 min), sent 09:15 UTC as #78 → **0.945**: 🔁 +0.003 vs `v15c`,
  pick 2 only, P-81 closed. Then B21 placeholder infer v75 (green, 0.02 h), sent 10:16 UTC as #79 → **0.946**: 🔁 +0.002 vs B18,
  the C4c gate fires. At 11:50 UTC the frontier read (a public 0.954 = nartaa + S6) drops C4c for C5 / A14. Tian: "lets do C5, then focus on A14"; S6 allowed in pick 1. `src/build_fork954.py` written; C5 placeholder
  `rsna-knee-fork954` v1 green (12:07–12:12 UTC), sent 12:13 UTC as #80 → **0.954** (🔁 = the anchor at three decimals, 9th of 425 in
  the tie: the arm ran; rank 147, our best; experiments.md "Submission #80"); A14 placeholder v2 green (pulled 17:48 UTC), sends at
  00:00:30 UTC 10-12 (pid 15852). All 5 slots of 10-11 used.
- **10-12:** A14 sent 00:00:34 UTC by pid 15852 as #81 (the 00:20 backup task was a no-op), read 03:28 UTC → **0.944** (= B19; experiments.md
  "Submission #81"). S6 is no pick-1 lever by itself, P-83 is dropped, and C6 is re-priced to 0.946–0.953 (Tian's go). The A14-night
  power settings and the backup task were undone at 07:35 UTC (traps 53). Then, on Tian's go: C6 placeholder `rsna-knee-fork954` v3
  green, sent 07:53 UTC as #82 (⏳); the forum / public-notebook read (experiments.md "Single-model levers"); Tian chose "Band trim +
  Claude-only targets"; critic GO WITH CHANGES (v15cr gated on C6); P-84 `v15cb` training on a RunPod L40S since 08:35 UTC. Shipped
  10:12 UTC, backed up, pod deleted ≈ 10:52 ($1.75); infer placeholder v76 green, sent 11:01 UTC as #83 → **0.943** (🔁 +0.001 vs
  `v15c`; P-84 closed). 2 slots left. C6 read 0.950 (#82) at 11:42; `v15cr` trained on a RunPod A40 (12:07–14:26 UTC, ≈ $1.86, pod deleted);
  sent 16:08 UTC as #84 (fork on nartaa's 0.949) → **0.950** and #85 (solo) → **0.932**: k ≈ 0.026–0.035, P-85 closed (experiments.md
  "Submissions #84–#85"). All 5 slots of 10-12 used. Then B22's placeholder (infer v78) green, armed for 10-13 00:00:30 UTC (pid 27732).
- **10-13:** pid 27732 died when the laptop was shut down overnight (traps 53); B22 sent by hand at 08:07:55 UTC as #86 → **0.945** (🔁 +0.001 vs B18; the C6′ gate fires; experiments.md "Submission #86"). The 09:19 UTC LB: the frontier moved to 0.956 (heliosli's nartaa 0.5 + R7 0.5) → C7 / C8 queued, C5′ dropped. 4 slots left. Then, on Tian's go (C6′ held): `build_fork954.py --anchor 0956`; C7 placeholder `rsna-knee-fork956` v1 green (09:53–10:04 UTC), sent 10:05:59 UTC as #87; C8 placeholder v2 green (10:06–10:18 UTC), sent 10:19:19 UTC as #88. 2 slots left. Read by 16:57 UTC: C7 **0.956** (🔁, 14th of 547 in the tie, rank 159 → pick 2) and C8 **0.953** (✅ +0.003 vs C6 → pick 1 if fastMRI is accepted); the external-data check on both (experiments.md "Submissions #87 / #88"). 2 slots left, none staged.
- **10-14:** C8′ #89 **0.953** and C7′ #90 **0.956** were read at 08:21 UTC (`v15cr` closed as a leg member). C9's placeholder was green
  and it was sent at 09:02 UTC as #91 (read 10-16: **0.956**, > C7 by sign). 1 slot used. **10-15:** no submission.
- **10-16:** the frontier moved to 0.958 (experiments.md "Submission #91 and the 0.957 / 0.958 frontier"). On Tian's go, C11 / C10 /
  C10w were built and sent at 12:19 / 12:23 / 12:32 UTC as #92–#94 and read at 18:29 UTC: **0.952 / 0.957 / 0.957** (experiments.md
  "Submissions #92–#94"). 2 slots left, nothing worth one. Then, on Tian's go, C10w-b05 sent at 20:13 UTC as #95 (read 10-17:
  **0.957**, ❌ by its rule).
- **10-17:** P-86 `v15cj` trained on RunPod (≈ $1.6). On Tian's go ("submit our latest models"), #96 `v15cj`, #97 C13, #98 B23, #99
  C12 and #100 C13-V2 were sent at 23:17–23:33 UTC (all 5 slots) and read by 10-18 10:04 UTC: **0.936 / 0.958 / 0.943 / 0.954 /
  0.958** (experiments.md "Submissions #96–#100"). P-86 closed ❌; C12 → pick 1 if fastMRI passes; C13-V2 → pick 2.

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
| ~~T14~~ | ~~P-83 `v15s`~~ = `v15c` with ConvNeXt-S, 288 px (never built) | the size step inside our best family | **DROPPED 10-12: its gate (A14 ≥ 0.950) failed. A14 = S6 alone = 0.944: a public ConvNeXt-S at 384 with knee-centred views reads our ConvNeXt-T level** (experiments.md 2026-10-12 "Submission #81") | — | P-83 (retired) |
| **T9** | **P-68** pseudo-label pair `v13ex` ‖ `v13ex2` (B0, seeds 42 / 43, on 0.5 LLM + 0.25 Raptor + 0.25 CoAtNet cross-fit OOF `xfit_v09k`); `v11o` / `v13eo` after the floor run, only if the pair reads ✅ | the forum's largest single-model lever (different-source pseudo-labels, Raymond / SpeedSci +0.011), the one target change we have not tested | **READ 10-10: #74 `v13ex` 0.937 / #75 `v13ex2` 0.936 → pair mean 0.9365 = the B0 seed mean → 🔁, P-68 closed, not adopted** (experiments.md "Submissions #74–#75"); `v11o` / `v13eo` / `cnnoof_v1` not run. Trained 10-10 (`rsna-knee-train-b` v10, 4.02 h, green), shipped `rsna-knee-ckpt-v13ex` / `-v13ex2` | ≈ 6 h, one session | P-68 |
| **T7** | **P-67 loop** (**the floor pair ✅ read 10-10: floor 0.0014, bar ≈ 0.002, `v14p` pooled 0.8628; `v14p2` fold 0 NaN, traps 63**): proxy baseline × 2 seeds (the floor), then one variable per 5-fold run: `v14lr` / `v14th` / `v14gd` / `v14bl` / `v14ns` / `v14sh` (augmentation components), `v14mx` (mixup), `v14r288` (B0 @ 288), `v14db` (blank windows), `v14ep20` (longer schedule); `v14tx` / `v14prog` if P-73 / P-74 are approved; later drop-path / EMA, slot layout | the single-model recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method. The epoch-budget variant answers the epoch-selection question too | **SESSION C READ 10-11 (`rsna-knee-train` v51, 7.56 h): `v14lr` 0.8609 (−0.0019, mixed folds, 🔁), `v14gd` 0.8607 (−0.0021, 5/5 folds down, ❌) vs `v14p` 0.8628 → no transfer arm** (experiments.md 2026-10-11 "P-67 session C read"); next `v14mx` / `v14r288` / `v14ep20` …, on Tian's go; 5.0 Kaggle GPU-h until 10-17. Was: **Tian's go 10-10 for session C** (`v14lr` ‖ `v14gd`, `rsna-knee-train`, the floor run's slot and image); traps 63's guard coded (`4faecd0`); smoke v50 green; **real session v51 pushed 14:43 UTC** (watcher `artifacts/watch_sC_1010.log`) | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67 |
| **T8** | **P-80 ConvNeXt seed twin `v15c2`** (= `v15c` at seed 43), then `v15c` + `v15c2` as one vote (`INFER_VOTE_GROUPS`, built 10-08) | the ConvNeXt family's seed spread; the final ConvNeXt vote | **READ 10-09: #70 `v15c2` 0.941 → s_c = 0.001, the bands stand; #71 B19 (the twins as one vote + `v13b3`) 0.944 = B18 → pick 1; P-80 closed ✅** (experiments.md "Submissions #70–#72"). **✅ DONE 10-08: trained 20:14–21:41 UTC, gold-58 SWA 0.9180 (`v15c` 0.9234; the pair's rank-mean 0.9234), shipped `rsna-knee-ckpt-v15c2` 21:42, backed up (`artifacts/kaggle_out/pod_v15c2/`), pod deleted, 1.59 pod-h ≈ $1.41** (experiments.md "P-80 on RunPod"). Ran on secure pod `nvjvi86joqceu9` (RTX 4090, $0.89/h, 100 GB local NVMe `/workspace`, `ssh root@69.145.85.70 -p 10179`; created 20:07:05 UTC, `POD_T0` 1791490025; repo `30c8a8d`; stopper 22:43 UTC = 2.6 h ≈ $2.3 cap; AUTO_STOP, `MAX_POD_H=2.6`; the chain's first step run in the foreground first, traps 61; cache verify 71 blobs / 0 bad / 4,407 studies; the T8 config confirmed in the training log: ConvNeXt-T @ 288, lr 1e-4, decay 0.9, 20 epochs, SWA 3, `train_all`, seed 43, Raptor mix 0.5). `_lastema.pt` was lost to AUTO_STOP (traps 56; unused). The first 10-08 attempt **died 40–80 s into the chain at 19:38 UTC** on community pod `kygt8m42pscb6y` (its AUTO_STOP trap fired in pip / the CUDA check; that host's fault, the package set re-tested clean; its log was removed with the container, traps 61; deleted 20:05). Was: launched on community pod `kygt8m42pscb6y` (RTX 4090, $0.34/h: no 4090 for 40 min, then Tian chose "retry community, then secure"), repo at `30c8a8d`, stopper deadline 22:28 UTC (3.0 h ≈ $1.0), AUTO_STOP; no public IP, so the local backup comes from the shipped Dataset (traps 60). Was: approved (Tian's go + critic GO), on hold until the evening of 10-08. Pod `88cdnd5fmxjqn6` was created 11:23 and deleted 11:33 during the cache pull (≈ $0.13, nothing trained). **Relaunch recipe:** create-pod (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, SECURE, RTX 4090, disk 40, persistent 100 GB at /workspace, ports 22/tcp + 8888/http, startSsh); check `stat -f -c %T /workspace` (10-08: local NVMe xfs → `CACHE_ROOT=/workspace/cache`; on MooseFS split per traps 55); `apt-get install bc git`, clone the repo into /workspace/repo, scp `~/.kaggle/credentials.json` (check its expiry ≥ 3 h away, traps 20), `ln -sfn /workspace/kaggle /kaggle`; stopper `runpod_stopper.sh /workspace/job_v15c2.log $((POD_T0 + 9360))`; export the pod key from /proc/1/environ, run the chain's first step in the foreground (traps 61), then `CACHE_PREFIX=rsna-knee-cache3 RSNA_TEACHER_TABLES='("raptor_teacher",)' AUTO_STOP=1 SEQ_ARMS=1 SHIP_TRIES=3 SHIP_WAIT_S=120 MAX_POD_H=2.6 POD_T0=<creation epoch> runpod_chain.sh v15c2` on a line of its own | ≈ 2.1 pod-h ≈ $1.9 at $0.89/h (cap 2.6 h ≈ $2.3) | P-80 |
| T11 | **`v15c320`** (arm built and pushed 10-08, `4b241bc`: `v15c` at 320 px; local smoke green) | the resolution probe | **NOT TRAINED (10-09): A8 #72 read `v15c` at 320 px 0.942 = at 288, under its ≥ 0.946 gate** (experiments.md "Submissions #70–#72"). Was: **HELD 10-08 by the critic:** 288 → 320 is only 1.11× finer and P-43's 1.43× step read +0.002, so it would read inside the floor and is really a third ConvNeXt vote for ≈ $1.4; the money is better kept for a ConvNeXt on the P-68 mix if that pair reads ✅ (10-10/11). The free read first: A8 | ≈ $1.4 | P-80 |
| **T12** | **P-81 `v15co`** (built and smoke-green 10-08, `8a0c1b2`): one RunPod 4090 job with `OAI=1` (download 7,196 OAI series ≈ 58 GB with nda-tools + build shard 90 on the pod, then train) | OAI as our own training data | **READ 10-11: A9 #78 `v15co` 0.945 vs `v15c` 0.942 → 🔁 +0.003, pick 2 only; P-81 closed; follow-up B21 → C4c** (experiments.md 2026-10-11 "Submission #78"). Was: **critic GO WITH CHANGES (10-08)**, applied: 3 labels, `.env` shredded after the download, OAI_MAX_PREP_H 1.2 abort, a US datacentre, stopper cap ≈ 4.3 h; the label-split infer read (`v15co` on its 3 labels + `v15c` on the other 9) is built: `INFER_MEMBER_LABELS` (unit-checked; {} = the old blend). **✅ DONE 10-10 (Tian's top-up + go): trained 13:46–15:57 UTC on secure 4090 `3lutummotrslpb` (US, CUDA 12.8, cgroup 10.2 CPUs, 200 GB NVMe, proxy-only SSH; created 13:08:27 after three broken community 4090s, traps 65), 6,748 studies (2,399 OAI: Lateral OA / PF OA 2,399, Synovitis 1,659 supervised), gold-58 SWA 0.9211; shipped `rsna-knee-ckpt-v15co` 15:57, backed up (`artifacts/kaggle_out/pod_v15co/`), pod deleted ≈ 17:45, $2.64** (experiments.md "P-81 on RunPod"). Recipe notes: inputs went as a 0.31 MB base64 bundle over the proxy (`image03.txt` trimmed 91 → 7 MB), repo `3c199ba` (traps 66), OAI ready 37 min after creation | 2.2 h trained, 2.8 pod-h billed: $2.64 | P-81 |
| **T13** | **P-82 DINOv2-S pair `v16d1` ‖ `v16d3`** (`0df06f3`; an LR bracket, Tian): DINOv2-S at 280 px on `v15c`'s data side, top block 1e-4 / LLRD 0.8 vs 3e-4 / 0.75, wd 0.05, drop path 0 → 0.1, 20 epochs | a third family (a ViT) for pick 1 and C3 | **READ 10-11 ❌: trained 7.35 h (`rsna-knee-train-b` v13; gold-58 SWA 0.9066 / 0.8604), shipped `rsna-knee-ckpt-v16d1` / `-v16d3`; #76 `v16d1` 0.932, #77 `v16d3` 0.893 → the better arm ≤ 0.934, no B20, DINOv2 stays dropped; the LR direction is low** (experiments.md 2026-10-11 "P-82 read"). Was: **Tian's go 10-10**; smoke `rsna-knee-train-b` v11 green (5.79 GiB peak at 280 px × 34 windows × 2 studies); timing smoke v12 (240 studies, both arms at once): 0.25–0.26 s / study ≈ 19 min / epoch → 20 epochs ≈ 6.5 h under the 8.3-h guard, so 20 stays; **real session v13 pushed 14:43 UTC** (watcher `artifacts/watch_sD_1010.log`) | ≈ 6.5 h, one session | P-82 |
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
