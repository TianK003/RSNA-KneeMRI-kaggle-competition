# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-05 (12:15 UTC). All five 10-05 reads are in (#49–#53). Session E is trained and shipped. **Tian, 10-05 ≈ 12:00 UTC:
no training on 10-05 or 10-06; the next days only submit work we already have.** Training (section D) resumes at the 10-10 reset.
5 slots on every UTC day to the 10-22 deadline.

## Priority — every candidate, ranked, with its day

**P1** is queued. **P2** decides the own final pick and the composition of the week-2 retrains. **P3** is contingent or explains a
P2 read. **P4** is useful only if slots are free. **P5** is held: the blend rule prices it under B6, or a P2 row supersedes it.
"Build" means a `rsna-knee-infer` placeholder (≈ 5–10 GPU-min each; 3.07 h of Kaggle GPU is left until 10-10, enough for ≈ 15).

| Prio | Day | Candidate | What it decides | Gate | Status |
|---|---|---|---|---|---|
| **P1** | 10-06 | **C2**: the public stack + B6 as our leg at β 0.45 | Fork final pick (pick 2). ≈ 6 h to score: send 1 | — | fork **v12** ✅, queued |
| **P1** | 10-06 | **A4** `v13ecp`: 0.5 Claude dose | The Claude share of the week-2 retrains (with A5) | — | **v53** ✅, queued (send 2) |
| **P1** | 10-06 | **A3** `v13es` + `v13rs`: the P-62 pair, two sends | The silent mix for every later arm | — | **v49** + **v50** ✅, queued (sends 3–4) |
| **P1** | 10-06 | **A5** `v13ec`: 0.25 Claude dose | as A4 (the dose-response read) | — | **v54** ✅, queued (send 5) |
| **P2** | 10-07 | **B12**: B6 + every 10-06 solo that reads ≥ 0.936 | The own final pick's successor (pick 1) | ≥ 1 of the four 10-06 solos ≥ 0.936; else the slot goes to B8 | build after the 10-06 solos (≈ 01:00 UTC) |
| **P2** | 10-07 | **B14**: B6 − `v11a` (the four CNNs) | Is the CoAtNet family needed in pick 1 and the week-2 retrains? | — | build |
| **P2** | 10-07 | **B13**: B6 + `v11n` + `v11n2` (three CoAtNets, four CNNs) | Does more CoAtNet weight help? Read with B14 | — | build |
| **P3** | 10-07 | **B4**: B6 − `v13e2` | B3 or a second B0 seed inside the blend? Read with B5. Decides B3 × 2 seeds (RunPod) vs B0 × 2 (Kaggle) in week 2 | — | build |
| **P3** | 10-07 | **B9**: `v11a` + `v13rs` + `v13es` | #48 with the silent-mix members | A3 ✅ only; else the slot goes to B11 | build |
| **P3** | 10-08 | **C3**: the public stack + the best own blend as the leg, β 0.45 | Fork pick 2 with a better leg. Send 1 of its day | C2 ≥ 0.942 **and** a 10-07 own blend ≥ 0.943 (beats B6) | build (`src/build_fork.py`) on 10-07 |
| **P3** | 10-08 | **B5**: B6 − `v13b3` | the second half of B4's read | — | build |
| **P3** | 10-08 | **B15**: B12 + `v11n` + `v11n2` | B12 and B13 combined | B12 and B13 both read ≥ 0.943 | build |
| **P4** | 10-08 / 10-09 | **B8**: every member with a solo ≥ 0.930 | Count vs curation (Tian prefers 3–5) | free slot | build |
| **P4** | 10-08 / 10-09 | **B11**: `v13b3` + `v13e2` + `v13e` | The strong EfficientNet triple | free slot | build |
| P5 | hold | B1, B2, B7, B10 | — | B14 / B13 supersede B1 / B2; B7 / B10 predict under B6 | hold |
| — | 10-09 | **Freeze the shortlist for P-50** | pick 1 = the best own blend, pick 2 = the best fork | — | Tian, by 10-22 |

**What the blend rule expects** (section B): every P2–P4 row predicts 0.940–0.943, inside B6's 🔁 band. These sends are worth their
slots for what they decide (pick 1, the fork leg, the week-2 retrain composition), not as likely new bests. A slot left empty costs
nothing; a repeat of a read is worth nothing.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Public-stack fork | **0.942** | #13 / #15 alone; **0.943** with our #48 trio at β 0.45 (#49, rank 373) |
| Our best own | **0.942** | #52 = B6: flat rank-mean of `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` (= the public-stack fork). Best solo #50 `v13b3` 0.940 |
| Solos | **0.940** / 0.938 / 0.935 / 0.934 / 0.932 / 0.932 / 0.932 / 0.931 | `v13b3` / `v13e2` / `v13e` / `v13r` / `v11a` / `v11n` / `v11n2` / `v13h` |
| CNN seed spread | s = 0.003 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

All four are queued for 10-06 (pid 2104). Delete each row once its score is logged.

| # | Candidate | What it tests / contributes | Decides | Read rule | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| A4 | `v13ecp` (P-65, session E chain 1): `v13e` on 0.5 Raptor + 0.5 Claude, no LLM-blend share | Does the Claude relabel lift the CNNs at the full dose? | The Claude share of the week-2 retrains (with A5) | vs the B0 seed mean 0.9365: ✅ ≥ 0.9405 / 🔁 0.933–0.940 / ❌ ≤ 0.932 | **v53** ✅ (10:30: `smoke False`, `v13ecp/fold0` at 0.9063, decode-once verified, `constant labels 0`); send 2 | 0.9063 |
| A5 | `v13ec` (P-65, session E chain 2): 0.25 LLM + 0.5 Raptor + 0.25 Claude | The half dose: with A4 and the dose-0 seeds, a dose-response read | as A4 | as A4 | **v54** ✅ (11:50: `smoke False`, `v13ec/fold0` at 0.9116, decode-once verified, `constant labels 0`); send 5 | 0.9116 |
| A3 | `v13es` + `v13rs` (P-62: Raptor 0.75 on report-silent cells) — **one read, two submissions** | Does weighting the image teacher on report-silent cells lift the CNNs? | Whether the final members train with `TEACHER_SILENT_MIX` | mean of the two vs 0.9345: ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299. On gold, both move less than a seed change, so 🔁 is likely | **v49** + **v50** ✅; sends 3–4 | 0.9107 / 0.9160 |

## B. Submission candidates — ensembles of members we have

Since 10-05 every row here is read against **#52 (B6) 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938**, unless the row says otherwise
(B6 is the bar to beat for our own final pick). Inside 🔁, the pick-1 tie rule is the one used since #10: an equal score with more
members wins. "Build" means a new `rsna-knee-infer` placeholder (recipe at the bottom).

**Blend rule (10-05, all 10 flat blends with solo-read members, experiments.md "Submissions #49–#53" + its CORRECTED note):**
a flat rank-mean reads ≈ its members' mean solo LB + a gain that depends on family diversity and count:
- same recipe (seeds, small variants): + 0.001–0.0033 (#26, #29, #32, #36);
- cross-family: + 0.0025–0.0047 at 2–3 members (#23, #38, #47, #48, #53), + 0.006 at 5 (#52).
B0 and B3 count as same-recipe (within-class ρ 0.888 = a seed pair). The 10-06 target variants (`v13es`, `v13rs`, `v13ecp`, `v13ec`)
also sit at seed distance from their parents on gold (ρ 0.88–0.95). "Pred." below is that rule's range, LB rounded to 0.001. Member
solos: `v13b3` 0.940, `v13e2` 0.938, `v13e` 0.935, `v13r` 0.934, `v11a` / `v11n` / `v11n2` 0.932, `v13h` 0.931.

| # | Prio | Members | What it tests / contributes | Read rule (beyond the header bands) | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| B12 | P2 | B6 + every 10-06 solo ≥ 0.936 (of `v13es`, `v13rs`, `v13ecp`, `v13ec`) | The pick-1 successor. A qualifying member sits at or above B6's members' mean (0.9358), so it cannot lower it. Pred. ≈ B6 + 0.000–0.002; more if a solo reads ≥ 0.940 | ≥ 0.943 → the new pick 1 and the C3 leg | build on 10-06 after the solos | 0.9221 with all four |
| B14 | P2 | B6 − `v11a` = `v13r` + `v13e` + `v13b3` + `v13e2` | The CoAtNet ablation of B6. Pred. ≈ 0.940–0.941 (mean 0.9368, CNN-only diversity) | ≥ 0.942 → the CoAtNet is not needed: week 2 drops the CoAtNet retrain. ≤ 0.940 → it stays | build | 0.9211 |
| B13 | P2 | B6 + `v11n` + `v11n2` | Family weight: three CoAtNet votes (two `v11a`-recipe variants with heavy aug, 12 ep) against four CNNs. Pred. ≈ 0.941–0.942 (mean 0.9347, n 7) | ≥ 0.943 → more CoAtNet weight helps: week 2 adds a CoAtNet seed. Read with B14 | build | 0.9256 |
| B4 | P3 | `v11a` + `v13r` + `v13e` + `v13b3` (B6 − `v13e2`) | Pred. ≈ 0.939–0.941 (mean 0.9353, n 4). With B5: does B3 or the second B0 seed carry B6? | B4 − B5 ≥ 0.002 → B3 carries it: week 2 trains B3 × 2 (RunPod). ≤ −0.002 → B0 seeds suffice | build | 0.9250 |
| B5 | P3 | `v11a` + `v13r` + `v13e` + `v13e2` (B6 − `v13b3`) | Pred. ≈ 0.939–0.940 (mean 0.9348, n 4) | with B4 | build | 0.9238 |
| B9 | P3 | `v11a` + `v13rs` + `v13es` | #48 with the P-62 members swapped in | A3 ✅ only; vs #48 0.938: ≥ 0.941 → the silent mix also helps inside a blend | build | 0.9224 |
| B15 | P3 | B12 + `v11n` + `v11n2` | B12 and B13 combined | only if both read ≥ 0.943 | build | — |
| B8 | P4 | every member with a solo ≥ 0.930: B6 + `v11n` + `v11n2` + `v13h` + `v11d`, plus the 10-06 solos ≥ 0.930 | Does "everything" beat a curated 3–5? (Tian prefers 3–5) | — | build | 0.9248 (the nine without the 10-06 arms) |
| B11 | P4 | `v13b3` + `v13e2` + `v13e` | The strong EfficientNet triple; the robust three-member shape. Pred. ≈ 0.941–0.942 (mean 0.9377, same-recipe gain) | — | build | — |
| B1 | P5 | `v11a` + `v13h` + `v13r` + `v13e` | A fourth member (ResNet-34) on the trio. Pred. ≈ 0.937–0.939: under B6 | hold | **v45** ✅ | 0.9186 |
| B2 | P5 | `v13h` + `v13r` + `v13e` (CNN trio) | Is the CoAtNet needed? **Superseded by B14** (the same question on B6's own members) | hold | **v46** ✅ | 0.9129 |
| B7 | P5 | `v11a` + `v13b3` | The two best families as a pair. Pred. ≈ 0.938–0.940 | hold | build | 0.9267 |
| B10 | P5 | `v13b3` + `v13e2` | The strongest pair. Pred. ≈ 0.940–0.942 (same-recipe gain) | hold | build | — |

## C. Submission candidates — the public stack plus our models (P-50)

| # | Prio | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|---|
| C2 | P1 | Public stack + **B6** (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`) as our leg at β 0.45 | C1 (#49, the #48 trio as the leg) read **0.943** = +0.001 over the stack (🔁) and rank 373. A 0.942 leg should add more than a 0.938 one did. **Scores slowly: #49 took 5.3 h with three members; expect ≈ 6 h with five** | vs #49 0.943: ✅ ≥ 0.946 / 🔁 0.942–0.945 / ❌ ≤ 0.941 | `rsna-knee-fork` **v12** ✅ (placeholder green 08:56 UTC 10-05: `beta0.45`, our arm rc 0 in 69 s, 5 members in one decode-once pass, verified, constant labels 0); send 1 of 10-06 |
| C3 | P3 | Public stack + **the best own blend after 10-07** (B12, B13 or B15) as our leg at β 0.45 | Whether a better own leg lifts the fork further | Gate: C2 ≥ 0.942 and the leg read ≥ 0.943. vs C2: ✅ ≥ C2 + 0.003 / 🔁 within ±0.002 / ❌ ≤ C2 − 0.003 | build on 10-07 with `src/build_fork.py --members …`; send 1 of 10-08 |
| — | — | Fork at other β | **Not planned:** it tunes a weight to the public LB | — | — |

## Order

- **10-06 (queued, `auto_submit.py` pid 2104, 00:00:30 UTC):** C2 (fork v12) → A4 (v53) → A3a (v49) → A3b (v50) → A5 (v54).
- **10-06, after the four solos read (≈ 00:20–01:00 UTC):** build B12 (if a solo qualifies), B14, B13, B4 and B9 (if A3 ✅; else B11 or
  B8) as placeholders; write `artifacts/submit_plan_1007.json` in that order; start `auto_submit.py --at 2026-10-07T00:00:30Z` before
  23:30 UTC, with no other submitter alive.
- **10-07, after C2 reads (≈ 06:00 UTC) and the 10-07 blends read:** build C3 if its gate opens; then B5, B15 (if gated in), B8 / B11 for
  `submit_plan_1008.json` (C3 first).
- **10-09:** only rows that still decide something; otherwise leave the slots empty. Freeze the P-50 shortlist.
- **10-10:** training resumes (section D). The submissions after that are the new arms' solos.

## D. Training candidates (GPU) — paused until the 10-10 reset

**Paused (Tian, 2026-10-05 ≈ 12:00 UTC):** no training on 10-05 or 10-06; the days until 10-10 only submit. Kaggle GPU minutes go to
placeholders only.

**Budget (2026-10-05, 11:55 UTC).** Kaggle: **3.07 h** left until 10-10 (`kaggle quota`), then 30 h on 10-10 and 30 h on 10-17; the quota
counts *session* hours and every session has two T4s (`PARALLEL_ARMS`), so ≈ 60 T4-GPU-h per week. RunPod: **≈ $5** ≈ 6.5 h on a 4090
($0.74/h); every pod needs a written case checked by a critic subagent, then Tian's go. Measured speeds: a 30-ep CNN arm ≈ 5.9 h on a
T4 (B0 / R50 / R34), B0 ≈ 71 min and B3 @ 288 ≈ 2.2 h on a 4090; B3 @ 288 is RunPod-only (12–15 h and a memory risk on a T4).
Deadline 10-22; final picks can be changed until then (10-15 is the entry / merger deadline).

**The decision for Tian (research.md 2.10):** the 10-10 week goes either to **the ablation loop (T7)**, which can lift every final
member, or to **more members of known recipes (T6 only)**, which the blend rule prices at + 0.001–0.003. Recommendation: T7 + one
family arm (T8) in week 1, final retrains (T6) in week 2. The P2 / P3 reads of 10-07 / 10-08 (B14, B13, B4 / B5) set T6's composition.

| # | Arm(s) | What it tests / adds | Gate | Est. cost | Card |
|---|---|---|---|---|---|
| **T7** | **P-67 loop**: proxy baseline × 2 seeds (the floor), then one variable per 5-fold run (aug components, 50 ep, head, drop-path / EMA, mixup, slot layout, smoothing) | the training recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method. The epoch-budget variant answers the epoch-selection question too (experiments.md "Session E chain 2") | the 10-10 reset | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67 |
| **T8** | **P-69** ConvNeXt-T on the `v13h` recipe | a sixth family for B6 (+ 0.002–0.003 by the blend rule) | 10-10 quota; loader check first | 1 arm ≈ 6 h (shares a session with T7) | P-69 |
| T9 | **P-68** teacher arm: one production arm on LLM + Raptor + a CNN-family OOF (≥ 0.5 on silent cells) | the forum's multi-source pseudo-label gain | the A3 read + an OOF table (free from T7's best proxy) | 1 arm ≈ 6 h | P-68 |
| T6 | Final members (week of 10-17): B3 × 2 seeds (RunPod), B0 × 2, R50, the T8 family, a CoAtNet if B14 / B13 say so — all on the winning stack and target | the two final picks (B6-successor; the fork leg) | T7 / T9 reads; A3 / A4 / A5; B14 / B13 / B4 / B5 | ≈ 30 Kaggle session-h + ≈ $4 RunPod for the B3s | P-50 |
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
