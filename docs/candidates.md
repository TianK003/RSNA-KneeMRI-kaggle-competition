# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-05 (06:30 UTC). All five 10-05 reads are in (#49–#53). 5 slots on every UTC day to the 10-22 deadline.

## Priority (Tian, 2026-10-05: gather the research first, then build; nothing is built yet)

| Prio | Candidate | Why it is next | Status |
|---|---|---|---|
| **1** | **C2** — the public stack + B6 as our leg at β 0.45 (section C) | The fork with the 0.938 trio read 0.943 (#49, rank 373); a 0.942 leg should add more. The highest-stake single read left for the final picks. ≈ 6 h to score: first send of its day | **built: fork v12, placeholder green** (Tian's go 10-05 after the research read); sends 10-06 00:00:30 UTC via `auto_submit.py` |
| **2** | **A3** — the P-62 pair, `v13es` + `v13rs` (section A) | Decides the target of every arm trained after it (silent mix or flat). Placeholders green | ready to send (two slots) |
| **3** | **T2 (+ T1 on the same pod)** — B3 at seed 43 on A3's winning target, optionally session E (section D) | Critic-vetted 10-05: de-biases the one 0.940 B3 draw, adds a RunPod-only final member; E settles the Claude-label question for ≈ $2 more. ≈ $4 of the ≈ $7 | case written; waits for A3, then Tian's go |
| **4** | **New families on the `v13h` recipe** (Kaggle after 10-10; section D, T5 / T6) | The blend rule says a sixth family at ≥ 0.935 lifts B6 more than another EfficientNet seed | pick the families from the 10-05 research |
| — | B1 / B2 / B4 / B5 / B7 / B10 / B11 | All predicted under B6 by the blend rule | hold |

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Public-stack fork | **0.942** | #13 / #15 alone; **0.943** with our #48 trio at β 0.45 (#49, rank 373) |
| Our best own | **0.942** | #52 = B6: flat rank-mean of `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` (= the public-stack fork). Best solo #50 `v13b3` 0.940 |
| Solos | **0.940** / 0.938 / 0.935 / 0.934 / 0.932 / 0.931 | `v13b3` / `v13e2` / `v13e` / `v13r` / `v11a` / `v13h` |
| CNN seed spread | s = 0.003 | #51 `v13e2` 0.938 vs `v13e` 0.935 (10-05): the one-seed bands stand (≥ 0.004). The B0 recipe's seed mean is 0.9365 |

**Floors.** A one-seed solo delta needs ≥ 0.004 (P-44). A two-arm mean uses ±0.0045. A blend counts only if it reads ≥ its best
member + 0.004.

**Gold-58 is direction only.** It has called the LB direction of recipe and target changes wrong several times (traps 39).

## A. Submission candidates — single models

| # | Candidate | What it tests / contributes | Decides | Read rule | Placeholder | Gold-58 |
|---|---|---|---|---|---|---|
| A3 | `v13es` + `v13rs` (P-62: Raptor 0.75 on report-silent cells) — **one read, two submissions** | Does weighting the image teacher on report-silent cells lift the CNNs? | Whether session E and the final members train with `TEACHER_SILENT_MIX` | mean of the two vs 0.9345: ✅ ≥ 0.9390 / 🔁 0.9300–0.9389 / ❌ ≤ 0.9299. On gold, both move less than a seed change, so 🔁 is likely | **v49** + **v50** ✅ | 0.9107 / 0.9160 |

## B. Submission candidates — ensembles of members we have

Since 10-05 every row here is read against **#52 (B6) 0.942: ✅ ≥ 0.946 / 🔁 0.939–0.945 / ❌ ≤ 0.938**, unless the row says otherwise
(B6 is the bar to beat for our own final pick). "Build" means a
new `rsna-knee-infer` placeholder (≈ 10 GPU-min; recipe at the bottom).

**Blend rule (10-05, all 10 flat blends with solo-read members, experiments.md "Submissions #49–#53" + its CORRECTED note):**
a flat rank-mean reads ≈ its members' mean solo LB + a gain that depends on family diversity and count:
- same recipe (seeds, small variants): + 0.001–0.0033 (#26, #29, #32, #36);
- cross-family: + 0.0025–0.0047 at 2–3 members (#23, #38, #47, #48, #53), + 0.006 at 5 (#52).
B0 and B3 count as same-recipe (within-class ρ 0.888 = a seed pair). "Pred." below is that rule's range. It is a working
rule, LB rounded to 0.001. Member solos: `v13b3` 0.940, `v13e2` 0.938, `v13e` 0.935, `v13r` 0.934, `v11a` 0.932, `v13h` 0.931.

| # | Members | What it tests / contributes | Placeholder | Gold-58 |
|---|---|---|---|---|
| B1 | `v11a` + `v13h` + `v13r` + `v13e` | Does a fourth member (ResNet-34) add to the trio? Pred. ≈ 0.937–0.939 (mean 0.933, n 4): **low value now, under B6** | **v45** ✅ | 0.9186 |
| B2 | `v13h` + `v13r` + `v13e` (CNN trio) | Is the CoAtNet needed? Read: ≥ 0.938 means it is not. Pred. ≈ 0.937–0.938 (mean 0.9333): **low value now, under B6** | **v46** ✅ | 0.9129 |
| B4 | `v11a` + `v13r` + `v13e` + `v13b3` | Does adding B3 lift our best ensemble? Pred. ≈ 0.939–0.941 (mean 0.9353, n 4). An ablation of B6 (B6 − `v13e2`): explains B6, cannot beat it | build | 0.9250 |
| B5 | `v11a` + `v13r` + `v13e` + `v13e2` | Does a second B0 seed help inside the ensemble? Pred. ≈ 0.939–0.940 (mean 0.9348, n 4). An ablation of B6 (B6 − `v13b3`) | build | 0.9238 |
| B7 | `v11a` + `v13b3` | The two best families alone, as a pair. Pred. ≈ 0.938–0.940 (mean 0.936) | build | 0.9267 |
| B8 | all 8: `v11a`, `v13h`, `v13r`, `v13e`, `v13b3`, `v13e2`, `v13es`, `v13rs` | Does "everything" beat a curated 3–5? (Tian prefers 3–5) | build | 0.9217 |
| B9 | `v11a` + `v13rs` + `v13es` | #48 with the P-62 members swapped in | build | 0.9224 |
| B10 | `v13b3` + `v13e2` | **The strongest pair (new 10-05).** The blend rule's best bet: two members within 0.002. Pred. ≈ 0.940–0.942 (mean 0.939, same-recipe gain). Strength vs count: does the best pair match the five? Read vs B6 0.942 (header bands). **Low value after the 10-05 correction** | build | — |
| B11 | `v13b3` + `v13e2` + `v13e` | **The strong EfficientNet triple (new 10-05).** B10 + the seed-42 B0; three members, the more robust shape for the private split. Pred. ≈ 0.941–0.942 (mean 0.9377, same-recipe gain). Read vs B6 0.942. **Low value after the 10-05 correction** | build | — |

## C. Submission candidates — the public stack plus our models (P-50)

| # | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|
| C2 | Public stack + **B6** (`v11a` + `v13r` + `v13e` + `v13b3` + `v13e2`) as our leg at β 0.45 | C1 (#49, the #48 trio as the leg) read **0.943** = +0.001 over the stack (🔁) and rank 373. A 0.942 leg should add more than a 0.938 one did. **Scores slowly: #49 took 5.3 h with three members; expect ≈ 6 h with five. Send first in its day** | vs #49 0.943: ✅ ≥ 0.946 / 🔁 0.942–0.945 / ❌ ≤ 0.941 | `rsna-knee-fork` **v12** ✅ (built 10-05, placeholder green 08:56 UTC: status `beta0.45`, our arm rc 0 in 69 s, 5 members in one decode-once pass, verified, constant labels 0). First entry of `artifacts/submit_plan_1006.json` |
| — | Fork at other β | **Not planned:** it tunes a weight to the public LB | — | — |

## Order

- **10-05, Tian's pick (= my recommendation):** C1, A1, A2, B6, B3.
  - **Sent 00:41–00:45 UTC as #49–#53** (fork v11, infer v47, v48, v51, v52; each message carries its read rule). The scheduled
    00:00:30 run sent nothing: its client held a token that had expired at 19:17 (traps 20 addendum, fixed). The fallback run
    (`artifacts/auto_submit_1005r.log`) sent all five. Per-ref watchers: `artifacts/watch_<ref>.log`. Scores ⏳.
- **10-06 (all 10-05 reads are in):**
  - **C2 first** (the fork + B6 leg; ≈ 6 h to score);
  - A3, the P-62 pair (two slots): decides the target of every arm trained after it;
  - the remaining two slots: B1 / B2 / B10 / B11 all predict under B6 (blend rule), so hold them unless a new member exists; a repeat is worth nothing.
- **Then:** the remaining B rows, as the reads make them relevant.

## D. Training candidates (GPU)

**Budget (2026-10-05).** Kaggle: 3.33 h left until 10-10, then 30 h on 10-10 and 30 h on 10-17; the quota counts *session* hours and
every session has two T4s (`PARALLEL_ARMS`), so ≈ 60 T4-GPU-h per week. RunPod: ≈ $7 ≈ 9 h on a 4090 ($0.74/h); every pod needs a
written case checked by a critic subagent, then Tian's go. Measured speeds: a 30-ep CNN arm ≈ 5.9 h on a T4 (B0 / R50 / R34), B0 ≈ 65
min and B3 @ 288 ≈ 2.2 h on a 4090; B3 @ 288 is RunPod-only (12–15 h and a memory risk on a T4). Deadline 10-22; final picks can be
changed until then (10-15 is the entry / merger deadline).

**The decision for Tian (research.md 2.10):** the 10-10 week goes either to **the ablation loop (T7)**, which can lift every final
member, or to **more members of known recipes (T6 only)**, which the blend rule prices at + 0.001–0.003. Recommendation: T7 + one
family arm (T8) in week 1, final retrains (T6) in week 2.

| # | Arm(s) | What it tests / adds | Gate | Est. cost | Card |
|---|---|---|---|---|---|
| **T7** | **P-67 loop**: proxy baseline × 2 seeds (the floor), then one variable per 5-fold run (aug components, 50 ep, head, drop-path / EMA, mixup, slot layout, smoothing) | the training recipe, judged on a pooled 5-fold report-label OOF; the forum's 0.95 teams' method | Tian's go on the 10-10 week | ≈ 5 variants per 9-h session, ≈ 16 per week; or $0.6 each on a 4090 | P-67 |
| **T8** | **P-69** ConvNeXt-T on the `v13h` recipe | a sixth family for B6 (+ 0.002–0.003 by the blend rule) | 10-10 quota; loader check first | 1 arm ≈ 6 h (shares a session with T7) | P-69 |
| T1 | Session E: `v13ec` ‖ `v13rc` (`claude_rap_v1` at mix 0.75) | does the Claude relabel improve the CNNs? Read: their mean vs 0.9345 | A3 (10-06), so E knows whether to add the silent mix | RunPod ≈ 2.6 h ≈ $2, or Kaggle ≈ 6 h | P-65 |
| T9 | **P-68** teacher arm: one production arm on LLM + Raptor + a CNN-family OOF (≥ 0.5 on silent cells) | the forum's multi-source pseudo-label gain | the A3 read + an OOF table (free from T7's best proxy) | 1 arm ≈ 6 h | P-68 |
| T6 | Final members (week of 10-17): B3 × 2 seeds (RunPod), B0 × 2, R50, the T8 family — all on the winning stack and target | the two final picks (B6-successor; the fork leg) | T7 / T9 / T1 reads | ≈ 30 Kaggle session-h + ≈ $4 RunPod for the B3s | P-50 |
| T2 | B3 at seed 43 **now** | de-biases the single 0.940 draw; the critic's case is written | folded into T6: the final B3 retrain on the winning recipe *is* the second draw. Run now only if A3 ✅ makes a new-target B3 urgent | ≈ $1.9 | P-50 |
| T3 / T4 | R50 seed 2; B3 on the winning target | — | = T6 | — | P-50 |
| ~~T5~~ | a bigger CNN (B4 @ 288 / 336) | **dropped 10-05:** the forum's ≥ 0.949 singles are R50-class at 224–288, "bigger is null", no gain above 288; the critic priced B4 @ 336 at ≈ 25 GB VRAM and ≈ $3.7 | — | — | → P-69 (a new family instead) |

**Dropped directions (research.md 2.10, item 5):** resolution > 288, B4-class capacity, 3D, MIL / bags, DINOv3 / RadImageNet / medical
foundation backbones, fork β or blend-weight tuning, geometric TTA, co-teaching, external datasets, multimodal-LLM image labelling.

## How to build and send a candidate

**Ensemble or solo placeholder** (one at a time; it needs a free GPU slot, traps 50):
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
- `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <N> -f submission.csv -m "<what + its read rule>"`;
- then `python src/watch_submission.py --ref <ref> --every 90`.
- After the read: `/update` (experiments.md row, card status), and delete the row here.
