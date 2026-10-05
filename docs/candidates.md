# Candidates — what to score or train next

The queue of concrete models and ensembles, with what each one tests or contributes. **Read this when choosing the next
submissions or the next GPU run.**

How this file relates to the others:
- **Hypotheses and full read-rule reasoning** live in the cards of [proposals.md](proposals.md).
- **Results** live in [experiments.md](experiments.md).
- **This file is only the queue.** When a candidate is read, its row is deleted here and its score goes to experiments.md
  (Submissions table + Scoreboard) through `/update`. Never keep a score in two places.

Updated 2026-10-04 (18:20 UTC). The 10-05 five go out automatically at 00:00:30 UTC (see "Order"); 5 more on every later UTC day to the 10-22 deadline.

## Baselines every read is compared against

| | LB | What |
|---|---|---|
| Public-stack fork | **0.942** | #13 / #15 |
| Our best own | **0.940** | #50 `v13b3` solo = #53 B3 swap (`v11a` + `v13r` + `v13b3`). Previous best ensemble: #48 0.938 (`v11a` + `v13r` + `v13e`) |
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

Every row here is read against #48 0.938: ✅ ≥ 0.941 / 🔁 0.936–0.940 / ❌ ≤ 0.935, unless the row says otherwise. "Build" means a
new `rsna-knee-infer` placeholder (≈ 10 GPU-min; recipe at the bottom).

**Blend rule (10-05, n = 7, experiments.md "Submissions #49–#53"):** a flat rank-mean reads ≈ its members' mean solo LB + 0.002–0.005
(pairs +0.0015–0.0045, triples +0.0043–0.0047). It beats its best member only when the members are within ≈ 0.004 of each other.
Since B3 solo reads 0.940, every row with `v11a` (0.932), `v13h` (0.931) or `v13r` (0.934) predicts *under* B3 alone. "Pred."
below is that rule's range. Member solos: `v13b3` 0.940, `v13e2` 0.938, `v13e` 0.935, `v13r` 0.934, `v11a` 0.932, `v13h` 0.931.

| # | Members | What it tests / contributes | Placeholder | Gold-58 |
|---|---|---|---|---|
| B1 | `v11a` + `v13h` + `v13r` + `v13e` | Does a fourth member (ResNet-34) add to the trio? Pred. ≈ 0.935–0.938 (mean 0.933): **low value now** | **v45** ✅ | 0.9186 |
| B2 | `v13h` + `v13r` + `v13e` (CNN trio) | Is the CoAtNet needed? Read: ≥ 0.938 means it is not. Pred. ≈ 0.935–0.938 (mean 0.9333): **low value now** | **v46** ✅ | 0.9129 |
| B4 | `v11a` + `v13r` + `v13e` + `v13b3` | Does adding B3 lift our best ensemble? Pred. ≈ 0.937–0.940 (mean 0.9353) | build | 0.9250 |
| B5 | `v11a` + `v13r` + `v13e` + `v13e2` | Does a second B0 seed help inside the ensemble? Pred. ≈ 0.937–0.939 (mean 0.9348) | build | 0.9238 |
| B6 | `v11a` + `v13r` + `v13e` + `v13b3` + `v13e2` | Our strongest own lineup: the candidate for our own final pick (P-50) | **v51** ✅ · sent as #52 (ref 56838060), ⏳ | 0.9256 |
| B7 | `v11a` + `v13b3` | The two best families alone, as a pair. Pred. ≈ 0.938–0.940 (mean 0.936) | build | 0.9267 |
| B8 | all 8: `v11a`, `v13h`, `v13r`, `v13e`, `v13b3`, `v13e2`, `v13es`, `v13rs` | Does "everything" beat a curated 3–5? (Tian prefers 3–5) | build | 0.9217 |
| B9 | `v11a` + `v13rs` + `v13es` | #48 with the P-62 members swapped in | build | 0.9224 |
| B10 | `v13b3` + `v13e2` | **The strongest pair (new 10-05).** The blend rule's best bet: two members within 0.002. Pred. ≈ 0.941–0.944 (mean 0.939). Read vs `v13b3` 0.940: ✅ ≥ 0.944 / 🔁 0.937–0.943 / ❌ ≤ 0.936 | build | — |
| B11 | `v13b3` + `v13e2` + `v13e` | **The strong EfficientNet triple (new 10-05).** B10 + the seed-42 B0; three members, the more robust shape for the private split. Pred. ≈ 0.940–0.942 (mean 0.9377). Read as B10 | build | — |

## C. Submission candidates — the public stack plus our models (P-50)

| # | Candidate | What it tests / contributes | Read rule | Placeholder |
|---|---|---|---|---|
| C1 | Public 0.942 stack + #48 as our leg at β 0.45 | Does our own model set lift the public stack, as the 0.946–0.947 teams' own legs do? The biggest single stake for the final pick. **Scores slowly (hours, #17 ≤ 8 h): send first in a day** | vs 0.942: ✅ ≥ 0.945 / 🔁 0.941–0.944 / ❌ ≤ 0.940 | `rsna-knee-fork` **v11** ✅ · sent 10-05 as #49 (ref 56838006), ⏳ |
| C2 | Public stack + the best ensemble from B (e.g. B6) at β 0.45 | The same question with a stronger own leg | as C1 | build (`src/build_fork.py`) after C1 and B are read |
| — | Fork at other β | **Not planned:** it tunes a weight to the public LB | — | — |

## Order

- **10-05, Tian's pick (= my recommendation):** C1, A1, A2, B6, B3.
  - **Sent 00:41–00:45 UTC as #49–#53** (fork v11, infer v47, v48, v51, v52; each message carries its read rule). The scheduled
    00:00:30 run sent nothing: its client held a token that had expired at 19:17 (traps 20 addendum, fixed). The fallback run
    (`artifacts/auto_submit_1005r.log`) sent all five. Per-ref watchers: `artifacts/watch_<ref>.log`. Scores ⏳.
- **10-06:**
  - A3, the P-62 pair: it only needs reading before E is trained;
  - B1 and B2;
  - one follow-up chosen from the 10-05 reads, e.g. C2 if C1 ✅, or B5.
- **Then:** the remaining B rows, as the reads make them relevant.

## D. Training candidates (GPU)

Kaggle GPU is out until 2026-10-10 (3.33 h left: enough for placeholders, not a training session).

RunPod works at any time:
- **budget:** ≈ $7 left of Tian's top-up, about 9 pod-hours on an RTX 4090 at $0.74/h;
- **rule:** every RunPod run needs a written case checked by a critic subagent, then Tian's go;
- **measured speed** on a 4090: B0 ≈ 65 min for 30 epochs, B3 @ 288 ≈ 2.2 h, setup + c03 pull ≈ 20 min. ResNet-50 is estimated at
  ≈ 1.5 h.

| # | Arm(s) | What it tests / adds | Gate (wait for) | Est. cost | Card |
|---|---|---|---|---|---|
| T1 | Session E: `v13ec` ‖ `v13rc` (`claude_rap_v1` at mix 0.75) | Does the Claude relabel improve the CNNs? Read: their mean vs 0.9345 | A3, so E knows whether to add the silent mix | RunPod ≈ 2.6 h ≈ $2, or Kaggle ≈ 6 h after 10-10 | P-65 (the critic rated its expected gain under its own bar) |
| T2 | B3 at seed 43 (arm to add) | A second B3 seed: a 2-seed B3 member for the final ensemble | **gate met:** A1 ✅ (#50 0.940, 10-05); needs a critic-checked case + Tian's go | ≈ 2.4 h ≈ $1.8 | P-66 → P-50 |
| T3 | `v13r` at seed 43 (arm to add) | A 2-seed ResNet-50 member | B5 showing that seed averaging pays (A2 read 10-05: s = 0.003) | ≈ 1.8 h ≈ $1.3 | P-50 |
| T4 | B3 on the winning target (silent mix and/or Claude) | The best target on the best backbone | A3 / T1 ✅ | ≈ 2.4 h ≈ $1.8 | P-62 / P-65 |
| T5 | A bigger or new CNN on the `v13h` recipe (e.g. EfficientNet-B4, ConvNeXt-T) | Does more capacity keep paying? | A1 ✅ by a clear margin: **not met** (#50 0.940 is +0.0035 over the B0 seed mean, under the 0.004 bar) | ≈ 3–4 h + a weight Dataset + smoke | new card first |
| T6 | Final members (Kaggle, after 10-10) | Retrains of the chosen recipes and seeds for the two final picks | the reads above | Kaggle quota | P-50 |

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
