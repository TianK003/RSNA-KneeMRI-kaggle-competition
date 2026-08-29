# Handoff — where we left off

**Update this at the end of every working session.** Newest entry on top. Keep it short and
concrete: what changed, what state things are in, what the next action is. This is the file
to read first after a break.

---

## 2026-08-30 (00:00) — `v06c` (ConvNeXt-Tiny, P-10) real fold-0 run IN FLIGHT as `rsna-knee-train` v15

Tian approved the launch at 23:55. This supersedes the "awaits go-ahead" entry below (same
session, minutes apart); everything else there still holds. **Read this entry, then the 23:35 one
for the blend results.**

### ⏳ Still in flight as this was written (00:00)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-train` v15** — arm `v06c` | **P-10**: ConvNeXt-Tiny backbone (HF `facebook/convnext-tiny-224`), concat head, `cache_jitter`, **8 epochs**, `ckpt_policy=best_oof`, `lr_backbone=1e-4` (LLRD 0.75 per stage), fold 0 only, from the cache. The first non-DINOv2 member | 2026-08-29 **23:57** | `kaggle kernels status tiankljucanin/rsna-knee-train` — expect `COMPLETE` around **02:00–02:15** (if the token has expired: `kaggle auth login`, traps 20). Then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v15 --file-pattern "(oof|no_match)"` | **First**: `cache: 4407 studies indexed` and the arm banner `v06c … backbone convnext_tiny … folds (0,) epochs 8`. **Throughput**: `N studies in Ns = X s/study` should read ~0.17–0.25 (the smoke's 3.51 on 4 studies was cuDNN warm-up); ~1.0+ means the cache path was not taken — stop and investigate. Eight `fold 0 epoch k:` lines; `EMA score … -> checkpoint = epoch k` / `not taken` lines show `best_oof` at work — the checkpointed epoch is whichever peaked. **Own OOF** (`auc_soft` at the checkpointed epoch) must be ≥ ~0.83 to be usable; DINOv2 concat sits at 0.847–0.851. If the run is `ERROR`, pull the log with `--file-pattern no_match` and read the tail |

### Where things stand

| | Status |
|---|---|
| Best LB | **0.896** (infer v5 = two heads; infer v8 = two heads + five concat folds; prefer v8). 1 submission was left for 2026-08-29; quota resets 02:00 local |
| P-10 | 🔧 code shipped (`c6ed3cd`), smoke green (train v14), **real run = v15 in flight** |
| ⚠️ Infer mounts | **v15 repoints infer v5/v8**: `rsna-knee-infer` mounts the *latest* `rsna-knee-train` output, which will now hold only `v06c_fold0_*`. Scores on the board are unaffected, but a **re-run/re-submit of the two-head blend would fail loudly** until the DINOv2 checkpoints (`v05a`/`v05b` from train v13, `v05g` from folds v4) are pinned to a Kaggle Dataset. Local copies of `v05a_fold0_best.pt` / `v05b_fold0_best.pt` are in `artifacts/kaggle_out/v13/` (88 MB each) — a Dataset can be built from them |
| GPU quota | ~13 h at 23:30 minus v15 (~2 h) → **~11 h** for the week (resets ~2026-09-05) |
| Repo | pushed; `crazy_good_rsna.ipynb` in the root is Tian's, untracked, not committed |

### What we talked about and decided

- Tian chose P-10 (a second architecture family) over the 9 h attention 5-fold after #6/#7 showed
  folds add nothing on top of head diversity. Go-ahead for the real run given at 23:55.
- HF ConvNeXt-**Tiny** via our own private Dataset (no official Kaggle Model exists; Tiny costs
  the same as ViT-S; LayerNorm-only so batch-of-1 is safe); one change per arm (head stays concat).
- Publishing hit two Windows CLI traps → traps 21.

### ⏭ Next action, in order

1. **Read v15** (table above). Then, on fold 0, the three-way check — same Python as the 23:35
   blend check, now with `c = artifacts/kaggle_out/v15/v06c_fold0_oof.csv`:
   own OOF; **ρ(c, v05a) and ρ(c, v05b)**; rank-mean a+b+c (equal votes) vs a+b **0.8670**.
   **Adopt** only if ρ < 0.77 against both **and** the 3-way blend clears **+0.008**, with own OOF
   ≥ ~0.83. Log with `/update` (P-10 card → pointer; experiments.md entry with the ρ table).
2. **If adopted**: pin the DINOv2 checkpoints to a Dataset (see the ⚠️ row), add
   `tiankljucanin/rsna-knee-train` (now holding v06c) + that Dataset + `rsna-knee-folds` to the
   infer kernel's sources, set `INFER_MEMBERS = ["v05a","v05b","v05g","v06c"]` via the sed, push,
   read the log (4 versions, `by_version` blend), submit once. Rule vs 0.896: < 0.005 is 🔁.
3. **If ρ ≈ 0.84** (another concat-like profile): the family bet fails at this size; next
   diversity bets are input geometry (P-11 336 px, P-08 more slices) or DINOv3, not more of this.
4. Regardless: make `convnext-tiny-224-hf` public before any final submission uses it.

### Open decisions for Tian

- Dataset pin for the DINOv2 weights (needed for any new two-head submission).
- `crazy_good_rsna.ipynb`: read for the 0.056 gap, ignore, or delete.
- Delete `artifacts/kaggle_out/v9smoke/` (1.8 GB).

### Things that will bite if forgotten

- **Token expires ~12 h after `kaggle auth login`** (last login 22:01 on 08-29 → expect failure
  ~10:00 on 08-30; the error text blames the slug — traps 20).
- **Mid-run logs are browser-only**; `kernels output`/`logs` are blank until the run ends.
- **Infer v5/v8 now point at v06c's output** (⚠️ row). **Infer v7 never** (flat 7-member).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True`; v15 ran the sed'd `False`.
- Everything in the 23:35 / 22:35 lists still applies.

---

## 2026-08-30 (00:05) — P-10 ConvNeXt-Tiny member implemented and smoke-green; real fold-0 run awaits go-ahead

Code through the "P-10: ConvNeXt-Tiny as a second backbone family" commit. Nothing new is
measured; this entry is state. The 23:35 and 22:35 entries below still hold.

### ⏳ Still in flight — nothing running

`rsna-knee-train` v14 (smoke, `COMPLETE`), no submission pending. **The real `v06c` run has NOT
been launched** — the try-out rule requires Tian's explicit go-ahead in a fresh message.

### Where things stand

| | Status |
|---|---|
| Best LB | **0.896** (infer v5 = two heads; infer v8 = two heads + five concat folds — prefer v8 as the default blend) |
| P-10 code | ✅ `Config.backbone` ∈ {`dinov2`, `convnext_tiny`}; `BACKBONES` resolves weights per family; `KneeNet` loads either; LLRD per ConvNeXt stage; infer members carry their family |
| P-10 weights | ✅ private Kaggle Dataset `tiankljucanin/convnext-tiny-224-hf` (HF `facebook/convnext-tiny-224`, Apache-2.0, 111 MB); mounted at `/kaggle/input/convnext-tiny-224-hf` in v14. **Must become public (or be replaced by an official Model) before a final submission relies on it** |
| P-10 smoke | ✅ local CPU + Kaggle v14 green: cache indexed, arm banner `v06c … backbone convnext_tiny`, `backbone LR range 2.37e-05 .. 1.00e-04 over 4 blocks`, inference `v06c/fold0 (convnext_tiny, concat)` 8 s/100 studies |
| GPU quota | ~13 h left this week (v14 smoke ≈ 0.1 h) |
| Repo | pushed; **`crazy_good_rsna.ipynb` in the root is Tian's, untracked, deliberately not committed** |

### What we talked about and decided

- After #6/#7 (folds +0.009 alone, +0.000 on top of heads) Tian chose the "third source of diverse
  errors" — P-10 — over the 9 h attention 5-fold.
- **HF ConvNeXt-Tiny over timm**: no official HF/timm ConvNeXt exists on Kaggle Models; publishing
  the Apache-2.0 HF checkpoint as our own Dataset keeps the code path identical to DINOv2
  (`from_pretrained(dir)`), needs no new dependency, and avoids the BatchNorm-at-batch-1 problem
  the card warned about (ConvNeXt is LayerNorm-only).
- **Tiny, not Small**: 4.5 GFLOPs ≈ ViT-S/14's 4.6, so the arm costs the same ~0.17–0.2 s/study;
  Small would double it for an unmeasured gain.
- **8 epochs under `best_oof`**, concat head, jitter, `lr_backbone = 1e-4` (card value; an
  ImageNet-supervised CNN tolerates ~5× the LR DINOv2's SSL features need). One change per arm:
  the head stays concat so the comparison to `v05b`/`v05g` isolates the family.
- Kaggle CLI dataset upload has two Windows traps (title ≤ 50 chars; run `kaggle datasets create
  -p .` from *inside* the directory or it builds a bad temp path) — noted below, not yet in traps.md.

### ⏭ Next action, in order

1. **On Tian's go-ahead, launch the real `v06c` run** (~1.8–2.0 h; the smoke's 3.51 s/study on 4
   studies is CUDA/cuDNN warm-up, not throughput — inference ran at the ViT members' speed):
   ```bash
   sed 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' src/kaggle_pipeline.py > /tmp/train_real.py
   python src/nbgen.py /tmp/train_real.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
   kaggle kernels push -p kaggle/rsna-knee-train        # -> v15
   # 10-min gate (browser, the CLI is blind mid-run): s/study ~0.17-0.25; ~1.0+ = something is wrong
   ```
   ⚠️ **Pushing `rsna-knee-train` changes what infer v5/v8 mount** (`kernel_sources` = latest
   output). v15's output will hold only `v06c_fold0_*`, so **infer v5/v8 would fail loudly**
   (`no v05a_fold*_best.pt is mounted`) if re-run — the 0.896 scores already on the board are
   unaffected, but a *new* submission of the two-head blend needs the weights pinned first
   (Dataset) or `rsna-knee-train` re-pushed with the old arms. Do the Dataset pin **before** or
   right after this launch.
2. **Read `v06c`** when it finishes: pull `--file-pattern "(oof|no_match)"` into
   `artifacts/kaggle_out/v15/`, then on fold 0 compute (the same Python as tonight's blend check):
   own OOF at the checkpointed epoch; ρ against `v05a` and `v05b`; rank-mean a+b+c (by-version,
   equal votes) vs a+b 0.8670. **Adopt** only if ρ < 0.77 against both **and** the 3-way blend
   clears +0.008; own OOF should be ≥ ~0.83. Log via `/update`; if adopted, add `v06c` to
   `INFER_MEMBERS`, push infer (with the train mount holding v06c and the folds mount holding
   v05g — the v05a/v05b checkpoints then need the Dataset pin), submit once.
3. If it fails on ρ (≈0.84 like the concat variants): the family is not diverse enough at this
   size; the next diversity bets are a different input geometry (P-11 336 px, P-08 more slices)
   or DINOv3, not more of the same.

### Open decisions for Tian

- Go / no-go on the ~2 h `v06c` run (item 1) and, with it, the Dataset pin for the DINOv2 weights.
- Make `convnext-tiny-224-hf` public (needed for any final submission that uses it).
- What to do with `crazy_good_rsna.ipynb` — read it for the 0.056 gap, ignore it, or delete.

### Things that will bite if forgotten

- **The 10-minute throughput gate is browser-only** (CLI `output`/`logs` are blank mid-run).
- **Pushing `rsna-knee-train` repoints infer v5/v8** — see item 1.
- `kaggle datasets create`: title ≤ 50 chars; run from inside the directory with `-p .`.
- All of the 23:35 / 22:35 lists (infer v7 never; token expires ~12 h; `v05f` never mount).

---

## 2026-08-29 (23:35) — #6/#7 scored: folds +0.009 alone, +0.000 on top of heads; attn 5-fold NOT launched

Supersedes the in-flight table of the 22:35 entry below; everything else there still holds.
Findings logged in experiments.md (Submissions #6/#7, Scoreboard, RESOLVED note on the 5-fold entry),
proposals.md (P-10 raised, P-13 supported), CLAUDE.md state line.

### ⏳ Still in flight — nothing

No kernel is running and no submission is pending. Background watchers from this session are dead.

### Where things stand

| | Status |
|---|---|
| Submissions | **seven**; best **0.896** twice (#5 infer v5 = two heads; #7 infer v8 = two heads + five concat folds). #6 (five folds alone) 0.886. 1 submission left today (resets 02:00) |
| Decision taken | **The 9 h attention 5-fold is not launched** — its gate (#7 > #5 by ≥ 0.005) failed by exactly 0.000 |
| GPU quota | ~13 h left this week (resets ~2026-09-05) |
| Repo | pushed through this entry |

### What we figured out (tonight's two numbers)

1. **Five folds of one model: +0.009 LB** (0.877 → 0.886), 1.8× the floor. Real, but half the
   second head's +0.019. The OOF→LB offset does **not** apply to fold ensembles (+0.039 here) —
   pooled OOF cannot see variance reduction.
2. **Five folds on top of the head blend: +0.000** (0.896 → 0.896). Fold-averaging and
   head-blending remove the same variance; once both heads are in, folds of one head are redundant.
   Caveat: the attention vote also fell 1/2 → 1/3 in #7, so a small gain may have cancelled a small
   dilution — deliberately *not* chased with more submissions (public-LB weight tuning is the trap).

### ⏭ Next action, in order

1. **A third diverse member, fold 0 first (~1 h), not more folds.** P-10 with a licence-clean timm
   ConvNeXt (Tiny or Small, 224, ImageNet weights) as an `ARMS` arm on `rsna-knee-train`, same
   cache/slots/jitter/EMA, 4 epochs first. Decision rule from P-21's own template: report the
   rank correlation of its fold-0 OOF against `v05a` and `v05b` (the two heads sit at 0.77) and the
   fold-0 rank-mean gain over 0.8670 (a+b). Adopt only if the 3-way blend clears +0.008 OOF **and**
   ρ < 0.77 against both — a member that is merely another concat-like error profile adds nothing,
   as #7 just showed. Smoke first; `FORCE_SMOKE = True`; check the cache line.
   Cheaper alternatives if ConvNeXt is more than an evening of code: an attention head at a
   different schedule (4 epochs, ~0.9 h) or a different seed of `v05a` — each is a weaker diversity
   bet (same backbone, ρ likely ≥ 0.84 like `v05b`–`v05g`).
2. **Pin the submission weights to a Kaggle Dataset** before any further push to `rsna-knee-train`
   or `rsna-knee-folds`: infer v5/v8 mount the *latest* outputs of both slugs.
3. **Final-submission hygiene**: v5 and v8 both score 0.896; v8 contains more models and is the
   safer private-LB bet (five folds average site variance even if the public LB cannot see it) —
   prefer v8's blend as the default going forward, and add any new member via `INFER_MEMBERS`.

### Open decisions for Tian

- ConvNeXt (P-10) vs cheaper same-backbone diversity for the remaining ~13 h this week.
- Kaggle Dataset pinning (browser or `kaggle datasets create`).
- Delete `artifacts/kaggle_out/v9smoke/` (1.8 GB of smoke checkpoints with real names).

### Things that will bite if forgotten

- All of the 22:35 entry's list, plus: **infer v7 must never be submitted** (flat 7-member; fold 0
  says 0.8611); the two 0.896 kernels are **v5** and **v8**.

---

## 2026-08-29 (22:35) — LB 0.896 from the two-head blend; valid 5-fold done; two blends submitted and pending

Findings logged through `cad29d7`. This entry is state only.

### ⏳ Still in flight as this was written (22:35)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #6** — ref `55874877` | `rsna-knee-infer` **v6**: `INFER_MEMBERS=["v05g"]` — the five concat folds alone, rank-meaned | 22:30 | `kaggle competitions submissions rsna-knee-abnormality-detection --csv \| head -4` (a background watcher was polling; it dies with the session) | **This is the fold-ensemble gain**, read against `v04d`'s one-fold **0.877** (same recipe). ≥ 0.882 = folds are worth having on their own; 0.877–0.882 = 🔁 (inside the LB floor); < 0.877 would mean the 4-epoch fold models do not average well and needs a look at the per-fold OOF csvs in `artifacts/kaggle_out/folds_v4/`. Expect the +0.02–0.03 OOF→LB offset on pooled 0.8467 → roughly 0.87–0.88 *if* five folds add nothing, higher if they do |
| **Submission #7** — ref `55874878` | `rsna-knee-infer` **v8**: `INFER_MEMBERS=["v05a","v05b","v05g"]`, `INFER_BLEND="by_version"` — attn + concat-8ep + 5-fold concat, one vote per version | 22:30 | same command | Read against **#5 = 0.896**. The fold-0 proxy predicts only +0.001 OOF, so a move under 0.005 either way is 🔁 and the two-head blend stays the reference. A clear gain says folds add on top of heads → the attn 5-fold run is the next spend. A clear loss says the 4-epoch concat folds dilute → drop `v05g` from the blend or weight it down. **Infer v7 (flat 7-member) exists and must not be submitted** — fold 0 says it would score below #5 |

Both submissions run the same notebook, so a scoring error on one is a scoring error on both; a
missing score with status `ERROR` means the rerun crashed — pull `kaggle kernels output
tiankljucanin/rsna-knee-infer -p artifacts/kaggle_out/x --file-pattern no_match` and read the tail.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **seven**: 0.500 → 0.841 → 0.871 → 0.877 → **0.896** (#5, P-21 two-head blend) → #6 ⏳ → #7 ⏳. 1 submission left today (resets 02:00 local) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF 0.8574 |
| Best OOF blend (fold 0) | ✅ a+b 0.8670 → LB 0.896; a+b+g by-version 0.8680 (submitted as #7) |
| 5-fold ensemble | ✅ **`v05g` valid** (`rsna-knee-folds` v4, 4.27 h): per-fold 0.843–0.851, pooled 0.8467, gold 0.8476 on all 58. `v05f` (v2) = invalid, never mount |
| Checkpoint policy | ✅ switched to `ckpt_policy="best_oof"` (P-22); inert at 4 epochs, matters for concat past epoch 4 |
| Infer path | ✅ `INFER_MEMBERS` × `INFER_BLEND="by_version"`, decode-once with in-kernel equality check; infer kernel mounts `rsna-knee-train` **and** `rsna-knee-folds` |
| Kaggle layout | ✅ observed and printed at startup: kernel outputs at `/kaggle/input/notebooks/<owner>/<slug>/` (platform-wide since today) |
| GPU quota | ~19 h at 17:30 → **~13 h left** for the week after `v05g` (4.3 h) + three infer runs + smokes |
| Local env | ✅ `.venv` (torch 2.13.0+cpu, scikit-learn), `requirements.txt` pinned |
| Repo | ✅ clean, pushed through `cad29d7` |

### What we talked about and decided

- **Verified the traps-6f diagnosis before spending anything**: the folds smoke log (local copy +
  Tian's paste) and then v2's own log (`no cache is mounted`, 1.17 s/study, fold-0 OOF 0.8198 =
  the v02 number). Tian stopped v2 at ~10 h wall-clock to save quota; the log's own clock read 8.0 h
  — it had queued ~2 h, so "hours since push" overestimates execution.
- **Reordered the handoff's plan**: P-22 (local) before P-21 before the folds re-run, because P-22
  could change the checkpoint policy of the run about to launch (it did) and the slug was busy.
- **Decode-once was included in the P-21 change** (Tian's call), verified by array equality in the
  kernel, and the arm was renamed `v05g` so the invalid `v05f` files can never be confused.
- **GPU strategy with 19 h/week**: spend ~4.5 h on the concat 5-fold now, hold the ~9 h attn 5-fold
  until the fold-ensemble number is in. Still the plan; #6/#7 decide it.
- **Did not submit the flat 7-member blend (v7)** after the fold-0 weighting check showed it below
  the two-head blend; built the by-version rule instead and submitted that (v8) plus the 5-fold-alone
  kernel (v6), both approved by Tian.
- **Rejected tuning blend weights on fold 0** (attn 2:1:1 = 0.8688 vs 0.8680 by-version — inside
  the floor); by-version is the principled, unfitted rule.

### What we figured out

1. **P-21 transferred: LB 0.896 (+0.019, 3.8× the floor)** from two heads on one backbone, one
   fold, and the OOF→LB offset (+0.029) held for a blend — n=4 now. → experiments.md Submissions.
2. **P-22: best-OOF checkpointing is +0.0128 split-half for the concat head, ~0 for attn, no
   teacher-chasing; P-09 becomes a tie (−0.0024 at best epochs)** — the +0.0103 was concat's late
   decay. Policy switched. → experiments.md "P-22", `src/oof_epoch_analysis.py`.
3. **The valid 5-fold run**: fold spread 0.843–0.851 (one floor), fold 0 is representative, `v04d`
   reproduced within 0.005, pooled 0.8467. → experiments.md "First valid 5-fold run".
4. **Vote weighting beats member count**: a third same-head member adds +0.001 on fold 0; giving
   the concat side 6/7 of the vote *loses* 0.007 vs the two-head blend. Hence `INFER_BLEND`.
5. **Kaggle changed `/kaggle/input` for everyone today** (`notebooks/<owner>/<slug>/`), not just new
   slugs; traps 6f corrected, layout printed in every log. → traps 6f.
6. **Per-version kernel output is not retrievable** (`<slug>/11` returned v13's files) and
   `kernels logs` is blank mid-run; the v11 OOF csvs are gone. → traps 12e.
7. **A local smoke can "resume" from a stale `_last.pt` and train nothing**; smoke mode no longer
   resumes. → traps 19.
8. **The Kaggle OAuth token expires after ~12 h** with a misleading "wrong slug" error. → traps 20.
9. From v2's wreckage: a second seed-pair of the v02 recipe differs by 0.002–0.009 per epoch
   (corroborates the 0.008 floor), and fold 1 ≈ fold 0 at epoch 0. → experiments.md (invalid-run
   entry is unchanged; noted here only).

### ⏭ Next action, in order

1. **Read #6 and #7** (command in the in-flight table) and log both in experiments.md's
   Submissions table + Scoreboard via `/update`, using the rules written in those rows. Then decide:
   - #7 > #5 by ≥ 0.005 **and** #6 ≥ 0.882 → **launch the attn 5-fold** (`ARMS` arm
     `("v05h", {"head_type": "attn", "cache_jitter": True, "folds": (0,1,2,3,4), "epochs": 8})`
     via the `FIVE_FOLD` block — 5 × 8 × ~12.8 min ≈ 8.5 h, so raise `runtime_limit_hours` to 8.8
     for that run or plan a resume; ~9 h of the ~13 h quota). Smoke first (`FORCE_SMOKE = True`,
     gate on `cache: 4407 studies indexed`), then real.
   - #7 ≈ #5 (within 0.005) → folds add nothing on top of heads; spend the quota on a second
     *head* or *schedule* member instead (attn 4-ep fold 0 is cheap: ~0.9 h), not on more folds.
   - #7 < #5 → drop `v05g` from `INFER_MEMBERS` (or weight it down) before any further submission;
     #5's kernel (infer v5) remains the best submitted.
2. **Pin the submission's weights to a Kaggle Dataset** (open since two sessions): `rsna-knee-infer`
   mounts the *latest* `rsna-knee-train` and `rsna-knee-folds` outputs; any new push to either slug
   changes what a re-run of the infer kernel loads. Do this before the next training push to
   `rsna-knee-train`.
3. Only then P-10 / P-14 / P-15 / P-16.

### Open decisions for Tian

- The attn 5-fold (~9 h of the remaining ~13 h this week) — gated on #6/#7 as above.
- Kaggle Dataset pinning (item 2) — a browser or `kaggle datasets create` job.
- `artifacts/kaggle_out/v9smoke/` (1.8 GB of smoke checkpoints with real-run filenames) is still
  safe to delete; also `artifacts/kaggle_out/v11oof/` was removed this session (it held v13 copies).
- Browser-only questions still block P-10/P-16/P-18: rules text, Efficiency Prize formula,
  radimagenet.com T&C, hidden test size.

### Things that will bite if forgotten

- **`kaggle auth login` every ~12 h** (traps 20); the failure looks like a wrong slug.
- **Kernel outputs live under `/kaggle/input/notebooks/<owner>/<slug>/`** now; every resolver
  searches depth 4 and the layout is printed at the top of each log — read it.
- **Infer kernel versions**: v5 = a+b (0.896), v6 = 5-fold alone, v7 = flat 7 (**do not submit**),
  v8 = by-version 7. The committed `.ipynb` is v8's; `INFER_MEMBERS` in `src/` still defaults to
  `["v05a","v05b"]` and is sed'd per push like `MODE`.
- `kaggle kernels output <slug>/<version>` returns the **latest** version; download small results
  with the log at run time (traps 12e).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True`; the real kernels had `False`.
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by filename,
  never edit the `.ipynb`, `experiments.md` is append-only).

---

## 2026-08-29 (17:11) — The 5-fold run was invalid, not slow: the cache never mounted

Findings logged as of `f50e6a4`. This entry is state only. **Read traps 6f first** — it is the
most transferable thing this session produced.

### ⏳ Still in flight as this was written (17:11)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-folds` v2 — ABANDON IT** | Was meant to be 5 folds × 4 epochs of the `v04d` recipe. It is **training the wrong recipe** (v02 decode path — no crop, no laterality, no per-series norm) because the cache never mounted | 2026-08-29 ~08:00 | `kaggle kernels status tiankljucanin/rsna-knee-folds` | **Do not use its output for anything.** ~9.2 h in; the 8.3 h guard has been reached, and the tail is a post-guard validation pass at ~1 s/study over 882 studies (~15 min) plus checkpoint writes. When it stops, `all folds complete: False`, inference skipped by design. `v05f_fold*` checkpoints are v02-recipe models — **delete them mentally; do not mount them.** Nothing needs to be salvaged |

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **four**: 0.500 → 0.841 → 0.871 → **0.877** (`v04d`) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF **0.8574** — not yet submitted |
| Best OOF of any kind | ✅ **0.8670** — rank-mean of `v05a` + `v05b` on fold 0 (P-21), not yet submitted |
| Noise floor | ✅ measured: 0.008 macro, ~0.03 per label |
| P-09 / P-04 / P-05 / P-08-jitter | ✅ all closed — see experiments.md |
| **5-fold ensemble** | ❌ **wasted run, must be redone** (traps 6f) |
| Cache loader | ✅ fixed in `src/` (depth 4, fallback now fatal) — **not yet exercised on Kaggle** |
| Repo | ✅ clean, pushed through `f50e6a4` |

### What we talked about and decided

- **Ran all five prior suggestions** (submit `v04d`; 8-epoch P-09 retest; 5 folds; the direct
  worker-RNG check; P-04 as the retest's control), two kernels concurrently.
- **The 8-epoch retest shipped a matched control** (`v05b`) rather than comparing back to
  `v04d`, so head and schedule could not confound each other again.
- **P-10 de-prioritised** behind the new P-21: head diversity is free, a CNN member costs a
  session, RadImageNet's licence is unresolved.
- **Decided to ask rather than auto-launch the re-run.** ~9 h of GPU was just burned; the
  re-run is another ~4.5 h and the quota is shared.

### What we figured out

1. **THE BIG ONE — a new kernel slug does not mount inputs at the same depth as an old one.**
   `load_cache_manifests` globbed at `max_depth=2`; `rsna-knee-train` mounts the cache at
   `/kaggle/input/rsna-knee-cache-a/…` (depth 2) but the newly created `rsna-knee-folds` mounts
   it type-prefixed at depth 4, like the datasets. It was **the only glob in the file capped at
   2** — every other resolver uses 3 or 4. → **traps 6f**.
   - **Cost:** `cfg.use_cache` flipped to `False`, the dataset silently took the **v02 decode
     branch**, and ~9 h of GPU trained a superseded recipe at 0.99 s/study. Five folds at that
     rate is ~19.6 h — it could never have finished.
   - **How it was found:** Tian opened the kernel page and pasted the **smoke log**, which says
     `! use_cache=True but no cache is mounted` in plain text at line 61. That log had already
     been read once, for the arm banners and the RNG check, and the cache line was skipped.
   - **Why it went unnoticed for six hours:** the overrun *was* the evidence and it was
     explained away. At 14:38 the previous handoff recorded "~6.6 h in against a ~4.5 h estimate
     — the estimate was wrong, the run is not." A 1.7× miss against a throughput figure measured
     three separate times that day should have been treated as a symptom, not as estimator
     error. → experiments.md, "The first 5-fold run was invalid, not slow".
   - **Fixed:** glob depth 2 → 4, and a missing cache in train mode is now a `SystemExit` unless
     `ALLOW_DECODE_FALLBACK = True`.
   - **Second incident in one day from the same root cause:** `use_cache` means both *which
     preprocessing* and *is the file present*. traps 6d was the inference-side twin.
2. **Submission #4 = 0.877, a new best**, and a third OOF→LB point at +0.024 (three for three in
   the +0.02–0.03 band). The LB delta is only 1.2× its floor; jitter is carried by its
   11-of-12 per-label sign pattern. → experiments.md.
3. **P-09 ✅ KEEP (+0.0103 at matched 8 epochs) but the card's reasoning was wrong** — it loses
   on the plane-specific labels it was predicted to help (MCL −0.040, Lateral Meniscus −0.032);
   what it buys is overfit resistance. → experiments.md.
4. **The two heads rank-blend to OOF 0.8670** at ρ = 0.773 — free error diversity, no second
   backbone. → new card **P-21**, now the highest-value untested item.
5. **P-04: 8 epochs does not beat 4** for the concat head; the epoch count is **head-specific**.
6. **traps 6e was wrong and is corrected** — PyTorch already seeds numpy/`random` per worker, so
   the fork pathology does not exist here and `v04d`'s jitter was never confounded.
7. **traps 12e was wrong and is corrected** — per-version kernel output *is* retrievable via
   `<slug>/<version>`; a **run in flight on that slug** is what blocks it.
8. **New trap 4b** — the cosine schedule spans `cfg.epochs`, so epoch N of a 4-epoch run and
   epoch N of an 8-epoch run are at different LRs and are not comparable.

### ⏭ Next action, in order

1. **Do P-21 first — it needs no GPU.** In the inference block, `ckpt_paths` maps
   `fold -> path` for one `cfg.version`; make it accept (version, fold) pairs so `rank_mean`
   blends across *versions*. Then submit the `v05a` + `v05b` blend from `rsna-knee-infer`.
   Decision rule: expect a **small** LB move; a sub-0.005 change is **not** confirmation, and
   the +0.02–0.03 OOF→LB offset was calibrated on single models so ~0.891 is **not** a
   prediction. Push the infer kernel **before** any new `rsna-knee-train` push.
2. **Re-run the 5 folds with the fixed loader.** `sed 's/^FIVE_FOLD = False/FIVE_FOLD = True/'`,
   `FORCE_SMOKE = True` first, push, and **before promoting to real, grep the smoke log for the
   cache line**:
   ```bash
   kaggle kernels output tiankljucanin/rsna-knee-folds -p artifacts/kaggle_out/x --file-pattern "no_match"
   # REQUIRED in the log:  cache: 4407 studies indexed (c01_p224_s16_crop130_lat20)
   # If instead you see "no cache is mounted", it now raises - but check anyway.
   ```
   Then `FORCE_SMOKE = False` and push. **Sanity gate at ~10 min in: `s/study` must read
   ~0.17–0.18.** If it reads ~1.0, kill it immediately — that is the decode path again.
3. **Run P-22** (~0.1 session, analysis only, no training): best-epoch vs last-epoch OOF from
   the `_ep{e}_oof.csv` files already in `artifacts/kaggle_out/v13/`, with the gold curve
   alongside to detect teacher-chasing. `v05b` ships 0.013 below its own peak, and this choice
   silently decides P-09's verdict.
4. Only then: P-10 / P-14 / P-15 / P-16.

### Open decisions for Tian

- **Re-run 5 folds now, or do P-21 first, or both?** Asked and not yet answered. P-21 costs no
  GPU and ships something; the folds re-run is another ~4.5 h after ~9 h already burned today.
- **Pin the final submission's weights to a Kaggle Dataset instead of `kernel_sources`.**
  `rsna-knee-infer` mounts *the latest version* of `rsna-knee-train`, which forced
  submit-before-push ordering twice today. A dataset is immutable and kills the whole failure
  class before the deadline. **Traps 6f raises the priority of this** — mount layout is
  evidently not stable across slugs either.
- **`artifacts/kaggle_out/v9smoke/` is 1.8 GB of smoke checkpoints** whose filenames are
  identical to the real ones (`v9smoke/v04d_fold0_best.pt` is a 4-study toy, not the 0.877
  model). Safe to delete.
- Browser-only questions still block P-10/P-16/P-18: rules text, Efficiency Prize formula,
  radimagenet.com T&C, hidden test size.

### Things that will bite if forgotten

- **Read the cache line in every smoke log before promoting a run to real.** Scan the log for
  the *known* failure modes, not only for the new thing being tested (traps 6f).
- **Treat a large runtime overrun as a symptom, not as a bad estimate** — especially when the
  throughput figure it contradicts was measured the same day.
- A run in flight on a slug makes `kernels output` return nothing for that whole slug, for every
  version form (traps 12e). Past versions are retrievable as `<slug>/<version>` once idle.
- `--file-pattern` is a **regex, not a glob**; `"(oof|manifest)"` pulls results + log,
  `"no_match"` pulls the log alone.
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose; the kernels that
  ran had `False`.
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by
  filename, never edit the `.ipynb`, `experiments.md` is append-only).

---

## 2026-08-29 (14:50) — LB 0.877, P-09 and P-04 closed, head-blend found; 5-fold still running

Findings are all logged as of `0a131ea`. This entry is state only. Continues the 00:13 entry
below, which described the five-arm batch while it was still in flight.

### ⏳ Still in flight as this was written (14:50)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **`rsna-knee-folds` v2** | First real ensemble: **5 folds × 4 epochs** of the confirmed `v04d` recipe (concat head + `cache_jitter`), arm `v05f`. New kernel slug so it could run beside `rsna-knee-train` | 2026-08-29 ~08:00 | `kaggle kernels status tiankljucanin/rsna-knee-folds`, then **`kaggle kernels output tiankljucanin/rsna-knee-folds -p artifacts/kaggle_out/folds --file-pattern "(oof\|manifest)"`** — pulls log **and** the OOF csvs together (traps 12e) | **~6.9 h in at 14:50 against a ~4.5 h estimate — the estimate was wrong, the run is not.** The 8.3 h guard fires ~16:20. Log has 20 `fold N epoch M:` lines (5 folds × 4 epochs) and a `=== v05f fold K ===` banner per fold. Per-fold OOF ~0.85 is normal; the number that matters is the **rank-mean across folds**, printed by the inference block only if *all five* folds completed. If the guard fired mid-fold, `all folds complete: False`, inference is skipped by design, and each fold has `v05f_fold{k}_last.pt` — attach this run's output as an input to a new run and it resumes |

**Compare the 5-fold ensemble gain against P-21's head-blend gain (+0.0096 on fold 0).** That
comparison is the point of the run: it decides whether the final submission spends its budget on
folds or on head diversity.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **four**: 0.500 → 0.841 → 0.871 → **0.877** (`v04d`, new best) |
| Best single model | ✅ `v05a` attn + jitter, 8 ep: OOF **0.8574**, gold 0.9266 (n=11) — **not yet submitted** |
| Best OOF of any kind | ✅ **0.8670** — rank-mean of `v05a` + `v05b` on fold 0 (P-21, not yet submitted) |
| Noise floor | ✅ measured: **0.008 macro, ~0.03 per label** |
| P-09 attention head | ✅ KEEP (+0.0103 at matched 8 ep) |
| P-04 8 epochs | 🔁 does not beat 4 for concat; epoch count is **head-specific** |
| P-05 laterality | ✅ confirmed, ≈ +0.015 of v03's +0.022 |
| 5-fold ensemble | ⏳ running |
| Repo | ✅ clean, pushed through `0a131ea` |

### What we talked about and decided

- **Ran all five of the previous turn's suggestions rather than picking.** Submitted `v04d`; the
  8-epoch P-09 retest; the 5-fold run; the direct worker-RNG check; and P-04 folded into the same
  kernel as the retest's control arm.
- **The 8-epoch retest ships a matched control (`v05b`) rather than comparing back to `v04d`.**
  Comparing a new head against a differently-scheduled run would have confounded head with
  schedule — the exact mistake that made v03 ambiguous for a day.
- **The 5-fold run uses today's confirmed winner (`v04d`), not "whatever wins".** #3 was
  contingent on #2, but sequencing them wastes a sitting, and Kaggle runs two GPU sessions at
  once. The ensemble and the first trustworthy multi-fold number are worth having either way.
- **Deliberately did not launch a third GPU run while both were busy** — quota is shared, and the
  right next arm depended on results an hour away.
- **P-10 (second architecture family) de-prioritised** behind the new P-21: head diversity is free,
  a CNN member costs a session, and RadImageNet's licence is still unresolved.

### What we figured out

1. **Submission #4 = 0.877, a new best**, and the **third** OOF→LB point at a +0.024 offset —
   three for three inside +0.02–0.03. Honest caveat recorded: +0.006 LB is only 1.2× the LB
   floor, so jitter is carried by its 11-of-12 per-label sign pattern, not the leaderboard.
   → experiments.md, Submissions.
2. **P-09 ✅ KEEP at a matched schedule (+0.0103), but the card's reasoning was wrong.** It
   predicted gains on plane-specific findings; those are where it *loses* (MCL −0.040, Lateral
   Meniscus −0.032). What it buys is **overfit resistance** — it plateaus at 0.857 while concat
   peaks at epoch 4 and decays to 0.8471. → experiments.md.
3. **That verdict is policy-dependent.** At each head's own best epoch they are indistinguishable
   (0.8576 vs 0.8600); attn wins only under fixed-last-epoch checkpointing. → new card **P-22**.
4. **The two heads rank-blend to OOF 0.8670** (+0.0096 over the best single arm) at mean rank
   correlation **0.773**, despite sharing backbone, data, fold, schedule and seed. Free error
   diversity. → new card **P-21**, the highest-value untested item now.
5. **P-04: 8 epochs does not beat 4 for the concat head** (0.8471 vs 0.8528). The epoch count is
   **head-specific**, not a project setting. → experiments.md.
6. **traps 6e was wrong and is corrected**: the on-Kaggle check shows PyTorch already seeds numpy
   and `random` per worker, so the fork pathology does not exist here. `v04d`'s jitter was never
   confounded. → traps 6e.
7. **traps 12e was also wrong and is corrected**: per-version output *is* retrievable
   (`<slug>/<version>`); what actually blocks it is a **run in flight on that slug**. → traps 12e.
8. **New trap 4b — the cosine schedule spans `cfg.epochs`**, so epoch N of a 4-epoch run and
   epoch N of an 8-epoch run sit at different learning rates and are not comparable. Compare
   final-to-final across budgets. → traps 4b.

### ⏭ Next action, in order

1. **Read the 5-fold run** (command in the in-flight table). Take the **rank-mean across folds**
   from the inference block, not the per-fold numbers. Decision rule: compare the ensemble's gain
   over a single fold (`v04d` 0.8528) against **P-21's +0.0096 head-blend gain**. If 5 folds buys
   less than ~0.01 over one fold, head diversity is the better use of the budget and P-21 comes
   first. If it buys more, folds win and P-21 becomes an addition rather than a substitute.
2. **Ship P-21** (~0.2 session, no training): make the infer path rank-mean across *versions*, not
   just folds — `ckpt_paths` currently maps `fold -> path` for one `cfg.version`; it needs
   (version, fold) pairs. Then submit the `v05a` + `v05b` blend. Expect a *small* LB move; do not
   read a sub-0.005 change as confirmation.
3. **Run P-22** (~0.1 session, analysis only, no training): for every arm, compare best-epoch vs
   last-epoch OOF from the `_ep{e}_oof.csv` files already downloaded in
   `artifacts/kaggle_out/v13/`, with the gold curve alongside to detect teacher-chasing. If
   best-epoch beats fixed-epoch by more than 0.008 **and** gold does not diverge, switch the
   policy and re-read P-09 and P-04 under it.
4. **Only then** consider more architecture (P-10/P-14/P-15) or P-16's re-labelling.

### Open decisions for Tian

- **Pin the final submission's weights to a Kaggle Dataset instead of `kernel_sources`.** Right
  now `rsna-knee-infer` mounts *the latest version* of `rsna-knee-train`, so pushing a training
  run changes what a submission would load. It forced submit-before-push ordering twice today. A
  dataset is immutable and removes the whole failure class before the deadline.
- **`artifacts/kaggle_out/v9smoke/` is 1.8 GB of smoke checkpoints** (4-study models) with
  **filenames identical to the real ones** — `v9smoke/v04d_fold0_best.pt` is *not* the 0.877
  model. Safe to delete; flagged rather than deleted.
- Browser-only questions still block P-10/P-16/P-18: rules text (hosted-LLM clause, winner
  licence), Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- **The remaining 0.075 to the public top is still unexplained.** We use the same public label
  tables the leaders use; ensembling plausibly accounts for 0.01–0.02.

### Things that will bite if forgotten

- **A run in flight on a slug makes `kernels output` return nothing for that whole slug**, for
  every version form. Wait for it to finish (traps 12e).
- Pull the log **and** the small result files in one command: `--file-pattern "(oof|manifest)"`.
  `--file-pattern` is a **regex, not a glob**; `"no_match"` is the log-only trick.
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose; the kernels that ran
  had `False`. Regenerating a notebook locally produces a *smoke* notebook.
- The 5-fold kernel is generated by `sed 's/^FIVE_FOLD = False/FIVE_FOLD = True/'` — never edit
  the `.ipynb`.
- Push `rsna-knee-infer` **before** re-pushing `rsna-knee-train` (see the Kaggle Dataset item).
- Everything from the entries below still applies (`PYTHONUTF8=1`, never sort DICOMs by filename,
  do not download the 21 GB cache, `experiments.md` is append-only).

---

## 2026-08-29 (00:13) — LB 0.841 → 0.871, four silent bugs fixed, five-arm batch launched

All findings are logged in experiments.md / traps.md / proposals.md / CLAUDE.md as of commit
`531a923`. This entry is state only.

### ⏳ Still in flight as this was written (00:13)

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Kernel v11** — `rsna-knee-train` v11 | Five fold-0 arms, 4 epochs each, from the cache, in run order: `v04base` (reference, seed 42) → `v04a` (seed 43) → `v04c` (`head_type="attn"`, P-09) → `v04b` (`lat_undo`, P-05) → `v04d` (`cache_jitter`, P-08) | 2026-08-28 ~23:35 | `kaggle kernels status tiankljucanin/rsna-knee-train`, then `kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out/v11 --file-pattern "no_match"` for the log **alone** (a plain `output` pulls ~1 GB of checkpoints) | ~0.9 h/arm, ~4.6 h total against an 8.3 h guard. Each arm prints a `##########` banner with its resolved folds/epochs/seed, then a per-label table per epoch. **Read `v04a` vs `v04base` FIRST** — that difference *is* the noise floor and nothing else can be judged without it. An arm that raised is logged `!! arm <v> FAILED` and the remaining arms still ran |

**The machine can be shut down without affecting this.** The kernel runs on Kaggle's servers; the
local polling loops watching it die with the session and matter to nothing. Every arm writes
`v04*_fold0_best.pt` as it completes, so even a guard-stopped run keeps whatever finished.

### Where things stand

| | Status |
|---|---|
| Submissions | ✅ **three**: 0.500 (constant, #1) → **0.841** (v02, #2) → **0.871** (v03, #3) |
| Best model | ✅ v03 fold 0: OOF-vs-teacher 0.843, gold 0.906 (n=11), **public LB 0.871** — one fold, one backbone, no ensemble, no TTA |
| Infer path | ✅ fixed and verified — predictions byte-identical to the training kernel's on the same studies (traps 6d) |
| Cache (P-01) | ✅ closed as measured: 5.4× end to end, and the largest scoring gain so far |
| Noise floor (P-02 step 1) | ⏳ being measured for the first time by v11 — still **asserted** at 0.01 until it reports |
| P-09 attention head | 🔧 shipped, ⏳ running |
| Repo | ✅ clean, pushed through `531a923` |

### What we talked about and decided

- **Read the previous session's two in-flight items before proposing anything.** That is what
  surfaced both the 0.871 and the confound; going straight to new ideas would have missed both.
- **Chose P-09 over P-16, P-08 and "just run 5 folds"** as the impact card. P-16 has the highest
  ceiling but its own cited result (Qwen3-14B-AWQ, 0.881 gold) sits *below* our current blend's
  0.8948, so it is not obviously a ceiling-raiser and costs 1–2 sessions. P-08 turned out to buy
  token granularity rather than coverage (card corrected). P-09 is free in GPU time, *removes*
  head parameters in a regime that is measurably overfitting, and targets the plane-specific weak
  labels. Five folds is the reliable +0.01–0.02 and stays queued as breadth.
- **Ablated laterality from the existing cache instead of rebuilding it.** The cache's transforms
  are involutions, so right knees can be de-canonicalised at load time — free, and a *cleaner*
  test than v03-vs-v02 because the crop stays constant. Rejected the no-crop rebuild: two CPU
  kernels, another 21 GB, 42 GB mounted, and it does not change what we do next.
- **Added a fifth arm (`v04base`) mid-build.** Once `seed_worker()` changed how augmentation is
  randomised, kernel v8 stopped being a same-code baseline, so arms b/c/d had nothing to differ
  from in exactly one thing. One extra hour buys single-variable attribution for all three.
- **Submitted v03 before touching `rsna-knee-train`.** `kernel_sources` mounts the *latest*
  version's output, so the infer push had to happen while that was still v8.

### What we figured out

1. **The v03 gain is real and transferred: LB 0.841 → 0.871, +0.030 against a 0.005 floor.** The
   +0.022 OOF was not a teacher-agreement artefact. → experiments.md, Submissions.
2. **OOF-vs-teacher predicts the LB and under-reads by +0.02–0.03** (two points: 0.821→0.841,
   0.843→0.871) — expected when the teacher's own gold is 0.8948. An OOF gain is now evidence
   rather than a hope. n=2, so treat it as an offset, not a law. → experiments.md.
3. **P-01 was mis-specified, not merely under-measured.** Its "OOF within ±0.01 = faithful
   speed-up" rule assumed v03 replayed v02's inputs; v6's config has no `crop_mm` and no
   `lat_dead_zone_mm` at all. Four things moved at once. `v04b` splits them. → experiments.md.
4. **The cache is a 5.4× speed-up end to end, not the ~60× the decode arithmetic implied** — the
   T4 is the bottleneck now, at ~0.18 s/study for ~29 ViT forwards. **This corrects finding #9 of
   the 2026-08-28 entry below, which says 60×.** Consequence: I/O work is now worthless, and
   extra slices cost linearly in GPU time. → experiments.md.
5. **Inference was one line away from scoring a v03 model on v02 pixels.** `use_cache` meant both
   "which preprocessing" and "read a .npy"; with no cache mounted the infer kernel took the v02
   decode branch — no crop, no laterality. Nothing would have raised. → traps 6d.
6. **A real-mode `MODE="auto"` infer kernel would have re-trained at rerun**, because fold
   narrowing was gated on `cfg.smoke` and only fold 0 has a checkpoint. → traps 12c/12d.
7. **Smoke mode structurally cannot reveal real-mode defaults.** Every arm inherited
   `folds=(0,1,2,3,4)` — 25 folds, ≈18 h — and four green smokes said nothing about it, because
   `__post_init__` forces `folds=(0,)`. → traps 12d.
8. **The pipeline's augmentation may never have been random.** numpy and `random` are
   fork-inherited per DataLoader worker, so jitter and noise repeat identically every epoch.
   Fixed with `seed_worker`, **but not empirically verified** — Windows spawns workers, so the
   pathology cannot be reproduced locally. → traps 6e.
9. **v8's train loss falls monotonically while OOF turns over at epoch 2.** That is the
   overfitting signature, and it is the argument against P-04's 8 epochs and for augmentation and
   fewer head parameters. → experiments.md.

### ⏭ Next action, in order

1. **Read v11's log** (command in the in-flight table). Take the epoch-3 `auc_soft` of each arm,
   then in this order:
   - `floor = |v04a − v04base|`. **This replaces the asserted 0.01 everywhere.** If it exceeds
     ~0.02, the v03 +0.022 was never established and that is the session's headline finding.
   - `v04c − v04base` (P-09). Adopt only if it clears the floor. Report the macro **and**
     Effusion~Synovitis, Medial OA~Medial Meniscus, Contusion~Fracture — research.md's stated
     risk is that a per-label head hurts exactly those pairs.
   - `v04b − v04base` (P-05). A **drop** attributes the v03 gain to laterality; flat means the
     130 mm crop or per-series normalisation carried it. Read the five side-specific labels, not
     the macro.
   - `v04d − v04base` (P-08 jitter). Judge on the *shape* of the epoch-2→3 turn, not the peak. If
     flat, "the augmentation still is not random" remains live (finding 8) — settle it by logging
     epoch-0 vs epoch-1 jitter offsets in one Kaggle run before calling it a dead end.
2. **Log the four verdicts with `/update`**, each stating its delta and the floor it beat.
3. **Submit the winning arm** if any clears the floor: set `cfg.version` to that arm, regenerate
   `kaggle/rsna-knee-infer` with the sed recipe in CLAUDE.md, push, submit. Expect roughly
   LB ≈ OOF + 0.02–0.03 (finding 2).
4. **Then five folds of the best recipe** — ~4.6 h now, and the cheapest standing claim on part of
   the 0.081 gap to the public top.

### Open decisions for Tian

- **Depth or breadth next.** Five folds of the current recipe (a real ensemble, a trustworthy LB
  number, +0.01–0.02) versus more fold-0 arms. With ~0.9 h arms, depth-then-breadth still looks
  right, but the 2026-10-22 deadline makes it a judgement call.
- Browser-only questions still block P-10/P-16/P-18: rules text (hosted-LLM clause, winner
  licence), Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- **Where the remaining 0.081 comes from is genuinely unexplained.** We use the same public label
  tables the leaders use, and ensembling accounts for maybe 0.01–0.02. Worth reading the top
  notebooks' configs before assuming more of the same recipe closes it.

### Things that will bite if forgotten

- `kaggle kernels output` with no `--file-pattern` pulls **every** checkpoint — ~1 GB for a
  five-arm run. Use `--file-pattern "no_match"` when you only want the log.
- The infer notebook is generated from a **sed'd copy** (`MODE="infer"`, `FORCE_SMOKE=False`);
  `MODE="auto"` is wrong there, and `FORCE_SMOKE=True` sets a 0.4 h runtime guard (traps 12c/12d).
- `src/kaggle_pipeline.py` is committed with `FORCE_SMOKE = True` on purpose. Kernel v11 on Kaggle
  has `False`; regenerating the notebook locally produces a *smoke* notebook.
- Push the infer kernel **before** re-pushing `rsna-knee-train`: `kernel_sources` mounts the
  latest version's output.
- Everything from the entry below still applies (`PYTHONUTF8=1`, never edit the `.ipynb`, do not
  download the 21 GB cache, `experiments.md` is append-only).

---

## 2026-08-28 (evening) — Research, 2 submissions, v02 baseline, cache built, v03 training from cache

**Two things are still in flight as this was written (22:35). Check them first:**

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|
| **Submission #2** — ref `55852527` | The v02 fold-0 model, submitted from `rsna-knee-infer` v1 (which mounts `rsna-knee-train` v6's output and runs `MODE=infer`) | 22:00 local (19:59:50 UTC) | `kaggle competitions submissions rsna-knee-abnormality-detection --csv` | Its rerun decodes ~1,300 hidden studies at ~150–320 s/100 studies, so ~40–70 min is normal. **Anything except exactly 0.500 means the submission path finally works.** Expect roughly 0.80–0.88 if the OOF transfers (public DINOv2-S/224 baselines sit at 0.809). Log it in the experiments.md Submissions table next to OOF 0.821 / gold 0.847 |
| **Kernel v8** — `rsna-knee-train` v8 | v03: the same recipe as v6 but **training from the mounted cache** instead of decoding DICOMs, fold 0, 4 epochs | 22:04 local | `kaggle kernels status tiankljucanin/rsna-knee-train`; logs only become readable when it finishes | **The runtime is itself the measurement.** v6 took 5.0 h (58 min/epoch). If v8 finishes in well under an hour the cache works; if it takes ~5 h it silently fell back to per-epoch decode (the notebook prints `! use_cache=True but no cache is mounted` in that case) |

### Where things stand

| | Status |
|---|---|
| Research | ✅ 18-agent workflow (8 researchers → 8 skeptics → synthesis → critic, 817 tool calls) → [research.md](research.md); 21 cards → [proposals.md](proposals.md) |
| Submission #1 | ✅ scored **0.500 exactly** (kernel v2, smoke) = constant output at rerun; root-cause class found and fixed |
| Submission #2 | ⏳ **PENDING** (see table above) |
| Pipeline v02 | ✅ green (Kaggle v3–v5): prob targets, LR 2e-5 + LLRD 0.75, EMA 0.998, fixed-epoch `best.pt`, per-label + OOF logging every epoch, `MODE=infer`, loud-failure submission, resume fixed |
| Real fold-0 run (v02, decode path) | ✅ kernel v6, 5.0 h: **0.99 s/study**, OOF-vs-teacher **0.821**, gold 0.847 (n=11, CI 0.72–0.94) |
| Cache | ✅ built: `rsna-knee-cache-a` (2,115 studies, 10.2 GB) + `-b` (2,292, 11.0 GB), 25 min each, **0 decode failures**, 4,407/4,407 studies |
| Cache loader (v03) | ✅ shipped and smoke-verified on Kaggle (v7: 4,407 studies indexed, no header scan needed) |
| Label audit | ✅ `src/label_audit.py` → `artifacts/label_audit.md`, findings in experiments.md |
| Repo | ✅ 4 commits pushed today (`3bf7085..8b63927`) |

### What we figured out (the findings that changed what we do)

**1. Our training targets were on the wrong scale (P-00, fixed).** The soft targets were
rank percentiles. `rank(pct=True)` gives tied values their *average* rank, so on labels where
most reports say 0, every confident negative got a target of 0.28–0.39 while the 58 gold rows
sat at a hard 0/1 with 8× weight — **no study on any label had a target below 0.1**. AUC is
rank-invariant, which is why this looked principled, but BCE fits *values*. Now the mean of
the sources' probabilities (teacher gold 0.8948 vs 0.8934; the Δ is noise — the fix is about
scale, not AUC).

**2. A submission can score exactly 0.500 and tell you nothing.** Submission #1 completed and
scored 0.500 to three decimals. A near-random model on ~1,300 studies scores 0.47–0.53, never
0.500 — that is a *constant* submission, i.e. our own `fillna(0.5)` fallback fired because the
hidden test tree did not match the assumed layout. Code-competition rerun logs are invisible,
so nothing reported it. v02 probes the test root by glob, tolerates non-`.dcm` names, writes
**no placeholder**, and **raises** when < 90% of test studies have an image slot or > 6 labels
are constant. A visible scoring error beats an invisible 0.500.

**3. The first real model works and its weak labels are visible per label.** v6 fold 0:
OOF-vs-teacher 0.821 over 882 val studies, plateauing at epoch 2–3; `pred_std` 0.12 → 0.23
(no base-rate collapse). Weakest against the teacher: **Lateral Meniscus 0.72, MCL 0.75,
Lateral OA 0.78, PF OA 0.79** — three of the four are side-specific or small focal findings,
which is exactly what P-05 (laterality), P-08 (more slices) and P-11 (resolution) target.

**4. Synovitis is a teacher-ceiling problem, not a model problem.** The student reproduces
the teacher almost perfectly where the teacher is confident (0.94 on confidently-labelled
rows) while gold sits at chance (0.50) — it has faithfully learned that "not mentioned" means
negative, and Synovitis is unaddressed in **84%** of reports. Only better targets move this
(P-07/P-16), never a better backbone.

**5. Our three LLM label sources are about 1.5 sources.** `hans_v4` and `sol56` make identical
decisions at the 0.5 cut on 99.45% of studies (error-φ = 1.000 on gold, every label). Adding
more public label tables is therefore pointless, and the `agreement` term in
`confidence_weights` is inflated by a duplicate.

**6. Silence is not down-weighted the way the code claims.** A report that never mentions a
finding blends to ~0.18 — a confident-looking negative — so its weight is 0.69 versus 0.80–0.89
on addressed rows. The docstring's "silent reports pull far less" was wrong; gating the weight
on pilkwang's `UNK` verdict is an open card.

**7. The published Synovitis←Effusion back-fill does not reproduce on our blend.** Gold AUC
0.788 → 0.729, paired-bootstrap Δ −0.059, 95% CI [−0.164, +0.042] → 🔁 INCONCLUSIVE, not
adopted. The public card's +0.11 came from a 0.678 baseline we are already above.

**8. Laterality can be recovered from geometry, on our own data.** Over all 4,407 studies:
`Laterality` tag present on 49.6%, geometry (image-centre x, 20 mm dead zone) resolves 96.9%,
**tag-vs-geometry agreement 0.988** (n = 2,116), 26 conflicts left unmirrored, 2.1%
unresolved. Previously community-sourced, now measured here.

**9. The cache is a 60× epoch speed-up and is bit-exact.** 4,407 studies in 25 min per shard,
0 decode failures, 21.2 GB in two shards (needed: a single kernel output caps around 20 GB).
The training notebook carries the builder's exact functions, so a *test* study is built on the
fly by the same code — verified bit-identical to the cached array on the local sample for both
a left and a right knee.

**10. Inference cost, not training cost, will bound the ensemble.** Measured 150–320 s per 100
test studies for **one** fold. Five folds each decoding the test set again would spend hours on
inference alone in the rerun, so decode-once-predict-all-folds is a prerequisite for any
multi-fold submission (P-18), not an optimisation.

**11. Two silent bugs the code review caught before they cost a session.** Resume never
resumed (checkpoints were looked for in `WORK`, but a mounted previous run is read-only under
`/kaggle/input`) — every fold would have restarted at epoch 0. And a smoke-mode `MODE=infer`
would have fed 2 slices/slot to a model trained on 6 while passing every assert; the infer path
now reads the input geometry from the checkpoint's saved config.

### ⏭ Next action, in order

1. **Read the two in-flight results** (table at the top). Log submission #2's score in the
   experiments.md Submissions table; for v8 pull `kaggle kernels output tiankljucanin/rsna-knee-train
   -p artifacts/kaggle_out/v8 --file-pattern "(oof|log)"` and compare **epoch time** and
   **OOF-vs-teacher** with v6's 0.821. P-01's verdict rule: within ±0.01 = the cache is a
   faithful speed-up; a real drop means the crop / per-series normalisation / laterality changed
   the inputs, and that must be understood before adopting it.
2. **Submission #3 = v03** if its OOF holds: re-push `kaggle/rsna-knee-infer` (it mounts
   `rsna-knee-train`'s latest output and reads the cache config from the checkpoint), then
   `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer
   -v <ver> -f submission.csv -m "..."`.
3. **Then the cache-era cards, cheapest first** — they only make sense now that an epoch is
   minutes: P-02 seed-noise baseline (2 seeds of fold 0 — this *measures* the 0.01 OOF floor we
   have only asserted), P-04 8 epochs, P-08 16 slices/slot, P-03b LLRD vs uniform, P-02 proper
   (site-grouped folds), then 5 folds for a real ensemble.

### Open decisions for Tian

- Browser-only questions still block P-10/P-18: rules text (hosted-LLM clause, winner licence),
  Efficiency Prize formula, radimagenet.com T&C, hidden test size.
- After v03 lands, the next GPU session is a choice between **breadth** (5 folds of the current
  recipe → a real ensemble and a trustworthy LB number) and **depth** (P-04/P-08 arms on fold 0
  → a better single model first). With cheap epochs, depth-then-breadth is the better order,
  but it is a judgement call about the 2026-10-22 deadline.

### Things that will bite if forgotten

- `PYTHONUTF8=1` for **any** Python that prints non-ASCII, not just the Kaggle CLI (traps 15b).
- The submitted notebook must be the **infer-mode** one (`rsna-knee-infer`), never the training
  notebook — a code competition re-runs what you submit (traps 12c).
- A public LB of exactly **0.500** is a constant submission, not a bad model (traps 12b).
- `kaggle/rsna-knee-cache-b` and the real-run notebooks are generated by `sed`ing
  `src/*_pipeline.py` before `nbgen.py` — never edit an `.ipynb`.
- Do **not** download the cache locally (21 GB); mount `rsna-knee-cache-a`/`-b` as
  `kernel_sources` and read `manifest_shard{0,1}.csv` + `<study>.npy` on Kaggle.
- `experiments.md` is append-only: superseded rows are marked, not deleted.

---

## 2026-08-28 — Pipeline built and verified green on Kaggle

### Where things stand

| | Status |
|---|---|
| Competition entry | ✅ rules accepted, `userHasEntered: True` |
| Kaggle CLI | ✅ authenticated as `tiankljucanin` (OAuth) |
| Data locally | ✅ CSVs + LLM labels + models. ⚠️ sample DICOMs **459/557** (rate-limited) |
| Targets & folds | ✅ built, verified, reproducible across machines |
| Pipeline | ✅ runs green end-to-end on a Kaggle T4 (smoke mode) |
| Real training run | ❌ **not started** |
| Submissions | ❌ **zero** |
| Repo | ✅ git initialised, remote set, pushed |

### What got done

- Verified every competition fact against the API and the CSVs (see
  [../CLAUDE.md](../CLAUDE.md)). Corrected an earlier wrong hypothesis: reports are
  **not** available at inference time.
- Scored all five public LLM report-label sources against the 58 gold studies. Best blend:
  **teacher gold macro-AUC 0.8934**.
- Built leak-safe grouped folds: **882/882/881/881/881**, gold **11/12/12/12/11**, no report
  group spanning a fold. Byte-identical on Kaggle and locally.
- Built `src/kaggle_pipeline.py` — the whole pipeline in percent format, runnable as a
  script *and* convertible to the notebook via `src/nbgen.py`.
- Pushed and ran `tiankljucanin/rsna-knee-train` (private). **Kernel v2 completed green** in
  9.5 min in smoke mode: targets → folds → header scan → slot selection → triplets → DINOv2
  → training → checkpoints → rank-mean inference → validated `submission.csv`.
- Found and fixed three silent bugs, all recorded in [traps.md](traps.md): a fold could
  finish with **no `best.pt`** and vanish from the ensemble; teacher AUC was being graded
  against itself (printed 1.0000); an empty header scan was being cached and would poison
  resumes.
- Split the docs into [traps.md](traps.md), [experiments.md](experiments.md),
  [brainstorm.md](brainstorm.md), [setup.md](setup.md), and this file.

### ⏭ Next action, in order

1. **Benchmark real throughput on Kaggle.** Everything else depends on this number. Fold 0
   took 36 s for 8 study-passes at 2 slices/slot with `num_workers=0` (~4.5 s/pass), which
   naively extrapolates to **6–8 h per fold** — five folds ≈ five sessions. That is an
   extrapolation from 8 studies, *not a measurement*. Get the real figure with
   `num_workers=2` at the production slice count.
2. **Build the preprocessing cache kernel** (backlog #1). Almost certainly the bottleneck is
   DICOM decode: ~90 file reads per study, every epoch. Decode/resize to uint8 once in a CPU
   kernel, mount it from training.
3. **Add site-grouped folds** (backlog #2). Largest correctness gap in our validation.
   Expect OOF numbers to drop — that is the point.
4. Only then: flip `FORCE_SMOKE = False` and launch a real multi-fold run.

**Do not skip to step 4.** A 5-fold run at current throughput burns several sessions on work
the cache makes ~10× cheaper, and its OOF would be inflated by site leakage anyway.

### Open decisions for Tian

- **Whether to train a single fold now** anyway, to get a first real LB number for
  orientation, accepting that it costs a session and the OOF is not yet trustworthy. There
  is a real argument for it — zero submissions means zero feedback — but it is a judgement
  call about spending a session.
- Several questions need a **browser** and cannot be answered from the CLI: the exact
  runtime limit, the Efficiency Prize scoring formula, and the winner-licence clause (which
  gates using RadImageNet in a final submission). See the open-questions table in
  [brainstorm.md](brainstorm.md).

### Things that will bite if forgotten

- `export PYTHONUTF8=1` before any `kaggle` CLI call on Windows.
- Run scripts from the **repo root** with `PYTHONPATH=src`.
- Edit `src/kaggle_pipeline.py`, **never** the generated `.ipynb`.
- `FORCE_SMOKE = True` for the first push after any edit.
- Never select the **P100**.
- The 98 missing sample DICOMs are a **rate limit**, not a bug, and block nothing.

---

## Template for the next entry

```markdown
## YYYY-MM-DD — one-line summary

### What got done
-

### Results (also log in experiments.md with a verdict)
-

### ⏭ Next action
1.

### Open decisions / blockers
-
```
