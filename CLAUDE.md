# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this repo

Competition workspace for the Kaggle **RSNA Knee Abnormality Detection** challenge (RSNA
2026 AI Challenge). Predict **12 independent binary findings per knee MRI study**, scored
by **macro ROC-AUC** (unweighted mean of 12 per-label AUCs).

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

**State as of 2026-10-01 (09:40 UTC) — work resumes 2026-10-03 after the GPU reset.** **P-62 (silence-aware teacher mix: Raptor 0.75 on report-silent cells, 0.5 elsewhere; gold-58 target 0.9300 vs 0.9268, direction only) is implemented: `TEACHER_SILENT_MIX` + arms `v11s` ‖ `v11s2`, Kaggle smoke `rsna-knee-train` v38 green (0.04 h; GPU ≈ 0.5 h left until the reset); `kaggle/rsna-knee-train/kernel-metadata.json` now mounts the c03 caches only — the P-60 part-2 push copies `artifacts/p60_part2_kernel-metadata.json` over it first** (proposals.md P-62; experiments.md 2026-09-30 "Silence-aware teacher mix"). **State as of 2026-09-30 (21:15 UTC).** **P-60 part 1 (`rsna-knee-folds` v11, pushed 12:10 UTC) COMPLETE in 2.61 h, green: `v11n` / `v11n2` (c03 + drop-path 0.1 + heavy aug + 12 epochs) guard-stopped in epoch 5 by design, gold-58 EMA 0.9166 / 0.9129 at epoch 4 = `v11a` / `v11b` at theirs (direction only); `_last.pt` ×2 in the v11 output → part 2 = resume in `rsna-knee-train` after the 2026-10-03 reset (≈ 3.0 h). Do not push `rsna-knee-folds` before then; its epoch-5 `_best.pt` files are not members (traps 47). Kaggle GPU 0.58 h left** (experiments.md 2026-09-30 "P-60 part 1"). **State as of 2026-09-30 (12:00 UTC).** **Submissions #34–#38 read (sent 11:06 UTC from `rsna-knee-infer` v30–v34): the P-55 student `v09o` / `v09o2` 0.927 / 0.927, pair 0.928 → ❌ (m = 0.927 = the c02 twins; the OOF-target line closes, gold-58's +0.007 did not transfer — traps 39 extended); ResNet-34 `v13c` 0.921 ❌ solo, c03 pair + `v13c` 0.932 = the c03 pair ❌ (P-57 closes; the ResNet is an Efficiency candidate only, P-18). Best solo still 0.932 (`v11a`, the c03 pair); best LB 0.942. Kaggle GPU 3.19 h left until the 2026-10-03 reset (`kaggle quota`); P-60 waits for it** (experiments.md 2026-09-30 "Submissions #34–#38"). **State as of 2026-09-29 (19:30 UTC).** **Evening research (3 agents) → P-59 ✅: a CNN learning rate lifts ResNet-34 gold-58 0.8306 → 0.8992 (`v13b`; frozen BN `v13c` 0.9014 🔁) — P-57's ❌ was our ViT-tuned LLRD (traps 46); P-60 ("noisy student" c03 CoAtNet `v11n` / `v11n2`: drop-path 0.1, heavy aug, 12 epochs) training on a RunPod A100 (traps 45: pod 1 had no bandwidth); 5 solos queued for 00:03 UTC 2026-09-30 (3 student + `v11n` / `v11n2`)** (experiments.md 2026-09-29 "P-59"; proposals.md P-60 / P-61). **State as of 2026-09-29 (17:00 UTC).** **Session E (`rsna-knee-train` v35, P-55) COMPLETE in 2.94 h, green: the student `v09o` / `v09o2` gold-58 SWA 0.9121 / 0.9104 (pair 0.9119 vs the c02 pair 0.9065, direction only); shipped as `rsna-knee-ckpt-v09o` / `-v09o2`; three solo reads after 00:00 UTC 2026-09-30** (experiments.md 2026-09-29 "Session E"). Sessions C (re-push, `rsna-knee-train` v34) ‖ D (`rsna-knee-folds` v10) COMPLETE: P-54 has all five cross-fit folds → table `xfit_v09k` (gold-58 0.9028 vs LLM 0.8948) → loose gate OPEN → **session E (`v09o` ‖ `v09o2`, P-55) = `rsna-knee-train` v35 RUNNING**; c03 `v11a` / `v11b` gold-58 0.9204 / 0.9167 (pair +0.014 over the c02 pair, direction only — solo reads today); ResNet-34 `v13a` 0.8306 = ❌ on gold; P-52 placeholder = `rsna-knee-infer` v25; new Datasets `rsna-knee-ckpt-v09k` / `-v11a` / `-v11b` / `-v13a` (experiments.md 2026-09-29 "Sessions C ‖ D", "P-54 cross-fit table"). **State as of 2026-09-28 (15:45). **`rsna-knee-train` v30 (P-43 + P-44) COMPLETE in 5.88 h — `v09x` = `v09r` at 320 px, gold-58 SWA 0.9094 ‖ `v09u` = `v09r` seed 43 (Kaggle), 0.8995, vs `v09r` 0.9093 (direction only; experiments.md 2026-09-28 "rsna-knee-train v30"); shipped as Datasets `rsna-knee-ckpt-v09x` / `-v09u`. **Solo reads 2026-09-28 (#24–#26): `v09u` 0.927 = `v09r` → the Kaggle-retrain spread is ≈ 0 and one-seed deltas need ≥ 0.004 (P-44 ✅; P-39 re-confirmed on Kaggle); `v09x` (320 px) 0.929 → 🔁 (+0.002, P-43); `v09r` + `v09u` rank-mean **0.930 = our best solo and the production member** (+0.003, adopted, not proven); gold-58 had both same-recipe directions wrong (traps 39 extended)** (experiments.md 2026-09-28 "Submissions #24–#26"). **#27 = the fork with `v09r` at β 0.20 reads 0.941 (🔁 vs 0.942) → P-40 closed, the fork is for final builds only; #28 = the P-53 per-label probe reads 0.785 → ACL / MCL / PF OA / Lat Men average 0.935 on the public test vs 0.9275 for the other eight, so the gold-58 "structural deficit" is gold-specific; P-49 (Raptor over gold-58): Raptor 0.9254, the 0.5/0.5 target 0.9268, `v09r` ~ Raptor within-class ρ 0.835** (experiments.md 2026-09-28 "Submissions #27–#28", "P-49"). **The discussion-735304 single-model plan is running: cross-fit folds 0–3 done (A ‖ B), session C = `rsna-knee-train` v33 (`v09k4` ‖ ResNet-34 `v13a`) and session D = `rsna-knee-folds` v10 (c03 `v11a` ‖ `v11b`, c03 cache mounted alone) pushed 15:35 UTC ⏳**; next card P-52 (`v09r` + `v09u` + `v09x`, no training); `proposals.md` rewritten after a four-reviewer audit (11 live cards; closed cards as pointers); traps 42 = per-arm `seed` / `teacher_mix` were inert (fixed).**** **Scoring speed (read 2026-09-27):** the public "0.943 Speedy Raptors CoAtNet D4" notebook is our anchor's cells plus two more CoAt readers — its "< 30 min" is its 3-study commit run, not the hidden test; the fork is hours by construction (#17 ≤ 8 h 06 min) while a solo submission is minutes (#19 ≤ 42 min), so **solo submissions are the fast instrument**; P-41 (16-thread header scan + 8 decode workers in the infer path, byte-identical) is ✅ — **a two-member solo scored 28.3–29.3 min after sending (#23)**, so a solo read is ≈ 30 min (experiments.md 2026-09-27 "Submission #23"; time every submission with `src/watch_submission.py` and keep the machine awake). **Best single member of ours: `v09r` = 0.927 solo (#20)** — the S2 `v09a` recipe trained on 0.5 · LLM + 0.5 · quantile-matched Raptor teacher table (P-39 ✅: +0.009 vs #18 `v09a` 0.918, the first change of training targets that transfers to the LB; ≈ the public stack's best member, 0.928 — experiments.md 2026-09-27 "Submission #20"); **P-40 in progress:** `v08r` (the DINOv2-S recipe on the same Raptor table) **reads 0.918 solo (#21)** — −0.009 vs `v09r`, under the ≈ 0.920 bar, so not a fork member on solo strength (experiments.md 2026-09-27 "Submission #21"; its gold-58 ρ with `v09r` is 0.924 vs 0.874 for the LLM-target pair — the shared teacher pulls families together, P-42); the fork v9 (`v09r` alone, β 0.10, vs 0.942) waited ≈ 3 h 40 min in the Kaggle queue and was **submitted as #22** (16:36 UTC) and **reads 0.942 = #13 / #15 (🔁, P-40)** — at β 0.10 the fork is flat over member strength 0.913 → 0.927 (experiments.md 2026-09-27 "Submission #22"), and **P-42** (`v09r` + `v08r` flat rank-mean solo) **reads 0.927 = `v09r` alone (#23, 🔁)** — a second backbone on the same Raptor table adds nothing measurable; the next gain has to come from the targets or the input. Best public LB **0.942** (#13 / #15; #16 0.940; **#17 = 0.941** — the fork with the S2 production members at β 0.10, read 20:00: −0.001, 🔁, identical to #14's 16-epoch members — the fork at β 0.10 does not read member quality, the solo submission does; experiments.md 2026-09-23 "Submission #17 read 0.941"). **#18 = the S2 `v09a` ALONE reads 0.918 solo** (`rsna-knee-infer` v15), above our own 12-member blend (#11, 0.913) and inside the public stack's member range 0.90–0.928 — the fold-0→LB offset under-predicted the all-data SWA member by ≈ 0.02 (experiments.md 2026-09-23 "Submission #18"). **Member-strength programme (spec + 12-task plan, branch `member-strength`, executed subagent-driven 2026-09-23, final review clean, merged to `main` the same evening):** teacher tables mixed into the *training* targets (`TEACHER_TABLES` / `TEACHER_MIX` / `TEACHER_PATHS` in the config cell, `yt__*` columns, `y__*` evaluation targets unchanged; `src/build_targets.py --teacher-tables`), `Config.pos_weight_max` (P-37), arms `v09e` / `v09f` / `v09s`, the Raptor teacher-pass kernel (`src/build_teacher_pass.py` → `kaggle/rsna-knee-teacher/`, `src/merge_teacher.py`), Dataset `tiankljucanin/rsna-knee-teacher-tables` (`selfdistill_v1.csv`). **Measured:** round 2 `v09d` (LR 3e-5) **0.8596 ❌ harmful** vs `v09c` 0.8730 → `v09e` dropped; `v08c` (DINOv2 + aug) 0.8650 🔁; RunPod 4090 (≈ 34 min per 8-epoch CoAtNet-1 arm): `v09f` (pos_weight) **0.8717 🔁**, **`v09s` (self-distillation, P-38) 0.8839 ✅ KEEP — +0.0109, 12/12 labels up, the only recipe change since `v09h` that clears the 0.008 floor**; the Raptor teacher pass is green (smoke 6/6, spike 100/100 at **5.1 s/study → full pass ≈ 6.2 GPU-h after Saturday's reset**; teacher vs LLM teacher AUC 0.914 on the spike). **Every recipe knob between our c02 CoAtNet and the public 0.928 member is now measured and none clears the floor (P-29 / P-32 / P-33 / P-34 / P-37); the target source is the lever — but only a teacher with *more information than ours*: P-38 ✅ on fold 0 turned ❌ in production, P-39 next.** **`v09t`** — the production `v09a` recipe on `("selfdistill_v1",)` under its own version name, trained on a RunPod 4090 in 35 min — reads gold-58 SWA **0.9009** vs `v09a` 0.8922 (direction only) but **reads 0.917 solo (#19) vs 0.918 → ❌ by the pre-registered rule: self-distillation does not transfer to the production member, and the fold-0 OOF-vs-LLM-teacher gain was agreement with the teacher, not truth — target-source changes are judged by gold-58 direction + solo LB only (traps 39)** (experiments.md 2026-09-24 "Submission #19"). `v09a` stays the production member. **The Raptor pass is complete** (3 shards, 2026-09-24 / 26: 4,349 studies, 0 failed, 5.6–7.5 s/study, shard 2 after a 3 h 10 min Kaggle T4 queue — traps 41; Raptor vs the hard LLM teacher macro AUC **0.9075**; `raptor_teacher.csv` in Dataset `rsna-knee-teacher-tables` — experiments.md 2026-09-26 "Raptor pass complete") and **`v09r` trained on it** (RunPod 4090 via `scripts/runpod_chain.sh`, 48 min, gold-58 SWA 0.9093) **reads 0.927 solo**. The Kaggle CLI refreshes its OAuth token itself on the first call ≥ 30 min after expiry (traps 20, corrected 2026-09-26). The pass cannot leave Kaggle — it reads the DICOMs. The competition's DICOM trees are **`train_series/` and `test_series/`** (traps 37). Session history, in-flight items and next actions: [docs/handoff.md](docs/handoff.md); the LB progression 0.500 → … → 0.913 → 0.939 → 0.942 (and 0.918 solo) is the Scoreboard in [docs/experiments.md](docs/experiments.md).

## 📚 Documentation map — read the relevant one before acting

| File | What it holds | Read it when |
|---|---|---|
| [docs/handoff.md](docs/handoff.md) | Session state, what changed last, next action | **First, always** |
| [docs/traps.md](docs/traps.md) | Bugs and **silent** failure modes, tiered by damage | Before writing pipeline code |
| [docs/experiments.md](docs/experiments.md) | Every measurement, with a verdict | Before proposing an experiment |
| [docs/proposals.md](docs/proposals.md) | **Ranked backlog as testable cards** — live cards (P-45…P-58, P-18) on top with full bodies, P-00…P-44, P-49, P-53 closed as one-line pointers (rewritten 2026-09-27 after a four-reviewer audit) | When choosing what to do next |
| [docs/research.md](docs/research.md) | Literature + prior-competition research behind the cards (18-agent workflow, critic-fixed) | Before changing a training parameter or model |
| [docs/brainstorm.md](docs/brainstorm.md) | Open questions and strategy notes only | When a question needs a browser |
| [docs/setup.md](docs/setup.md) | Bootstrapping a new machine | New clone / new laptop |

Two conventions that keep these useful:

- **`experiments.md` is append-only.** Every entry carries a verdict (✅ KEEP / ❌ DEAD END /
  🔁 INCONCLUSIVE / ⏳ PENDING). Check it before proposing anything so we never re-run a
  settled question or resurrect a dead end. Untried ideas are **cards in `proposals.md`**,
  written *before* running: hypothesis → origin → measure → noise floor → if-works / if-fails.
- **Update `handoff.md` at the end of every session.** It is the only file that answers
  "what was I doing?"

## 🛠 Project skills (`.claude/skills/`)

Three slash commands encode the workflows above so they are followed the same way every time:

| Command | What it does | Owns |
|---|---|---|
| `/try-out` | Turns an idea or a `P-nn` card into one edit to `src/` + a **smoke** kernel run, then stops. Never pushes a real run, never submits — both need your go-ahead. | `src/`, `kaggle/*/`, card status |
| `/update` | Routes every new finding to exactly one doc, with a verdict gated by the noise floor. Commits and pushes. | `experiments.md`, `proposals.md`, `traps.md`, `brainstorm.md`, this file |
| `/handoff` | Writes the new `docs/handoff.md` session entry (in-flight table, decisions, next actions). Runs `/update` first if findings are unlogged. Commits and pushes. | `docs/handoff.md` |

**The noise floor governs whether any result counts as evidence.** With 58 gold studies the
Hanley–McNeil SE of an AUC near 0.8 is ≈0.09 (a 95% interval of ±0.17), and the top ten
public-LB teams span 0.006 in total. So a gold-AUC difference under ~0.05, or a public-LB
difference under ~0.005, is **inconclusive, not a win**.

**The OOF floor is measured, not assumed** (2026-08-29, kernel v11: two seeds of the same fold-0
config): **0.008 macro, and ~0.03 per label** — the same two runs move Fracture by 0.028 on seed
alone. A single per-label story below 0.03 is not evidence; a *consistent sign across many
labels* is, because seed changes scatter signs.

## ⛔ Hard constraints

1. **NEVER select the P100 accelerator.** Kaggle's PyTorch ships no Pascal CUDA kernels, so
   the session dies at the first convolution. Set `"machine_shape": "NvidiaTeslaT4"` in
   `kernel-metadata.json` and re-check it on every new or forked kernel.
2. **Never download the competition images in bulk** (~570 GB). Train on Kaggle.
3. **Never sort DICOM slices by filename** — measured ρ = −0.012 vs. true spatial order, and
   it fails silently.
4. **Never hard-code `/kaggle/input` paths** — all three of our inputs resolve to the
   non-obvious layout. Keep `resolve_dir()` and its glob fallback.
5. **`FORCE_SMOKE = True` on the first push after any edit.** A crash in the inference cell
   after six hours of training costs an entire session.
6. **Edit `src/kaggle_pipeline.py`, never the generated `.ipynb`.**

Full reasoning and 12 more failure modes in [docs/traps.md](docs/traps.md).

## Layout

```
CLAUDE.md               this file — index + verified facts
docs/                   handoff, traps, experiments, proposals, research, brainstorm, setup
src/kaggle_pipeline.py  THE PIPELINE, percent-format (runs as .py AND becomes the notebook)
src/cache_pipeline.py   preprocessing-cache kernel (P-01): DICOM -> uint8 once, laterality, site proxy
src/nbgen.py            percent-format .py -> .ipynb
src/build_targets.py    targets + leak-safe folds -> artifacts/targets.csv
src/label_audit.py      per-language / per-label audit of the LLM label sources
src/oof_epoch_analysis.py  P-22: checkpoint-policy analysis on the per-epoch OOF csvs (no GPU)
src/dicom_probe.py      DICOM header / ordering / normalisation audit
src/baseline_infer.py   standalone inference smoke test
src/cache_selftest.py   builder vs on-the-fly preprocessing, both cache schemes, bit for bit (run before any cache push)
src/window_head_test.py unit checks: windows, WindowAttnHead, timm offline load, param_groups coverage
src/blend_check.py      P-23 acceptance rule on fold-0 OOF csvs (rho, blend gain, per-label table) -- NOT for train_all members (traps 32)
src/build_fork.py       P-27: notebook_score_0.942.ipynb (cells 0-49 verbatim) + our infer pipeline as a payload -> kaggle/rsna-knee-fork/
src/targets_test.py     unit checks for the prediction-table teacher path in build_targets.py (quantile_match, mix_teacher, validation, default md5)
src/build_distill_table.py  P-38: rank-mean of complete 5-fold OOF sets -> artifacts/teacher/selfdistill_v1.csv (the self-distillation teacher table)
src/build_teacher_pass.py   P-39: notebook_score_0.942.ipynb's Raptor branch (verbatim, 2 token patches) + chunk preamble -> kaggle/rsna-knee-teacher/ (--shard/--n-shards/--limit/--slug/--kernel-source/--check)
src/merge_teacher.py    P-39: raptor_teacher_shard*.npz (+ partial flushes) -> artifacts/teacher/raptor_teacher.csv (view-weighted mean probabilities)
src/teacher_pass_test.py  checks for build_teacher_pass.py + merge_teacher.py (7 cells, patches once, gold-free disjoint shards, resume accumulation, merge rules)
src/teacher_plausibility.py  P-39: a merged Raptor table vs the hard LLM teacher (per-label AUC / rho / operating points, coverage) -- a plausibility read of the pass, never a verdict (traps 39)
src/watch_submission.py  P-41: poll one submission until scored; logs the true sent -> scored time to artifacts/submission_timing.csv (run in the background after every submit)
scripts/runpod_bootstrap.sh  off-Kaggle runner: setup | train <arm> | ship <arm>   (requirements-gpu.txt)
scripts/runpod_chain.sh      one unattended pod job: inputs, 4 parallel c02 pulls, blob verify, teacher-table check, train <arm>, ship <arm>
notebook_score_0.942.ipynb  the public "DINOsaur V5" inference graph (public LB 0.942, trains nothing) -- input of build_fork.py
kaggle/rsna-knee-train/     generated training notebook + kernel-metadata.json (one production arm via ARM_ONLY sed)
kaggle/rsna-knee-folds/     second training slot (ARM_ONLY sed; historically FIVE_FOLD=True); mounts c02 + timm weights too
kaggle/rsna-knee-infer/     MODE="infer" copy of OUR blend alone (submissions #1-#11) / one member solo (#18-#21) / our Raptor-distilled pair (#23; committed render = **v20 = #23**, `v09r` + `v08r`, submittable; v19 = #21 `v08r` 0.918; v18 = P-41 SMOKE, NOT submittable; v17 = #20 0.927)
kaggle/rsna-knee-fork/      GENERATED by src/build_fork.py -- the public 0.942 graph + our arm; the kernel that gets SUBMITTED from #12 on
kaggle/rsna-knee-teacher/   GENERATED by src/build_teacher_pass.py -- the Raptor teacher pass over training-study chunks (committed render = shard 2/3 of the FULL pass, LIMIT 0: a push starts a ~3 h session; v4 = shard 0, v5 = shard 2, both done and pulled -- the pass is complete, do not re-push)
kaggle/rsna-knee-teacher-b/ GENERATED with --slug tiankljucanin/rsna-knee-teacher-b -- shard 1/3 of the full pass for the second GPU slot (v1 done 2026-09-24, output pulled)
kaggle/rsna-knee-cache-a/   c01 cache kernel, shard 0 of 2 (-b: shard 1) -- committed notebooks, do NOT regenerate (traps 27)
kaggle/rsna-knee-cache2-a/  c02 cache kernel, shard 0 of 4 (-b/-c/-d: SHARD=1/2/3 sed'd in at build time)
data/  models/  artifacts/   all gitignored (see docs/setup.md)
```

## The main workflow

`src/kaggle_pipeline.py` is the single source of truth. Percent-format (`# %%` /
`# %% [markdown]`) means the same file runs locally as a plain script *and* converts to the
Kaggle notebook.

```bash
export PYTHONUTF8=1 PYTHONPATH=src         # both needed; run from the repo root
python src/kaggle_pipeline.py              # local CPU smoke run
python src/nbgen.py src/kaggle_pipeline.py \
       kaggle/rsna-knee-train/rsna-knee-train.ipynb
kaggle kernels push   -p kaggle/rsna-knee-train
kaggle kernels status tiankljucanin/rsna-knee-train
kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out
```

Other local checks:

```bash
python src/build_targets.py                     # must print teacher gold macro-AUC 0.8948 (blend), 0.8934 (rank, diagnostic)
python src/label_audit.py                       # per-language / per-label label audit -> artifacts/label_audit.md
python src/dicom_probe.py                       # header / ordering audit
python src/baseline_infer.py --slices 3         # standalone inference smoke test
python src/oof_epoch_analysis.py                # P-22: best-epoch vs last-epoch from the per-epoch OOF csvs -> artifacts/oof_epoch_analysis.md
```

Use the repo's `.venv` (`docs/setup.md`): `.venv/Scripts/python.exe` on Windows — CPU torch,
scikit-learn, pandas; `requirements.txt` is the pin list, so the environment moves between
machines.

**`FORCE_SMOKE`** at the top of the config cell: `True` = minutes-long end-to-end check
(1 fold, 1 epoch, 2 slices/slot, 24 studies scanned); `False` = real 5-fold run; `None` =
auto (smoke locally, real on Kaggle).

**`MODE`** next to it: `"train"` trains then infers; `"infer"` loads `{version}_fold*_best.pt`
from a mounted kernel output and only predicts — **this is what gets submitted**, because a
code competition re-runs the notebook on the hidden test and a training notebook would
retrain there. `"auto"` picks `infer` when such checkpoints are mounted.

**Submitting a notebook version** (works from the CLI, no browser needed):

```bash
kaggle competitions submit rsna-knee-abnormality-detection \n       -k tiankljucanin/rsna-knee-train -v <version> -f submission.csv -m "<what changed>"
```

**The inference kernel** is generated from a sed'd copy, the same pattern the cache shards use —
`MODE="infer"` because `"auto"` requires *every* configured fold to have a checkpoint and would
otherwise decide `"train"` and re-train at rerun, and `FORCE_SMOKE=False` because smoke sets a
0.4 h runtime guard. **What it blends is `INFER_MEMBERS`** in the config cell (P-21): every mounted
`{version}_fold*_best.pt` of every listed version is one member of a flat rank-mean, a listed
version with no checkpoint is fatal, heads are per member, and the test set is decoded once and
shared by all members (equality-checked in the log). Submit with `-k tiankljucanin/rsna-knee-infer`:

```bash
sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "infer"/' src/kaggle_pipeline.py > /tmp/infer.py
python src/nbgen.py /tmp/infer.py kaggle/rsna-knee-infer/rsna-knee-infer.ipynb
kaggle kernels push -p kaggle/rsna-knee-infer
```

Read only a kernel's log, without pulling hundreds of MB of checkpoints, with a pattern that
matches nothing: `kaggle kernels output <slug> -p <dir> --file-pattern "no_match"`.

**Fold-0 A/B arms** run back to back in one kernel via the `ARMS` list at the top of the config
cell — each arm gets its own `version`, so `{version}_fold0_*` never collide, and an arm that
raises is logged and skipped rather than killing the session. `ARM_FOLDS` pins them to fold 0;
without it a real-mode run inherits `folds=(0,1,2,3,4)` and smoke mode cannot reveal that
(traps 12d).

**Production arms (P-28, 2026-09-21)** are the `PROD` preset — `train_all=True` (every non-gold study
trains, the 58 gold rows are the validation, reported only), **8 epochs** (P-29: 16 over-trained), `swa_last=3` (`_best.pt` = the mean
of the last three EMA snapshots, `_lastema.pt` beside it), `ckpt_policy="last"` — and **one arm per kernel**
through the `ARM_ONLY` build flag (historically one per slug; since P-31 two arms fit one session through
`PARALLEL_ARMS`, one per GPU — see below). Such members have **no OOF**: never run `blend_check.py` on them,
`oof_eval` scores gold-58 only (traps 32). A resume runs in the *sibling* slug with the other kernel's slug in
`kernel_sources` (traps 31). Build/push (smoke first, then `FORCE_SMOKE = False`):

```bash
sed -e 's/^ARM_ONLY = ""/ARM_ONLY = "v09a"/' src/kaggle_pipeline.py > artifacts/train_v09a.py
python src/nbgen.py artifacts/train_v09a.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
kaggle kernels push -p kaggle/rsna-knee-train        # grep the .py for 'ARM_ONLY = "v0' before a real push
```

**Two arms per session (P-31, 2026-09-22).** The `NvidiaTeslaT4` machine has **two** T4s (traps 34). `PARALLEL_ARMS`
(sed'd at build like `ARM_ONLY`, exclusive with it) makes Section 8 spawn one child process per arm on its own GPU —
this very file as the child's script (`RSNA_CHILD=1`, `RSNA_ARM=<arm>`, `CUDA_VISIBLE_DEVICES=<i>`, `RSNA_TRAIN_ONLY=1`),
each writing `/kaggle/working/<arm>.log`; `nbgen` embeds the pipeline text (zlib + base64 + sha256) only when
`PARALLEL_ARMS` is non-empty. The parent prints a heartbeat (log tails, `nvidia-smi`, host RAM) and judges each child by
its `_best.pt`, never by its exit code. Read the children's logs with `--file-pattern "\.log$"`.

```bash
sed -e 's/^PARALLEL_ARMS = ()/PARALLEL_ARMS = ("v09b", "v09c")/' -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' \
    src/kaggle_pipeline.py > artifacts/train_ab_real.py       # smoke first: leave FORCE_SMOKE = True
python src/nbgen.py artifacts/train_ab_real.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
grep -E '^(FORCE_SMOKE|PARALLEL_ARMS|ARM_ONLY) = ' artifacts/train_ab_real.py    # before every push
kaggle kernels push -p kaggle/rsna-knee-train
```

Window-mode arms may train several studies per step (`batch_studies`, P-32: one encoder pass over all their windows,
the BatchNorm batch; loss normalised per study; evaluation and inference stay at one study) and augment on the GPU
(`aug="light"`, P-33: affine + gamma/gain, no flips; never at inference). `src/window_head_test.py` checks both.
The fork builder's `--anchor-preset parent` builds the flat-0.60 hedge (submission #16).

The legacy **5-fold** run is a sed'd copy into the second kernel:

```bash
sed 's/^FIVE_FOLD = False/FIVE_FOLD = True/' src/kaggle_pipeline.py > /tmp/folds.py
python src/nbgen.py /tmp/folds.py kaggle/rsna-knee-folds/rsna-knee-folds.ipynb
kaggle kernels push -p kaggle/rsna-knee-folds
```

**Kaggle runs two GPU sessions at once** (verified 2026-08-29: `rsna-knee-train` and
`rsna-knee-folds` ran concurrently), so an A/B batch and an ensemble run can share a sitting.

**Cache kernels** (CPU, run in parallel with training):

```bash
python src/nbgen.py src/cache_pipeline.py kaggle/rsna-knee-cache-a/rsna-knee-cache-a.ipynb
sed 's/^SHARD = 0 /SHARD = 1 /' src/cache_pipeline.py > /tmp/cache_b.py
python src/nbgen.py /tmp/cache_b.py kaggle/rsna-knee-cache-b/rsna-knee-cache-b.ipynb
kaggle kernels push -p kaggle/rsna-knee-cache-a; kaggle kernels push -p kaggle/rsna-knee-cache-b
```

**Cache v2 (`c02`) kernels** (CPU, 2026-08-30) — `src/cache_pipeline.py` defaults to `SCHEME = "c02"`,
`N_SHARDS = 4`; the c01 kernels are committed notebooks and are never regenerated (traps 27):

```bash
for i in 0 1 2 3; do s=$(echo abcd | cut -c$((i+1)));
  sed "s/^SHARD = 0               #/SHARD = $i               #/" src/cache_pipeline.py > artifacts/cache2_$s.py
  python src/nbgen.py artifacts/cache2_$s.py kaggle/rsna-knee-cache2-$s/rsna-knee-cache2-$s.ipynb
  kaggle kernels push -p kaggle/rsna-knee-cache2-$s; done
python src/cache_selftest.py          # BEFORE any cache push: both schemes, both modules, bit for bit
```

**Two cache schemes, one pipeline.** `Config.cache_scheme` (`"c01"` 224/16 dense per-study files; `"c02"`
336, budgets 18/12/12/14/8/8, band 2–98 %, 64-study blobs) resolves through `cache_version_for(cfg)`; every
mounted shard of every scheme is indexed (`CACHE_INDEX[version]`) and each **arm** or **member** reads the
one its config names. Arms are one edit: `("v08w", {**C02, "backbone": "dinov2", "img_size": 224})`.
`window_mode="random"` + `head_type="window_attn"` is the P-25 member (Dataset ships uint8 + window
indices; the model gathers/resizes on the GPU). `backbone="timm:<arch>"` loads `<dir>/model.safetensors`
offline (Datasets `timm-coatnet-rmlp-1-rw-224`, `-2-rw-384`). **Inference** groups members by
`INFER_CACHE_KEYS` (one decode-once pass per cache geometry) and applies `INFER_MEMBER_KEYS` per member;
`INFER_OVERRIDES = {version: {member keys}}` gives old members TTA (`tta_offsets`, `tta_pool`) or an
`eval_windows` cap. **`MODE="oof_eval"`** scores each `INFER_MEMBERS` fold-0 checkpoint on its held-out
studies with those settings → `{version}_fold0_tta_oof.csv` for `src/blend_check.py` (P-12 measurement).
Local checks before any push: `python src/cache_selftest.py`, `python src/window_head_test.py`, then the
smoke run (`FORCE_SMOKE = True` locally trains both arms on the 3 sample studies and infers).

**Off-Kaggle training (RunPod, P-24):** `scripts/runpod_bootstrap.sh setup` lays the inputs out under
`/kaggle/input` exactly as Kaggle mounts them (CSVs, label tables, weights, the four c02 shards via
`kaggle kernels output`, verified by file count + bytes), `train <arm>` runs one arm (`RSNA_ARM`,
`RSNA_TRAIN_ONLY=1`, `RSNA_WORKERS=8`, `RSNA_RUNTIME_H=40`; resumable), `ship <arm>` publishes
`_best.pt` + `_oof.csv` as Dataset `rsna-knee-ckpt-<arm>` for the infer kernel. Inference stays on Kaggle.

**Resuming:** five folds do not fit in one 9 h session. When the runtime guard fires, each
fold has written `{version}_fold{k}_last.pt` and inference is skipped. Attach that run's
output as an input to a new run and it resumes from the last epoch. Inference runs only once
every fold is complete, so a half-trained ensemble is never submitted.

Local env is **CPU-only** and exists for CSV/report analysis, header work, notebook
authoring, and CLI orchestration — **not** training.

## Verified data facts

Checked directly against the downloaded CSVs on 2026-08-28.

`data/train.csv` — 4,407 studies:
`StudyInstanceUID, Report, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

- **All 4,407 studies have a `Report`. Exactly 58 have labels** (all 12 filled, strictly
  `0`/`1`); the other 4,349 have every label blank.
- Reports are multi-line free text in ~9–12 languages. The file is **58,556 physical lines
  for 4,407 records** — use a real CSV parser, never line splitting.
- Positive rate on the 58: Effusion 60%, Synovitis 47%, Medial Meniscus 45%, ACL 41%,
  Lateral Meniscus 40%, PF OA 36%, Contusion 33%, Fracture 31%, Medial OA 26%, Baker's 21%,
  Lateral OA 19%, MCL 16%.

`data/train_series.csv` — 24,371 series over all 4,407 studies. Planes: Sagittal 9,864 /
Coronal 8,609 / Axial 5,898. Series per study 3 / **5 median** / 14.

The DICOM trees on Kaggle are **`train_series/<study>/<series>/*.dcm`** and **`test_series/…`** under
`/kaggle/input/competitions/rsna-knee-abnormality-detection/` — there is no `train_images/` (verified 2026-09-23 from the
kernels' mount listing; traps 37).

`data/test.csv` — **`StudyInstanceUID` only, 3 studies** (placeholder; the real test set is
served at rerun). `sample_submission.csv` — the 12 label columns, all `0.5`.

### Three facts that determine the whole design

1. **`train.csv` has `Report`; `test.csv` does not.** Text exists when fitting and is absent
   when predicting, so a text branch is **impossible at inference** — it would have nothing
   to read. Reports are usable *only* as training targets, an auxiliary task dropped at
   inference, or a per-sample confidence weight. This is the most tempting wrong turn in the
   whole competition, since it is advertised as multimodal.
2. **58 labels is the entire supervised signal.** The real task is converting 4,349 reports
   into trustworthy targets — a weak-supervision problem wearing a computer-vision costume.
3. **`Fluid_Sensitive` and `Fat_Suppression` are degenerate as delivered** — only `(1,1)`
   and `(0,0)` occur across all 24,371 series, never a mixed pair. Recover both from the
   DICOM headers. (`Anatomical_Plane`, by contrast, is trustworthy.)

## What the metric implies

Macro ROC-AUC is invariant to any strictly increasing per-label transform, so:

- **Calibration and thresholds are worth nothing.** Only rank order is read.
- **Ensemble by averaging ranks, not probabilities.** Probability averaging lets the most
  confident model dominate; rank averaging combines exactly what AUC reads.
- **Every label costs the same.** One label stuck at chance forfeits ~(M−0.5)/12 — about
  0.029 at M=0.85 — however good the other eleven are. **Rare findings deserve more
  attention than common ones**, because that is where a model most easily lands at chance.
- Prevalence is not guaranteed to match across train / public / private. AUC largely survives
  that; a baked-in threshold does not.

## Where the field is

Public LB on 2026-08-28: **top 0.952**, ranks 2–9 spanning 0.946–0.949 — the top ten inside
a **0.006 band**, from 2,559 teams (2,957 teams on 2026-09-03; top 0.952, ranks 2–10 spanning 0.947–0.950 — API-verified). **We are at 0.942** (submissions #13 and #15, read 2026-09-22: the public 0.942 notebook's graph verbatim, with our c02 arm at β 0.10 and without it — P-27; β 0.20 scored 0.939; our own blend alone is **0.913**, submission #11, read 2026-09-03: two DINOv2 heads + five concat folds + one ConvNeXt-T + the three c02 window-attention members `v08w` / `v10c` / `v09h` with `v09h` as five folds, one vote per version; 0.912 with `v09h` as one fold, 0.909 without it, 0.900 before the c02 members; 0.877 for the best single model on the LB). The Efficiency Prize has its own leaderboard, published
as a notebook (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, readable via
`kaggle kernels output`); its leader is also top-5 on accuracy, so efficiency is not being
bought with score.

**The top public notebooks are one shared, heavily-forked community ensemble whose own
author warns it is "likely overfit to the public leaderboard"** after a fork-and-republish
race chasing 0.001–0.003 movements. Expect a private shakeup. Prefer a pipeline you can
validate over a blend you can only submit.

**Decomposition of one 0.936 notebook, read cell by cell on 2026-08-30** (`crazy_good_rsna.ipynb`, a
port of "DINOsaur V10"; [docs/research.md](docs/research.md) §2.7.1): DINOv2-S branch ≈ 0.899 → +
16-slices-as-channels ViT + RadImageNet R50 frozen-feature heads + stacking calibrator ≈ 0.920 → +
CoAtNet-2 @384 over 64 slices (0.924 alone) 0.935 → + gold-58-tuned weights 0.936. It trains nothing:
every member is a mounted public checkpoint. **Our DINOv2 recipe is at parity with theirs; the gap is
the number of families, which is P-23.** **Measured 2026-08-30 (evening):** the 0.924 member's gain is the **wide slice band +
per-label window attention (+0.007 on the same DINOv2-S) and the hybrid backbone (+0.004, different errors on
the menisci) — not the 384 px** (CoAtNet-1 @224 = 0.8683 beats CoAtNet-2 @384 = 0.8641 at ⅓ of the cost); the
three c02 arms blend with the c01 members to 0.8820 (experiments.md ⭐ "What made the 0.936 notebook good"). Its own counter-example: three backbones on the same input
blended to +0.001 — diversity has to come from the input representation and pretraining regime.

Consensus architecture there: DINOv2 ViT-S/14 as the workhorse (with DINOv3 and RadImageNet
ResNet-50 rank-blended alongside), 2.5D one-series-per-slot with a presence mask, laterality
normalisation, attention pooling over slices, and **LLM-read report labels as the de-facto
standard target source**. Not EfficientNet — that was the early-baseline era.

Our own measurements of these choices are in [docs/experiments.md](docs/experiments.md).

## Submitting

This is a **code competition**: you submit a notebook, and Kaggle re-runs it against the
hidden test set. You do not upload a CSV.

Working metadata: `enable_gpu: true`, `enable_internet: false`, weights mounted via
`dataset_sources` / `model_sources` (never downloaded at runtime), predictions written to
`/kaggle/working/submission.csv` with the exact `sample_submission.csv` columns.

```bash
kaggle competitions submissions rsna-knee-abnormality-detection
kaggle competitions leaderboard rsna-knee-abnormality-detection -s --csv
```

**Unverified** (community-sourced, never read from the competition pages): the **≤9 hour**
runtime limit and the internet-off requirement. `enable_internet: false` in every public
submission corroborates the latter.

## Compute strategy

~570 GB across ~819,000 training DICOMs. Keep only the CSVs, the LLM labels, and a handful
of sample DICOMs locally. Run **Kaggle-to-Kaggle**, each kernel mounting the previous one's
output so nothing large crosses your machine:

```
metadata/header scan (CPU)  →  cache build (CPU)  →  train (T4 GPU)  →  submit
```

The cache-build step is `src/cache_pipeline.py` (P-01 in [docs/proposals.md](docs/proposals.md));
the training-side loader that reads the cache is the follow-up.

**Training needs only the cache, never the DICOMs** (verified in the loader, 2026-08-30): the two cache
shards (~21 GB uint8 `.npy` + manifests), the CSVs, the LLM label tables and the backbone weights are the
whole training input, so a training run can leave Kaggle (free Colab, any GPU box) while **inference must
stay a Kaggle notebook** (code competition). Checkpoints come back as a private Kaggle Dataset, which the
infer kernel already resolves. Kaggle GPU quota is **30 h/week per account**; the P-24 card holds the
no-cost expansion options (2×T4 sessions, Colab runner). A derived cache must stay private.

## Rules: AI assistance and data handling

**Using Claude Code / AI agents to develop the solution is permitted.** Nothing in Kaggle's
framework prohibits AI coding assistance — it is ordinary tooling, an LLM-agent-assisted
team publicly won a Kaggle competition in March 2026, and the external-resources provision
turns on whether a resource is *publicly available at minimal cost*, not on who wrote the
code. The binding obligations are the usual ones: one account, no private code sharing
outside your team, winners deliver working code and documentation.

**Two real constraints:**

1. **Everything you rely on must be publicly available and free to all.** Pretrained weights
   and shared LLM label tables qualify because they are published as Kaggle Models/Datasets.
   A private or paid asset does not.
2. **Sending report text to a hosted third-party LLM API is genuinely open.** The Data
   Security provisions plausibly forbid transmitting competition data off-platform. The
   tension: it is now widespread practice — one of the most-downloaded public label sets is
   openly titled "GPT-5.6-Sol" — and the host has not visibly objected, which is evidence of
   tolerance but **not a ruling**. Safe path: mount an existing public label table, or run
   open-weights models locally or inside a Kaggle notebook. **This is about moving
   competition data off-platform; it is unrelated to using Claude Code on your own source
   code, which is fine.**

Read the rules text before relying on either point — the above is inference from Kaggle's
general framework plus observed community behaviour, not a quotation. Also: keep report text
and `StudyInstanceUID`s out of any public location. `artifacts/` contains both and is
gitignored for that reason.

⚠️ **RadImageNet weights carry no stated licence** (checked 2026-08-28: code MIT, paper CC BY
4.0, data "by request"; an earlier version of this file said CC-BY-NC-SA-4.0, which could not
be verified). Treat as restrictive until radimagenet.com's Terms & Conditions and the
competition's winner-licence clause are read in a browser. DINOv2 is Apache-2.0; timm
ConvNeXt weights are licence-clean and are the first choice for a CNN ensemble member.

## Timeline

| Date | Event |
|---|---|
| 2026-07-30 | Launched |
| **2026-10-15** | Entry deadline **and** team-merger deadline |
| **2026-10-22 23:59 UTC** | Final submission deadline (API-verified) |
| 2026-11-05 | Winners announced |
| 2026-11-29 – 12-03 | RSNA 2026, Chicago |

Category **Research**, reward **$77,000** covering the accuracy leaderboard **plus a
separate Efficiency Prize track**.

## Provenance

**API/CSV-verified (high confidence):** every number in "Verified data facts"; the deadline,
category, reward, team count; leaderboard standings; the existence and metadata of the public
notebooks and label datasets; CLI auth and entry status; everything in
[docs/experiments.md](docs/experiments.md) marked as measured.

**From public notebooks** (notably `pilkwang/rsna-knee-baseline-v1`, 454 votes — an unusually
rigorous write-up worth reading in full): the metric reasoning, sequence-slot design, DICOM
ordering trap, Hanley–McNeil noise argument, and the graded-vs-thresholded label insight.

**Community-sourced, still unverified:** the ≤9 h runtime limit, internet-off, the ~570 GB /
819k-file totals, and the fold-leakage magnitudes.

**Corrected 2026-08-28:** an earlier version of this file hypothesised that leaders were
exploiting report text available at inference time. That is **wrong** — `test.csv` has no
`Report` column. The high public scores come from LLM-derived training labels plus large rank
ensembles of self-supervised ViTs, and partly from public-LB overfitting.

Kaggle competition pages are JS-rendered, so `WebFetch`/`curl` return only the SPA shell and
the CLI exposes no command for the overview, rules, or discussion prose. Anything depending
on those remains unread and is flagged as such.
