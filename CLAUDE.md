# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this repo

Competition workspace for the Kaggle **RSNA Knee Abnormality Detection** challenge (RSNA
2026 AI Challenge). Predict **12 independent binary findings per knee MRI study**, scored
by **macro ROC-AUC** (unweighted mean of 12 per-label AUCs).

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Current state (2026-10-16, 20:15 UTC)

One block, kept current by `/update`. Session history is in [docs/handoff.md](docs/handoff.md); every number, with its verdict,
is in [docs/experiments.md](docs/experiments.md).

| | |
|---|---|
| Public LB (2026-10-16, 18:30 UTC) | **Top 0.968**; 10th 0.964; 100th 0.959. **605 teams ≥ 0.958 (481 at exactly 0.958; 278 at 11:29)**, 920 ≥ 0.957 (315 at exactly), 1,513 ≥ 0.956, 1,756 ≥ 0.954, 1,996 ≥ 0.950; 6,103 teams. Both blocks are copies of heliosli's notebooks (experiments.md 2026-10-16 "Submission #91 and the 0.957 / 0.958 frontier"). **0.957** = the 0.956 recipe with nartaa's 224 px CoAtNet in the Raptor half (0.65 × 384 + 0.35 × 224) and **R6 + flip TTA** in place of R7. This R6 is bundle `fe055406`: ConvNeXt-S on Opus targets, R7's trainer, fastMRI unconfirmed; it is not the S6 re-upload. **0.958** = that plus four per-finding Raptor-half weights (Synovitis 0.80, Lateral OA 0.20, PF OA 0.60, Baker's 0.35): "user decision 10-16" in the notebook, `PROBE_CLASS = None` in the published version, so public probing is possible but unconfirmed. nartaa's **s6fall** is public too: S6's architecture on JEV labels, **0.950 alone, no OAI, no fastMRI**, CC BY-NC. Nothing public reads above 0.958 (18:35 UTC); the top 9 did not move all day. **We are rank 617 at 0.957** (12th of 315 at 0.957; C10 / C10w). Our best OAI-free number is **0.953** (C8 #88 = R7 + B18; fastMRI inside R7); free of OAI and fastMRI **0.952** (C11 #92); own models **0.945** (B22 #86) |
| Our best | **LB 0.957** = #93 **C10**: heliosli's public R6 + flip TTA notebook + B21 at β 0.20 (`rsna-knee-fork957` v1), **pick 2's candidate**; and #94 **C10w**: his 0.958 `blend-mine-in` + B21 at β 0.20 (`rsna-knee-fork958` v1), one tick under its own anchor. Both carry OAI on three paths (both CoAtNets, `v15co`); Tian accepted the OAI rule risk for pick 2 on 10-09. **OAI-free 0.953** = #88 **C8**: R7 alone + B18 at β 0.45 (`rsna-knee-fork956` v2), **the pick-1 candidate if fastMRI is acceptable**, an open check (brainstorm.md "Does fastMRI pass the host's 10-14 test?"). **Free of OAI and fastMRI, 0.952** = #92 **C11**: nartaa's s6fall alone + B18 at β 0.45 (`rsna-knee-forks6f` v1), pick 1 otherwise. **0.956** = #91 C9, #87 C7. **Own models 0.945** = #86 **B22** = `v15c` + `v13b3` + `v15cr` (`rsna-knee-infer` v78). **0.944** = #67 **B18** = `v15c` + `v13b3` (C8 / C11's leg). **0.946 with OAI** = #79 **B21** = `v15co` + `v13b3` (the pick-2 leg). **Solo:** `v15co` 0.945 (OAI). OAI-free: `v15c` 0.942 (ConvNeXt-T `in12k_ft_in1k` @ 288, own optimiser), `v15cb` 0.943, `v15c2` 0.941, `v13b3` 0.940. **Blend rule** (held six times): a flat rank-mean ≈ its members' mean + a gain that grows with the number of *families*. A same-family vote on different targets (no Raptor) counts like a new family (B22 #86). **Fork rule** (held on every fork read): a fork ≈ (1 − β) × anchor + β × leg + a diversity gain k β (1 − β): +0.002–0.004 (+0.0014–0.0026 on heliosli anchors at β 0.20). **On a ≥ 0.957 anchor B21 at β 0.20 is break-even to a cost** (C10w #94), so the own leg pays in pick 1 only (≈ +0.00045 per +0.001 at β 0.45). Some anchors decorrelate more: S6 on the CoAtNet gains +0.0055–0.0075, and label lineage matters (#84). CNN seed spread s = 0.003; ConvNeXt s_c = 0.001 (#70) |
| Production recipe | CNNs on the `v13h` recipe: c03 input, CNN LR 3e-4 uniform, frozen BN, heavy aug, drop-path 0.1, 30 epochs, SWA of 27–29. **ConvNeXt-T on its own optimiser** (P-69, ✅ 10-08): AdamW 1e-4 + per-stage decay 0.9, head 1e-3, 20 epochs, SWA 17–19, the same data side (traps 46: a family needs its own optimiser). Targets 0.5 LLM + 0.5 quantile-matched Raptor: the only target change that ever transferred (the Claude relabel, the silent-cell mix, D4, self-distillation, the same-family OOF student and, 10-10, the cross-family CoAtNet OOF mix P-68 read no lift; 10-12, report-only `v15cr` −0.010 solo, a diversity effect only, P-85; the label side is closed) |
| Read 10-16 | **#92 C11 = 0.952** (🔁: the fastMRI-free pick 1, in place of C6; k ≈ 0.015–0.023 vs B18). **#93 C10 = 0.957** (🔁 = its anchor; pick 2's candidate, in place of C9). **#94 C10w = 0.957** (❌: one tick under its 0.958 anchor; the arm ran, since the anchor is deterministic). Only our best entry shows on the LB (rank 617, most likely C10w), so C10 vs its anchor is unread (experiments.md 2026-10-16 "Submissions #92–#94"). **Public vs private** (`src/pubpriv_sim.py` + the forum read): the public SD is ≈ 0.003, and the 0.956 → 0.957 → 0.958 steps sit inside it. Nothing predicts a drop of the plateau's level (± 0.004), but its internal order, our forks included, is a lottery on private. **Image teachers:** no public model is worth a teacher pass for pick 1's leg: each is OAI, fastMRI or the anchor's own lineage (experiments.md 2026-10-16 "Public vs private", "Public image models as a second teacher"). **The host ruled on 10-14 at 13:18 UTC** (Chris Carr, 743416). No dataset is named. Registration + licence acknowledgement passes; an institutional research agreement or an affiliation limit fails. Teams list the datasets and shared weights they used (744056: public shared weights are award-eligible). Whether fastMRI requires an affiliation is an open check. **Forum:** colum2131 (rank 31) builds targets from image-model pseudo-labels blended with LLM labels at a fixed ratio, which is our recipe in kind. The split is ≈ 30 / 70 public / private (unverified). No publishing-lock date was posted |
| Next | **The queue: [docs/candidates.md](docs/candidates.md).** **In flight: #95 C10w-b05** (heliosli's 0.958 + B21 at β 0.05, Tian's test against the generic copy; sent 20:13 UTC, ref 57282094). It is read by its position in the 0.958 tie against the anchor's verbatim copies (heliosli 517 / degnonguidi 518 at 20:12 UTC; candidates.md). **Tian, 10-16: no pick decisions until one day before the deadline (10-21).** **Shortlist (P-50):** pick 1 = **C8 #88 0.953** if fastMRI passes the host's 10-14 test (brainstorm.md), else **C11 #92 0.952**. Pick 2 = **C10 #93 0.957**, or **C10w #94 0.957** (+ the per-finding weights). **Decided 10-09 (Tian):** an OAI-trained pick 2 is accepted; no training at 336 px. **Tian, 10-10: improve the models until the picks are due.** That now means pick 1's leg: own-leg work for pick 2 is closed (proposals.md Dropped directions). **Open for Tian:** fastMRI in pick 1; **C12** (R7 + s6fall + B18, pick 1 with fastMRI, pred. 0.953–0.956 vs C8; a builder spec + a 9-h runtime check; send by 10-20); a Qwen 3.8 Max arm of P-87 (no zero-retention endpoint; **the Fable arm was dropped by Tian on 10-16**); without it, P-87 closes. **Labels (P-87):** Opus 🔁 0.9140, Gemini 3.7 Flash 🔁 0.9119, Mistral Large 4 ❌ 0.8786. Runner `artifacts/vlm_pilot/run_openrouter.py`, `OPENROUTER_API_KEY` in `.env`, key capped at $13, $2.29 spent. **After the 10-17 00:00 UTC Kaggle reset:** **P-86** (`v15cj`, one JEV arm), then **the P-67 recipe loop** (next `v14mx`, `v14r288`, `v14ep20`, …); both feed pick 1 only. **Tian's bar (10-12): +0.001 is worth considering; on a same-seed ConvNeXt pair ≥ +0.002 is a win, +0.001 "leaning" (s_c = 0.001).** A variant that clears 0.002 gets one production transfer arm. A fork on a heliosli anchor takes ≈ 4–7 h to score: send it early. Final picks by 10-22 23:59 UTC: **select both by hand** (Kaggle's default would take the two best public entries, C10 / C10w, both OAI forks). Before the picks, re-read `kaggle kernels list --sort-by dateRun` for a newer public anchor. An unattended send needs the laptop on AC with the lid open and a crash check the evening before (traps 53) |
| Budgets | Kaggle GPU: 30 h/week of *session* time, two T4s per session. **26.63 h used, 3.37 h left** (`kaggle quota` 10-16 20:15 UTC); the reset is **10-17 00:00 UTC**; a fork placeholder takes ≈ 0.2 h. Submissions: 5 per UTC day to 10-22. 10-14 used 1 (#91), 10-15 none, **10-16: 4 used (#92–#95), 1 left**. RunPod: **Tian topped up to $12.15 on 10-10, the last top-up ("tactical" spending); ≈ $5.8 is left** after P-81 / P-84 / P-85 (all pods deleted). Every pod still needs a critic-checked case and Tian's go. Community 4090s on CUDA 13.0 boot broken (traps 65); size pod worker pools from the cgroup quota (traps 66) |

## 📚 Documentation map — read the relevant one before acting

| File | What it holds | Read it when |
|---|---|---|
| [docs/handoff.md](docs/handoff.md) | Session state, what changed last, next action | **First, always** |
| [docs/traps.md](docs/traps.md) | Bugs and **silent** failure modes, tiered by damage | Before writing pipeline code |
| [docs/experiments.md](docs/experiments.md) | Every measurement, with a verdict | Before proposing an experiment |
| [docs/candidates.md](docs/candidates.md) | **The queue:** which models / ensembles to submit and which arms to train next, what each tests, its read rule, placeholder status | Before choosing the next submissions or GPU run |
| [docs/proposals.md](docs/proposals.md) | **Ranked backlog of live, testable cards** (full bodies) + **Dropped directions, each with the why** (forum / literature / our reads). A measured card leaves this file: its one-line pointer is the closed-cards index at the end of experiments.md | When choosing what to do next, and before proposing anything |
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
| `/update` | Routes every new finding to exactly one doc, with a verdict gated by the noise floor. Commits and pushes. | `experiments.md`, `proposals.md`, `candidates.md`, `traps.md`, `brainstorm.md`, this file |
| `/handoff` | Writes the new `docs/handoff.md` session entry (in-flight table, decisions, next actions). Runs `/update` first if findings are unlogged. Commits and pushes. | `docs/handoff.md` |

**Noise floors decide what counts as evidence.** The working rules are in proposals.md, "Decision-metric hierarchy".
- **Gold-58** (the 58 labelled studies): the Hanley–McNeil SE is ≈ 0.09 per label, so a macro difference under ~0.05 is direction
  only. Gold has also inverted the LB direction of recipe and target changes several times (traps 39). Members are judged by
  their solo LB.
- **Public LB:**
  - a one-seed delta needs ≥ 0.004 (P-44);
  - a two-arm mean read uses a ±0.0045 band;
  - the top ten span ≈ 0.004.
- **Fold-0 OOF** (the legacy arms): 0.008 macro and ~0.03 per label on seed alone. A consistent sign across many labels is
  evidence; one label's story is not.

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

Full reasoning and ~60 more failure modes in [docs/traps.md](docs/traps.md).

## Layout

```
CLAUDE.md               this file — index + verified facts
docs/                   handoff, candidates, traps, experiments, proposals, research, brainstorm, setup
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
src/build_fork949.py    C4 / C4b: notebook_score_0.949.ipynb (nartaa's 5 cells verbatim) + our arm -> kaggle/rsna-knee-fork949/
src/build_fork954.py    C5 / A14 / C6 (--anchor 0954), C7 / C8 (--anchor 0956) and C9 (--anchor 0956v, varshithakolupuri's 224 px variant): heliosli's notebook (7 cells verbatim, its docker image pinned) + our arm; --s6-only (S6 alone, 0954 only), --compact-anchor / --s6-anchor (no CoAtNet stage, OAI-free); --anchor 0957 (C10: heliosli's R6 + flip notebook, C_SLUG patched to the -w1 copy), 0958 (C10w, verbatim) and 0950s6f (C11: nartaa's s6fall script with component A, the OAI CoAtNet, removed; refuses a render holding any competition StudyInstanceUID, traps 73) -> kaggle/rsna-knee-fork954/ (and -fork957 / -fork958 / -forks6f)
src/targets_test.py     unit checks for the prediction-table teacher path in build_targets.py (quantile_match, mix_teacher, validation, default md5)
src/build_distill_table.py  P-38: rank-mean of complete 5-fold OOF sets -> artifacts/teacher/selfdistill_v1.csv (the self-distillation teacher table)
src/build_teacher_pass.py   P-39: notebook_score_0.942.ipynb's Raptor branch (verbatim, 2 token patches) + chunk preamble -> kaggle/rsna-knee-teacher/ (--shard/--n-shards/--limit/--slug/--kernel-source/--check)
src/merge_teacher.py    P-39: raptor_teacher_shard*.npz (+ partial flushes) -> artifacts/teacher/raptor_teacher.csv (view-weighted mean probabilities)
src/teacher_pass_test.py  checks for build_teacher_pass.py + merge_teacher.py (7 cells, patches once, gold-free disjoint shards, resume accumulation, merge rules)
src/teacher_plausibility.py  P-39: a merged Raptor table vs the hard LLM teacher (per-label AUC / rho / operating points, coverage) -- a plausibility read of the pass, never a verdict (traps 39)
src/watch_kernel.py     poll one kernel run until it ends (status timeline; fresh CLI per poll), then pull its .log / _oof.csv outputs + key lines (mid-run logs are blank, traps 20)
src/watch_submission.py  P-41: poll one submission until scored; logs the true sent -> scored time to artifacts/submission_timing.csv (run in the background after every submit)
src/build_d4_teacher_pass.py  P-45: the 0.942 notebook's D4 CoAtNet child over training-study chunks -> kaggle/rsna-knee-teacher-d4/ (checks: src/d4_teacher_pass_test.py)
src/fold_oof_summary.py  per-fold and pooled OOF summary of a k-fold version (no GPU)
src/blend_diversity_gold.py  gold-58: blend gain by pair type (same recipe / sibling / cross-family) + a fitted marginal-family model (research.md 2.7.8)
src/kaggle_log.py       print a Kaggle kernel log (the CLI's JSON) as plain lines, optionally filtered
scripts/runpod_bootstrap.sh  off-Kaggle runner: setup | train <arm> | ship <arm>   (requirements-gpu.txt)
scripts/runpod_chain.sh      one unattended pod job: inputs, 4 parallel cache pulls (CACHE_PREFIX c02 / c03), blob verify, teacher-table check, train (SEQ_ARMS; ships each arm as it ends), ship, AUTO_STOP; MAX_POD_H skips an arm that would overrun
scripts/runpod_stopper.sh    on-pod bill cap: podStop on the job's last line, at a deadline, or on /workspace/STOP_NOW (key from /proc/1/environ)
scripts/harvest_forum.py     harvest the competition forum through the API (global topic search) -> artifacts/forum/
notebook_score_0.942.ipynb  the public "DINOsaur V5" inference graph (public LB 0.942, trains nothing) -- input of build_fork.py
notebook_score_0.949.ipynb  nartaa's public 0.949 single model (OAI-trained CoAtNet-2) -- input of build_fork949.py
notebook_score_0.954.ipynb  heliosli's public 0.954 blend (nartaa 0.7 + S6 0.3), pulled 2026-10-11 -- input of build_fork954.py
notebook_score_0.956.ipynb  heliosli's public 0.956 blend (nartaa 0.5 + R7 0.5), pulled 2026-10-13 -- input of build_fork954.py --anchor 0956
notebook_score_0.956v.ipynb varshithakolupuri's 0.956 variant (+ nartaa's unused 224 px CoAtNet in the Raptor half), pulled 2026-10-14 -- input of --anchor 0956v (C9)
notebook_score_0.957.ipynb  heliosli's R6 + flip TTA notebook (0.957 by others' reads; titled 0.9576), pulled 2026-10-16 -- input for C10 (no builder spec yet; R6 from the -w1 Dataset, traps 72)
notebook_score_0.958.ipynb  heliosli's blend-mine-in (the 0.957 + per-finding Raptor weights, 0.958), pulled 2026-10-16 -- C10w
artifacts/anchors/notebook_score_0.950s6fall.py  nartaa's s6fall solo script (0.950, no OAI / fastMRI in the model; mounts his OAI CoAtNet as a fallback), pulled 2026-10-16 -- C11; gitignored: it embeds the 3 placeholder test StudyInstanceUIDs
kaggle/rsna-knee-train/     training slot 1 (PARALLEL_ARMS / ARM_ONLY builds sed'd from src); mounts c03 + the CNN weights
kaggle/rsna-knee-train-b/   training slot 2 (since 2026-10-03; c03-only mounts)
kaggle/rsna-knee-folds/     older training slot (historically FIVE_FOLD=True); mounts c03 + CoAtNet-1 weights
kaggle/rsna-knee-infer/     MODE="infer": INFER_MEMBERS as a flat rank-mean -- every submission of our own models (#1-#11, #18 on)
kaggle/rsna-knee-fork949/   GENERATED by src/build_fork949.py -- the public 0.949 model + our leg (#68 C4, #73 C4b)
kaggle/rsna-knee-fork954/   GENERATED by src/build_fork954.py -- heliosli's public 0.954 blend + our leg (v1 = C5 #80), S6 alone (v2 = A14 #81 0.944), S6 + B18 (v3 = C6 #82)
kaggle/rsna-knee-fork956/   GENERATED by src/build_fork954.py --anchor 0956 -- heliosli's public 0.956 blend + our leg (v1 = C7 #87), R7 alone + B18 (v2 = C8 #88, --compact-anchor), C8′ (v3 #89), C7′ (v4 #90), C9 (v5 #91, --anchor 0956v)
kaggle/rsna-knee-fork957/   GENERATED by src/build_fork954.py --anchor 0957 -- heliosli's R6 + flip TTA notebook + B21 at beta 0.20 (v1 = C10 #93)
kaggle/rsna-knee-fork958/   GENERATED by src/build_fork954.py --anchor 0958 -- heliosli's 0.958 blend-mine-in + B21 (v1 = C10w #94 at beta 0.20; v2 = C10w-b05 #95 at beta 0.05)
kaggle/rsna-knee-forks6f/   GENERATED by src/build_fork954.py --anchor 0950s6f -- nartaa's s6fall (no OAI stage) + B18 at beta 0.45 (v1 = C11 #92)
kaggle/rsna-knee-fork/      GENERATED by src/build_fork.py -- the public 0.942 graph + our members at weight beta (#12-#17, #22, #27; final builds only, P-40)
kaggle/rsna-knee-teacher/   GENERATED by src/build_teacher_pass.py -- the Raptor teacher pass (complete with -b; never re-push)
kaggle/rsna-knee-teacher-b/ the Raptor pass's second GPU slot (shard 1/3, done)
kaggle/rsna-knee-teacher-gold/  build_teacher_pass.py --gold: Raptor over the 58 gold studies (P-49, done)
kaggle/rsna-knee-teacher-d4/    GENERATED by src/build_d4_teacher_pass.py -- the D4 teacher pass (P-45, complete; never re-push)
kaggle/rsna-knee-eval/      MODE="oof_eval" kernel (P-12 TTA reads; historical)
kaggle/rsna-knee-stack/     the 16-slices-as-channels DINOv2 member v07s (a dead end; historical)
kaggle/rsna-knee-cache-a/   c01 cache kernel, shard 0 of 2 (-b: shard 1) -- committed notebooks, do NOT regenerate (traps 27)
kaggle/rsna-knee-cache2-a/  c02 cache kernel, shard 0 of 4 (-b/-c/-d: SHARD=1/2/3 sed'd in at build time)
kaggle/rsna-knee-cache3-a/  c03 cache kernel (C03_KW budgets), shard 0 of 4 (-b/-c/-d); its files are named c02_* on disk (traps 52)
kaggle/convnext-tiny-224-hf/  dataset metadata of the ConvNeXt-Tiny weight Dataset (Apache-2.0)
data/  models/  artifacts/   all gitignored (see docs/setup.md)
```

Which version of each kernel is committed, and whether re-pushing it starts a real run, is in the newest
handoff.md entry ("Committed renders").

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
(1 fold, 1 epoch, 2 slices/slot, 24 studies scanned); `False` = real run (one `train_all` fold per production
arm; five folds only with `FIVE_FOLD = True`); `None` =
auto (smoke locally, real on Kaggle).

**`MODE`** next to it: `"train"` trains then infers; `"infer"` loads `{version}_fold*_best.pt`
from a mounted kernel output and only predicts — **this is what gets submitted**, because a
code competition re-runs the notebook on the hidden test and a training notebook would
retrain there. `"auto"` picks `infer` when such checkpoints are mounted.

**Submitting a notebook version** (works from the CLI, no browser needed):

```bash
kaggle competitions submit rsna-knee-abnormality-detection \
       -k tiankljucanin/rsna-knee-infer -v <version> -f submission.csv -m "<what changed + its read rule>"
```

**The inference kernel** is generated from a sed'd copy, the same pattern the cache shards use —
`MODE="infer"` because `"auto"` requires *every* configured fold to have a checkpoint and would
otherwise decide `"train"` and re-train at rerun, and `FORCE_SMOKE=False` because smoke sets a
0.4 h runtime guard. **What it blends is `INFER_MEMBERS`** in the config cell (P-21): every mounted
`{version}_fold*_best.pt` of every listed version is one member of a flat rank-mean, a listed
version with no checkpoint is fatal, heads are per member, and the test set is decoded once and
shared by all members (equality-checked in the log). Keep its `kernel-metadata.json` at ≤ 49 input sources: more is refused with a bare `400 … SaveKernel` (traps 64); drop a dead
checkpoint Dataset before mounting a new one. Submit with `-k tiankljucanin/rsna-knee-infer`:

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
`kernel_sources` (traps 31). The CNN arms (`v13h` / `v13r` / `v13e` / `v13b3` / `v13e2`) override the epochs: 30 (SWA 27–29),
uniform LR 3e-4, frozen BN, heavy aug, drop-path 0.1 (P-64). Build/push (smoke first, then `FORCE_SMOKE = False`):

```bash
sed -e 's/^ARM_ONLY = ""/ARM_ONLY = "v09a"/' src/kaggle_pipeline.py > artifacts/train_v09a.py
python src/nbgen.py artifacts/train_v09a.py kaggle/rsna-knee-train/rsna-knee-train.ipynb
kaggle kernels push -p kaggle/rsna-knee-train        # grep the .py for 'ARM_ONLY = "v0' before a real push
```

**Training targets (teacher tables).** `TEACHER_TABLES` / `TEACHER_MIX` in the config cell (sed'd per session) mix
quantile-matched prediction tables into the *training* targets only: `yt__* = (1 − mix) · LLM + mix · teacher`. The `y__*`
evaluation targets stay the LLM blend. `TEACHER_SILENT_MIX` (P-62) sets a different mix on report-silent cells.
`DISTILLED_ARMS` / `DISTILLED_MIX` / `DISTILLED_SILENT_MIX` pin each distilled arm to its exact tables and mixes, and any other
build refuses to start (traps 40).
- **Where the tables live:** the private Dataset `tiankljucanin/rsna-knee-teacher-tables`, resolved through `TEACHER_PATHS`;
  locally, `src/build_targets.py --teacher-tables`.
- **Tables:** `raptor_teacher`, `d4_teacher`, `xfit_v09k`, `selfdistill_v1`, `claude_v1`, `claude_rap_v1`.
- **That Dataset is derived competition data: never make it public.**

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
**P-67 / P-68 switches (2026-10-05), all off by default:** `aug_extra` (any of lowres / thick / grid / blur / noise / sharpen, each at
`aug_extra_p`, after the heavy stack, training only), `mixup_p` (study-level mixup of a two-study batch, training only), `drop_blank_frac`
(drops near-empty windows in training **and** inference: an inference key), `snapshot_every` (EMA snapshots `_ep{e}_ema.pt`, submittable),
`train_gold` (a `train_all` arm also trains on the 58 gold rows). The arms are `v14lr` … `v14ep20` (P-67) and `v13eo` (P-68, on the
`cnnoof_v1` table built from the P-67 floor run's OOFs).
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
336, budgets 18/12/12/14/8/8, band 2–98 %, 64-study blobs; **c03** = c02 with budgets 24/24/24/14/8/8 and a 150 mm crop,
`C03_KW`, kernels `rsna-knee-cache3-*`, ≈ 51 GB, files named `c02_*` on disk, traps 52) resolves through `cache_version_for(cfg)`; every
mounted shard of every scheme is indexed (`CACHE_INDEX[version]`) and each **arm** or **member** reads the
one its config names. Arms are one edit: `("v08w", {**C02, "backbone": "dinov2", "img_size": 224})`.
`window_mode="random"` + `head_type="window_attn"` is the P-25 member (Dataset ships uint8 + window
indices; the model gathers/resizes on the GPU). `backbone="timm:<arch>"` loads `<dir>/model.safetensors`
offline (Datasets `timm-coatnet-rmlp-1-rw-224`, `-2-rw-384`, `timm-resnet50-a1`, `timm-efficientnet-b0-ra`,
`timm-efficientnet-b3-ra2`, `timm-convnext-tiny-in12k`; each kernel keeps the Kaggle image pinned at its creation, so `rsna-knee-train`
runs timm 1.0.26 / Python 3.12 and `rsna-knee-train-b` timm 1.0.29 / Python 3.13, traps 54; local and RunPod 1.0.28). **Inference** groups members by
`INFER_CACHE_KEYS` (one decode-once pass per cache geometry) and applies `INFER_MEMBER_KEYS` per member;
`INFER_OVERRIDES = {version: {member keys}}` gives old members TTA (`tta_offsets`, `tta_pool`) or an
`eval_windows` cap. **`MODE="oof_eval"`** scores each `INFER_MEMBERS` fold-0 checkpoint on its held-out
studies with those settings → `{version}_fold0_tta_oof.csv` for `src/blend_check.py` (P-12 measurement).
Local checks before any push: `python src/cache_selftest.py`, `python src/window_head_test.py`, then the
smoke run (`FORCE_SMOKE = True` locally trains both arms on the 3 sample studies and infers).

**Off-Kaggle training (RunPod).** Only after a written justification, checked by a critic subagent, and Tian's go (2026-10-04).
`scripts/runpod_chain.sh <arm> [<arm> ...]` is one unattended job:
- lays the inputs out under `/kaggle/input` exactly as Kaggle mounts them: CSVs, label and teacher tables, weights, and the four
  cache shards of `CACHE_PREFIX`, pulled in parallel and blob-verified;
- trains each arm (`RSNA_ARM`, `RSNA_TRAIN_ONLY=1`, 8 workers; `SEQ_ARMS=1` runs them one after another);
- ships `_best.pt` + OOF + log as Dataset `rsna-knee-ckpt-<arm>`, never from a guard-stopped run (traps 47);
- with `AUTO_STOP=1`, stops the pod (traps 51).

Put `/kaggle` on the persistent volume and the cache on local disk. Ship and back up each arm as soon as it finishes, and
delete the pod only after the ships are confirmed (traps 45, 46). Inference stays on Kaggle.

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

Public LB on 2026-10-16, 18:30 UTC (full CSV via `kaggle competitions leaderboard -d`, `artifacts/lb_1016b/`):
- **Top 0.968** (CentralyDS); 10th 0.964; 100th 0.959.
- **605 ≥ 0.958 (481 at exactly 0.958), 920 ≥ 0.957 (315 at exactly 0.957)**, 1,513 ≥ 0.956 (593 at exactly), 1,756 ≥ 0.954, 1,996 ≥ 0.950; 6,103 teams. (11:29 UTC the same day: 388 ≥ 0.958 (278 at exactly), 748 ≥ 0.957; 6,079 teams. 10-14 08:21 UTC: 203 ≥ 0.957, 964 ≥ 0.956; 5,958 teams.)
- **The plateau moved to 0.957 / 0.958 (10-14 → 10-16), both from heliosli's notebooks** (experiments.md 2026-10-16 "Submission #91 and the 0.957 / 0.958 frontier"):
  - `this-is-best-public-dataset-new-version-0-9576` (10-14) = the 0.956 recipe + nartaa's 224 px CoAtNet at 0.35 of the Raptor half + **R6 + flip TTA** in place of R7 → 0.957 by others' reads (goodpjw2008 v21–24, prvsiyan). R6 = bundle `fe055406`, ConvNeXt-S on Opus targets, built 10-11 with R7's trainer; fastMRI unconfirmed.
  - `blend-mine-in-private-lb-luck-to-all-upvoters` (10-15) = the same + Raptor-half weights Synovitis 0.80 / Lateral OA 0.20 / PF OA 0.60 / Baker's 0.35 → 0.958. heliosli's code carries a per-class public probe (`PROBE_CLASS`), set to `None` in the published version. The notebook calls the weights "user decision 10-16", and goodpjw2008 "fixed before any run" (R6 strong on the lateral compartment, weaker on synovitis), so probing is unconfirmed. His four-decimal titles (0.9573 / 0.9576) are unverifiable claims.
  - nartaa's `s6fall` (749359): S6's architecture on his JEV labels, competition data only, **0.950 alone**, no OAI, no fastMRI, CC BY-NC 4.0. leoprovorov reads 0.957 with 0.6 R7 + 0.4 s6fall as the compact half.
  - Most of the 0.958 block arrived on 10-16 (258 of 278 last-submitted that day). The top 10 moved only +0.001–0.002. nartaa puts the public split's SD at ≈ 0.003, wider than either step. The split is ≈ 30 % public / 70 % private (the board header per two participants, unread by us).
- **Rank 6, azamat1ch (0.964, solo, 34 submissions; claimed a 5-fold ConvNeXt-T at 0.959 on 10-09), on 10-12 / 10-13 (748295):** "Short answer is labels", "AI, not by hand. And don't rely on reports alone", and only the competition data, no external sets. That points at image-informed labels, e.g. a multimodal LLM reading the report and the slices (colum2131's 0.953 recipe), which we dropped on 10-05 (experiments.md 2026-10-13 "Top-10 hints on labels").
- **The plateau moved to 0.956 (10-13):** heliosli's public notebook `0-957-latest-dataset-score-in-one-click` = nartaa's 0.949 CoAtNet (0.5) + heliosli's **R7** (0.5): S6's ConvNeXt-S recipe plus a fastMRI auxiliary loss, newer Opus report targets and refitted Effusion / Baker state stackers; no OAI in R7, Dataset licence "other" (experiments.md 2026-10-13 "The public frontier moved to 0.956"). R7 alone is unmeasured (≈ 0.945–0.950 by the fork rule).
- **The plateau moved to 0.954 (10-11):** heliosli's public notebook `rsna-knee-blend-gold-dataset-restored` = nartaa's 0.949 CoAtNet (0.7) + heliosli's **S6** (0.3): ConvNeXt-small, 4 folds, joint-anchored views from an SKM-TEA / KneeXNet anatomy model, a side classifier, "report-derived targets"; OAI not mentioned (experiments.md 2026-10-11 "The public frontier moved to 0.954"). **S6 alone = 0.944 from our account (#81, 10-12)**, not the ≈ 0.948–0.955 our fork rule implied: the blend gains +0.0055–0.0075 over its parts (experiments.md 2026-10-12 "Submission #81"). heliosli's 0.956 team score of 10-12 is the R7 blend, public since (bullet above).
- **Kaggle orders a tie by the unrounded score** (checked 10-11 on the full CSV: inside the 0.954 / 0.950 / 0.943 ties, rank does not follow submission time). So the rank inside a tie shows the sign of a sub-tick difference, but only for a team's best entry (experiments.md 2026-10-11 "Submission #80").
- 10-12 17:34 UTC: 210 ≥ 0.955, 955 ≥ 0.954 (745 at exactly), 1,404 ≥ 0.950; 5,846 teams. 10-12 07:27 UTC: 182 ≥ 0.955, 802 ≥ 0.954 (620 at exactly), 1,304 ≥ 0.950; 5,808 teams. 10-11 17:49 UTC: 563 ≥ 0.954 (425 at exactly), 1,190 ≥ 0.950; 5,768 teams. 10-11 11:50 UTC: top 0.965, 125 ≥ 0.955, 432 ≥ 0.953 (≈ 278 at 0.954), 1,107 ≥ 0.950; 5,733 teams. 10-10: top 0.964, 90 ≥ 0.955, 797 ≥ 0.950; 5,626 teams.
- **The previous plateau, 0.950 (485 teams on 10-10, 513 on 10-11):** It is made of: forks of `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` ("Apex" fusion) = **`nartaa`'s ONE retrained CoAtNet-2 @ 384 (0.949 alone) + the public 0.929 ConvNeXt reader** at per-label weights (+0.001). The 0.949 model's edge is **2,399 OAI knees as masked external labels** (+0.005 in its author's own table; without OAI the author read 0.942 = our `v15c`); OAI's legality is unresolved (experiments.md 2026-10-08 "The 0.949 / 0.950 public notebooks, read"). The old plateau, 0.943 (the community stack, our anchor plus two CoAt readers), still holds 817 teams; 0.944 holds 218, among them the public fork `goodpjw2008`
  (its 2.5D ConvNeXt-T reader, 0.929 solo, at 30 %), whose author measured the stack's run-to-run spread at one tick
  (0.944 / 0.944 / 0.943).
- **We are at 0.957 (rank 617 of 6,103 on 10-16 18:30 UTC, 12th of the 315 tied at 0.957; rank 774 at 11:29)** with C10 #93 (heliosli's R6 + flip notebook + B21 at β 0.20) and C10w #94 (his 0.958 notebook + B21, one tick under its anchor); OAI rule risk, accepted by Tian for pick 2 on 10-09. Before them, C9 #91 / C7 #87 0.956, C5 #80 0.954 and C4b #73 0.951. Our best OAI-free fork is **0.953** (C8 #88: R7 alone + B18 at β 0.45; fastMRI inside R7); free of OAI and fastMRI, **0.952** (C11 #92: nartaa's s6fall + B18); with S6, 0.950 (C6 #82); with no heliosli or nartaa model, 0.946 (C3 #69).
- Our own models read **0.945** (#86 B22, `v15c` + `v13b3` + `v15cr`, 10-13), **0.944** (#71 B19, the ConvNeXt twins as one vote + `v13b3`; #67 B18, `v15c` + `v13b3`, equal) and **0.942** solo (#64 `v15c`, ConvNeXt-T on its own optimiser; its seed twin `v15c2` 0.941; then #50 `v13b3` 0.940, EfficientNet-B3). With OAI as external labels, `v15co` reads **0.945** solo (#78, 10-11) and **0.946** with `v13b3` (B21 #79); pick 2 only.
- Earlier snapshots are in research.md §2.7.3–2.7.4 and experiments.md (2026-10-06 "The public frontier moved"; 2026-10-08 "Submissions #64–#66").

The Efficiency Prize has its own leaderboard, published
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
every member is a mounted public checkpoint. Our DINOv2 recipe was at parity with theirs. (The 2026-08-30 reading, "the gap is the number of families", is
**superseded 2026-10-03:** our gap is image-side and the forum's best singles are CNNs, research.md §2.7.3.) **Measured 2026-08-30 (evening):** the 0.924 member's gain is the **wide slice band +
per-label window attention (+0.007 on the same DINOv2-S) and the hybrid backbone (+0.004, different errors on
the menisci) — not the 384 px** (CoAtNet-1 @224 = 0.8683 beats CoAtNet-2 @384 = 0.8641 at ⅓ of the cost); the
three c02 arms blend with the c01 members to 0.8820 (experiments.md ⭐ "What made the 0.936 notebook good"). Its own counter-example: three backbones on the same input
blended to +0.001 — diversity has to come from the input representation and pretraining regime.

Consensus architecture there: DINOv2 ViT-S/14 as the workhorse (with DINOv3 and RadImageNet
ResNet-50 rank-blended alongside), 2.5D one-series-per-slot with a presence mask, laterality
normalisation, attention pooling over slices, and **LLM-read report labels as the de-facto
standard target source**. (**Corrected 2026-10-04:** "not EfficientNet" no longer holds. The forum's best singles are CNNs,
ResNet-34/50 and EfficientNet-B0, and so is ours: EfficientNet-B3 `v13b3` 0.940; B0 `v13e` / `v13e2` 0.935 / 0.938. **10-08:** our best
single is now a ConvNeXt-T on its own optimiser, `v15c` 0.942.)

Our own measurements of these choices are in [docs/experiments.md](docs/experiments.md).

## Submitting

This is a **code competition**: you submit a notebook, and Kaggle re-runs it against the
hidden test set. You do not upload a CSV.

Working metadata: `enable_gpu: true`, `enable_internet: false`, weights mounted via
`dataset_sources` / `model_sources` (never downloaded at runtime), predictions written to
`/kaggle/working/submission.csv` with the exact `sample_submission.csv` columns.

**5 submissions per UTC day** (the count resets at 00:00 UTC).
- **Scoring time:** a solo of our own models scores in ≈ 15–45 min. A fork on the public 0.942 stack takes ≈ 6.7 h (#69; #17 ≤ 8 h),
  so send it first in a day; a fork on the 0.949 anchor ≈ 1.75 h (#73).
- **Timing:** time each one with `src/watch_submission.py --ref <ref>`.

```bash
kaggle competitions submissions rsna-knee-abnormality-detection
kaggle competitions leaderboard rsna-knee-abnormality-detection -d -p <dir>    # the full CSV; -s shows one page
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
the training-side loader has read the cache since v03.

**Training needs only the cache, never the DICOMs** (verified in the loader, 2026-08-30): the four cache
shards (c03: ≈ 51 GB uint8 blobs + manifests), the CSVs, the LLM label tables and the backbone weights are the
whole training input, so a training run can leave Kaggle (free Colab, any GPU box) while **inference must
stay a Kaggle notebook** (code competition). Checkpoints come back as a private Kaggle Dataset, which the
infer kernel already resolves. Kaggle GPU quota is **30 h/week per account** (`kaggle quota`), two sessions at once,
each a 2×T4 machine (P-31: one arm per GPU). Paid RunPod runs follow the rule in "The main workflow". A derived cache must stay private.

## Rules: AI assistance and data handling

**Using Claude Code / AI agents to develop the solution is permitted.** Nothing in Kaggle's
framework prohibits AI coding assistance — it is ordinary tooling, an LLM-agent-assisted
team publicly won a Kaggle competition in March 2026, and the external-resources provision
turns on whether a resource is *publicly available at minimal cost*, not on who wrote the
code. The binding obligations are the usual ones: one account, no private code sharing
outside your team, winners deliver working code and documentation.

**Two real constraints:**

1. **Everything you rely on must be reasonably accessible to all participants at minimal cost** (Rules 2.6.b). Pretrained weights
   and shared LLM label tables qualify because they are published as Kaggle Models/Datasets.
   A private or paid asset does not.
2. **Sending report text to a hosted third-party LLM API: PERMITTED. The host made it a formal
   rule update** (pasted by Tian 2026-10-04, "Use of Commercially Hosted LLMs", citing Rules
   Section 2.6.b, EXTERNAL DATA AND TOOLS).
   - **What the rule says:**
     - Commercially hosted LLMs and other external inference services are permitted, if the
       service and method otherwise comply with the Rules. That includes being reasonably
       accessible to all participants and of minimal cost.
     - Submitting Competition Data, including report text, to an external LLM or API for
       inference or processing (e.g. extracting labels from reports) "will not, by itself, be
       considered prohibited PRIVATE SHARING".
     - PRIVATE SHARING still prohibits sharing Competition Data, code or competition-specific
       work product with other participants, teams or third parties for collaboration.
     - We stay responsible for the service's own terms of use.
     - The host may still rule that a service, model or configuration is not reasonably
       accessible, is prohibitively costly, or creates an unfair advantage.
   - **What it means for us:**
     - Labelling the reports with Claude, GPT or Gemini is allowed at a cost any team could pay.
       Raymond Yuen, a top-50 team, reports < $5 of API cost for all his labels.
     - The output (a label table) must stay team-private like everything else in `artifacts/`.
   - **History:** Po-Hao "Howard" Chen already said this on the forum on 08-09 and again on
     08-27 (topic 733965; docs/research.md §2.7.3): "You can use LLM API, such as those from
     OpenAI, to read the reports to generate the labels."
   - **Tian's choice:** on 2026-09-28 he chose "no hosted-API labels". On 2026-10-04 he reopened it, and the full
     Claude (Opus) relabel ran (P-65: `claude_v1` / `claude_rap_v1`). Its only judge was the LB, through session E:
     **read 2026-10-06, no lift** (#57 0.935 / #58 0.932 vs the B0 seed mean 0.9365); P-65 is closed and the tables are unused.
   - Same source on external data: KneeCoT is banned (it needs an institutional agreement).
     Click-through datasets (OAI, MRNet, fastMRI+, SKM-TEA) are not excluded for being
     non-commercial. The winners'-licence fit is the team's problem.
   - **External data as it stands (forum re-read 2026-10-13; experiments.md "Submissions #87 / #88"):**
     - The host's two tests:
       - 733965 (08-27): a straightforward registration or a click-through DUA generally passes. Institution-specific
         approvals, negotiated agreements, IRB approval or lengthy credentialing may not.
       - 741819 (09-21): OAI is allowed only "if it is generally accessible to all participants without an escalated (i.e.
         institutional) review and sign-off … and is free of charge".
     - **OAI likely fails the second test.** NIH has barred NDA access from "countries of concern" since 2025-04-04, China
       included, and the competition does not exclude China.
     - **fastMRI** (NYU's form + a Data Sharing Agreement e-signed by the applicant) **likely passes the first.**
     - Neither is ruled by name.
     - **The host's 10-14 ruling** (Chris Carr, RSNA Director of Informatics, 743416, 13:18 UTC): "we are not prepared to designate
       specific datasets as permitted or off limits." "Datasets that require only registration and acknowledgement of licensing terms
       would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with
       institutional affiliations clearly do not." Teams should disclose the datasets used. 744056: "submissions incorporating publicly
       shared weights are eligible for awards", listed with their source datasets. Later asks about OAI by name (10-14, 10-16) are
       unanswered.
     - **Open since 10-16:** does NYU's fastMRI form require an institutional affiliation? A participant's table says so (unverified;
       brainstorm.md). If it does, Carr's second clause covers it.
   - **Data retention by an API provider (read 2026-10-13, forum 733873 / 733965):**
     - The host's 2.6.b ruling neither forbids retention nor requires zero retention. Sending competition data for processing "will
       not, by itself" be private sharing, and we answer for the service's terms of use.
     - The rules' §2.4.b (do not "provide or make available the Competition Data to any party not participating") and the RSNA MIRA
       research-use agreement it incorporates (§4 no reproduction, §5 no sharing access) argue against providers that store the data
       or train on it. These clauses are quoted by a participant (733873); the rules page itself is unread. The host never answered
       that participant's zero-retention question.
     - **Tian's rule (10-13): competition data goes only to zero-data-retention endpoints** (OpenRouter `data_collection: deny` +
       `zdr: true`). Qwen 3.8 Max (Alibaba) and Muse Spark (Meta) have none, so they are not used.
   - **This is about moving competition data off-platform; it is unrelated to using Claude Code
     on your own source code, which is fine.**

Point 2 is the host's 2.6.b text as Tian pasted it. The paragraph on AI assistance is inference from Kaggle's
general framework, not a quotation. Also: keep report text
and `StudyInstanceUID`s out of any public location. `artifacts/` contains both and is
gitignored for that reason.

⚠️ **RadImageNet weights carry no stated licence** (checked 2026-08-28: code MIT, paper CC BY
4.0, data "by request"; an earlier version of this file said CC-BY-NC-SA-4.0, which could not
be verified). Treat as restrictive until radimagenet.com's Terms & Conditions and the
competition's winner-licence clause are read in a browser. DINOv2 is Apache-2.0; timm
ConvNeXt **V1** weights are Apache-2.0 (`convnext_tiny.in12k_ft_in1k` is P-69's); ConvNeXt **V2** weights are CC-BY-NC-4.0 (timm
`convnext.py`, the HF cards), so they are excluded.

## Timeline

| Date | Event |
|---|---|
| 2026-07-30 | Launched |
| **2026-10-15** | Entry deadline **and** team-merger deadline; ≈ the start of the public-notebook publishing lock ("one week, in this competition", Kaggle staff, forum 748181, 10-12; the rules page is unread) |
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
the CLI exposes no command for the overview or rules. The forum is read through the API
(`scripts/harvest_forum.py`, research.md §2.7.3). Anything depending on the overview or rules pages remains unread and
is flagged as such.
