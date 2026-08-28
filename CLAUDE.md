# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this repo

Competition workspace for the Kaggle **RSNA Knee Abnormality Detection** challenge (RSNA
2026 AI Challenge). Predict **12 independent binary findings per knee MRI study**, scored
by **macro ROC-AUC** (unweighted mean of 12 per-label AUCs).

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

**State as of 2026-08-28 (evening):** pipeline v02 runs green end to end (local + Kaggle
smoke). **One submission made** — a smoke run that scored exactly 0.500 and thereby exposed a
silent image-root failure at rerun (fixed). The first real fold-0 run and the preprocessing
cache are the next actions; see [docs/handoff.md](docs/handoff.md).

## 📚 Documentation map — read the relevant one before acting

| File | What it holds | Read it when |
|---|---|---|
| [docs/handoff.md](docs/handoff.md) | Session state, what changed last, next action | **First, always** |
| [docs/traps.md](docs/traps.md) | Bugs and **silent** failure modes, tiered by damage | Before writing pipeline code |
| [docs/experiments.md](docs/experiments.md) | Every measurement, with a verdict | Before proposing an experiment |
| [docs/proposals.md](docs/proposals.md) | **Ranked backlog as testable cards P-00…P-20** (hypothesis, evidence, measure, noise floor, cost) | When choosing what to do next |
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
src/dicom_probe.py      DICOM header / ordering / normalisation audit
src/baseline_infer.py   standalone inference smoke test
kaggle/rsna-knee-train/     generated training/inference notebook + kernel-metadata.json
kaggle/rsna-knee-cache-a/   cache kernel, shard 0  (-b: shard 1, SHARD=1 sed'd in at build time)
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
```

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

**Cache kernels** (CPU, run in parallel with training):

```bash
python src/nbgen.py src/cache_pipeline.py kaggle/rsna-knee-cache-a/rsna-knee-cache-a.ipynb
sed 's/^SHARD = 0 /SHARD = 1 /' src/cache_pipeline.py > /tmp/cache_b.py
python src/nbgen.py /tmp/cache_b.py kaggle/rsna-knee-cache-b/rsna-knee-cache-b.ipynb
kaggle kernels push -p kaggle/rsna-knee-cache-a; kaggle kernels push -p kaggle/rsna-knee-cache-b
```

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
a **0.006 band**, from 2,559 teams. The Efficiency Prize has its own leaderboard, published
as a notebook (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, readable via
`kaggle kernels output`); its leader is also top-5 on accuracy, so efficiency is not being
bought with score.

**The top public notebooks are one shared, heavily-forked community ensemble whose own
author warns it is "likely overfit to the public leaderboard"** after a fork-and-republish
race chasing 0.001–0.003 movements. Expect a private shakeup. Prefer a pipeline you can
validate over a blend you can only submit.

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
