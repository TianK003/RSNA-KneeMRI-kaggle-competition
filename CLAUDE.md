# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this folder

Competition workspace for the Kaggle **RSNA Knee Abnormality Detection** challenge (RSNA
2026 AI Challenge). Predict **12 independent binary findings per knee MRI study**, scored
by **macro ROC-AUC** (unweighted mean of 12 per-label AUCs).

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

**State as of 2026-08-28:** the only contents are this file and `data/` holding the five
competition CSVs. No model code, no pipeline, no build or test commands exist yet. Do not
reference commands or files that don't exist — check first.

## ⛔ Hard constraints — read before running anything on Kaggle

1. **NEVER select the P100 accelerator.** Kaggle's current PyTorch ships no Pascal kernels,
   so the session dies at the first convolution — after you have already burned setup time.
   Always set the T4: `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`, and
   check it on every new or forked kernel, since the default is not guaranteed.
2. **Never download the competition images in bulk** (~570 GB). Train on Kaggle; see
   "Compute strategy" below.
3. **Never sort DICOM slices by filename** — it fails silently. See "Non-obvious traps".

## Layout and commands

```
data/                  CSVs (9 MB) + sample_dicom/ + llm_labels/ (gitignored)
models/                dinov2_small/, radimagenet_r50/ResNet50.pt (gitignored)
src/dicom_probe.py     header/ordering/normalisation audit on real DICOMs
src/baseline_infer.py  standalone inference smoke test
src/build_targets.py   targets + leak-safe folds -> artifacts/targets.csv
src/kaggle_pipeline.py THE PIPELINE, percent-format (runs as .py AND becomes the notebook)
src/nbgen.py           percent-format .py -> .ipynb
kaggle/rsna-knee-train/ generated notebook + kernel-metadata.json
artifacts/             derived outputs (gitignored — embeds StudyInstanceUIDs)
```

### The main workflow

`src/kaggle_pipeline.py` is the single source of truth — never edit the `.ipynb` by
hand, it is generated. Percent-format (`# %%` / `# %% [markdown]` markers) means the
same file runs locally as a plain script and converts to the Kaggle notebook.

```bash
export PYTHONUTF8=1
python src/kaggle_pipeline.py                 # local smoke run (CPU, 3 sample studies)
python src/nbgen.py src/kaggle_pipeline.py \
       kaggle/rsna-knee-train/rsna-knee-train.ipynb
kaggle kernels push -p kaggle/rsna-knee-train
kaggle kernels status tiankljucanin/rsna-knee-train
kaggle kernels output tiankljucanin/rsna-knee-train -p artifacts/kaggle_out
```

**`FORCE_SMOKE` at the top of the config cell is the switch that matters:**
`True` = minutes-long end-to-end check (1 fold, 1 epoch, 2 slices/slot, 24 studies
scanned); `False` = real 5-fold run; `None` = auto (smoke locally, real on Kaggle).
Always push with `True` first on a new or edited notebook — a crash in the inference
cell after six hours of training costs an entire session.

**Resuming:** five folds do not fit in one 9 h session. When the runtime guard fires,
each fold has written `{version}_fold{k}_last.pt` and inference is skipped. Attach that
run's output as an input dataset to a new run and it resumes from the last epoch.
Inference only runs once every fold is complete, so a half-trained ensemble is never
submitted.

Both scripts import from `src/`, so **run them from the repo root with `PYTHONPATH=src`**:

```bash
export PYTHONUTF8=1 PYTHONPATH=src
python src/dicom_probe.py                       # audit headers & ordering
python src/baseline_infer.py --slices 3         # full smoke test
python src/baseline_infer.py --limit 1 --slices 2   # fast smoke
```

Local env is CPU-only (`torch 2.13.0+cpu`) and exists for CSV/report analysis, header
work, and shape verification — **not** training. See `requirements.txt`; install torch from
the CPU index or pip pulls a ~2.5 GB CUDA build.

Weights on disk, and how to re-fetch them:

```bash
kaggle models instances versions download metaresearch/dinov2/PyTorch/small/1 \
  -p models/dinov2_small --untar            # ViT-S/14, HF format, 88 MB, hidden_size 384
kaggle datasets download marwanmath/resnet-50-radimagenet-marwan \
  -p models/radimagenet_r50 --unzip         # RadImageNet ResNet-50, 94 MB
```

DINOv2-small loads with `transformers.Dinov2Model.from_pretrained("models/dinov2_small")`.

⚠️ **RadImageNet is CC-BY-NC-SA-4.0 — non-commercial and share-alike.** It is widely used
in public notebooks here, but an NC/SA licence can conflict with a prize competition's
winner-licensing obligations. Check the competition's winner-licence clause before making
RadImageNet load-bearing in a final submission; DINOv2 (Apache-2.0) carries no such
question. Flagging, not resolving — I could not read the rules text.

## Measured on the sample studies (2026-08-28)

`src/dicom_probe.py` over 12 series / 459 slices / 3 studies confirmed, on real data:

- **Filename order is anatomically meaningless.** Spearman rho between filename order and
  `ImagePositionPatient` projection: mean **−0.012**, range −0.31 … +0.25, and `|rho|>0.99`
  in **0 of 12** series. Sorting by filename is not a style preference — it silently
  destroys slice adjacency, which is the whole point of a 2.5D triplet.
- **Header recovery of the acquisition flags pays off.** The CSVs carry 2 of 4
  `(Fluid_Sensitive, Fat_Suppression)` combinations; headers recovered **3 of 4** on just
  12 series, including `(fluid=True, fat_sat=False)` — a combination the CSV cannot express
  at all. Weighting split: PD 6, T2 4, GRE 1, T1 1.
- **`Anatomical_Plane` in the CSVs is trustworthy** — 100% agreement with the plane derived
  from `ImageOrientationPatient`. Don't waste effort recomputing it; the two *flags* are
  the untrustworthy columns, not the plane.
- **Per-series intensity normalisation is mandatory.** Max intensity spans 690 … 8,736
  across series (12.7x); a global window would not transfer. All sample series are
  `MONOCHROME2` and need no rescale, but the pipeline must still handle `MONOCHROME1`
  inversion and `RescaleSlope/Intercept` since the hidden test set spans 16–19 sites.

`src/baseline_infer.py` runs the full path (slot selection → ordering → triplets → DINOv2
ViT-S/14 → per-slot mean pool + presence mask → 12 logits → CSV) and validates the output
against `sample_submission.csv`. **Smoke test passes**; the head is randomly initialised so
the values carry no signal — it verifies plumbing, not accuracy.

One design lesson from running it: strict slot matching (requiring fluid **and** fat-sat)
left 2 of 12 series unassigned and one study at 2/6 slots. A **relaxed second tier** —
right plane + fluid, ignoring fat-sat — lifted coverage to 4/6 and 5/6. Real studies carry
axial fluid series with no fat suppression; strict-only matching silently discards usable
sequences. Slot assignment is greedy, strict tier first across all slots, so a series
claimed strictly is not stolen by another slot's fallback.

## Verified Kaggle run (kernel v2, 2026-08-28)

`tiankljucanin/rsna-knee-train` (private) ran the full pipeline **green on a T4** in smoke
mode — targets → grouped folds → header scan → slot selection → 2.5D triplets → DINOv2 →
training → checkpoints → rank-mean inference → validated `submission.csv`.

What it settled:

- **All three mount layouts differ from the obvious guess.** The competition resolved to
  `/kaggle/input/competitions/rsna-knee-abnormality-detection`, the backbone to
  `/kaggle/input/models/metaresearch/dinov2/pytorch/small/1`, and the label datasets to
  `/kaggle/input/datasets/<owner>/<slug>/...` — that last one was found only by the
  filename-glob fallback. Keep `resolve_dir()` and the glob; do not "simplify" them.
- **Folds are reproducible across machines** — 882/882/881/881/881 with gold
  11/12/12/12/11, byte-identical to the local run. Teacher AUC 0.8934 both places.
- **Slot coverage on real training data is good**: mean **4.96 of 6** slots per study.
  `SAG_FLUID_FS` 100%, `AX_FLUID_FS` 100%, `COR_FLUID_FS` 95.8%, `SAG_FLUID_NOFS` 87.5%,
  `COR_T1` 62.5%, `SAG_T1` 50%. The two T1 slots are the weak ones — first candidates to
  drop if compute needs cutting. (Measured on the 24 studies the smoke scan covered.)
- `auc_soft 0.3182` is **meaningless** — 4 validation studies, 1 epoch, near-random head.
  Not a signal of anything. `pred_std 0.127` is healthy (no base-rate collapse).

### ⚠ Throughput is the open risk — measure before launching 5 folds

Fold 0 took **36 s for 8 study-passes at 2 slices/slot with `num_workers=0`** (~4.5 s per
pass). The real config is 3× the slices, so a naive extrapolation to 4,407 study-passes ×
4 epochs lands around **6–8 h per fold** — meaning five folds needs roughly five sessions,
and one fold barely fits in one. This is an extrapolation from a tiny sample, not a
measurement; the real per-pass cost with `num_workers=2` is unknown.

**Do not start a 5-fold run before benchmarking.** The bottleneck is almost certainly DICOM
decode, not the ViT: at 6 slices/slot × 3 channels × ~5 slots that is ~90 file reads per
study, repeated every epoch. The standard fix is the Kaggle-to-Kaggle **cache build** — a
separate CPU kernel that decodes, resizes, and writes uint8 arrays once (community reports
~15.9 GB, ~1 h for all 4,407 studies), which the training kernel then mounts. That
converts 90 DICOM reads per study per epoch into one array read. Build that before scaling
up folds or epochs.

Checkpoint sizes to plan around: `best.pt` 88 MB (weights), `last.pt` 266 MB (with
optimizer state). Five folds of both is ~1.8 GB, well inside Kaggle's output limit.

## Status and access

- Kaggle CLI **v2.2.4, authenticated** (OAuth, user `tiankljucanin`). Rules **accepted**
  (`userHasEntered: True`), so `competitions download` works.
- **Zero submissions made so far.**
- Set `PYTHONUTF8=1` before any `kaggle ... --csv` call — team names and reports contain
  non-ASCII and Windows `charmap` otherwise throws.
- **Kaggle rate-limits file downloads with HTTP 429.** `xargs -P 10` over ~550 files got
  ~80% through and then failed the rest; `-P 4` also tripped it. The API also returns
  **exit code 0 on a 429**, so a loop that discards stderr reports success while silently
  skipping files — always verify with a file count, not the exit status. Use sequential
  downloads with a short sleep, and note the CLI **flattens nested paths** into `-p`, so
  rebuild the `study/series/` tree yourself.
- `kaggle competitions download -f` takes one file at a time; there is no folder download,
  and a bare `-c` with no `-f` would pull all ~570 GB.
- Kaggle competition pages are JS-rendered; `WebFetch`/`curl` return only the SPA shell.
  Use the CLI. The CLI exposes **no** command for the overview, data-description, rules, or
  discussion prose — those remain unread, and anything depending on them is flagged below.

## Timeline

| Date | Event |
|---|---|
| 2026-07-30 | Launched |
| **2026-10-15** | Entry deadline **and** team-merger deadline |
| **2026-10-22 23:59 UTC** | Final submission deadline (API-verified) |
| 2026-11-05 | Winners announced |
| 2026-11-29 – 12-03 | RSNA 2026, Chicago |

Category **Research**, reward **$77,000**, **2,559 teams** entered (2026-08-28). The pool
covers the accuracy leaderboard **plus a separate Efficiency Prize track** with its own
leaderboard — runtime is a scored objective, not merely a limit.

## Verified data facts

All of the following was checked directly against the downloaded CSVs on 2026-08-28.

`data/train.csv` — 4,407 studies, columns:
`StudyInstanceUID, Report, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

- **All 4,407 studies have a `Report`. Exactly 58 have labels** (all 12 filled, values
  strictly `0`/`1`); the other 4,349 have every label blank.
- Reports are multi-line free text in ~9–12 languages (Spanish, Greek, Turkish, Croatian,
  … ). The file is 58,556 physical lines for 4,407 records — **you must use a real CSV
  parser**, never line-based splitting.
- Positive rate on the 58 labeled studies: Effusion 60%, Synovitis 47%, Medial Meniscus
  45%, ACL 41%, Lateral Meniscus 40%, PF OA 36%, Contusion 33%, Fracture 31%, Medial OA
  26%, Baker's 21%, Lateral OA 19%, MCL 16%.

`data/train_series.csv` — 24,371 series over all 4,407 studies. Columns:
`StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`.
Planes: Sagittal 9,864 / Coronal 8,609 / Axial 5,898. Series per study 3 / **5 median** / 14.

`data/test.csv` — **`StudyInstanceUID` only, 3 studies.** `test_series.csv` — 15 series.
`sample_submission.csv` — `StudyInstanceUID` + the 12 label columns, all filled `0.5`.

### Three facts that determine the whole design

1. **`train.csv` has `Report`; `test.csv` does not.** Text exists when fitting and is
   absent when predicting. A multimodal fusion model with a text branch is therefore
   **impossible at inference** — it would have nothing to read. Reports are usable *only*
   as a source of training targets, as an auxiliary task dropped at inference, or as a
   per-sample confidence weight.
2. **58 labels is the entire supervised signal**, so the real task is converting 4,349
   reports into trustworthy targets. This is a weak-supervision problem wearing a
   computer-vision costume.
3. **`Fluid_Sensitive` and `Fat_Suppression` are degenerate as delivered** — verified: only
   `(1,1)` (14,010 rows) and `(0,0)` (10,361) occur, never a mixed pair. Two physically
   independent properties arrive as one bit. Recover them from the DICOM headers
   (`ScanningSequence`, `SeriesDescription`, `SequenceName`, `ScanOptions`, and $T_R$/$T_E$)
   rather than trusting these columns.

## What the metric implies

Macro ROC-AUC is invariant to any strictly increasing per-label transform, so:

- **Calibration and thresholds are worth nothing.** Only rank order is read.
- **Ensemble by averaging ranks, not probabilities.** Probability averaging lets the most
  confident model dominate; rank averaging combines exactly the information AUC reads.
  This is what every strong public notebook does.
- **Every label costs the same.** One label stuck at chance forfeits ~(M−0.5)/12 of the
  score — about 0.029 at M=0.85 — no matter how good the other eleven are. **Rare findings
  deserve more attention than common ones** (MCL and Lateral OA are the scarcest here).
- Prevalence is not guaranteed to match across train / public / private, which AUC largely
  survives — but any threshold you bake in will not.

## Where the field actually is

Public LB on 2026-08-28: **top 0.952**, ranks 2–9 spanning 0.946–0.949 — the top ten sit
inside a **0.006 band**. Efficiency LB is published as a notebook
(`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, downloadable via
`kaggle kernels output`); its leader, Scott Willis at 0.948, is also top-5 on accuracy, so
efficiency is not being bought with score.

**Read this before chasing the leaderboard.** The top public notebooks are one shared,
heavily-forked community ensemble, and its own author warns in the notebook that it is
"likely overfit to the public leaderboard" after a fork-and-republish race chasing
movements of 0.001–0.003. Expect a substantial private shakeup. Treat sub-0.005 public
deltas as noise, hold out the 58 labeled studies honestly, and prefer a pipeline you can
validate over a blend you can only submit.

Consensus architecture in the strong public notebooks:

- **Self-supervised ViT encoders**: DINOv2 ViT-S/14 is the workhorse (available as a Kaggle
  Model, `metaresearch/dinov2/PyTorch/small/1`); DINOv3 ViT-S/16 and RadImageNet ResNet-50
  are rank-blended alongside. Not EfficientNet — that was the early baseline era.
- **2.5D**, one primary series per plane/weighting slot, with a per-slot presence mask
  because few studies have all slots. Slots worth carrying: sagittal/coronal/axial
  fluid-sensitive+fat-sat, sagittal fluid no-fat-sat, coronal/sagittal T1.
- **Laterality normalization** (mirror right knees), attention pooling over slices rather
  than mean pooling (mean pooling washes out small focal findings).
- **LLM-read report labels are the de-facto standard target source.** Multiple public
  datasets exist and are widely mounted — `pilkwang/rsna-knee-llm-labels` (~2.2k
  downloads), `stevenleehans/rsna-knee-llm-report-labels` (~2.1k),
  `lixin73/rsna-knee-llm-report-labels-sol56`, plus merged/refined variants. Using several
  and cross-checking them is common practice.
- **Grade the mention, don't binarize it.** The reporting radiologist and the annotator do
  not share a threshold — a report saying "small joint effusion" can sit against a negative
  annotation, because annotators marked only findings they judged significant, with "on the
  fence" graded negative. So `term present ⇒ positive` is wrong by construction; emit
  trace / unqualified / marked grades, which costs nothing since only order is read.
- **Negation and multilingual handling are the hard part** of any rule-based extractor.
  Reports list what was checked and found intact, so for several findings most mentions are
  negative; explicit normality is evidence of absence. This is exactly why LLM labels beat
  lexicons here.

### Non-obvious traps other teams hit

- **Never sort DICOM slices by filename.** The filename is the SOP Instance UID, assigned
  to be unique rather than ordered, so filename order is uncorrelated with anatomy — and it
  fails silently. Sort by projecting `ImagePositionPatient` onto the normal from
  `ImageOrientationPatient`; fall back to `InstanceNumber`.
- **Group your folds.** Random K-fold reportedly inflates AUC ~0.05 through scanner/site
  memorization (one team measured a +0.136 grouped-vs-random gap). Additionally, **some
  reports are shared verbatim across studies**, which yields one target vector for several
  studies — those must not straddle a fold boundary.
- **Gold-subset metrics are extremely noisy.** With ~58 studies and a handful of positives
  per label, the Hanley–McNeil SE of an AUC near 0.8 is ≈0.09, a 95% interval of ±0.17 —
  far wider than the differences you'd be trying to resolve. Judge lexicon/label changes on
  *coverage* (does the rule fire at all, by language) rather than on gold agreement.
- **Do not select the P100 accelerator** — Kaggle's PyTorch ships no Pascal kernels and the
  session dies at the first convolution. Use T4 (`"machine_shape": "NvidiaTeslaT4"`, which
  `kaggle kernels push` accepts in `kernel-metadata.json`).
- **Never hard-code `/kaggle/input` paths.** Kaggle mounts the competition at *both*
  `/kaggle/input/rsna-knee-abnormality-detection` **and**
  `/kaggle/input/competitions/rsna-knee-abnormality-detection` depending on how the kernel
  was created, and Models at either `/kaggle/input/dinov2/pytorch/small/1` or
  `/kaggle/input/models/metaresearch/dinov2/pytorch/small/1`. Every strong public notebook
  probes both. This bit us: kernel v1 died instantly with
  `FileNotFoundError: /kaggle/input/rsna-knee-abnormality-detection/train.csv` even though
  the competition was correctly attached. `resolve_dir()` in the pipeline probes candidates
  and prints the actual directory listing on failure.

## Submitting

This is a **code competition** — confirmed by `test.csv` containing only 3 placeholder
studies while the real test set is served at rerun time. You do **not** upload a CSV; you
submit a notebook, and Kaggle re-runs it against the hidden test set.

Reference metadata from working public submissions:
`enable_gpu: true`, `enable_internet: false`, pretrained weights mounted via
`dataset_sources` / `model_sources`, never downloaded at runtime. Write predictions to
`/kaggle/working/submission.csv` with the exact `sample_submission.csv` columns.

Iterate with the CLI rather than the browser:

```bash
export PYTHONUTF8=1
kaggle kernels pull <user>/<kernel> -p <dir> -m     # fetch a notebook + metadata
kaggle kernels push -p <dir>                        # push (kernel-metadata.json drives it)
kaggle kernels status <user>/<kernel>
kaggle kernels output <user>/<kernel> -p <dir>      # retrieve submission.csv / artifacts
kaggle competitions submissions rsna-knee-abnormality-detection
kaggle competitions leaderboard rsna-knee-abnormality-detection -s --csv
```

**Unverified** (community-sourced, not read from the competition pages): the **≤9 hour**
runtime limit and the internet-off requirement. The `enable_internet: false` in every
public submission corroborates the latter. Confirm both on the overview/rules pages before
planning a long run.

## Compute strategy — do not download the images

The image data is ~570 GB across ~819,000 training DICOMs. **Do not pull it locally.** Keep
only the CSVs (~9 MB, already in `data/`) plus a handful of sample DICOMs for header work.

Run **Kaggle-to-Kaggle**: each kernel mounts the previous kernel's output directly, so
nothing large crosses the local machine.

```
metadata/header scan (CPU)  →  cache build (CPU)  →  train (T4 GPU)  →  submit
```

Local machine is for: CSV and report analysis, label tooling, notebook authoring, and
CLI orchestration. All DICOM decoding and training happens on Kaggle.

## Rules: AI assistance and data handling

**Using Claude Code / AI agents to develop the solution is permitted.** Nothing in Kaggle's
competition framework prohibits AI coding assistance — it is ordinary tooling, an
LLM-agent-assisted team publicly won a Kaggle competition in March 2026, and the standard
external-data-and-models provision turns on whether a resource is *publicly available and
reasonably accessible at minimal cost*, not on whether a human or a model wrote the code.
The obligations that do bind you are the usual ones: one account, no private code sharing
outside your team, and winners must deliver working code and documentation.

**Two real constraints, in decreasing confidence:**

1. **Everything you rely on must be publicly available and free to all.** Pretrained
   weights (DINOv2/v3, RadImageNet) and shared LLM label tables satisfy this because they
   are published as Kaggle Models/Datasets. A private model or paid asset does not.
2. **Sending report text to a hosted third-party LLM API is the one genuinely open
   question.** The Data Security / data-use provisions plausibly forbid transmitting
   competition data to an outside service. Note the tension: this is now widespread
   practice — one of the most-downloaded public label sets is openly titled "GPT-5.6-Sol" —
   and the host has not visibly objected, which is evidence of tolerance but **not a
   ruling**. Safe path: mount an existing public label table (the work is already done and
   shared), or run open-weights multilingual models locally or inside a Kaggle notebook.
   **This concern is about moving competition data off-platform — it is unrelated to using
   Claude Code on your own source code, which is fine.**

Read the rules text yourself before relying on either point; I could not fetch it, and the
above is inference from Kaggle's general framework plus observed community behaviour, not a
quotation. Also: keep report text and `StudyInstanceUID`s out of any public repo — that is
competition data.

## Provenance

**API/CSV-verified (high confidence):** every number in "Verified data facts"; the
deadline, category, reward, and team count; leaderboard standings; the existence and
metadata of the public notebooks and label datasets; CLI auth and entry status.

**From public notebooks** (notably `pilkwang/rsna-knee-baseline-v1`, 454 votes — an
unusually rigorous write-up worth reading in full): the metric reasoning, sequence-slot
design, DICOM ordering trap, Hanley–McNeil noise argument, and the graded-vs-thresholded
label insight.

**Community-sourced, still unverified:** the ≤9 h runtime limit, internet-off, the ~570 GB
/ 819k-file totals, and the fold-leakage magnitudes.

**Corrected on 2026-08-28:** an earlier version of this file hypothesized that leaders were
exploiting report text available at inference time. That is **wrong** — `test.csv` has no
`Report` column. The high scores come from LLM-derived training labels plus large rank
ensembles of self-supervised ViTs, and partly from public-LB overfitting.
