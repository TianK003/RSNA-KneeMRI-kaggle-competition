# Member strength: a stronger teacher and the last recipe knobs (design, 2026-09-23)

**Problem.** Our best single model is fold-0 OOF 0.8730 (`v09c`), about 0.90 solo on the public LB by the
measured +0.02–0.03 offset. The public stack's best single member (Raptor CoAtNet-2 @384, `maxspan-v5`) is
0.924 solo. A member below ~0.90 solo adds nothing to the 0.942 stack at any β (experiments.md 2026-09-23
"#16 … why nothing of ours moves 0.942"), so member strength, not blend weights, is the lever left.

**Diagnosis behind the design.** Our teacher (the 3-source LLM blend) reads 0.8948 on gold-58; a student cannot
learn much past its teacher's label quality. Every recipe knob we A/B'd moved ≤ 0.005. The Raptor member's own
gold-58 (held out) is 0.905–0.917, i.e. its *predictions* on our 4,349 training studies are a better teacher than
the reports. Two tracks, agreed with Tian: **A** build that teacher and retrain on it; **B** the last untested
recipe knobs, cheap, on a rented 4090.

Decisions taken by Tian (2026-09-23): compute = Kaggle + RunPod (Tian creates the pod, I drive it); recipe
knobs judged by fold-0 OOF vs the unchanged teacher, the distilled member by gold-58 direction + one solo LB
read; levers in scope = recipe, label source, distillation (not bigger inputs / pretraining); the Raptor
predictions enter the targets **blended** with the LLM teacher (mix 0.5), not as a replacement; CoAt family
children (resgated, D4) are **not** part of the teacher; a solo-LB **baseline** submission of the current best
member is included.

---

## 1. Track A — the Raptor teacher pass (Kaggle)

**What.** Run the public 0.942 notebook's Raptor branch — unchanged code, the same three CC0 checkpoints
(`dreaddevelopment/raptor-knee-maxspan` v5 swa, `-native384` v8 swa, `-native384dense` v10; 293 MB each), the
same four views (`maxspan-v5` w 0.60, `native384dense-v10` 0.10, `maxspan-v5-reverse` 0.10, `native384-v8` 0.20)
— over the 4,349 report-labelled training studies, and keep the **raw per-view probabilities**.

**Why the DICOM mount.** Their reader is their own preprocessing (140 mm crop, per-slot-series [2, 98] percentile,
336 → 384, no laterality, slot-crossing windows); our c02 cache is a different geometry. The pass therefore runs
on Kaggle, where `train_images` is mounted (hard constraint 2: never download the DICOMs).

**Builder: `src/build_teacher_pass.py` → `kaggle/rsna-knee-teacher/`** (notebook + `kernel-metadata.json`):

- Extracts **verbatim** from `notebook_score_0.942.ipynb`: cell 45's embedded module `_KE_SRC` (reader, `RaptorClassifier`,
  `eval_windows`, `_pick_series_for_slot`, `find_test_root`, `ARMS`), cell 45's cached-reader / prefetch / two-GPU
  runner (`_ke_make_cached_reader`, `_ke_prepare_windows`, `_ke_infer_input`, `_ke_prefetched_windows`,
  `_ke_run_raptor_arms`), the helpers they name from cell 16 (`rsna_sha`, `rsna_event`, `rsna_rank01`,
  `rsna_strict_load`, `rsna_frame`, `rsna_save_predictions`) and cell 12 (`_asset_walk`, `_asset_find_asset`), and
  the `RUN["raptor_view_w"]` dict from cell 2. Nothing else of the graph (DINO ×20, A5, RadImageNet, calibrator,
  CoAt family, FineSpacing) is included; the `_DINOV2_MATCHED_MEMBERS` gate and the `_pipeline_stage.csv` read
  are removed (the study list comes from the chunk, not from the transformer stack).
- **Chunk preamble** (new code, ~40 lines): constants `SHARD = 0`, `N_SHARDS = 1`, `LIMIT = 0` (sed'd at build like the
  cache shards; `LIMIT = 100` = the spike); reads `train.csv` / `train_series.csv`, takes the report-labelled
  studies (the 58 gold rows are excluded — they are validation), sorts by UID, slices shard `SHARD` of `N_SHARDS`
  (then the first `LIMIT`), writes `/kaggle/working/chunk/test.csv` (UID column only, like the real one),
  `chunk/test_series.csv` (their rows), `chunk/sample_submission.csv`, symlinks `chunk/test_images →
  /kaggle/input/rsna-knee-abnormality-detection/train_images`, sets `RSNA_COMP_ROOT=/kaggle/working/chunk`. The
  Raptor code then believes this is the test set.
- **Outputs**: `raptor_teacher_shard{SHARD}.npz` (`study_uids`, `raw_probabilities` shaped 4 × N × 12, view names,
  per-view weights, checkpoint sha256s) and `raptor_teacher_shard{SHARD}.csv` (UID + the 12 view-weighted mean
  probabilities), plus a `teacher_receipt.json` (studies, seconds per study, failures). No `submission.csv`.
- **Runtime contract**: the runner requires exactly two GPUs (their code raises otherwise) — `NvidiaTeslaT4`; the
  T4 budget guard is kept at 8.3 h with a checkpointed flush every 200 studies (the npz is rewritten
  incrementally so a guard stop keeps what ran; a resume re-runs only missing UIDs). Kaggle sources: the three
  Raptor datasets, `mattiaangeli/opencv-python-headless-4120088-x86` (their cv2 wheel), the competition.
- **Merge (CPU, `src/merge_teacher.py`)**: concatenates the shard npz files, asserts 4,349 unique UIDs, writes
  `artifacts/teacher/raptor_teacher.csv` (UID + 12 columns = view-weighted mean of raw probabilities — the same
  combination the notebook rank-blends, in probability space) and publishes it with the self-distill table as the
  private Dataset **`tiankljucanin/rsna-knee-teacher-tables`**.

**Spike first.** `LIMIT = 100`, one session (~1 h): read seconds/study from the receipt → chunk count =
`ceil(4,349 × s_per_study / (8 h × 3600))`, run the chunks in the two concurrent GPU slots after Saturday's quota
reset (estimate before the spike: 10–12 GPU-h total). A spike that fails on our chunk files (their validators
compare ids against `test.csv` of the root) is fixed in the preamble, never in their code.

**Licence.** All three checkpoints and their datasets are CC0-1.0 (verified 2026-09-21); pseudo-labelling one's
own training data with a public model is ordinary practice. The RadImageNet stage (CC-BY-NC-SA) is not used.

## 2. Teacher mixing in the targets (shared by both tracks)

`build_targets` exists twice (`src/build_targets.py` for the local artefact, `build_targets(train_csv)` in
`src/kaggle_pipeline.py` for the kernel); both gain the same feature, with `src/build_targets.py` as the reference
and a byte-identity check of the default teacher (md5 `29f641ed…`, gold 0.8948) as the regression test.

- **Prediction-table sources.** `TEACHER_TABLES: tuple[str, ...] = ()` and `TEACHER_MIX: float = 0.5` are constants
  in the config cell, sed'd per kernel session (one teacher per session, like `ARM_ONLY`). Each table is a CSV
  `StudyInstanceUID` + 12 label columns in [0, 1], resolved by the same `first_existing` / shallow-glob path
  search as the LLM tables (local `artifacts/teacher/<name>.csv`, Kaggle `/kaggle/input/.../<name>.csv`).
- **Quantile matching per label.** For label *L*, the table's values are replaced by
  `np.quantile(llm_blend[L], rank_pct(table[L]))` over the report-only rows the table covers: ranks are preserved,
  the LLM blend's value distribution is adopted, so the `decisive = |mean − 0.5|·2` confidence weight keeps its
  meaning and no source dominates by scale. Studies absent from a table keep the LLM value (weight unchanged).
- **Training vs evaluation targets.** `yt__{L} = (1 − MIX) · llm_blend[L] + MIX · matched[L]` on report-only rows;
  gold rows keep the hard 0/1 (weight `gold_weight`). `y__{L}` stays the 3-source LLM teacher. The Dataset emits
  `yt` when the columns exist; the training loop calls `weighted_bce(logits, b.get("yt", b["y"]), b["w"])`;
  `evaluate()` keeps `b["y"]`, so OOF-vs-teacher, per-epoch csvs and `blend_check.py` remain comparable with every
  earlier arm. The checkpoint's saved config records `teacher_tables` and `teacher_mix`; inference ignores both
  (not `INFER_MEMBER_KEYS`).
- **Self-distillation table (`src/build_distill_table.py`, CPU).** Rank-mean per label of the two complete 5-fold
  OOF sets on disk (`artifacts/kaggle_out/pod_v09h_5fold/v09h_fold*_oof.csv`, pooled 0.8625; `folds_v4/v05g_fold*_oof.csv`,
  0.8467; 4,407 studies each) → `artifacts/teacher/selfdistill_v1.csv`. Every prediction is out-of-fold, so no study
  is taught by a model that saw it.
- **Weights** stay the LLM-derived agreement/decisiveness weights (floor 0.15); folds are unchanged (they depend
  on report groups and gold only).

## 3. Track B — recipe arms (RunPod 4090, fold 0, each vs `v09c` 0.8730, floor 0.008)

Three cards (P-36 … P-38), one arm each, ~1 h per CoAtNet-1 fold-0 arm on a 4090 (50 min measured for `v09h`):

| arm | recipe | knob under test | card |
|---|---|---|---|
| `v09e` | `v09c` + `epochs 16`, `lr_backbone 3e-5` | the public schedule length at the public LR (P-29's over-training was at 1e-4; P-34 reads 3e-5 at 8 epochs first) | P-36 |
| `v09f` | `v09c` + `pos_weight_max 10` | per-label `pos_weight = clip((1 − p) / p, 1, 10)`, `p` = positive rate of `y > 0.5` on the training rows; the public member's loss | P-37 |
| `v09s` | `v09c` + `TEACHER_TABLES=("selfdistill_v1",)`, `MIX 0.5` | self-distillation (P-17's round-2 targets) | P-38 |

`pos_weight` is one `Config` field and one argument to `F.binary_cross_entropy_with_logits` in `weighted_bce`
(computed once from the training targets in `make_loaders`, passed as a tensor; 0 = off, byte-identical loss).
The "Rejected without testing" row for `pos_weight` is amended: the rejection reasoned about calibration, not
training dynamics; this is a measured A/B.

**RunPod flow.** Tian creates a pod (4090 / A5000, ≥ 60 GB NVMe) and provides SSH. `scripts/runpod_bootstrap.sh
setup` (adds the `rsna-knee-teacher-tables` Dataset to its label pulls), `train v09e` / `v09f` / `v09s`, `ship <arm>`
(Dataset `rsna-knee-ckpt-<arm>` with `_best.pt` + `_oof.csv`). The OOF csvs are pulled to
`artifacts/kaggle_out/pod_<arm>/` and read with the S1 per-label script pattern. If P-34 (`v09d`, 3e-5 at 8 epochs,
reading ≈ 15:30) is harmful (< 0.865), `v09e` is dropped.

## 4. Measurement, order, budget

| what | judge | floor | verdict rule |
|---|---|---|---|
| `v09e`, `v09f`, `v09s` | fold-0 OOF vs the unchanged teacher (882 studies), per-label table | 0.008 (per label 0.03) | ≥ 0.881 ✅ / 0.865–0.881 🔁 / < 0.865 harmful; ≥ 9/12 labels up as support |
| Raptor-distilled `v09a` (PROD, all 4,349, 8 ep, S1 knobs, `TEACHER_TABLES=("raptor_teacher",)`) | gold-58 (held out under `train_all`) vs S2 `v09a` 0.8922 | 0.05 | direction only |
| the same member, solo | public LB via `rsna-knee-infer` with `INFER_MEMBERS=["v09a"]` vs the **baseline solo submission of the S2 `v09a`** | 0.005 | ≥ +0.005 ✅ the teacher works → retrain `v08a` the same way and rebuild the fork; ≤ +0.004 🔁; a drop ❌ |

Two solo submissions in total (baseline, distilled). Nothing is tuned on the LB.

**Order.** (1) Today, while round 2 runs: sections 2 and 3 code + unit checks + local smoke; P-36…P-38 cards; the
teacher-pass builder + its local dry build. (2) This week's remaining Kaggle quota (≈ 7 h after round 2): the
100-study spike (≈ 1 h). (3) RunPod as soon as the pod exists: `v09e`, `v09f`, `v09s`. (4) After Saturday's reset:
the full Raptor pass in two concurrent sessions, merge, publish the tables Dataset. (5) Distilled `v09a` retrain
(Kaggle both T4s with `v08a` on the same teacher, or the pod), baseline + distilled solo submissions, `/update`.
(6) If ✅: production members on the mixed teacher, fork at β 0.10.

**Budget.** Kaggle: spike ≈ 1 h this week; pass ≈ 10–12 h + retrain ≈ 2.7 h next week (of 30 h). RunPod: setup 0.7 h +
three arms ≈ 3.7 h ≈ $3–4 at $0.74/h. Submissions: 2 solo reads (+ 1 fork later).

## Risks and how they are handled

- **Their validators reject our chunk root** (id checks against `test.csv`, `sample_submission.csv` columns): the
  preamble writes all three files in the competition's exact schema; the spike catches the rest.
- **Rank-vs-probability mismatch across chunks**: raw probabilities are saved per view; ranking happens once, in
  the merge, over all 4,349 studies.
- **Quota guard mid-chunk**: incremental npz flush + resume by missing UIDs.
- **The distilled student just copies the teacher's errors**: the mix keeps half the LLM signal; the LB solo read
  is the arbiter, and a ❌ ends the track without touching production members.
- **Kaggle CLI hangs / phantom pushes** (traps 36): every push is followed by `kernels status` before any retry.
- **Scope creep**: no CoAt family, no DINO stage, no label-source swap in this round (the `--sources dread` path
  exists but is not part of these arms).
