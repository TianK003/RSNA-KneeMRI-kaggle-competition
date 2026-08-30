# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this repo

Competition workspace for the Kaggle **RSNA Knee Abnormality Detection** challenge (RSNA
2026 AI Challenge). Predict **12 independent binary findings per knee MRI study**, scored
by **macro ROC-AUC** (unweighted mean of 12 per-label AUCs).

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

**State as of 2026-08-30 (22:15):** **Best public LB 0.912 (#10, seven versions)**; ⭐ **5-fold `v09h` is done and shipped** (RunPod chain4: per-fold 0.8683/0.8668/0.8589/0.8546/0.8653, **pooled OOF 0.8625** vs `v05g`'s 0.8467, gold-58 0.874; `rsna-knee-ckpt-v09h` = 5 checkpoints, versioned from the local pull after the pod's ship died silently on the expired token; **infer v14 pushed** with all five mounted — not submitted). **The pod is deleted**; all pod artifacts live in `artifacts/` (md5-verified). Next: a new input representation (P-23 #3/#4) or P-17 self-training on the multi-family OOF. See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (19:50):** ⭐ **Best public LB 0.912 — submission #10** (infer v13: v05a+v05b+v05g+v06c + `v08w` + `v10c` + `v09h`, fold-0 proxy OOF 0.8820; #9 without `v09h` = 0.909). The c02 lane moved the LB **0.900 → 0.912 in one day** (2.4× the floor); the `v09h` increment alone (+0.003) is 🔁, as the OOF predicted — the three c02 members are one family, so the next LB move needs a new *input representation* (P-23 #3/#4, P-17). `INFER_MEMBERS` in `src/` = the seven. **5-fold `v09h` training on the pod** (chain4; folds 0/1 = 0.8683/0.8668, fold 2 running; ≈ 21:25 done → `ship` → infer v14 push → pod deleted). Offset OOF→LB ≈ +0.030 (n=7). See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (19:05):** ⭐ **Best public LB 0.909 — submission #9** (infer v12: v05a+v05b+v05g+v06c + `v08w` + `v10c`, fold-0 proxy OOF 0.8795, +0.009 over 0.900 = 1.8× the 0.005 floor, offset +0.0295): the c02 band + window-attention + hybrid recipe carries to the hidden test. #10 (+`v09h`, OOF 0.8820) scoring. **5-fold `v09h` training on the RunPod pod** (chain4; fold 1 = 0.8668, fold 0 0.8683; ≈ 21:25 done, then `ship` → `rsna-knee-ckpt-v09h`, infer push, pod deleted). `v08w` pinned (`rsna-knee-ckpt-v08w`); the infer kernel mounts only Dataset pins now. Rerun cost: #9 ≈ 1 h 50 min (2× #8; `v10c` + a second decode pass — experiments.md Infrastructure 2026-08-30). See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (17:50):** ⭐ **The 0.936 notebook's mechanism is measured**: band + window head (+0.007) + hybrid backbone (+0.004, menisci), *not* resolution (CoAtNet-1 @224 0.8683 > CoAtNet-2 @384 0.8641). Default member recipe is now `c02` + `window_mode="random"` + `head_type="window_attn"`; `v09h` costs 50 min/fold on a 4090. **Submissions #9 (six versions, OOF 0.8795) and #10 (seven, 0.8820) are scoring.** P-12 TTA: 🔁, not adopted. Next pod job: 5-fold `v09h` (~4 h, ≈ $3) on Tian's go-ahead; the pod idles at $0.74/h. See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (17:35):** three RunPod/Kaggle arms on the c02 cache with the window-attention head are the three best single models — **`v09h` 0.8683 (CoAtNet-1 @224, 50 min)**, `v08w` 0.8648, `v10c` 0.8641 — and one family (ρ 0.84). Blends: 6-member 0.8795 (**submitted #9, infer v12, 17:08**), 7-member 0.8820 (infer v13 → #10 on Tian's instruction). P-12 TTA evals running on the pod. See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (16:45):** **`v10c` (CoAtNet-2 @384, c02, RunPod) fold 0 = OOF 0.8641**, parity with `v08w` from a second family, the least correlated member (ρ 0.73–0.81) and the best on both menisci (LatMen 0.858); **6-member blend v05a+v05b+v05g+v06c+v08w+v10c = 0.8795 (+0.0073 over the LB-0.900 blend)** — infer v12 with those six pushed for an end-to-end check; submission pending Tian. `v09h` training on the pod; P-12 evals after it. Next arm: `v10c` at 10–12 epochs (still rising). See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (14:15):** **`v08w` fold 0 = OOF 0.8648, the best single model** (v05a 0.8574; 12/12 labels up; MCL +0.028 confirms half of P-26) but it blends to only +0.0044 as a fifth member (ρ 0.866) — the DINOv2 family saturates at ≈ 0.876; no submission on it alone. RunPod: `v10c` (CoAtNet-2 @384) trains at 0.30 s/study (~22 min/epoch, fold 0 ≈ 16:50) after an fd-limit crash (traps 29), then `v09h`, then the P-12 evals. Kaggle OAuth tokens today last **3 h** (traps 20). See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (13:10):** `v08w` fold 0 runs on Kaggle (train v17, expect ~15:00). The Kaggle P-12 `oof_eval` pass (eval v2) **died OOM on its second member** (traps 28) after scoring v05a at 0.8621 with mean TTA (+0.0047, 🔁 alone); the measurement moved to **RunPod**: a secure RTX 4090 pod (`2wend9j0lr7zf3`, EUR-IS-2, $0.74/h, MCP-driven, direct SSH) runs `oof_eval` mean + focal, then `v10c`, then `v09h`, shipping each as `rsna-knee-ckpt-<arm>` — the P-24 runner's first real run (traps 29 for the first-run fixes). See [docs/handoff.md](docs/handoff.md).
**State as of 2026-08-30 (afternoon):** the 0.936 notebook was re-read cell by cell; its gap to our 0.900 is
(a) a **0.924 single model** — CoAtNet-2 @384 over 64 slices at a **2–98 % slice band** with **per-label
attention over every window** — and (b) three more input-representation families rank-fused on top;
same-geometry backbones gave head-like diversity only. Our two weakest labels (MCL 0.836, Lateral Meniscus
0.833 OOF) are the ones its docstring says the discarded outer slices carry. **Shipped, locally verified,
zero GPU spent:** cache v2 (`c02`: 6 slots × 18/12/12/14/8/8 slices, 2–98 %, 336 px, 64-study blobs —
kernels `rsna-knee-cache2-a..d` building), the window-attention head + random-window training (P-25),
offline timm hybrids (`timm:coatnet_rmlp_1_rw_224`, `timm:coatnet_rmlp_2_rw_384`), mixed-geometry
inference (c01 + c02 members in one blend), slice-offset TTA + `MODE="oof_eval"` (P-12), a RunPod runner
(P-24; Tian chose RunPod for the hybrids), pins `rsna-knee-ckpt-v06` / `-v05g`, `src/cache_selftest.py` +
`src/window_head_test.py`. Pre-change and post-change inference of the existing members are byte-identical.
Next: Kaggle smoke of arms `v08w` + `v09h` once the c02 kernels finish, then (go-ahead) `v08w` fold 0
(~2 h) and the `oof_eval` TTA pass (~0.5 h); `v09h` / `v10c` on RunPod. Backlog: P-25, P-26, P-23 #2, P-12.
**State as of 2026-08-30 (morning):** **eight submissions, best public LB 0.900** (#8, infer v10: the P-21 two-head
blend + `v05g` 5-fold + **`v06c` ConvNeXt-T**, one vote per version; **+0.004 over 0.896 is 🔁 under the 0.005
floor**, but it is the default blend now). Overnight
P-23 results: **`v06c` ConvNeXt-Tiny is a second family at parity** (fold-0 OOF 0.8562 vs 0.8574 for the best
DINOv2 head) but with head-like diversity (ρ 0.83, blend +0.006, 10/12 labels up — a narrow reject by the
pre-registered rule; #8 = 0.900); **`v07s` 16-slices-as-channels DINOv2 is dead
as built** (OOF 0.74 on all five folds, 4.8 h). GPU quota left this week ≈ 6 h. The 2026-08-29 state follows.
**State as of 2026-08-29 (night):** **seven submissions, best public LB 0.896** — #5, the P-21
rank-mean of an attention-head and a concat-head model on one DINOv2-S/14 224 backbone, fold 0,
OOF 0.8670 (+0.019 LB over the best single model, 3.8× the LB floor). Progression 0.500 → 0.841 →
0.871 → 0.877 → **0.896** → 0.886 (#6, five concat folds alone: folds are worth +0.009) → 0.896
(#7, folds + both heads: **folds add nothing on top of head diversity**, so the next GPU spend is a
third diverse member, not more folds — P-10). An epoch costs ~11 min, so a fold-0 arm is ~0.9 h. Measured today: the **OOF noise floor** (0.008 macro / ~0.03 per label), **slice
jitter +0.011**, **laterality ≈ +0.015 of v03's +0.022**, and **two heads rank-blend to 0.8670** at
ρ = 0.773 (P-21, now shipped as `INFER_MEMBERS` with decode-once inference). **P-22 switched the
checkpoint policy** to best-OOF-epoch (`ckpt_policy="best_oof"`: +0.013 split-half for the concat
head, ~0 for attn, gold flat) and re-read **P-09 as a tie** — the attention head's +0.0103 was the
concat head's late decay. The first 5-fold run was **wasted** (`v05f`: the cache never mounted,
traps 6f — Kaggle moved kernel outputs to `/kaggle/input/notebooks/<owner>/<slug>/` platform-wide
that day; the loader now searches depth 4, fails loudly, and prints the mount layout). **The valid
5-fold run `v05g` is done** (4.27 h, per-fold OOF 0.843–0.851, pooled 0.8467, gold 0.8476 on all
58); its LB value is being measured. **2026-08-30:** `crazy_good_rsna.ipynb` (public **0.936**) was read
cell by cell — it trains nothing; its DINOv2 branch alone is ≈ 0.899, **at parity with our 0.896**, and
the remaining ≈ 0.036 is three more model families rank-fused on top (16-channel ViT, RadImageNet
heads, CoAtNet-2@384 at 0.924 alone). **P-23 multi-family fusion is now the top of the backlog**;
decomposition in [docs/research.md](docs/research.md) §2.7.1. `v06c` (ConvNeXt-T, P-23 candidate #1)
is in flight as train v15. See [docs/handoff.md](docs/handoff.md).

## 📚 Documentation map — read the relevant one before acting

| File | What it holds | Read it when |
|---|---|---|
| [docs/handoff.md](docs/handoff.md) | Session state, what changed last, next action | **First, always** |
| [docs/traps.md](docs/traps.md) | Bugs and **silent** failure modes, tiered by damage | Before writing pipeline code |
| [docs/experiments.md](docs/experiments.md) | Every measurement, with a verdict | Before proposing an experiment |
| [docs/proposals.md](docs/proposals.md) | **Ranked backlog as testable cards P-00…P-23** (hypothesis, evidence, measure, noise floor, cost) | When choosing what to do next |
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
src/blend_check.py      P-23 acceptance rule on fold-0 OOF csvs (rho, blend gain, per-label table)
scripts/runpod_bootstrap.sh  off-Kaggle runner: setup | train <arm> | ship <arm>   (requirements-gpu.txt)
kaggle/rsna-knee-train/     generated training/inference notebook + kernel-metadata.json
kaggle/rsna-knee-folds/     5-fold ensemble run (FIVE_FOLD=True sed'd in at build time)
kaggle/rsna-knee-infer/     MODE="infer" copy -- the kernel that gets SUBMITTED
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
(traps 12d). The **5-fold** run is a sed'd copy into a second kernel:

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
a **0.006 band**, from 2,559 teams. **We are at 0.912** (2026-08-30 evening, submission #10: two DINOv2 heads + five concat folds + one ConvNeXt-T + the three c02 window-attention members `v08w` / `v10c` / `v09h`, one vote per version; 0.909 without `v09h`, 0.900 before the c02 members; 0.877 for the best single model on the LB). The Efficiency Prize has its own leaderboard, published
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
