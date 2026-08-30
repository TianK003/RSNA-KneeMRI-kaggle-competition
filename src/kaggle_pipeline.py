# %% [markdown]
# # RSNA Knee — 12 findings from one MRI study
#
# A 2.5D DINOv2 baseline with report-derived weak labels, grouped folds, a runtime
# guard, and resumable checkpoints.
#
# **The shape of the problem.** Only 58 of 4,407 training studies carry official
# labels. The other 4,349 carry a radiology report. `train.csv` has a `Report`
# column and `test.csv` does **not** — text exists when fitting and is absent when
# predicting. So reports can only ever be a source of *targets*, never a model
# input. A text branch would have nothing to read at inference.
#
# **What the metric changes.** Macro ROC-AUC is the unweighted mean of 12 per-label
# AUCs, and AUC is invariant to any strictly increasing transform. Three
# consequences drive design choices below: calibration is worthless (only rank
# order matters), ensembles must average **ranks** not probabilities, and every
# label costs the same — one label left at chance forfeits ~(M−0.5)/12 of the
# score, so rare findings deserve *more* attention than common ones.
#
# **Order of sections** follows what constrains what: config → targets → which
# series to show the encoder → how to read pixels → model → training → OOF →
# inference.

# %%
# ── Section 0: environment ────────────────────────────────────────────────────
# Detects Kaggle vs local so the same file runs in both places. Locally it can
# only smoke-test shapes (there are 3 sample studies and no GPU); on Kaggle it
# trains for real.
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import pandas as pd

T_START = time.time()

ON_KAGGLE = os.path.exists("/kaggle/input")


def resolve_dir(candidates, must_contain=None):
    """First candidate that exists (and holds `must_contain`, if given).

    Kaggle mounts competitions at BOTH /kaggle/input/<comp> and
    /kaggle/input/competitions/<comp> depending on how the kernel was created, and
    Models at either /kaggle/input/<name>/... or /kaggle/input/models/<owner>/...
    Hard-coding one path is the single most common reason a CLI-pushed kernel dies
    instantly, so probe instead of assuming.
    """
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        if must_contain and not os.path.exists(os.path.join(c, must_contain)):
            continue
        return c
    return None


if ON_KAGGLE:
    COMP = resolve_dir([
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    ], must_contain="train.csv")
    WORK = "/kaggle/working"
    if COMP is None:
        print("!! competition data not found. /kaggle/input contains:")
        for root in ("/kaggle/input", "/kaggle/input/competitions"):
            if os.path.isdir(root):
                print(f"   {root}: {sorted(os.listdir(root))[:20]}")
        raise SystemExit("attach the competition to this kernel")
else:
    COMP = "data"
    WORK = "artifacts/local_run"


def print_input_layout(root="/kaggle/input", max_depth=3,
                       skip=("train_series", "test_series"), max_dirs=12):
    """Where did Kaggle mount things? A slug created today lays out /kaggle/input
    differently from one created last week (type-prefixed, one or two levels deeper), and
    a glob that is too shallow fails silently (traps 6f). Print the tree, minus the image
    trees, so the layout is read off the log instead of inferred after the fact."""
    if not os.path.isdir(root):
        return
    print(f"input layout under {root} (depth <= {max_depth}; image trees not descended):")

    def walk(d, depth):
        try:
            names = sorted(os.listdir(d))
        except OSError as e:
            print(f"  {d}: {e}")
            return
        dirs = [n for n in names if os.path.isdir(os.path.join(d, n))]
        files = [n for n in names if n not in dirs]
        print(f"  {d}: {len(dirs)} dirs, {len(files)} files"
              + (f"  e.g. {files[:4]}" if files else ""))
        if depth >= max_depth:
            return
        for n in dirs[:max_dirs]:
            if n in skip:
                print(f"  {os.path.join(d, n)}: (image tree, skipped)")
            else:
                walk(os.path.join(d, n), depth + 1)
        if len(dirs) > max_dirs:
            print(f"  {d}: ... {len(dirs) - max_dirs} more dirs not shown")

    walk(root, 0)


if ON_KAGGLE:
    print_input_layout()

os.makedirs(WORK, exist_ok=True)
print(f"ON_KAGGLE={ON_KAGGLE}  COMP={COMP}  WORK={WORK}")
if ON_KAGGLE:
    print(f"COMP contains: {sorted(os.listdir(COMP))[:12]}")

# %%
# ── Section 1: configuration ──────────────────────────────────────────────────
# Everything tunable lives here so an experiment is one edit and the config is
# saved next to the checkpoints.
#
# `smoke` is the important one: it shrinks every dimension so the whole pipeline
# runs end to end in a couple of minutes. Never trust a long run you have not
# smoke-tested first — a crash in the inference cell after six hours of training
# costs a whole session.

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Plane x acquisition slots, chosen so every finding has at least one sequence
# that shows it well: cruciates run obliquely (sagittal), collaterals and the
# meniscal body coronally, patellar cartilage axially.
SLOTS = [
    "SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
    "SAG_FLUID_NOFS", "COR_T1", "SAG_T1",
]


# ┌──────────────────────────────────────────────────────────────────────────┐
# │ FORCE_SMOKE: True  = fast end-to-end check (minutes) -- use for the first │
# │                      run of any new/edited notebook.                     │
# │              False = real training run (hours, resumable).               │
# │              None  = auto (smoke locally, real on Kaggle).               │
# └──────────────────────────────────────────────────────────────────────────┘
FORCE_SMOKE = True

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ MODE: "train" = train the configured folds, then infer if all complete.  │
# │       "infer" = load `{version}_fold*_best.pt` from a mounted kernel     │
# │                 output and only predict the test set. This is what gets  │
# │                 SUBMITTED: a code competition re-runs the notebook on    │
# │                 the hidden test, and re-training there would both blow   │
# │                 the runtime and change the model being scored.           │
# │       "oof_eval" = score each INFER_MEMBERS version's fold-0 checkpoint  │
# │                 on its held-out studies from the cache, with the TTA /  │
# │                 eval_windows in INFER_OVERRIDES -> {v}_fold0_tta_oof.csv │
# │                 for src/blend_check.py. No test prediction (P-12).       │
# │       "auto"  = "infer" if such checkpoints are mounted, else "train".   │
# └──────────────────────────────────────────────────────────────────────────┘
MODE = "auto"

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ INFER_MEMBERS: versions rank-meaned in "infer" mode (P-21). Every        │
# │ mounted `{version}_fold*_best.pt` of every listed version is one member  │
# │ of a flat rank-mean. A listed version with NO mounted checkpoint is      │
# │ fatal, so the blend can never silently shrink to a model that was not   │
# │ the one validated (traps 6d). Empty -> [cfg.version]. Ignored in "train".│
# │ Members must share preprocessing geometry; head_type may differ.        │
# └──────────────────────────────────────────────────────────────────────────┘
INFER_MEMBERS = ["v05a", "v05b"]        # attn + concat heads, fold 0: OOF 0.8670 rank-mean
# How members combine. "by_version": rank-mean the folds of each version, then rank-mean the
# versions -- every version gets one vote, however many folds it has. "flat": one vote per
# checkpoint. Measured on fold 0 (2026-08-29): attn + concat-8ep + concat-4ep flat = 0.8680,
# but with the concat-4ep version carrying 5 fold votes the flat mean drops to 0.8611 -- below
# the two-head blend alone (0.8670) -- because the attention head, the source of the
# diversity, becomes 1/7 of the vote. Versions are the unit of diversity; folds are replicates.
INFER_BLEND = "by_version"
# Per-version MEMBER-key overrides at inference (P-12 TTA for members whose checkpoints predate
# the fields, or an eval_windows cap). Only keys in INFER_MEMBER_KEYS are allowed -- an override
# can change how a member reads the decoded array, never which array is decoded. Example:
#   INFER_OVERRIDES = {"v05a": {"tta_offsets": (-1, 0, 1), "tta_pool": "focal"}}
INFER_OVERRIDES = {}

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ ARMS: run several fold-0 configurations back to back in ONE session.     │
# │ Each arm gets its own version string, so its checkpoints and OOF csvs    │
# │ (`{version}_fold0_*`) never collide. An arm that raises is logged and    │
# │ skipped -- the session, not the code, is the scarce resource.            │
# │ Set ARMS = None for a single run of the plain config.                    │
# └──────────────────────────────────────────────────────────────────────────┘
# v11 measured the floor: |v04a - v04base| = 0.008 macro (up to 0.03 per label). Verdicts:
# jitter +0.011 KEEP; lat_undo -0.015 confirms P-05; attn -0.005 INCONCLUSIVE *because it had
# not converged* (still rising at ep3, train loss 0.447 vs 0.398). So the retest gives the head
# a schedule it can converge in, with a matched control that changes only the head.
# v13 (v05a attn / v05b concat, 8 ep) closed P-09 and gave the 0.896 two-head blend; the 5-fold
# v05g run showed folds add nothing on top of head diversity (#6/#7). P-10: the next member must
# make *different* errors -- a second architecture family. ConvNeXt-Tiny, concat head, jitter,
# 8 epochs under ckpt_policy=best_oof (unknown peak epoch for a CNN), backbone LR 1e-4 per the
# card (ImageNet-supervised CNN tolerates 5x the LR that DINOv2's SSL features need).
# 2026-08-30 (P-25 / P-26 / P-23 #2): members on the wide-band c02 cache with the window-attention
# head. `v08w` = DINOv2-S at 224 (isolates band + windows + head from resolution; ~2 h fold 0 on a
# T4). `v09h` = the timm CoAtNet-1 hybrid probe at 224 (RunPod). `v10c` = CoAtNet-2 @384, the 0.936
# notebook's strongest-member recipe (RunPod; grad_checkpoint for 24 GB cards, eval_windows 42 so
# the hidden-test rerun stays inside the budget -- oof_eval must use the same value).
C02 = {"cache_scheme": "c02", "window_mode": "random", "head_type": "window_attn",
       "train_windows": 24, "epochs": 8}
ARMS = [
    ("v08w", {**C02, "backbone": "dinov2", "img_size": 224}),
    ("v09h", {**C02, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 1e-4}),
]
ARM_V10C = ("v10c", {**C02, "backbone": "timm:coatnet_rmlp_2_rw_384", "img_size": 384,
                     "lr_backbone": 1e-4, "eval_windows": 42, "grad_checkpoint": True})
PRIMARY_ARM = "v08w"
ARM_FOLDS = (0,)
# Off-Kaggle runner (scripts/runpod_bootstrap.sh): RSNA_ARM=<version> runs exactly that arm
# (from ARMS or ARM_V10C) and makes it PRIMARY_ARM; RSNA_WORKERS / RSNA_RUNTIME_H override the
# loader worker count and the session guard. Unset on Kaggle, so nothing changes there.
if os.environ.get("RSNA_ARM"):
    _only = os.environ["RSNA_ARM"]
    ARMS = [a for a in list(ARMS) + [ARM_V10C] if a[0] == _only]
    if not ARMS:
        raise SystemExit(f"RSNA_ARM={_only!r} is not one of the defined arms")
    PRIMARY_ARM = _only
    print(f"RSNA_ARM: running only {_only}")

# Refuse to silently train the v02 decode path when the cache is expected (traps 6f).
ALLOW_DECODE_FALLBACK = False

# Flipped by sed for kaggle/rsna-knee-folds: five folds of the confirmed v04d recipe
# (concat + jitter, 4 epochs) for the first real ensemble. 5 x 4 epochs ~= 4.5 h; 5 x 8 would
# be ~9 h and needs the resume path instead.
# `v05f` is RETIRED: rsna-knee-folds v2 wrote v05f_fold*.pt trained on the v02 decode path
# (the cache never mounted, traps 6f). Never mount that output; the valid re-run is `v05g`.
FIVE_FOLD = False
if FIVE_FOLD:
    ARMS = [("v05g", {"cache_jitter": True, "folds": (0, 1, 2, 3, 4), "epochs": 4})]
    PRIMARY_ARM = "v05g"

# Flipped by sed for kaggle/rsna-knee-stack (P-23 candidate #3): five folds of the 16-channel
# member, 8 epochs under best_oof. It has its OWN kernel slug so pushing it never repoints the
# rsna-knee-train / rsna-knee-folds mounts that rsna-knee-infer reads (handoff 2026-08-30).
STACK_RUN = False
if STACK_RUN:
    ARMS = [("v07s", {"stack_mode": "channels", "cache_jitter": True,
                      "folds": (0, 1, 2, 3, 4), "epochs": 8})]
    PRIMARY_ARM = "v07s"


@dataclass
class Config:
    smoke: bool = field(default_factory=lambda:
                        (not ON_KAGGLE) if FORCE_SMOKE is None else bool(FORCE_SMOKE))
    version: str = "v03"             # v01 rank targets (smoke only) · v02 prob targets, decode per epoch · v03 from cache

    # data
    img_size: int = 224              # DINOv2 ViT-S/14 patches 14 -> 224 = 16x16 tokens
    slices_per_slot: int = 6         # uniformly sampled centres per slot
    triplet_gap: int = 2             # channels are slices [i-gap, i, i+gap]  (decode path only)

    # Cache path (P-01). When a cache built by src/cache_pipeline.py is mounted, training
    # reads one uint8 array per study; TEST studies are built on the fly by the very same
    # functions (crop, per-series normalisation, laterality), so train and test share one
    # preprocessing code path. Triplets are neighbouring cached slices [c-1, c, c+1].
    use_cache: bool = True
    cache_n_slices: int = 16         # stored slices per slot (must match the mounted cache)
    cache_px: int = 224
    crop_mm: float = 130.0
    lat_dead_zone_mm: float = 20.0
    # P-05 ablation. The cache stores every knee in a canonical left-knee frame; this puts
    # the right knees back into their own chirality at load time (both cache operations are
    # involutions), so laterality can be ablated without rebuilding 21 GB of cache.
    lat_undo: bool = False
    # P-08 sub-arm: jitter the K sampled slice centres by +-1 cached slice each epoch. The
    # only real augmentation this pipeline has (the other is Gaussian noise at sigma 0.01).
    cache_jitter: bool = False
    # P-23 candidate #3: how the 16 cached slices of a slot reach the encoder. "triplet" = K
    # centres, each a 3-channel [c-1, c, c+1] image (v03..v06). "channels" = ONE image per slot
    # with all 16 cached slices as its input channels -- the whole stack in one forward pass, a
    # different input representation from every triplet member (the 0.936 notebook's second
    # family works this way). The patch-embedding conv is widened 3 -> 16 (RGB-mean weights
    # x 3/16, response scale preserved) and trained at `lr_stem`. 6 encoder passes per study
    # instead of 36, so an epoch is ~6x cheaper. With `cache_jitter` the whole stack shifts +-1.
    stack_mode: str = "triplet"
    lr_stem: float = 2e-4            # channels mode only: the widened patch-embedding conv

    # Cache SCHEME (2026-08-30). "c01" = the original cache: dense [6, 16, 224, 224] per study,
    # one .npy each, per-plane band sag 8-92 / cor 20-80 / ax 10-90 -- described by cache_px /
    # cache_n_slices above. "c02" = the wide-band rebuild: the same six slots with RAGGED slice
    # budgets (18/12/12/14/8/8 = 72 slices, order = SLOTS), band 2-98 % for every plane, 336 px,
    # stored FLAT [72, 336, 336] inside multi-study blob files. Why: the 0.936 notebook's best
    # member uses 2-98 % and reports the outer slices carry the collaterals and the lateral
    # meniscus -- our two weakest labels. Both caches can be mounted at once; each Config resolves
    # to exactly one of them through cache_version_for(). The c02 fields below are ignored for c01.
    cache_scheme: str = "c01"
    cache_px_wide: int = 336         # c02 stored resolution (cache_px stays the c01 value)
    cache_slot_slices: tuple = ()    # c02 budgets per slot; () -> (18, 12, 12, 14, 8, 8)
    cache_band: tuple = ()           # c02 (lo, hi) for every plane; () -> (0.02, 0.98)

    # WINDOWS (P-25). "fixed" = K equidistant triplet centres per slot (every member through
    # v06c; array_to_tensor). "random" = the study is a set of (slot, centre) windows: training
    # samples `train_windows` of them (stratified, >= 2 per present slot) as its augmentation,
    # evaluation feeds every valid window (or `eval_windows` equidistant ones when > 0 -- the
    # SAME value must be used by oof_eval and infer so the OOF number predicts the LB number).
    # The Dataset ships the uint8 array + indices; the model gathers/normalises/resizes on the
    # GPU, so 60 windows never travel through DataLoader shared memory as float tensors.
    window_mode: str = "fixed"
    train_windows: int = 24
    eval_windows: int = 0
    # Slice-offset TTA for fixed-window members (P-12): the K centres are shifted by each offset
    # (clipped to the stack), one forward per offset, probabilities pooled per label.
    # tta_pool "mean" = average; "focal" = the 0.936 notebook's rule: max over views for
    # Fracture / Contusion / both Menisci / Baker's, top-2 mean for ACL / MCL, mean otherwise.
    tta_offsets: tuple = (0,)
    tta_pool: str = "mean"

    # model
    # P-10: a second architecture family as a blend member. "dinov2" = DINOv2 ViT-S/14 (CLS
    # token); "convnext_tiny" = HF facebook/convnext-tiny-224 (ImageNet-1k, Apache-2.0,
    # LayerNorm throughout so batch-of-1 is safe; pooled 768-d output). Same 224x3 ImageNet-
    # normalised triplets feed both, so a study array is shared across families at inference.
    backbone: str = "dinov2"
    backbone_dir: str = ""           # resolved from `backbone` below (and per arm / per member)
    dropout: float = 0.1
    # P-09. "concat" = v03 baseline (6 slot vectors + mask -> one Linear); "attn" = 12
    # learned label queries doing masked attention over the present slot vectors.
    # P-25. "window_attn" = 12 label queries attending over EVERY (slot, window) token of the
    # study (per-label softmax over windows, slot embedding added), with no label-agnostic
    # per-slot pooling in between -- the 0.936 notebook's strongest member pools this way.
    head_type: str = "concat"
    slot_dropout: float = 0.0        # P-09 sub-arm; 0 keeps the head A/B clean
    slot_embed: bool = True          # window_attn: add a learned per-slot embedding to each token
    # timm hybrids (P-23 #2): `backbone="timm:<arch>"` loads <dir>/model.safetensors offline.
    # Gradient checkpointing halves activation memory for coatnet_2 @384 x 24 windows on 24 GB.
    grad_checkpoint: bool = False

    # optimisation
    folds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 8         # v11: with jitter the OOF curve had not peaked by epoch 3
    lr_head: float = 1e-3
    # Backbone LR and layer-wise decay (P-03). Every medical DINOv2 fine-tuning
    # recipe we found lands at 1e-6..2e-5 for the top block; a uniform 5e-5 is the
    # regime described as catastrophic forgetting of the self-supervised features.
    # Block i gets lr_backbone * llrd_decay ** (n_blocks - 1 - i); the patch/pos
    # embeddings get one more decay step. 0.75 is the BEiT/MAE convention.
    lr_backbone: float = 2e-5
    llrd_decay: float = 0.75
    weight_decay: float = 0.02       # not applied to biases / LayerNorm
    # EMA of the weights is what gets validated and saved (robust to label noise,
    # and makes fixed-epoch selection safe). 0 disables.
    ema_decay: float = 0.998
    batch_studies: int = 1           # one study = up to 6 slots x 6 slices of ViT work
    grad_accum: int = 4
    warmup_frac: float = 0.1
    max_grad_norm: float = 1.0
    amp: bool = True

    # supervision
    gold_weight: float = 8.0
    weak_weight_floor: float = 0.15

    # runtime
    runtime_limit_hours: float = float(os.environ.get("RSNA_RUNTIME_H", 8.3))   # headroom under Kaggle's 9 h
    seed: int = 42
    num_workers: int = int(os.environ.get("RSNA_WORKERS", 2))     # 8 on a local-NVMe box
    # Which epoch `_best.pt` holds. "best_oof": the epoch with the highest OOF-vs-teacher
    # macro-AUC so far (P-22: +0.013 split-half for the concat head, ~0 for attn, gold flat).
    # "last": EMA weights after the last completed epoch (fixed-epoch, used through v05).
    ckpt_policy: str = "best_oof"
    # Smoke only: cap the header scan so a verification run does not spend minutes
    # reading all ~24k series headers before it reaches the training loop.
    smoke_max_studies: int = 24

    def __post_init__(self):
        if self.cache_scheme not in ("c01", "c02"):
            raise SystemExit(f"unknown cache_scheme {self.cache_scheme!r}")
        if self.cache_scheme == "c02":
            self.cache_slot_slices = tuple(self.cache_slot_slices) or (18, 12, 12, 14, 8, 8)
            self.cache_band = tuple(self.cache_band) or (0.02, 0.98)
            if self.stack_mode != "triplet" or self.lat_undo:
                raise SystemExit("stack_mode='channels' and lat_undo are c01-only (v07s is dead, "
                                 "P-05 is closed); they were not ported to the flat c02 layout")
        self.tta_offsets = tuple(self.tta_offsets)
        if self.smoke:
            self.folds = (0,)
            self.epochs = 1
            self.slices_per_slot = 2
            self.train_windows = 4
            if not str(self.backbone).startswith("timm:"):
                # a fixed-resolution timm hybrid (coatnet_rmlp_2_rw_384) crashes at 224; DINOv2
                # and ConvNeXt take any size, and 224 keeps a CPU smoke fast
                self.img_size = 224
            self.runtime_limit_hours = 0.4
            self.ema_decay = 0.9      # 8 steps of smoke would leave a 0.998 EMA ~= init


CACHE_BAND = {"Sagittal": (0.08, 0.92), "Axial": (0.10, 0.90), "Coronal": (0.20, 0.80)}
PLANE_OF_SLOT = {"SAG_FLUID_FS": "Sagittal", "COR_FLUID_FS": "Coronal", "AX_FLUID_FS": "Axial",
                 "SAG_FLUID_NOFS": "Sagittal", "COR_T1": "Coronal", "SAG_T1": "Sagittal"}
CACHE_PCT = (1.0, 99.0)      # per-series percentile window (the cache builder's pct_lo / pct_hi)


def cache_version_of(scheme, px, slot_slices, band, crop_mm, lat_dead_zone_mm):
    """Name of the directory a cache lives in. It must encode EVERYTHING that changes the
    stored bytes: c01's string left out the band and the percentiles, so a band change at the
    same px/slices would have been silently accepted by the loader (traps 23). Byte-identical
    copy in src/kaggle_pipeline.py -- src/cache_selftest.py asserts the two agree."""
    if scheme == "c01":
        return f"c01_p{px}_s{slot_slices[0]}_crop{int(crop_mm)}_lat{int(lat_dead_zone_mm)}"
    lo, hi = band["Sagittal"]                       # c02: one band for every plane
    return (f"c02_p{px}_b{'-'.join(str(int(s)) for s in slot_slices)}"
            f"_band{int(round(lo * 100))}-{int(round(hi * 100))}"
            f"_crop{int(crop_mm)}_lat{int(lat_dead_zone_mm)}")


def slot_offsets(slot_slices):
    """Start index of each slot inside the flat (sum(slot_slices), P, P) array, plus the total."""
    starts, acc = [], 0
    for n in slot_slices:
        starts.append(acc)
        acc += int(n)
    return tuple(starts), acc


def _cfg_get(c):
    """Uniform reader over a Config object or a checkpoint's saved-config dict (old checkpoints
    lack the new fields, so every read carries the c01-era default)."""
    if isinstance(c, dict):
        return lambda k, d=None: c.get(k, d)
    return lambda k, d=None: getattr(c, k, d)


def cache_geom(c):
    """(scheme, px, slot_slices, band_dict) that Config `c` resolves to -- the one place the two
    schemes' field conventions meet. Works on a Config or on a saved-config dict."""
    g = _cfg_get(c)
    scheme = g("cache_scheme", "c01")
    if scheme == "c01":
        n = int(g("cache_n_slices", 16))
        return "c01", int(g("cache_px", 224)), (n,) * len(SLOTS), dict(CACHE_BAND)
    ss = tuple(int(s) for s in (g("cache_slot_slices", ()) or (18, 12, 12, 14, 8, 8)))
    band = tuple(float(b) for b in (g("cache_band", ()) or (0.02, 0.98)))
    return "c02", int(g("cache_px_wide", 336)), ss, {p: band for p in ("Sagittal", "Coronal", "Axial")}


def cache_version_for(c):
    g = _cfg_get(c)
    scheme, px, ss, band = cache_geom(c)
    return cache_version_of(scheme, px, ss, band, float(g("crop_mm", 130.0)),
                            float(g("lat_dead_zone_mm", 20.0)))


cfg = Config()
CACHE_VERSION = cache_version_for(cfg)     # the DEFAULT config's cache; arms/members recompute
# cache_version -> {StudyInstanceUID -> locator}; a locator is a .npy path (c01, one study per
# file) or (blob_path, row) (c02). Filled per cache version in Section 8 / at inference.
CACHE_INDEX = {}

# Weight locations differ between Kaggle (mounted Model, two possible layouts) and
# local (models/). config.json is the marker that a real HF checkpoint dir is there.
BACKBONES = {
    "dinov2": ([
        "/kaggle/input/dinov2/pytorch/small/1",
        "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1",
        "/kaggle/input/dinov2-small/pytorch/small/1",
        "models/dinov2_small",
    ], "metaresearch/dinov2 PyTorch/small/1 as a Model input"),
    "convnext_tiny": ([
        "/kaggle/input/datasets/tiankljucanin/convnext-tiny-224-hf",
        "/kaggle/input/convnext-tiny-224-hf",
        "models/convnext_tiny",
    ], "tiankljucanin/convnext-tiny-224-hf as a Dataset input"),
    # timm hybrids (P-23 #2). Each Dataset holds the HF timm repo files: config.json (the marker
    # resolve_dir probes) + model.safetensors; timm itself ships in the Kaggle image.
    "timm:coatnet_rmlp_1_rw_224": ([
        "/kaggle/input/datasets/tiankljucanin/timm-coatnet-rmlp-1-rw-224",
        "/kaggle/input/timm-coatnet-rmlp-1-rw-224",
        "models/coatnet_rmlp_1_rw_224",
    ], "tiankljucanin/timm-coatnet-rmlp-1-rw-224 as a Dataset input"),
    "timm:coatnet_rmlp_2_rw_384": ([
        "/kaggle/input/datasets/tiankljucanin/timm-coatnet-rmlp-2-rw-384",
        "/kaggle/input/timm-coatnet-rmlp-2-rw-384",
        "models/coatnet_rmlp_2_rw_384",
    ], "tiankljucanin/timm-coatnet-rmlp-2-rw-384 as a Dataset input"),
}


def resolve_backbone_dir(backbone: str) -> str:
    """HF checkpoint dir for a backbone family; both mount layouts probed (traps 6f/10)."""
    if backbone not in BACKBONES:
        raise SystemExit(f"unknown backbone {backbone!r}; known: {sorted(BACKBONES)}")
    candidates, attach = BACKBONES[backbone]
    d = resolve_dir(candidates, must_contain="config.json")
    if d is None:
        raise SystemExit(f"{backbone} weights not found -- attach {attach}")
    return d


cfg.backbone_dir = resolve_backbone_dir(cfg.backbone)
print(f"backbone: {cfg.backbone} @ {cfg.backbone_dir}")
print(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=1))


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


seed_all(cfg.seed)


def elapsed_h() -> float:
    return (time.time() - T_START) / 3600.0


def out_of_time() -> bool:
    """Runtime guard. Five folds do not fit in one 9 h session, so training must be
    able to stop cleanly and resume in the next session rather than be killed."""
    return elapsed_h() > cfg.runtime_limit_hours

# %% [markdown]
# ## Section 2: where the targets come from
#
# The reports are the only way to supervise 4,349 studies, and reading them well
# is a multilingual NLP problem (~9–12 languages, and for several findings *most*
# mentions are negative because a report lists what was checked and found intact).
#
# Rather than rebuild a lexicon, this mounts the public LLM-read label tables and
# averages their probabilities. Measured gold macro-AUC (n=58): hans_v4 0.893,
# pilkwang 0.870, sol56 0.835, blend 0.895 (rank blend 0.893 -- same within noise,
# but the rank blend put confident negatives at ~0.3 instead of ~0; see P-00 in
# docs/proposals.md).
#
# Two details matter more than the blend:
#
# 1. **Grade the mention, don't binarise it.** The reporting radiologist and the
#    annotator do not share a threshold — a report saying *small joint effusion*
#    can sit against a negative annotation, because annotators marked only
#    findings they judged significant and graded "on the fence" as negative. So
#    `term present ⇒ positive` is wrong by construction. Soft targets cost nothing
#    because only rank order is read.
# 2. **Weight by how confidently the report could be read.** Source disagreement and
#    indecisiveness both lower the weight. Measured caveat: a report that never mentions
#    synovitis blends to ~0.18 and is *not* strongly down-weighted (0.69 vs 0.80 on
#    addressed rows) — silence looks like a confident negative. Open card P-07/P-16.
#
# The 58 official labels overwrite the weak ones and carry `gold_weight`.

# %%
# ── Section 2: targets ────────────────────────────────────────────────────────
LLM_SOURCES = [
    ("hans_v4", [
        "/kaggle/input/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "data/llm_labels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    ]),
    ("pilkwang", [
        "/kaggle/input/rsna-knee-llm-labels/report_labels_v2.csv",
        "data/llm_labels/rsna-knee-llm-labels/report_labels_v2.csv",
    ]),
    ("sol56", [
        "/kaggle/input/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv",
        "data/llm_labels/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv",
    ]),
]


def shallow_glob(root, name, max_depth=3, skip=("train_series", "test_series")):
    """`glob` for `name` at depth 1..max_depth below `root` WITHOUT descending into the
    image trees. A recursive `**` glob over /kaggle/input walks ~819k DICOM files on a
    network mount -- minutes of dead time on every run, invisible on the rerun."""
    import glob
    hits = []
    for d in range(0, max_depth + 1):          # depth 0 = directly under root
        pat = os.path.join(root, *(["*"] * d), name)
        hits += [h for h in glob.glob(pat)
                 if not any(f"{os.sep}{sk}{os.sep}" in h or f"/{sk}/" in h for sk in skip)]
    return sorted(hits)


def first_existing(paths):
    """Exact candidates first, then search /kaggle/input for the filename.

    Dataset mount slugs are predictable but not guaranteed, so fall back to finding
    the file by name rather than failing and silently training on prior-only targets.
    """
    for p in paths:
        if os.path.exists(p):
            return p
    if ON_KAGGLE:
        want = os.path.basename(paths[0])
        for hit in shallow_glob("/kaggle/input", want, max_depth=4):
            return hit
    return None


def auc_score(y, s) -> float:
    """Mann-Whitney AUC, hand-rolled so the notebook needs no sklearn."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def build_targets(train_csv: str):
    tr = pd.read_csv(train_csv)
    idx = pd.Index(tr.StudyInstanceUID)
    is_gold = tr[LABELS].notna().all(axis=1)

    loaded = {}
    for name, paths in LLM_SOURCES:
        p = first_existing(paths)
        if p is None:
            print(f"  ! {name}: not mounted, skipping")
            continue
        d = pd.read_csv(p).set_index("StudyInstanceUID").reindex(idx)
        if set(LABELS) <= set(d.columns):
            loaded[name] = d
            print(f"  loaded {name} from {p}")

    soft = pd.DataFrame(index=idx)
    wt = pd.DataFrame(index=idx)
    if loaded:
        for lab in LABELS:
            arr = np.vstack([d[lab].to_numpy(dtype=float) for d in loaded.values()])
            # Probability space, NOT rank space (P-00). Rank-percentiles give tied
            # values their average rank, so on a label where most reports say exactly
            # 0 every confident negative landed at ~0.3-0.4 while gold rows sit at a
            # hard 0/1. BCE fits the value, not the order. Ranks are for scoring and
            # for ensembling predictions, never for building a target.
            with np.errstate(invalid="ignore"):
                soft[lab] = np.nanmean(arr, axis=0)
                spread = np.nanstd(arr, axis=0)
                mean = np.nanmean(arr, axis=0)
            agree = 1.0 - np.nan_to_num(spread, nan=0.5) * 2.0
            decisive = np.abs(np.nan_to_num(mean, nan=0.5) - 0.5) * 2
            wt[lab] = np.clip(0.5 * np.clip(agree, 0, 1) + 0.5 * np.clip(decisive, 0, 1),
                              cfg.weak_weight_floor, 1.0)
    else:
        # No label tables mounted: fall back to prior-only targets so the pipeline
        # still runs. This trains nothing useful and says so loudly.
        print("  ! NO LLM LABELS MOUNTED — using prior-only targets (smoke only)")
        for lab in LABELS:
            soft[lab] = 0.5
            wt[lab] = cfg.weak_weight_floor

    gold = tr.set_index("StudyInstanceUID")[LABELS]

    # Score the teacher BEFORE the gold override, otherwise we are grading the gold
    # labels against themselves and always get 1.000.
    gold_pos = is_gold.to_numpy()
    teacher_auc = float("nan")
    if loaded and gold_pos.sum():
        gy = gold.loc[idx[gold_pos]].astype(float)
        a = [auc_score(gy[l].to_numpy(), soft.loc[gold_pos, l].to_numpy())
             for l in LABELS]
        teacher_auc = float(np.nanmean(a))
        print(f"  teacher (report labels only) gold macro-AUC: {teacher_auc:.4f}")
        print("  ^ this is the signal ceiling the vision model is distilling from")

    for lab in LABELS:
        g = gold[lab].reindex(idx)
        have = g.notna().to_numpy()
        soft.loc[have, lab] = g[have].to_numpy()
        wt.loc[have, lab] = cfg.gold_weight

    for lab in LABELS:
        m = soft[lab].isna()
        if m.any():
            soft.loc[m, lab] = float(soft[lab].mean())
            wt.loc[m, lab] = cfg.weak_weight_floor

    # ---- folds: group studies that share a report text -------------------
    # 49 report texts are shared by 183 studies (largest group 37). Studies sharing
    # a report share a target vector, so splitting them across folds leaks the
    # answer into validation.
    norm = tr.Report.fillna("").str.strip().str.lower()
    grp = norm.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()[:16])
    meta = pd.DataFrame({
        "StudyInstanceUID": tr.StudyInstanceUID.to_numpy(),
        "is_gold": is_gold.astype(int).to_numpy(),
        "report_group": grp.to_numpy(),
    })
    g = meta.groupby("report_group").agg(n=("StudyInstanceUID", "size"),
                                         gold=("is_gold", "sum"))
    g = g.sample(frac=1.0, random_state=cfg.seed).sort_values(
        ["gold", "n"], ascending=False)
    n_folds = 5
    sizes = np.zeros(n_folds)
    golds = np.zeros(n_folds)
    assign = {}
    for gid, row in g.iterrows():
        # Balance gold first (so every fold is scoreable), then total size.
        k = int(np.lexsort((sizes, golds))[0]) if row.gold > 0 else int(np.argmin(sizes))
        assign[gid] = k
        sizes[k] += row.n
        golds[k] += row.gold
    meta["fold"] = meta.report_group.map(assign)

    tgt = soft.reset_index(drop=True)
    tgt.columns = LABELS
    wdf = wt.reset_index(drop=True)
    wdf.columns = [f"w__{c}" for c in LABELS]
    out = pd.concat([meta.reset_index(drop=True), tgt, wdf], axis=1)

    print(f"  targets: {out.shape[0]} studies, {int((out.is_gold == 1).sum())} gold")
    print("  fold sizes:",
          out.groupby("fold").size().to_dict(),
          "gold:", out.groupby("fold").is_gold.sum().to_dict())
    return out


targets = build_targets(os.path.join(COMP, "train.csv"))
targets.to_csv(os.path.join(WORK, "targets.csv"), index=False)
targets.head(3)

# %% [markdown]
# ## Section 3: which series to show the encoder
#
# A study holds 3–14 series (median 5) in three planes. The encoder cannot see all
# of them, so each study is reduced to at most six slots.
#
# `train_series.csv` ships `Fluid_Sensitive` and `Fat_Suppression`, but **as
# delivered they carry one bit, not two** — verified on the full training set: only
# `(1,1)` (14,010 rows) and `(0,0)` (10,361) ever occur, never a mixed pair. Two
# physically independent properties collapsed into one axis. Fluid sensitivity is a
# property of the *contrast weighting* (set by TR/TE); fat suppression is a
# *preparation* applied on top of any weighting. So both are recovered from the
# DICOM headers.
#
# `Anatomical_Plane`, by contrast, **is** trustworthy — it agreed 100% with the
# plane derived from `ImageOrientationPatient` on the sample studies, so it is used
# as-is and only recomputed when missing.
#
# Slot matching runs in two tiers. Strict (right plane, fluid **and** fat-sat) left
# 2 of 12 sample series unassigned and one study at 2/6 slots, because real studies
# routinely carry an axial fluid series with no fat suppression. A relaxed second
# tier lifted that to 4/6 and 5/6.

# %%
# ── Section 3: series selection ───────────────────────────────────────────────
import pydicom

TR_SHORT_MAX = 800.0   # ms
TE_LONG_MIN = 60.0     # ms
FATSAT_TOKENS = ("fs", "fatsat", "fat_sat", "stir", "spir", "spair", "tirm",
                 "dixon", "chess", "sat", "supp")
FLUID_TOKENS = ("t2", "stir", "pd", "dess", "spair", "spir", "tirm")


def has_token(text: str, tokens) -> bool:
    t = text.lower().replace("-", "").replace(" ", "")
    return any(tok.replace("_", "") in t for tok in tokens)


def plane_from_iop(iop) -> str:
    if iop is None or len(iop) != 6:
        return "unknown"
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(np.argmax(np.abs(n)))]


def classify_weighting(tr, te, scanning_seq: str, desc: str) -> str:
    d = desc.lower()
    # Gradient echo has a short TR by design, so the TR/TE rule does not apply.
    if "gr" in scanning_seq.lower() or any(t in d for t in ("gre", "dess", "medic", "flash")):
        return "GRE"
    if tr is None or te is None:
        for k in ("t1", "t2", "pd"):
            if k in d:
                return k.upper()
        return "unknown"
    if tr <= TR_SHORT_MAX:
        return "T1"
    return "T2" if te >= TE_LONG_MIN else "PD"


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def centre_x_mm(h):
    """Patient-space x (LPS: +x = patient's left) of the image centre, in mm. The
    Laterality tag is missing on ~half the corpus; this is what decides the knee side."""
    ipp = getattr(h, "ImagePositionPatient", None)
    iop = getattr(h, "ImageOrientationPatient", None)
    ps = getattr(h, "PixelSpacing", None)
    rows, cols = getattr(h, "Rows", None), getattr(h, "Columns", None)
    if None in (ipp, iop, ps, rows, cols) or len(iop) != 6:
        return None
    r = np.array(iop[:3], float)          # direction of increasing column
    c = np.array(iop[3:], float)          # direction of increasing row
    centre = (np.array(ipp, float) + r * (float(cols) / 2) * float(ps[1])
              + c * (float(rows) / 2) * float(ps[0]))
    return float(centre[0])


def study_side(sdf, dead_zone_mm):
    """('L'|'R'|'', tag, geometry, conflict) for one study -- same rule as the cache."""
    tags = [t for t in sdf.get("laterality_tag", pd.Series(dtype=str)).tolist() if t in ("L", "R")]
    tag = max(set(tags), key=tags.count) if tags else ""
    xs = sdf["centre_x_mm"].dropna().to_numpy(dtype=float) if "centre_x_mm" in sdf else np.array([])
    geo = ""
    if len(xs):
        med = float(np.median(xs))
        if med > dead_zone_mm:
            geo = "L"
        elif med < -dead_zone_mm:
            geo = "R"
    conflict = int(bool(tag) and bool(geo) and tag != geo)
    side = "" if conflict else (tag if tag else geo)
    return side, tag, geo, conflict


def scan_series(series_csv: str, image_root: str, cache: str,
                max_studies: int = 0) -> pd.DataFrame:
    """One row per series with header-derived properties. Cached, because reading
    ~24k headers is slow and a resumed session must not pay for it twice."""
    if max_studies:                    # a smoke scan must never be mistaken for a full one
        cache = cache.replace(".csv", f"_smoke{max_studies}.csv")
    if os.path.exists(cache):
        print(f"  series cache hit: {cache}")
        return pd.read_csv(cache)

    meta = pd.read_csv(series_csv)
    if max_studies:
        keep = meta.StudyInstanceUID.drop_duplicates().head(max_studies)
        meta = meta[meta.StudyInstanceUID.isin(set(keep))]
        print(f"  smoke: scanning {len(meta)} series from {len(keep)} studies only")
    rows = []
    t0 = time.time()
    for i, r in enumerate(meta.itertuples(index=False)):
        d = os.path.join(image_root, r.StudyInstanceUID, r.SeriesInstanceUID)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".dcm"))
        if not files:
            # Do not assume the hidden test tree keeps the .dcm extension.
            files = sorted(f for f in os.listdir(d)
                           if os.path.isfile(os.path.join(d, f)))
        if not files:
            continue
        h = None
        for f in files[:5]:            # first file that parses, not blindly files[0]
            try:
                h = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
                break
            except Exception:
                continue
        if h is None:
            continue
        desc = " ".join(str(getattr(h, k, "") or "") for k in
                        ("SeriesDescription", "SequenceName", "ScanOptions", "ProtocolName"))
        trv = getattr(h, "RepetitionTime", None)
        tev = getattr(h, "EchoTime", None)
        w = classify_weighting(float(trv) if trv is not None else None,
                               float(tev) if tev is not None else None,
                               str(getattr(h, "ScanningSequence", "") or ""), desc)
        plane = getattr(r, "Anatomical_Plane", None)
        if not isinstance(plane, str) or plane not in ("Sagittal", "Coronal", "Axial"):
            plane = plane_from_iop(getattr(h, "ImageOrientationPatient", None))
        rows.append({
            "StudyInstanceUID": r.StudyInstanceUID,
            "SeriesInstanceUID": r.SeriesInstanceUID,
            "n_slices": len(files),
            "plane": plane,
            "weighting": w,
            "fat_sat": int(has_token(desc, FATSAT_TOKENS)),
            "fluid": int(w in ("T2", "PD") or has_token(desc, FLUID_TOKENS)),
            "laterality_tag": (str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "").upper()
                               if str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "").upper() in ("L", "R") else ""),
            "centre_x_mm": centre_x_mm(h),
        })
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(meta)} series  {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(cache, index=False)
        print(f"  scanned {len(df)} series in {time.time()-t0:.0f}s -> {cache}")
    else:
        # Never cache an empty scan: a resumed session would hit the empty cache and
        # silently train on nothing.
        print(f"  scanned 0 series under {image_root} (cache NOT written)")
    return df


SLOT_SPEC = {
    "SAG_FLUID_FS":   ("Sagittal", lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "COR_FLUID_FS":   ("Coronal",  lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "AX_FLUID_FS":    ("Axial",    lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "SAG_FLUID_NOFS": ("Sagittal", lambda r: r.fluid and not r.fat_sat, lambda r: r.fluid),
    "COR_T1":         ("Coronal",  lambda r: r.weighting == "T1", lambda r: not r.fluid),
    "SAG_T1":         ("Sagittal", lambda r: r.weighting == "T1", lambda r: not r.fluid),
}


def select_slots(sdf: pd.DataFrame) -> dict:
    """One series per slot; strict tier across all slots first, then relaxed, so a
    series claimed strictly is not stolen by another slot's fallback. Prefers a
    slice count near 32 to avoid unusually long 3D / high-resolution acquisitions."""
    out, used = {}, set()
    for tier in (1, 2):
        for slot, (plane, strict, relaxed) in SLOT_SPEC.items():
            if slot in out:
                continue
            pred = strict if tier == 1 else relaxed
            cand = sdf[(sdf.plane == plane) & sdf.apply(pred, axis=1)]
            cand = cand[~cand.SeriesInstanceUID.isin(used)]
            if len(cand) == 0:
                continue
            chosen = cand.iloc[(cand.n_slices - 32).abs().to_numpy().argmin()]
            out[slot] = chosen.SeriesInstanceUID
            used.add(chosen.SeriesInstanceUID)
    return out


def build_manifest(series_df: pd.DataFrame, cache: str) -> pd.DataFrame:
    if os.path.exists(cache):
        print(f"  manifest cache hit: {cache}")
        return pd.read_csv(cache)
    rows = []
    for study, sdf in series_df.groupby("StudyInstanceUID"):
        slots = select_slots(sdf)
        side, tag, geo, conflict = study_side(sdf, cfg.lat_dead_zone_mm)
        rows.append({"StudyInstanceUID": study,
                     **{s: slots.get(s, "") for s in SLOTS},
                     "n_slots": len(slots), "side": side, "side_tag": tag,
                     "side_geo": geo, "side_conflict": conflict})
    m = pd.DataFrame(rows)
    m.to_csv(cache, index=False)
    print(f"  manifest -> {cache}; mean slots/study {m.n_slots.mean():.2f}; side resolved "
          f"{(m.side != '').mean():.1%} (tag {(m.side_tag != '').mean():.1%}, conflicts "
          f"{int(m.side_conflict.sum())})")
    print("  slot fill rate:",
          {s: round(float((m[s] != '').mean()), 3) for s in SLOTS})
    return m

# %% [markdown]
# ## Section 4: reading pixels
#
# Four things that produce **no error** if you get them wrong:
#
# 1. **Slice order.** The filename is the SOP Instance UID, assigned to be unique
#    rather than ordered. Measured on the sample studies: Spearman ρ between
#    filename order and true spatial position is **−0.012** on average, and
#    `|ρ|>0.99` in **0 of 12** series. Sorting by filename silently destroys the
#    slice adjacency that makes a 2.5D triplet meaningful. Sort by projecting
#    `ImagePositionPatient` onto the slice normal from `ImageOrientationPatient`.
# 2. **Rescale and photometric.** Apply `RescaleSlope`/`Intercept`; invert
#    `MONOCHROME1`. The sample studies happen to be all `MONOCHROME2` with trivial
#    rescale, but the hidden test set spans 16–19 sites.
# 3. **Per-series normalisation.** Max intensity spans 690 … 8,736 across sample
#    series (12.7×). A global window would not transfer. Clip each triplet jointly
#    at its 1st/99th percentile so its three channels stay mutually comparable.
# 4. **Multi-frame files.** Some DICOMs hold a volume in one file; take the middle
#    frame rather than crashing on the extra axis.

# %%
# ── Section 4: pixels ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GRAY_MEAN, GRAY_STD = 0.449, 0.226     # ImageNet mean/std averaged over RGB, for N-channel stacks


def ordered_slice_paths(series_dir: str, plane: str = None, return_head: bool = False):
    """Spatially ordered slice paths. NEVER trust filename order.

    With `plane` given (cache path) the sort direction has a FIXED sign: sagittal
    stacks run along +x (patient left), other planes along the positive dominant axis,
    so "reverse for right knees" canonicalises rather than randomises between sites.
    Without `plane` (legacy decode path) the cross-product normal is used as before.
    `return_head=True` also returns the header of the FIRST FILE IN FILENAME ORDER -- the one
    the cache builder reads IOP / PixelSpacing from (src/cache_pipeline.py::ordered_slice_paths);
    reading the spatially-first slice instead was a latent divergence between the two."""
    files = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if not files:   # do not assume the hidden test tree keeps the .dcm extension
        files = [f for f in os.listdir(series_dir)
                 if os.path.isfile(os.path.join(series_dir, f))]
    if not files:
        return ([], None) if return_head else []
    paths = [os.path.join(series_dir, f) for f in sorted(files)]
    heads, kept = [], []
    for p in paths:
        try:
            heads.append(pydicom.dcmread(p, stop_before_pixels=True))
            kept.append(p)
        except Exception:
            continue                    # a stray non-DICOM file must not poison the order
    paths = kept
    if not heads:
        return ([], None) if return_head else []
    first = heads[0]

    def done(ordered):
        return (ordered, first) if return_head else ordered

    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        if plane == "Sagittal":
            n = np.array([1.0, 0.0, 0.0])
        elif plane is not None and n[int(np.argmax(np.abs(n)))] < 0:
            n = -n
        keys, ok = [], True
        for h in heads:
            ipp = getattr(h, "ImagePositionPatient", None)
            if ipp is None:
                ok = False
                break
            keys.append(float(np.dot(np.array(ipp, float), n)))
        if ok:
            return done([p for _, p in sorted(zip(keys, paths), key=lambda t: t[0])])
    inst = [getattr(h, "InstanceNumber", None) for h in heads]
    if all(i is not None for i in inst):
        return done([p for _, p in sorted(zip(inst, paths), key=lambda t: t[0])])
    print(f"  ! {series_dir}: no usable position/instance headers -- filename order")
    return done(paths)


def read_plane(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:                      # multi-frame: middle frame
        arr = arr[arr.shape[0] // 2]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def build_triplets(series_dir: str, n_samples: int, gap: int, size: int) -> torch.Tensor:
    """-> (n_samples, 3, size, size). Channels are slices [i-gap, i, i+gap], so the
    encoder sees local 3D context through a 2D backbone."""
    ordered = ordered_slice_paths(series_dir)
    if not ordered:
        return torch.zeros(n_samples, 3, size, size)
    n = len(ordered)
    centres = np.clip(np.linspace(gap, n - 1 - gap, n_samples).round().astype(int), 0, n - 1)
    out = []
    for c in centres:
        idx = [max(0, c - gap), int(c), min(n - 1, c + gap)]
        try:
            planes = [read_plane(ordered[i]) for i in idx]
        except Exception:
            out.append(torch.zeros(3, size, size))
            continue
        h = min(p.shape[0] for p in planes)
        w = min(p.shape[1] for p in planes)
        stack = np.stack([p[:h, :w] for p in planes], axis=0).astype(np.float32)
        lo, hi = np.percentile(stack, [1, 99])     # joint clip keeps channels comparable
        stack = (np.clip(stack, lo, hi) - lo) / max(hi - lo, 1e-6)
        t = torch.from_numpy(stack).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        t = (t.squeeze(0) - IMAGENET_MEAN) / IMAGENET_STD
        out.append(t)
    return torch.stack(out)

def centre_crop_mm(arr, pixel_spacing, crop_mm):
    if not crop_mm or pixel_spacing is None or pixel_spacing <= 0:
        return arr
    side_px = int(round(crop_mm / pixel_spacing))
    h, w = arr.shape
    if side_px >= min(h, w):
        return arr
    y0 = (h - side_px) // 2
    x0 = (w - side_px) // 2
    return arr[y0:y0 + side_px, x0:x0 + side_px]


def resize_u8(stack01, px):
    t = torch.from_numpy(np.ascontiguousarray(stack01)).unsqueeze(1)
    t = F.interpolate(t, size=(px, px), mode="bilinear", align_corners=False)
    return (t.squeeze(1).clamp_(0, 1) * 255).round().to(torch.uint8).numpy()


def cache_series(series_dir, plane, cfg, is_right, n_slices, band=None, px=None):
    """-> ((n_slices, px, px) uint8, n_failed) or (None, n_failed).
    IDENTICAL to src/cache_pipeline.py::cache_series -- keep them in sync (src/cache_selftest.py
    checks both schemes bit for bit). Used at test time so a test study gets exactly the
    preprocessing the cached training studies got. `band` is the plane's (lo, hi) fraction of
    the ordered stack and `px` the stored resolution; both default to the c01 values."""
    ordered, head = ordered_slice_paths(series_dir, plane, return_head=True)
    if not ordered:
        return None, 0
    n = len(ordered)
    lo_f, hi_f = band if band is not None else CACHE_BAND.get(plane, (0.0, 1.0))
    lo_i, hi_i = int(round(lo_f * (n - 1))), int(round(hi_f * (n - 1)))
    if hi_i <= lo_i:
        lo_i, hi_i = 0, n - 1
    # Repeated neighbours on short series are intended (no np.unique).
    idx = np.linspace(lo_i, hi_i, n_slices).round().astype(int)
    if plane == "Sagittal" and is_right:
        idx = idx[::-1]
    iop = getattr(head, "ImageOrientationPatient", None)
    col_to_left = (iop is not None and len(iop) == 6 and float(iop[0]) > 0)
    mirror = plane in ("Coronal", "Axial") and (col_to_left == is_right)
    ps = getattr(head, "PixelSpacing", None)
    ps = float(ps[0]) if ps is not None else None
    planes, n_fail = [], 0
    for i in idx:
        try:
            a = read_plane(ordered[int(i)])
        except Exception:
            a = None
            n_fail += 1
        planes.append(a)
    good = [a for a in planes if a is not None]
    if not good:
        return None, n_fail
    h = min(a.shape[0] for a in good)
    w = min(a.shape[1] for a in good)
    # A failed slice is replaced by its nearest good neighbour, never by zeros (zeros
    # would drag the per-series percentiles down and enter the model as a black slice).
    fixed = []
    for k, a in enumerate(planes):
        if a is None:
            near = min((j for j, b in enumerate(planes) if b is not None), key=lambda j: abs(j - k))
            a = planes[near]
        fixed.append(a[:h, :w])
    stack = np.stack(fixed).astype(np.float32)
    stack = np.stack([centre_crop_mm(x, ps, cfg.crop_mm) for x in stack])
    lo, hi = np.percentile(stack, [CACHE_PCT[0], CACHE_PCT[1]])   # per SERIES, whole stack
    stack = (np.clip(stack, lo, hi) - lo) / max(hi - lo, 1e-6)
    if mirror:
        stack = stack[:, :, ::-1]
    return resize_u8(stack, px if px is not None else cfg.cache_px), n_fail


def build_study_array(study, row, image_root, cfg):
    """On-the-fly equivalent of one cached study, in the layout of `cfg`'s cache scheme:
    c01 -> ([6, S, P, P] uint8, mask[6]); c02 -> ([sum(budgets), P, P] uint8, mask[6]) with slot
    `si` at rows slot_offsets()[si]. Mirrors cache_study / build_study_flat in the builder."""
    scheme, px, slot_slices, band = cache_geom(cfg)
    starts, total = slot_offsets(slot_slices)
    if scheme == "c01":
        arr = np.zeros((len(SLOTS), slot_slices[0], px, px), np.uint8)
    else:
        arr = np.zeros((total, px, px), np.uint8)
    mask = np.zeros(len(SLOTS), np.float32)
    is_right = str(row.get("side", "")) == "R"
    for si, slot in enumerate(SLOTS):
        sid = row[slot]
        if not isinstance(sid, str) or not sid:
            continue
        d = os.path.join(image_root, study, sid)
        if not os.path.isdir(d):
            continue
        plane = PLANE_OF_SLOT[slot]
        a, _ = cache_series(d, plane, cfg, is_right, slot_slices[si], band=band[plane], px=px)
        if a is None:
            continue
        if scheme == "c01":
            arr[si] = a
        else:
            arr[starts[si]:starts[si] + slot_slices[si]] = a
        mask[si] = 1.0
    return arr, mask


def slot_stacks(arr, cfg):
    """The six per-slot (n_i, P, P) views of a cached study, for either layout: c01 arrays are
    [6, S, P, P] (view = arr[si]); c02 arrays are flat [sum, P, P] (view = a row range)."""
    if arr.ndim == 4:
        return [arr[si] for si in range(len(SLOTS))]
    _, _, slot_slices, _ = cache_geom(cfg)
    starts, _ = slot_offsets(slot_slices)
    return [arr[s:s + n] for s, n in zip(starts, slot_slices)]


_NPY_HEADERS = {}     # blob path -> (shape, dtype, header_bytes); per process (DataLoader worker)


def npy_header(path):
    """(shape, dtype, header_bytes) of a .npy file, public numpy API only."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError(f"unsupported .npy version {version} in {path}")
        shape, fortran, dtype = reader(f)
        if fortran:
            raise ValueError(f"{path} is Fortran-ordered; blobs must be C-ordered")
        return tuple(shape), dtype, f.tell()


def read_cached(locator):
    """One study's uint8 array from its locator: a .npy path (c01) or (blob_path, row) (c02).
    The blob read is a single seek + read of that study's bytes -- no np.load(mmap_mode) on
    Kaggle's FUSE input mount, no mapping held open inside DataLoader workers, and the 8 MB
    buffer is freed with the item (the design review's memory concern, 2026-08-30)."""
    if isinstance(locator, str):
        return np.load(locator)
    path, row = locator
    hdr = _NPY_HEADERS.get(path)
    if hdr is None:
        hdr = _NPY_HEADERS[path] = npy_header(path)
    shape, dtype, header_bytes = hdr
    if not (0 <= row < shape[0]):
        raise IndexError(f"row {row} outside blob {path} with {shape[0]} studies")
    per_study = int(np.prod(shape[1:]))
    itemsize = np.dtype(dtype).itemsize
    with open(path, "rb") as f:
        f.seek(header_bytes + row * per_study * itemsize)
        buf = np.fromfile(f, dtype=dtype, count=per_study)
    if buf.size != per_study:
        raise IOError(f"short read on {path} row {row}: {buf.size} of {per_study} elements")
    return buf.reshape(shape[1:])


def valid_windows(mask, cfg):
    """Every (slot, centre) triplet window a study offers: centres 1 .. n_i-2 of each PRESENT
    slot. Returns (centres, slot_id) as int arrays; the centre indexes the slot's own stack."""
    _, _, slot_slices, _ = cache_geom(cfg)
    cs, ss = [], []
    for si, n in enumerate(slot_slices):
        if float(mask[si]) <= 0:
            continue
        c = np.arange(1, int(n) - 1)
        cs.append(c)
        ss.append(np.full(len(c), si, dtype=np.int64))
    if not cs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(cs), np.concatenate(ss)


def sample_train_windows(centres, slot_id, n, min_per_slot=2):
    """Training view: `n` windows without replacement, stratified so every present slot keeps at
    least `min_per_slot` (if it has that many), the rest uniform over what is left. Uses the
    global numpy RNG, which seed_worker re-seeds per worker and epoch."""
    W = len(centres)
    if n >= W:
        order = np.random.permutation(W)          # every window, shuffled
        return centres[order], slot_id[order]
    chosen = []
    for si in np.unique(slot_id):
        pool = np.flatnonzero(slot_id == si)
        k = min(min_per_slot, len(pool), max(0, n - len(chosen)))
        if k:
            chosen.extend(np.random.choice(pool, k, replace=False).tolist())
    rest = np.setdiff1d(np.arange(W), np.array(chosen, dtype=np.int64))
    need = n - len(chosen)
    if need > 0:
        chosen.extend(np.random.choice(rest, need, replace=False).tolist())
    ix = np.array(sorted(chosen), dtype=np.int64)
    return centres[ix], slot_id[ix]


def eval_windows_subset(centres, slot_id, n_eval):
    """Evaluation view: all windows when n_eval <= 0 or >= W; otherwise n_eval windows spread
    equidistantly over the (slot-ordered) list -- the same rule for oof_eval and infer."""
    W = len(centres)
    if n_eval <= 0 or n_eval >= W:
        return centres, slot_id
    ix = np.linspace(0, W - 1, n_eval).round().astype(np.int64)
    return centres[ix], slot_id[ix]


def array_to_tensor(arr, mask, cfg, train, centre_offset=0):
    """[6, S, P, P] uint8 -> (6, K, 3, img, img) float normalised for the encoder.
    Triplet channels are neighbouring cached slices [c-1, c, c+1]; the K centres are
    equidistant over the interior of the stack (eval) -- the same for train in v03 so the
    cache experiment isolates the cache, not a new augmentation. `centre_offset` shifts every
    centre by that many cached slices (clipped) -- the slice-offset TTA views (P-12); 0 is
    bit-identical to the pre-TTA code. A flat c02 array is handled slot by slot (ragged S)."""
    if arr.ndim == 3:                                   # c02 flat layout: per-slot stacks
        K = cfg.slices_per_slot
        views = []
        for st in slot_stacks(arr, cfg):
            S = st.shape[0]
            centres = np.linspace(1, S - 2, K).round().astype(int)
            if train and getattr(cfg, "cache_jitter", False):
                centres = centres + np.random.randint(-1, 2, size=K)
            centres = np.clip(centres + centre_offset, 1, S - 2)
            idx = np.stack([centres - 1, centres, centres + 1], axis=1)
            views.append(torch.from_numpy(st[idx].astype(np.float32) / 255.0))   # (K, 3, P, P)
        x = torch.stack(views)                                                    # (6, K, 3, P, P)
        if x.shape[-1] != cfg.img_size:
            x = F.interpolate(x.reshape(-1, 3, x.shape[-2], x.shape[-1]),
                              size=(cfg.img_size, cfg.img_size), mode="bilinear",
                              align_corners=False).reshape(len(SLOTS), K, 3, cfg.img_size, cfg.img_size)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        m = torch.as_tensor(np.asarray(mask, dtype=np.float32))
        return x * m.view(-1, 1, 1, 1, 1), m
    S = arr.shape[1]
    if getattr(cfg, "stack_mode", "triplet") == "channels":
        idx = np.arange(S)
        if train and getattr(cfg, "cache_jitter", False):
            idx = np.clip(idx + np.random.randint(-1, 2), 0, S - 1)   # shift the stack +-1 slice
        x = torch.from_numpy(arr[:, idx].astype(np.float32) / 255.0).unsqueeze(1)   # (6, 1, S, P, P)
        if x.shape[-1] != cfg.img_size:
            x = F.interpolate(x.reshape(-1, S, x.shape[-2], x.shape[-1]),
                              size=(cfg.img_size, cfg.img_size), mode="bilinear",
                              align_corners=False).reshape(len(SLOTS), 1, S, cfg.img_size, cfg.img_size)
        x = (x - GRAY_MEAN) / GRAY_STD
        m = torch.as_tensor(np.asarray(mask, dtype=np.float32))
        return x * m.view(-1, 1, 1, 1, 1), m
    K = cfg.slices_per_slot
    centres = np.linspace(1, S - 2, K).round().astype(int)
    if train and getattr(cfg, "cache_jitter", False):
        centres = np.clip(centres + np.random.randint(-1, 2, size=K), 1, S - 2)
    if centre_offset:
        centres = np.clip(centres + centre_offset, 1, S - 2)
    idx = np.stack([centres - 1, centres, centres + 1], axis=1)          # (K, 3)
    x = torch.from_numpy(arr[:, idx].astype(np.float32) / 255.0)         # (6, K, 3, P, P)
    if x.shape[-1] != cfg.img_size:
        x = F.interpolate(x.reshape(-1, 3, x.shape[-2], x.shape[-1]),
                          size=(cfg.img_size, cfg.img_size), mode="bilinear",
                          align_corners=False).reshape(len(SLOTS), K, 3, cfg.img_size, cfg.img_size)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    m = torch.as_tensor(np.asarray(mask, dtype=np.float32))
    x = x * m.view(-1, 1, 1, 1, 1)                # absent slots stay exactly zero
    return x, m


def undo_laterality(arr, cfg):
    """P-05 ablation: put a right knee back into its own chirality.

    The cache stores every study in a canonical left-knee frame -- coronal/axial mirrored
    left-right, sagittal stacks reversed. Both are involutions, so re-applying them to the
    R studies restores the two-chirality condition P-05 removed, with no cache rebuild.

    It does not reconstruct the original bytes: the per-series `col_to_left` sign that
    decided the mirror is not in the manifest. It reproduces the thing being ablated --
    chirality that varies with knee side -- which is what the arm is asking about. This is
    a cleaner test than v03-vs-v02, where the 130 mm crop varied at the same time.
    """
    out = arr.copy()
    for si, (slot, st) in enumerate(zip(SLOTS, slot_stacks(out, cfg))):
        if PLANE_OF_SLOT[slot] == "Sagittal":
            st[:] = st[::-1].copy()           # reverse the slice axis
        else:
            st[:] = st[:, :, ::-1].copy()     # mirror the width axis (coronal / axial)
    return np.ascontiguousarray(out)

# %% [markdown]
# ## Section 5: dataset
#
# One item = one study: a `(slot, slices, 3, H, W)` tensor plus a presence mask.
# Absent slots are zero-filled and masked, which is why the head receives the mask
# explicitly — "this study had no axial fluid series" is information, not noise.
#
# **Laterality normalisation:** right knees are mirrored so medial/lateral means the
# same thing in every image. Without it the model has to learn each finding twice,
# and `Medial OA` vs `Lateral OA` are separate labels — mirroring is not cosmetic.
# The DICOM tag is unreliable in this corpus, so this uses a light heuristic and
# leaves a hook for a better one.

# %%
# ── Section 5: dataset ────────────────────────────────────────────────────────
class KneeStudyDataset(Dataset):
    def __init__(self, manifest, targets_df, image_root, cfg, train=True,
                 studies=None):
        self.m = manifest.set_index("StudyInstanceUID")
        self.t = targets_df.set_index("StudyInstanceUID") if targets_df is not None else None
        self.root = image_root
        self.cfg = cfg
        self.train = train
        keep = studies if studies is not None else list(self.m.index)
        self.studies = [s for s in keep if s in self.m.index]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        study = self.studies[i]
        row = self.m.loc[study]
        if self.cfg.use_cache:
            locator = CACHE_INDEX.get(cache_version_for(self.cfg), {}).get(study)
            if locator is not None:
                arr = read_cached(locator)
                mk = str(row["mask"]) if "mask" in row and isinstance(row["mask"], str) else None
                if mk is None or len(mk) != len(SLOTS):
                    mk = "".join("1" if st.any() else "0" for st in slot_stacks(arr, self.cfg))
                mask_np = np.array([float(c) for c in mk], np.float32)
            else:                       # test study, or a study the cache missed
                arr, mask_np = build_study_array(study, row, self.root, self.cfg)
            if self.cfg.lat_undo and str(row.get("side", "")) == "R":
                arr = undo_laterality(arr, self.cfg)   # P-05 ablation arm; counted in train_fold
            if getattr(self.cfg, "window_mode", "fixed") == "random":
                # P-25: ship the uint8 study + window indices; the model gathers, normalises and
                # resizes on the GPU (60 float windows per study would otherwise cross the
                # DataLoader shared-memory boundary at ~80-100 MB each).
                centres, slot_id = valid_windows(mask_np, self.cfg)
                if self.train:
                    centres, slot_id = sample_train_windows(centres, slot_id, self.cfg.train_windows)
                else:
                    centres, slot_id = eval_windows_subset(centres, slot_id, self.cfg.eval_windows)
                out = {"study": study, "arr": torch.from_numpy(np.ascontiguousarray(arr)),
                       "centres": torch.from_numpy(centres.astype(np.int64)),
                       "slot_id": torch.from_numpy(slot_id.astype(np.int64)),
                       "mask": torch.as_tensor(mask_np)}
                if self.t is not None:
                    r = self.t.loc[study]
                    out["y"] = torch.tensor([float(r[l]) for l in LABELS])
                    out["w"] = torch.tensor([float(r[f"w__{l}"]) for l in LABELS])
                    out["is_gold"] = torch.tensor(float(r["is_gold"]))
                return out
            offsets = (0,) if self.train else tuple(getattr(self.cfg, "tta_offsets", (0,)))
            views = [array_to_tensor(arr, mask_np, self.cfg, self.train, centre_offset=o)
                     for o in offsets]
            imgs, mask = views[0]
            if len(views) > 1:
                imgs = torch.stack([v[0] for v in views])        # (n_views, 6, K, 3, H, W)
        else:
            imgs = torch.zeros(len(SLOTS), self.cfg.slices_per_slot, 3,
                               self.cfg.img_size, self.cfg.img_size)
            mask = torch.zeros(len(SLOTS))
            for si, slot in enumerate(SLOTS):
                sid = row[slot]
                if not isinstance(sid, str) or not sid:
                    continue
                d = os.path.join(self.root, study, sid)
                if not os.path.isdir(d):
                    continue
                imgs[si] = build_triplets(d, self.cfg.slices_per_slot,
                                          self.cfg.triplet_gap, self.cfg.img_size)
                mask[si] = 1.0

        if self.train:
            # Light augmentation. No vertical flip: knee anatomy is not
            # up/down symmetric, and no horizontal flip either because that
            # would swap medial and lateral -- which are different labels.
            if random.random() < 0.5:
                imgs = imgs + torch.randn_like(imgs) * 0.01

        out = {"study": study, "imgs": imgs, "mask": mask}
        if self.t is not None:
            r = self.t.loc[study]
            out["y"] = torch.tensor([float(r[l]) for l in LABELS])
            out["w"] = torch.tensor([float(r[f"w__{l}"]) for l in LABELS])
            out["is_gold"] = torch.tensor(float(r["is_gold"]))
        return out

# %% [markdown]
# ## Section 6: model
#
# ```
# study -> 6 slots -> N triplets each
#                       |
#             shared DINOv2 ViT-S/14  (one encoder for all slots: 4,407 studies
#                       |              cannot support six separate encoders)
#          attention pool over slices  (a torn ACL is visible on a few slices, so
#                       |               mean pooling dilutes it ~6x)
#            concat 6 slot vectors + 6-bit presence mask
#                       |
#                  linear -> 12 logits
# ```
#
# Two rates: the head gets `lr_head` (1e-3); the backbone gets `lr_backbone`
# (2e-5) at its top block, decaying by 0.75 per block downwards (layer-wise LR
# decay), and an EMA of the weights is what gets validated and saved. The
# pretrained self-supervised features are the asset here — with 58 gold labels
# there is nowhere near enough signal to relearn them, so they are nudged, not
# retrained. Every medical DINOv2 recipe we found sits at 1e-6..2e-5; a uniform
# 5e-5 (v01) is the "catastrophic forgetting" regime — see docs/research.md.

# %%
# ── Section 6: model ──────────────────────────────────────────────────────────
class AttnPool(nn.Module):
    """Attention pooling over the slice axis.

    Mean pooling weights every slice equally, so a finding visible on 1 of 6
    sampled slices is diluted. This learns which slices matter.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(),
                                   nn.Linear(dim // 4, 1))

    def forward(self, x):                    # x: (S, dim)
        a = torch.softmax(self.score(x).squeeze(-1), dim=0)
        return (a.unsqueeze(-1) * x).sum(0)


class SlotAttnHead(nn.Module):
    """P-09: 12 learned label queries attending over the slot vectors that are present.

    The concat head maps [6 x dim | mask] through one Linear, so every label reads all six
    slots through one shared weight matrix: "for MCL, weight coronal and ignore axial" has
    to be learned as 12 independent 2,310-dim rows from 3,525 studies of noisy targets.
    Here each label owns a query, a per-(label, slot) bias states that plane preference in
    72 parameters, and absent slots are masked out *before* the softmax so the context
    vector has the same scale whether a study has four slots or six (mean slots is 4.78 of
    6; COR_T1 fills 62.5%, SAG_T1 50%). 9,300 parameters against the concat head's 27,720.

    Risk on record (research.md): correlated label pairs may lose the shared-vector
    benefit -- report Effusion~Synovitis, Medial OA~Medial Meniscus and Contusion~Fracture
    separately, not just the macro.
    """

    def __init__(self, dim: int, n_labels=len(LABELS), n_slots=len(SLOTS)):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        # 2-D, so param_groups gives it weight decay. Decaying it toward zero is a
        # uniform-plane prior, which is the right default for a term with no data yet.
        self.slot_bias = nn.Parameter(torch.zeros(n_labels, n_slots))
        self.w = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_labels))
        self.scale = dim ** -0.5

    def forward(self, pooled, mask):             # pooled (B, NS, dim), mask (B, NS)
        att = torch.einsum("ld,bsd->bls", self.q, pooled) * self.scale
        att = att + self.slot_bias.unsqueeze(0)
        keep = (mask > 0.5).unsqueeze(1)                             # (B, 1, NS)
        att = att.masked_fill(~keep, torch.finfo(att.dtype).min)     # fp16-safe, not -inf
        # A study with no present slot cannot reach here (the manifest requires
        # n_slots > 0), but an all-masked row would softmax to NaN. Fall back to uniform.
        dead = (~keep).all(-1, keepdim=True).expand_as(att)
        att = torch.where(dead, torch.zeros_like(att), att)
        ctx = torch.einsum("bls,bsd->bld", torch.softmax(att, dim=-1), pooled)
        return (ctx * self.w.unsqueeze(0)).sum(-1) + self.b


def widen_patch_embedding(enc, in_chans):
    """3 -> `in_chans` input channels on a HF vision encoder (P-23 #3, stack_mode="channels").

    The pretrained RGB kernel is averaged over its three channels, replicated `in_chans` times and
    scaled by 3/in_chans, so a stack of identical slices produces exactly the response the grey
    image would have -- the model starts as "mean over the stack" and learns which slice offsets
    matter. Every `num_channels` bookkeeping attribute is updated because HF embeddings assert on
    it at forward time (Dinov2PatchEmbeddings, ConvNextEmbeddings)."""
    emb = enc.embeddings
    name, conv = next((n, m) for n, m in emb.named_modules() if isinstance(m, nn.Conv2d))
    new = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_chans, 1, 1)
                         * (3.0 / in_chans))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    parent, parts = emb, name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)
    for mod in (emb, getattr(emb, "patch_embeddings", None), enc.config):
        if mod is not None and hasattr(mod, "num_channels"):
            mod.num_channels = in_chans
    print(f"  patch embedding widened 3 -> {in_chans} channels (embeddings.{name})")


class WindowAttnHead(nn.Module):
    """P-25: 12 label queries over EVERY (slot, window) token of a study.

    The existing heads pool each slot's windows with a label-AGNOSTIC AttnPool first, so a
    Fracture slice and a meniscus slice in the same sagittal stack compete for one 384-d slot
    vector before any label reads it. Here each label runs its own softmax over all windows
    of the study (the 0.936 notebook's strongest member pools this way), with a learned slot
    embedding added to every token so "which sequence" survives the flattening. Gate =
    Linear(dim,256) -> Tanh -> Dropout -> Linear(256, 12); output = per-label context dot a
    per-label weight. Padded / absent windows are masked with finfo.min before the softmax
    (fp16-safe); an all-masked row falls back to uniform rather than NaN."""

    def __init__(self, dim, n_labels=len(LABELS), n_slots=len(SLOTS), slot_embed=True,
                 dropout=0.2, hidden=256):
        super().__init__()
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim)) if slot_embed else None
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_labels))
        self.w = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, feats, slot_id, valid=None):
        # feats (B, W, dim)   slot_id (B, W) long   valid (B, W) bool or None
        h = feats
        if self.slot_emb is not None:
            h = h + self.slot_emb[slot_id]
        h = self.norm(h)
        att = self.gate(h).transpose(1, 2)                       # (B, L, W)
        if valid is not None:
            keep = valid.unsqueeze(1)                            # (B, 1, W)
            att = att.masked_fill(~keep, torch.finfo(att.dtype).min)
            dead = (~keep).all(-1, keepdim=True).expand_as(att)
            att = torch.where(dead, torch.zeros_like(att), att)
        a = torch.softmax(att.float(), dim=-1).to(h.dtype)       # per-label softmax over windows
        ctx = torch.einsum("blw,bwd->bld", a, h)                 # (B, L, dim)
        return (ctx * self.w.unsqueeze(0)).sum(-1) + self.b


def load_timm_backbone(arch, backbone_dir, grad_checkpoint=False):
    """timm model built offline from <backbone_dir>/model.safetensors (the HF timm repo files,
    mounted as a Kaggle Dataset). Loads strictly except for the classifier head, and REFUSES a
    silent architecture mismatch -- `strict=False` alone would happily train from scratch."""
    import timm
    from safetensors.torch import load_file
    enc = timm.create_model(arch, pretrained=False, num_classes=0)
    sd = load_file(os.path.join(backbone_dir, "model.safetensors"))
    head_keys = [k for k in sd if k.startswith("head.fc")]        # ImageNet classifier
    for k in head_keys:
        sd.pop(k)
    res = enc.load_state_dict(sd, strict=False)
    bad_unexpected = [k for k in res.unexpected_keys if not k.startswith("head.")]
    if res.missing_keys or bad_unexpected:
        raise SystemExit(f"timm {arch}: weights do not match the architecture -- missing "
                         f"{res.missing_keys[:5]} ({len(res.missing_keys)}), unexpected "
                         f"{bad_unexpected[:5]} ({len(bad_unexpected)})")
    print(f"  timm {arch}: loaded {len(sd)} tensors from {backbone_dir} (dropped head "
          f"{len(head_keys)}); num_features {enc.num_features}, {len(enc.stages)} stages, "
          f"grad_checkpoint={grad_checkpoint}")
    if grad_checkpoint and hasattr(enc, "set_grad_checkpointing"):
        enc.set_grad_checkpointing(True)
    return enc


class KneeNet(nn.Module):
    def __init__(self, backbone_dir: str, n_labels=len(LABELS), dropout=0.1,
                 head_type="concat", slot_dropout=0.0, backbone="dinov2", in_chans=3,
                 slot_embed=True, grad_checkpoint=False, img_size=224):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.img_size = img_size
        if backbone == "convnext_tiny":
            from transformers import ConvNextModel
            self.enc = ConvNextModel.from_pretrained(backbone_dir)
            self.dim = self.enc.config.hidden_sizes[-1]          # 768 for Tiny
        elif str(backbone).startswith("timm:"):
            self.enc = load_timm_backbone(backbone.split(":", 1)[1], backbone_dir, grad_checkpoint)
            self.dim = self.enc.num_features
        else:
            from transformers import Dinov2Model
            self.enc = Dinov2Model.from_pretrained(backbone_dir)
            self.dim = self.enc.config.hidden_size
        if in_chans != 3:
            widen_patch_embedding(self.enc, in_chans)
        self.drop = nn.Dropout(dropout)
        self.head_type = head_type
        self.slot_dropout = slot_dropout
        if head_type == "window_attn":
            self.window_head = WindowAttnHead(self.dim, n_labels, slot_embed=slot_embed)
        else:
            self.pool = AttnPool(self.dim)
            if head_type == "attn":
                self.attn_head = SlotAttnHead(self.dim, n_labels)
            else:
                self.head = nn.Linear(self.dim * len(SLOTS) + len(SLOTS), n_labels)

    def encode(self, x):
        """(N, C, H, W) normalised images -> (N, dim) one vector per image."""
        if str(self.backbone).startswith("timm:"):
            return self.enc(x)                               # num_classes=0 -> pooled features
        out = self.enc(pixel_values=x)
        if self.backbone == "convnext_tiny":
            return out.pooler_output                         # LayerNorm(global-avg-pool), (N, 768)
        return out.last_hidden_state[:, 0]                   # CLS token, (N, 384)

    def forward(self, imgs, mask):
        # imgs: (B, SLOT, S, C, H, W)   mask: (B, SLOT)   C = 3 (triplet) or 16 (channels, S = 1)
        B, NS, S = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        flat = imgs.reshape(B * NS * S, *imgs.shape[3:])
        feats = self.encode(flat).reshape(B, NS, S, self.dim)
        if self.head_type == "window_attn":
            # fixed-window input through the window head: every (slot, centre) is a token,
            # tokens of absent slots are masked out
            slot_id = torch.arange(NS, device=feats.device).repeat_interleave(S).unsqueeze(0).expand(B, -1)
            valid = (mask > 0.5).repeat_interleave(S, dim=1)
            return self.window_head(self.drop(feats.reshape(B, NS * S, self.dim)), slot_id, valid)
        pooled = torch.stack([
            torch.stack([self.pool(feats[b, s]) for s in range(NS)])
            for b in range(B)
        ])                                                            # (B, NS, dim)
        pooled = pooled * mask.unsqueeze(-1)      # zero out absent slots
        if self.training and self.slot_dropout > 0:
            drop = (torch.rand_like(mask) > self.slot_dropout).float()
            # never drop a study's last remaining slot
            drop = torch.where((mask * drop).sum(1, keepdim=True) > 0,
                               drop, torch.ones_like(drop))
            mask = mask * drop
            pooled = pooled * mask.unsqueeze(-1)
        if self.head_type == "attn":
            return self.attn_head(self.drop(pooled), mask)
        x = torch.cat([pooled.reshape(B, -1), mask], dim=1)
        return self.head(self.drop(x))

    def forward_windows(self, arr, centres, slot_id, slot_starts):
        """P-25 window mode, one study per call. arr (T, P, P) uint8 on the device (c02 flat)
        or (6, S, P, P) (c01); centres / slot_id (W,) long index the slot's own stack. Gathers
        [c-1, c, c+1] triplets, scales, resizes to img_size and ImageNet-normalises ON THE GPU,
        then runs the encoder and the window head."""
        if arr.ndim == 4:                                   # c01 dense: flatten to (6*S, P, P)
            S = arr.shape[1]
            starts = torch.arange(arr.shape[0], device=arr.device) * S
            arr = arr.reshape(-1, *arr.shape[2:])
        else:
            starts = torch.as_tensor(slot_starts, device=arr.device, dtype=torch.long)
        base = starts[slot_id] + centres                    # (W,) row of each centre in `arr`
        idx = torch.stack([base - 1, base, base + 1], dim=1)  # (W, 3)
        x = arr[idx].float() / 255.0                        # (W, 3, P, P)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                              align_corners=False)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        if self.training and torch.rand(()) < 0.5:
            x = x + torch.randn_like(x) * 0.01              # the Dataset's noise aug, moved here
        feats = self.encode(x).unsqueeze(0)                 # (1, W, dim)
        if self.head_type != "window_attn":
            raise SystemExit("window_mode='random' needs head_type='window_attn'")
        return self.window_head(self.drop(feats), slot_id.unsqueeze(0))


def weighted_bce(logits, y, w):
    """Confidence-weighted soft-target BCE.

    No `pos_weight`: with soft targets it inflates every prediction and the metric
    reads only rank order, so there is nothing to gain and a collapse to overprediction
    to lose.
    """
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def build_model(c, device):
    """One factory for training and inference, from a Config or a checkpoint's saved config."""
    g = _cfg_get(c)
    backbone = g("backbone", "dinov2")
    sm = g("stack_mode", "triplet")
    in_ch = int(g("cache_n_slices", 16)) if sm == "channels" else 3
    m = KneeNet(resolve_backbone_dir(backbone), dropout=float(g("dropout", 0.1)),
                head_type=g("head_type", "concat"), slot_dropout=float(g("slot_dropout", 0.0)),
                backbone=backbone, in_chans=in_ch, slot_embed=bool(g("slot_embed", True)),
                grad_checkpoint=bool(g("grad_checkpoint", False)), img_size=int(g("img_size", 224)))
    return m.to(device)


def forward_batch(model, b, device, cfg):
    """Logits for one batch, whichever representation the Dataset produced: fixed windows
    (`imgs`, one view) or random/all windows (`arr` + indices). TTA views are NOT handled here
    (training only); predict_probs() does the multi-view pooling."""
    if "arr" in b:
        if b["arr"].shape[0] != 1:
            raise SystemExit("window_mode='random' runs one study per step (batch_studies=1)")
        _, _, slot_slices, _ = cache_geom(cfg)
        starts, _ = slot_offsets(slot_slices)
        return model.forward_windows(b["arr"][0].to(device), b["centres"][0].to(device),
                                     b["slot_id"][0].to(device), starts)
    imgs = b["imgs"]
    if imgs.ndim == 7:                                   # (B, n_views, 6, K, 3, H, W): view 0 only
        imgs = imgs[:, 0]
    return model(imgs.to(device), b["mask"].to(device))


FOCAL_MAX = {"Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus", "Baker's"}
FOCAL_TOP2 = {"ACL", "MCL"}


def pool_views(probs, how):
    """(n_views, B, L) probabilities -> (B, L). "mean" averages; "focal" is the 0.936 notebook's
    per-label rule (max for focal findings, top-2 mean for the cruciate/collateral, mean else)."""
    if probs.shape[0] == 1 or how == "mean":
        return probs.mean(0)
    out = probs.mean(0).clone()
    for i, lab in enumerate(LABELS):
        if lab in FOCAL_MAX:
            out[:, i] = probs[:, :, i].max(0).values
        elif lab in FOCAL_TOP2:
            k = min(2, probs.shape[0])
            out[:, i] = probs[:, :, i].topk(k, dim=0).values.mean(0)
    return out


@torch.no_grad()
def predict_probs(model, b, device, cfg):
    """Per-study probabilities with the member's TTA applied: for fixed-window members the
    Dataset stacks one view per `tta_offsets` entry along a leading axis; each view is a forward
    pass and the views are pooled per label with `tta_pool`. (0,) + "mean" == a single forward."""
    if "arr" in b or b["imgs"].ndim != 7:
        return torch.sigmoid(forward_batch(model, b, device, cfg)).float()
    views = []
    for v in range(b["imgs"].shape[1]):
        logits = model(b["imgs"][:, v].to(device), b["mask"].to(device))
        views.append(torch.sigmoid(logits).float())
    return pool_views(torch.stack(views), getattr(cfg, "tta_pool", "mean"))

# %% [markdown]
# ## Section 7: training
#
# Built around one operational fact: **five folds do not fit in one 9-hour Kaggle
# session.** So every fold writes a resumable `*_last.pt` after each epoch, the
# runtime guard stops cleanly before the ceiling, and re-running with the previous
# output attached picks up where it left off. A run that cannot resume wastes a
# whole session.
#
# Also here: AMP, gradient accumulation (batch of 1 study is already ~36 ViT
# forwards), cosine schedule with warmup, gradient clipping, and a
# **prediction-spread diagnostic**. That last one exists because the known failure
# mode of this setup is collapse to the base rate — every study gets the same score,
# AUC 0.5, and the loss looks fine. Near-zero spread is an alarm, never a target.

# %%
# ── Section 7: training ───────────────────────────────────────────────────────
def seed_worker(worker_id):
    """Re-seed numpy and `random` inside each DataLoader worker.

    PyTorch seeds only torch's RNG per worker; numpy and `random` are inherited from the
    parent by fork. Workers are recreated every epoch from the same parent state, so
    without this the "random" slice jitter (P-08) and the Gaussian noise are byte-identical
    in every epoch -- augmentation that never augments. `torch.initial_seed()` inside a
    worker is base_seed + worker_id, and base_seed advances each epoch.
    """
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def check_worker_rng():
    """Direct test of traps 6e on THIS platform, in seconds.

    Linux forks DataLoader workers from a parent whose numpy/`random` state has not moved
    between epochs, so without a `worker_init_fn` every epoch draws the same "random"
    numbers and slice jitter never jitters. Windows spawns instead, so this cannot be
    reproduced locally -- which is exactly why the check runs on Kaggle and prints both
    arms. Expect: without = True (identical, the bug), with = False (varying, fixed).
    """
    class _Probe(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return torch.tensor([np.random.randint(0, 10 ** 6), random.randint(0, 10 ** 6)])

    print("  worker RNG check (traps 6e):")
    for label, init in (("without worker_init_fn", None), ("with seed_worker", seed_worker)):
        try:
            dl = DataLoader(_Probe(), batch_size=4, num_workers=2, worker_init_fn=init)
            eps = [torch.cat([b for b in dl]).flatten().tolist() for _ in range(3)]
            same = eps[0] == eps[1] == eps[2]
            print(f"    {label:<24} identical across 3 epochs = {same}"
                  f"   {'<-- augmentation would never vary' if same else ''}")
        except Exception as e:
            print(f"    {label:<24} check failed: {type(e).__name__}: {e}")


def make_loaders(manifest, targets, image_root, cfg, fold):
    tr_studies = targets.loc[targets.fold != fold, "StudyInstanceUID"].tolist()
    va_studies = targets.loc[targets.fold == fold, "StudyInstanceUID"].tolist()
    if cfg.smoke:
        avail = set(manifest.StudyInstanceUID)
        tr_studies = [s for s in tr_studies if s in avail][:4]
        va_studies = [s for s in va_studies if s in avail][:4]
        if not tr_studies:      # local sample has no training studies at all
            tr_studies = va_studies = sorted(avail)[:3]
    tr_ds = KneeStudyDataset(manifest, targets, image_root, cfg, True, tr_studies)
    va_ds = KneeStudyDataset(manifest, targets, image_root, cfg, False, va_studies)
    print(f"  fold {fold}: train {len(tr_ds)} / val {len(va_ds)} studies")
    nw = 0 if cfg.smoke else cfg.num_workers
    return (DataLoader(tr_ds, batch_size=cfg.batch_studies, shuffle=True,
                       num_workers=nw, drop_last=False, worker_init_fn=seed_worker),
            DataLoader(va_ds, batch_size=cfg.batch_studies, shuffle=False,
                       num_workers=nw))


def bootstrap_macro_ci(Y_hard, P, n_boot=2000, seed=0):
    """Percentile-bootstrap 95% CI of the macro-AUC over studies. With ~12 gold
    studies per fold this interval is enormous -- which is the point of printing it."""
    rng = np.random.default_rng(seed)
    n = len(P)
    if n < 4:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        a = [auc_score(Y_hard[ix, i], P[ix, i]) for i in range(len(LABELS))]
        a = [v for v in a if np.isfinite(v)]
        if a:
            vals.append(float(np.mean(a)))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(model, loader, device, cfg):
    """Validation pass. Returns (metrics, table) where `table` is a DataFrame with the
    per-study predictions, targets, weights and gold flag -- the OOF rows. Per-label
    numbers are kept because the metric charges every label the same, so the label
    stuck at 0.5 is the thing we most need to see. TTA (tta_offsets / tta_pool, eval_windows)
    is whatever `cfg` says -- oof_eval and infer must run the same setting."""
    model.eval()
    P, Y, W, G, S = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            P.append(predict_probs(model, b, device, cfg).cpu().numpy())
            Y.append(b["y"].numpy())
            W.append(b["w"].numpy())
            G.append(b["is_gold"].numpy())
            S.extend(b["study"])
    if not P:
        return {}, None
    P, Y, W, G = (np.concatenate(x) for x in (P, Y, W, G))
    hard = (Y > 0.5).astype(int)
    gm = G > 0.5

    per_label = {}
    for i, lab in enumerate(LABELS):
        row = {"auc_soft": auc_score(hard[:, i], P[:, i]),
               "pred_std": float(P[:, i].std())}
        if gm.sum() >= 4:
            row["auc_gold"] = auc_score(hard[gm, i], P[gm, i])
        per_label[lab] = row

    def macro(key):
        vals = [r[key] for r in per_label.values() if np.isfinite(r.get(key, np.nan))]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    out = {"pred_std": round(float(P.std(0).mean()), 4),
           "auc_soft": macro("auc_soft"),
           "n_labels_scored": int(sum(np.isfinite(r["auc_soft"]) for r in per_label.values()))}
    if gm.sum() >= 4:
        out["auc_gold"] = macro("auc_gold")
        out["n_gold"] = int(gm.sum())
        lo, hi = bootstrap_macro_ci(hard[gm], P[gm])
        out["auc_gold_ci95"] = (round(lo, 3), round(hi, 3))
    out["per_label"] = per_label

    table = pd.DataFrame({"StudyInstanceUID": S, "is_gold": G.astype(int)})
    for i, lab in enumerate(LABELS):
        table[f"pred__{lab}"] = P[:, i]
        table[f"y__{lab}"] = Y[:, i]
        table[f"w__{lab}"] = W[:, i]
    return out, table


def print_per_label(per_label):
    print(f"    {'label':<18} {'auc_soft':>8} {'auc_gold':>8} {'pred_std':>8}")
    for lab, r in per_label.items():
        g = r.get("auc_gold", float("nan"))
        print(f"    {lab:<18} {r['auc_soft']:8.3f} {g:8.3f} {r['pred_std']:8.3f}"
              + ("   <-- near chance" if np.isfinite(r["auc_soft"]) and r["auc_soft"] < 0.55 else "")
              + ("   <-- collapsed" if r["pred_std"] < 0.01 else ""))


def param_groups(model, cfg):
    """Layer-wise LR decay for the DINOv2 encoder + no weight decay on 1-D params.

    HF Dinov2Model parameter names look like `embeddings.*`, `encoder.layer.<i>.*`,
    `layernorm.*`. The top block and the final LayerNorm get `lr_backbone`; each block
    below gets one more factor of `llrd_decay`; embeddings one more still. The head
    and the attention pool are freshly initialised, so they get `lr_head` undecayed.
    """
    # DINOv2: `encoder.layer.<i>` x 12 blocks. ConvNeXt (HF): `encoder.stages.<s>` x 4 stages
    # (depths 3/3/9/3) -- decay per stage, since a stage is the CNN's unit of feature level.
    # timm hybrids (coatnet_rmlp_*): `stem.*`, `stages.<s>.*` x 4, `norm.*` -- same per-stage rule.
    is_cnn = getattr(model, "backbone", "dinov2") == "convnext_tiny"
    is_timm = str(getattr(model, "backbone", "dinov2")).startswith("timm:")
    if is_timm:
        n_blocks = len(model.enc.stages)
    else:
        n_blocks = (len(model.enc.config.hidden_sizes) if is_cnn
                    else model.enc.config.num_hidden_layers)
    groups = {}

    def add(name, p, lr):
        no_decay = (p.ndim == 1 or name.endswith(".bias") or "token" in name
                    or "position_embeddings" in name)       # BEiT/MAE convention
        key = (round(lr, 12), no_decay)
        groups.setdefault(key, {"params": [], "lr": lr,
                                "weight_decay": 0.0 if no_decay else cfg.weight_decay})
        groups[key]["params"].append(p)

    for name, p in model.enc.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(model, "in_chans", 3) != 3 and "patch_embeddings" in name:
            add(name, p, cfg.lr_stem)     # widened conv = new capacity; under LLRD it would never move
            continue
        if name.startswith("embeddings.") or name.startswith("stem."):
            depth = 0
        elif name.startswith("encoder.layer.") or name.startswith("encoder.stages."):
            depth = int(name.split(".")[2]) + 1
        elif name.startswith("stages."):                 # timm: stages.<s>.blocks.<j>...
            depth = int(name.split(".")[1]) + 1
        else:                       # final layernorm
            depth = n_blocks + 1
        lr = cfg.lr_backbone * (cfg.llrd_decay ** (n_blocks + 1 - depth))
        add(name, p, lr)
    # Everything that is not the encoder is freshly initialised and gets lr_head undecayed.
    # Enumerated by name rather than hard-coded, so P-09's `attn_head` cannot silently end
    # up with no optimizer group when head_type="attn".
    n_head = 0
    for mname, mod in model.named_children():
        if mname == "enc":
            continue
        for name, p in mod.named_parameters():
            add(f"{mname}.{name}", p, cfg.lr_head)
            n_head += p.numel()
    out = list(groups.values())
    lrs = sorted({g["lr"] for g in out if g["lr"] < cfg.lr_head})
    print(f"  backbone LR range {lrs[0]:.2e} .. {lrs[-1]:.2e} over {n_blocks} blocks "
          f"(decay {cfg.llrd_decay}); head {cfg.lr_head:.0e} over {n_head:,} params "
          f"(head_type={getattr(model, 'head_type', 'concat')})")
    return out


class EMA:
    """Exponential moving average of the weights. Validated and saved instead of the
    raw weights: it is markedly more robust to label noise and makes a fixed epoch
    count a safe selection rule. Buffers are copied, not averaged."""

    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, e in self.module.state_dict().items():
            m = msd[k]
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def train_fold(fold, manifest, targets, image_root, cfg, device):
    ckpt_best = os.path.join(WORK, f"{cfg.version}_fold{fold}_best.pt")
    ckpt_last = os.path.join(WORK, f"{cfg.version}_fold{fold}_last.pt")
    oof_path = os.path.join(WORK, f"{cfg.version}_fold{fold}_oof.csv")

    model = build_model(cfg, device)
    opt = torch.optim.AdamW(param_groups(model, cfg))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    tr_loader, va_loader = make_loaders(manifest, targets, image_root, cfg, fold)
    steps_per_epoch = max(1, len(tr_loader) // cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_frac))

    def lr_at(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, best_epoch = 0, -1.0, -1
    # A smoke run never resumes: a stale `_last.pt` from an earlier local smoke made a
    # 1-epoch smoke "resume at epoch 1 of 1", skip training entirely and still finish
    # green -- the checkpoint code it was meant to exercise never ran (traps 19).
    if os.path.exists(ckpt_last) and not cfg.smoke:
        st = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None:
            # a checkpoint without an EMA (or with EMA switched on later) must not
            # leave the EMA copy at its random-head initialisation
            ema.module.load_state_dict(st.get("ema", st["model"]))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch = st["epoch"] + 1
        best = st.get("best", -1.0)
        best_epoch = st.get("best_epoch", st["epoch"])
        print(f"  resumed fold {fold} at epoch {start_epoch} (best {best:.4f} at epoch {best_epoch})")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        t_epoch = time.time()
        n_studies = 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr_loader):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = forward_batch(model, b, device, cfg)
                loss = weighted_bce(logits, b["y"].to(device), b["w"].to(device))
            scaler.scale(loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)
            running += float(loss.detach())
            nb += 1
            n_studies += int(b["mask"].shape[0])
            # Throughput is the open risk of this pipeline; print it early and often.
            if n_studies in (10, 50) or (n_studies % 500 == 0):
                dt = time.time() - t_epoch
                geom_note = (f"windows/study {cfg.train_windows}" if cfg.window_mode == "random"
                             else f"slices/slot {cfg.slices_per_slot}")
                print(f"    {n_studies} studies in {dt:.0f}s = {dt/n_studies:.2f} s/study "
                      f"({geom_note}, img {cfg.img_size}, workers "
                      f"{tr_loader.num_workers}) -> epoch ETA "
                      f"{dt/n_studies*len(tr_loader.dataset)/60:.0f} min")
            if out_of_time():
                print("  runtime guard hit mid-epoch")
                break
        train_secs = time.time() - t_epoch

        eval_model = ema.module if ema is not None else model
        t_eval = time.time()
        metrics, oof = evaluate(eval_model, va_loader, device, cfg)
        per_label = metrics.pop("per_label", {})
        print(f"  fold {fold} epoch {epoch}: loss {running/max(nb,1):.4f}  {metrics}")
        print(f"    train {train_secs/60:.1f} min ({train_secs/max(n_studies,1):.2f} s/study), "
              f"val {(time.time()-t_eval)/60:.1f} min")
        if per_label:
            print_per_label(per_label)
        if metrics.get("pred_std", 1.0) < 0.01:
            print("  !! prediction spread near zero -- base-rate collapse, not a "
                  "converged model")

        # Which epoch is "the" model? Selecting on the ~11 gold studies per fold is a coin
        # flip (Hanley-McNeil SE ~0.09) and stays banned. Through v05 `_best.pt` was simply
        # the EMA weights after the LAST completed epoch (fixed-epoch, P-03/P-04). P-22
        # (src/oof_epoch_analysis.py, 2026-08-29) then measured selection on OOF-vs-teacher
        # over the 882 held-out studies: +0.013 split-half for the concat head, which peaks
        # mid-schedule and decays, ~0 for the attention head, gold flat at the chosen epoch --
        # so `ckpt_policy="best_oof"` keeps the epoch with the highest auc_soft so far.
        # The score is never gold. A NaN score cannot drop a fold: the first epoch is always
        # written, and an undefined AUC falls back to the loss.
        score = metrics.get("auc_soft")
        if score is None or not np.isfinite(score):
            score = -running / max(nb, 1)
        take = (cfg.ckpt_policy == "last" or score > best
                or not os.path.exists(ckpt_best))
        if take:
            best, best_epoch = score, epoch
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch,
                    **({"ema": ema.module.state_dict()} if ema is not None else {})},
                   ckpt_last)
        if oof is not None:
            oof.insert(1, "epoch", epoch)
            oof.to_csv(oof_path.replace("_oof.csv", f"_ep{epoch}_oof.csv"), index=False)
        if take:
            torch.save({"model": eval_model.state_dict(), "score": score, "epoch": epoch,
                        "ema": ema is not None, "config": asdict(cfg)}, ckpt_best)
            if oof is not None:
                oof.to_csv(oof_path, index=False)        # always the checkpointed epoch
        print(f"    epoch {epoch} EMA score {score:.4f} -> "
              + (f"checkpoint = epoch {epoch} ({os.path.basename(ckpt_best)} + "
                 f"{os.path.basename(oof_path)})" if take else
                 f"not taken; best.pt stays epoch {best_epoch} ({best:.4f})")
              + f" [ckpt_policy={cfg.ckpt_policy}]")

        if out_of_time():
            print("  stopping: runtime guard. Attach this output and re-run to resume.")
            return model, best, False

    return model, best, True

# %% [markdown]
# ## Section 8: run
#
# On Kaggle this trains the configured folds; locally (`smoke=True`) it runs one
# fold over the 3 sample studies purely to prove the loop executes.

# %%
# ── Section 8: run training ───────────────────────────────────────────────────
if os.environ.get("RSNA_DEFS_ONLY"):
    raise SystemExit(0)          # src/cache_selftest.py imports Sections 1-7 and stops here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def resolve_image_root(series_csv: str, default_root: str) -> str:
    """Find the directory that actually holds `<study>/<series>/` for this CSV.

    Submission #1 (kernel v2, smoke) scored exactly 0.500 on the hidden test, which
    is what a constant submission scores -- i.e. on the rerun no test study was
    found under the assumed root and the 0.5 fallback fired, silently. Probing the
    tree beats assuming it, and failing loudly beats a silent 0.5 (see below).
    """
    meta = pd.read_csv(series_csv)
    if len(meta) == 0:
        return default_root
    first = meta.iloc[0]
    if os.path.isdir(os.path.join(default_root, first.StudyInstanceUID,
                                  first.SeriesInstanceUID)):
        return default_root
    # Shallow probe: <COMP>/<x>/<study>/<series> and one level deeper. Never `**` --
    # that walks the whole ~819k-file mount.
    hits = shallow_glob(COMP, first.SeriesInstanceUID, max_depth=3, skip=("train_series",))
    if not hits and ON_KAGGLE:
        hits = shallow_glob("/kaggle/input", first.SeriesInstanceUID, max_depth=4,
                            skip=("train_series",))
    if hits:
        root = os.path.dirname(os.path.dirname(hits[0]))
        print(f"  ! image root for {os.path.basename(series_csv)} is not {default_root}"
              f" -- found {root}")
        return root
    print(f"  ! could not locate any series of {os.path.basename(series_csv)} "
          f"under {default_root} or by glob")
    return default_root


TRAIN_IMG = os.path.join(COMP, "train_series")
TEST_IMG = os.path.join(COMP, "test_series")
if not os.path.isdir(TRAIN_IMG) and os.path.isdir(os.path.join(COMP, "sample_dicom",
                                                               "test_series")):
    # Local: only the public test tree exists, so use it for both.
    TRAIN_IMG = TEST_IMG = os.path.join(COMP, "sample_dicom", "test_series")
else:
    TEST_IMG = resolve_image_root(os.path.join(COMP, "test_series.csv"), TEST_IMG)
print(f"train images: {TRAIN_IMG}\ntest images:  {TEST_IMG}")

# ---- which mode are we in? ----------------------------------------------------
def find_mounted_checkpoints(version, kind="best"):
    """`{version}_fold<k>_{kind}.pt` files attached as a kernel/dataset input (Kaggle) or
    left in artifacts/kaggle_out (local). Shallow search only. Returns {fold: path}."""
    import re
    # Locally, WORK (this machine's own smoke checkpoints) is searched only when MODE asks for
    # inference explicitly -- in "auto" it would flip every local smoke run into infer mode.
    roots = (["/kaggle/input"] if ON_KAGGLE else
             ["artifacts/kaggle_out"] + ([WORK] if MODE in ("infer", "oof_eval") else []))
    found = {}
    for root in roots:
        # depth 4 like load_cache_manifests: a new slug mounts kernel outputs type-prefixed
        # (/kaggle/input/<type>/<owner>/<name>/...), an old one at /kaggle/input/<name>/ (traps 6f)
        for p in shallow_glob(root, f"{version}_fold*_{kind}.pt", max_depth=4):
            m = re.search(rf"{re.escape(version)}_fold(\d+)_{kind}\.pt$", p)
            if m:
                found.setdefault(int(m.group(1)), p)
    return found


mounted_ckpts = find_mounted_checkpoints(cfg.version, "best")
mounted_last = find_mounted_checkpoints(cfg.version, "last")
if MODE != "auto":
    mode = MODE
else:
    # infer only when EVERY configured fold has a finished checkpoint; a partial run
    # (guard fired) must resume training, not be submitted.
    mode = "infer" if mounted_ckpts and set(cfg.folds) <= set(mounted_ckpts) else "train"
print(f"MODE={mode}  mounted best: {sorted(mounted_ckpts)}  mounted last: {sorted(mounted_last)}")

# What a member's checkpoint decides, split in two (2026-08-30). CACHE keys describe the decoded
# test array -- members that agree on all of them share ONE decode-once pass (a "geometry group");
# c01 members (v05a/v05b/v05g/v06c) and c02 members (v08w, the hybrids) are two groups in one
# blend. MEMBER keys only change how a member READS the array and are applied per member around
# predict() -- the way stack_mode already was (P-21 heads, P-23 stack, P-25 windows, P-12 TTA).
INFER_CACHE_KEYS = ("use_cache", "cache_scheme", "cache_px", "cache_n_slices", "cache_px_wide",
                    "cache_slot_slices", "cache_band", "crop_mm", "lat_dead_zone_mm")
INFER_MEMBER_KEYS = ("slices_per_slot", "triplet_gap", "img_size", "stack_mode", "lat_undo",
                     "window_mode", "eval_windows", "tta_offsets", "tta_pool", "head_type",
                     "backbone", "slot_embed", "dropout", "slot_dropout")


def _norm_val(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def member_settings(saved, version=None):
    """Every CACHE + MEMBER key for one checkpoint: the saved config where present, else the
    dataclass default (old checkpoints predate the new fields and mean the c01-era value).
    INFER_OVERRIDES[version] then applies on top -- MEMBER keys only, TTA/eval_windows for
    members whose checkpoints predate them; it can never change what array is decoded."""
    out = {}
    for k in INFER_CACHE_KEYS + INFER_MEMBER_KEYS:
        if k in saved:
            out[k] = _norm_val(saved[k])
        else:
            out[k] = _norm_val(Config.__dataclass_fields__[k].default)
    for k, v in (INFER_OVERRIDES.get(version, {}) if version else {}).items():
        if k not in INFER_MEMBER_KEYS:
            raise SystemExit(f"INFER_OVERRIDES[{version}][{k}]: only member keys may be "
                             f"overridden at inference ({INFER_MEMBER_KEYS})")
        out[k] = _norm_val(v)
    return out


def cache_signature(settings):
    return tuple((k, settings[k]) for k in INFER_CACHE_KEYS)


def apply_settings(target_cfg, settings, keys):
    """setattr the chosen keys onto a Config (the module global, at inference); returns the
    previous values so they can be restored."""
    prev = {k: getattr(target_cfg, k) for k in keys}
    for k in keys:
        setattr(target_cfg, k, settings[k])
    return prev


infer_members = []          # [(version, fold, path)] -- the blend, in infer / oof_eval mode
infer_settings = {}         # (version, fold) -> resolved CACHE + MEMBER settings
if mode in ("infer", "oof_eval"):
    # P-21: the submission is a rank-mean over every mounted fold checkpoint of every version in
    # INFER_MEMBERS. Each version must be present -- a blend that silently lost a member is not
    # the model that was validated (the traps 6d failure class again). oof_eval scores fold 0
    # of each version on its held-out studies instead of predicting the test set.
    for v in (list(INFER_MEMBERS) or [cfg.version]):
        found = find_mounted_checkpoints(v, "best")
        if mode == "oof_eval":
            found = {f: p for f, p in found.items() if f in ARM_FOLDS}
        if not found:
            raise SystemExit(f"MODE={mode} but no {v}_fold*_best.pt is mounted (INFER_MEMBERS="
                             f"{INFER_MEMBERS}). Attach the training run's output as a kernel "
                             f"input (kernel_sources), or drop {v} from INFER_MEMBERS on purpose.")
        infer_members += [(v, f, found[f]) for f in sorted(found)]
    print(f"  {mode} members ({len(infer_members)}): "
          + ", ".join(f"{v}/fold{f}" for v, f, _ in infer_members))
    # The checkpoints decide the input geometry, not FORCE_SMOKE: a smoke-mode infer would
    # otherwise feed 2 slices/slot to a model trained on 6 and pass every assert.
    for v, f, p in infer_members:
        st0 = torch.load(p, map_location="cpu", weights_only=False)
        s = member_settings(st0.get("config", {}), v)
        infer_settings[(v, f)] = s
        # Fail here, in seconds, if a member's backbone weights are not mounted -- not after
        # seven other members have already predicted (infer v9, 2026-08-30: the ConvNeXt
        # dataset was missing from the infer kernel's sources).
        resolve_backbone_dir(s["backbone"])
        del st0
    groups = {}
    for (v, f), s in infer_settings.items():
        groups.setdefault(cache_signature(s), []).append(f"{v}/fold{f}")
    print(f"  {len(groups)} geometry group(s) (one decode-once pass each):")
    for sig, members in groups.items():
        d = dict(sig)
        print(f"    {cache_version_for(d)} x{len(members)}: {', '.join(members)}")
    for (v, f), s in infer_settings.items():
        print(f"    {v}/fold{f}: {s['backbone']}, {s['head_type']}, {s['window_mode']}"
              + (f", eval_windows {s['eval_windows']}" if s['window_mode'] == 'random' else
                 f", K {s['slices_per_slot']}, tta {s['tta_offsets']}/{s['tta_pool']}")
              + f", img {s['img_size']}")
    cfg.folds = tuple(sorted({f for _, f, _ in infer_members}))
else:
    # Resume: a previous session's output is mounted read-only; copy its checkpoints
    # into WORK so train_fold finds them (otherwise every fold restarts at epoch 0).
    import shutil
    for fold in cfg.folds:
        for kind, src_map in (("last", mounted_last), ("best", mounted_ckpts)):
            src = src_map.get(fold)
            dst = os.path.join(WORK, f"{cfg.version}_fold{fold}_{kind}.pt")
            if src and not os.path.exists(dst):
                shutil.copy(src, dst)
                print(f"  resume: copied {os.path.basename(src)} into WORK")

# ---- the caches (P-01 c01 / 2026-08-30 c02): shards written by src/cache_pipeline.py -----
def load_cache_manifests():
    """{cache_version: manifest DataFrame with a `locator` column}. EVERY mounted shard of every
    scheme is indexed; which cache an arm or a member reads is decided by cache_version_for(its
    config), so a c01 and a c02 cache can be mounted side by side."""
    roots = ["/kaggle/input"] if ON_KAGGLE else ["artifacts/cache_local"]
    frames = {}
    for root in roots:
        # depth 4, not 2: a NEWLY created kernel mounts kernel outputs type-prefixed
        # (/kaggle/input/<type>/<owner>/<name>/...) while older kernels mount them at
        # /kaggle/input/<name>/. max_depth=2 found the cache in rsna-knee-train and
        # silently missed it in rsna-knee-folds -- nine hours of the wrong recipe.
        for mpath in shallow_glob(root, "manifest_shard*.csv", max_depth=4):
            m = pd.read_csv(mpath, dtype={"mask": str})
            if "cache_version" not in m.columns or len(m) == 0:
                print(f"  ! {mpath}: no cache_version column or empty, ignored")
                continue
            version = str(m.cache_version.iloc[0])
            m = m[m.get("cached", 1) == 1].copy()
            arr_dir = os.path.join(os.path.dirname(mpath), version)
            if "blob" in m.columns:                     # c02: (blob path, row inside the blob)
                m["locator"] = [(os.path.join(arr_dir, str(b)), int(r)) for b, r in zip(m.blob, m.row)]
                m = m[[os.path.exists(loc[0]) for loc in m.locator]]
            else:                                       # c01: one .npy per study
                m["locator"] = [os.path.join(arr_dir, f"{u}.npy") for u in m.StudyInstanceUID]
                m = m[[os.path.exists(x) for x in m.locator]]
            m["mask"] = m["mask"].map(lambda v: str(v).zfill(len(SLOTS)) if isinstance(v, str) or v == v else "")
            frames.setdefault(version, []).append(m)
            print(f"  cache shard {mpath}: {len(m)} studies ({version})")
    return {v: pd.concat(fs, ignore_index=True) for v, fs in frames.items()}


cache_manifests = load_cache_manifests() if cfg.use_cache else {}
for _v, _m in cache_manifests.items():
    CACHE_INDEX[_v] = dict(zip(_m.StudyInstanceUID, _m.locator))
    print(f"  cache: {len(CACHE_INDEX[_v])} studies indexed ({_v})")
if cfg.use_cache and not cache_manifests and mode == "infer":
    # `use_cache` selects the PREPROCESSING (130 mm crop, per-series 1/99 normalisation,
    # laterality) as well as the array read. No TEST study is ever in the cache, so infer
    # builds every study through build_study_array -- the same functions the cache was
    # built with. Flipping it off here would take the v02 decode branch and score a v03
    # model on v02 pixels, and nothing would say so (traps.md 12d).
    print("  infer: no cache mounted (expected) -- test studies built on the fly by the "
          "cache-era preprocessing")


def ensure_cache(c):
    """The manifest of the cache `c` resolves to. Missing -> loud failure (traps 6f): every
    recipe since v03 depends on cache-era preprocessing and the decode branch would silently
    train v02 pixels at 5.5x the cost. ALLOW_DECODE_FALLBACK takes it deliberately (c01 only)."""
    if not c.use_cache:
        return None
    cv = cache_version_for(c)
    if cv in cache_manifests:
        return cache_manifests[cv]
    if ALLOW_DECODE_FALLBACK and cache_geom(c)[0] == "c01":
        print(f"  ! use_cache=True but cache {cv} is not mounted -- falling back to per-epoch "
              f"DICOM decode (ALLOW_DECODE_FALLBACK=True)")
        c.use_cache = False
        return None
    raise SystemExit(
        f"use_cache=True but cache {cv} is not mounted (mounted: {sorted(cache_manifests) or 'none'}). "
        f"Attach the matching cache kernels as kernel_sources (c01: rsna-knee-cache-a/-b; "
        f"c02: rsna-knee-cache2-a/-b/-c/-d), or set ALLOW_DECODE_FALLBACK=True to train on the "
        f"v02 decode path deliberately.")


def training_manifest(cache_manifest):
    """Train manifest for one cache (slots, side, mask straight from its manifest; a header scan
    only on the legacy decode path), plus placeholder target rows for imaged studies that are
    not in targets (the local sample). Mutates the module-level `targets`."""
    global targets
    if cache_manifest is not None:
        manifest = cache_manifest[["StudyInstanceUID", *SLOTS, "n_slots", "side", "mask"]].copy()
        print(f"  manifest from cache: {len(manifest)} studies; mean slots "
              f"{manifest.n_slots.mean():.2f}; side resolved {(manifest.side.fillna('') != '').mean():.1%}")
    else:
        train_series_csv = os.path.join(COMP, "train_series.csv")
        series_df = scan_series(train_series_csv, TRAIN_IMG,
                                os.path.join(WORK, "series_scan_train.csv"),
                                max_studies=cfg.smoke_max_studies if cfg.smoke else 0)
        if len(series_df) == 0:
            # Local sample: train_series.csv describes studies we do not have. Fall back to
            # scanning test_series.csv so the smoke test has something to chew on.
            series_df = scan_series(os.path.join(COMP, "test_series.csv"), TRAIN_IMG,
                                    os.path.join(WORK, "series_scan_fallback.csv"))
        manifest = build_manifest(series_df, os.path.join(WORK, "manifest_train.csv"))
    missing = set(manifest.StudyInstanceUID) - set(targets.StudyInstanceUID)
    if missing:
        print(f"  {len(missing)} imaged studies not in targets; adding placeholder "
              f"targets (smoke only)")
        add = pd.DataFrame({"StudyInstanceUID": sorted(missing)})
        add["is_gold"] = 0
        add["report_group"] = "local"
        add["fold"] = 0
        for l in LABELS:
            add[l] = 0.5
        for l in LABELS:
            add[f"w__{l}"] = cfg.weak_weight_floor
        targets = pd.concat([targets, add], ignore_index=True)
    return manifest


results = {}
if mode == "train":
    # Kaggle only: this script has no `if __name__ == "__main__"` guard, and Windows spawns
    # workers (re-importing __main__) instead of forking. The bug it tests is fork-specific.
    if ON_KAGGLE:
        check_worker_rng()
    base_cfg = replace(cfg)
    for arm_version, overrides in (ARMS or [(cfg.version, {})]):
        # Rebind the module-level `cfg`: out_of_time(), the dataset and the loaders all
        # read the global, so a local copy would silently leave them on the previous arm.
        # Merge, do not double-unpack: an override that sets `folds` (a 5-fold arm) would
        # otherwise be a duplicate keyword argument and raise TypeError. Overrides win.
        _ov = {**({"folds": ARM_FOLDS} if ARMS else {}), **overrides}
        cfg = replace(base_cfg, version=arm_version, **_ov)
        cfg.backbone_dir = resolve_backbone_dir(cfg.backbone)   # an arm may switch family (P-10)
        globals()["cfg"] = cfg
        # The cache and the manifest are per ARM: an arm may read a different cache scheme
        # than the default config (c02 arms next to c01 ones), so this cannot happen once
        # before the loop -- that would silently index the default config's cache for every arm.
        manifest = training_manifest(ensure_cache(cfg))
        if ARMS:
            print(f"\n########## arm {arm_version}: {overrides or 'baseline'} "
                  f"| folds {cfg.folds} epochs {cfg.epochs} seed {cfg.seed} ##########")
            print(f"  cache {cache_version_for(cfg)} | window_mode {cfg.window_mode}"
                  + (f" (train {cfg.train_windows}, eval {cfg.eval_windows or 'all'})"
                     if cfg.window_mode == "random" else f" (K {cfg.slices_per_slot})")
                  + f" | head {cfg.head_type} | backbone {cfg.backbone} | img {cfg.img_size}")
            if cfg.lat_undo:
                n_r = int((manifest["side"].astype(str) == "R").sum())                     if "side" in manifest.columns else 0
                print(f"  lat_undo: {n_r} of {len(manifest)} studies "
                      f"({n_r/max(len(manifest),1):.1%}) de-canonicalised at load time")
        try:
            for fold in cfg.folds:
                if out_of_time():
                    print(f"skipping fold {fold}: out of time")
                    continue
                print(f"\n=== {cfg.version} fold {fold} ===")
                _, best, done = train_fold(fold, manifest, targets, TRAIN_IMG, cfg, device)
                results[f"{cfg.version}/{fold}"] = {"best": best, "completed": done}
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception:
            # One arm failing must not cost the other three -- the Kaggle session is the
            # scarce resource here, not the code. Loud, logged, and on to the next arm.
            print(f"  !! arm {arm_version} FAILED -- continuing with the next arm")
            traceback.print_exc()
            results[f"{arm_version}/failed"] = {"best": float("nan"), "completed": False}
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # The inference below runs for ONE arm. It is a free smoke of the infer path, not a
    # submission -- what gets submitted is kaggle/rsna-knee-infer (traps.md 12c).
    if ARMS:
        cfg = replace(base_cfg, version=PRIMARY_ARM,
                      **{"folds": ARM_FOLDS, **dict(ARMS)[PRIMARY_ARM]})
        cfg.backbone_dir = resolve_backbone_dir(cfg.backbone)
        globals()["cfg"] = cfg
        print(f"\ninference uses PRIMARY_ARM={PRIMARY_ARM}")
    # members are (version, fold, path), the same shape the infer branch builds
    ckpt_members = [(cfg.version, f, os.path.join(WORK, f"{cfg.version}_fold{f}_best.pt"))
                    for f in cfg.folds]
elif mode == "oof_eval":
    # P-12 / P-25 measurement mode: score each member's fold-0 checkpoint on its own held-out
    # studies from the cache with the TTA / eval_windows it would use at inference, so the
    # `_tta_oof.csv` it writes is read by src/blend_check.py exactly like a training OOF file.
    base_cfg = replace(cfg)
    for v, f, p in infer_members:
        s = infer_settings[(v, f)]
        mcfg = replace(base_cfg, version=v)
        apply_settings(mcfg, s, INFER_CACHE_KEYS + INFER_MEMBER_KEYS)   # exact member settings, no smoke clamps
        mcfg.backbone_dir = resolve_backbone_dir(mcfg.backbone)
        globals()["cfg"] = mcfg
        cfg = mcfg
        print(f"\n=== oof_eval {v}/fold{f}: cache {cache_version_for(cfg)}, {s['window_mode']}, "
              f"eval_windows {s['eval_windows'] or 'all'}, tta {s['tta_offsets']}/{s['tta_pool']} ===")
        manifest = training_manifest(ensure_cache(cfg))
        _, va_loader = make_loaders(manifest, targets, TRAIN_IMG, cfg, f)
        model = build_model(cfg, device)
        st = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        t_eval = time.time()
        metrics, table = evaluate(model, va_loader, device, cfg)
        per_label = metrics.pop("per_label", {})
        print(f"  {v}/fold{f}: {metrics}  ({(time.time()-t_eval)/60:.1f} min)")
        if per_label:
            print_per_label(per_label)
        if table is not None:
            out_csv = os.path.join(WORK, f"{v}_fold{f}_tta_oof.csv")
            table.to_csv(out_csv, index=False)
            print(f"  -> {out_csv} ({len(table)} studies)")
        results[f"{v}/{f}"] = {"best": metrics.get("auc_soft", float("nan")), "completed": True}
        del model, st
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ckpt_members = []
else:
    results = {f"{v}/{f}": {"best": float("nan"), "completed": True} for v, f, _ in infer_members}
    ckpt_members = list(infer_members)

print("\nfold results:", json.dumps(results, indent=1))
if os.environ.get("RSNA_TRAIN_ONLY") and mode == "train":
    # Off-Kaggle (RunPod) training box: there is no test tree, so stop cleanly here instead of
    # dying at the coverage gate below. The checkpoints in WORK are the deliverable.
    print("RSNA_TRAIN_ONLY is set -- stopping before inference (train-only box)")
    raise SystemExit(0)
if mode == "infer":
    all_done = True                       # every member was verified mounted above
elif mode == "oof_eval":
    all_done = False                      # measurement only; nothing to submit
    print("oof_eval done -- no test prediction in this mode")
else:
    # With ARMS, `results` is keyed "<arm>/<fold>" across every arm, so completion has to be
    # judged on the arm inference will actually use -- otherwise the count never matches
    # len(cfg.folds) and the infer path is silently skipped.
    done_keys = ([k for k in results if str(k).startswith(f"{PRIMARY_ARM}/")]
                 if ARMS else list(results))
    all_done = len(done_keys) == len(cfg.folds) and all(results[k]["completed"] for k in done_keys)
print(f"all folds complete: {all_done}  elapsed {elapsed_h():.2f} h")

# %% [markdown]
# ## Section 9: inference and submission
#
# Ensembling is a **rank mean**, not a probability mean. AUC reads only order, so
# averaging probabilities lets whichever fold is most confident dominate, while
# averaging ranks combines exactly the information the metric uses.
#
# Inference only runs once every fold has finished. If the runtime guard fired,
# the notebook stops here — attach this output as input to a fresh run and it
# resumes rather than submitting a half-trained ensemble.

# %%
# ── Section 9: inference ──────────────────────────────────────────────────────
def predict(model, manifest, image_root, cfg, studies, device):
    ds = KneeStudyDataset(manifest, None, image_root, cfg, False, studies)
    dl = DataLoader(ds, batch_size=cfg.batch_studies, shuffle=False,
                    num_workers=0 if cfg.smoke else cfg.num_workers)
    ids, preds = [], []
    model.eval()
    with torch.no_grad():
        for b in dl:
            preds.append(predict_probs(model, b, device, cfg).cpu().numpy())
            ids.extend(b["study"])
    if not preds:
        return pd.DataFrame(columns=["StudyInstanceUID"] + LABELS)
    P = np.concatenate(preds)
    return pd.DataFrame({"StudyInstanceUID": ids,
                         **{l: P[:, i] for i, l in enumerate(LABELS)}})


def rank_mean(frames):
    """Average percentile ranks across folds -- the operation macro-AUC actually reads."""
    base = frames[0][["StudyInstanceUID"]].copy()
    for lab in LABELS:
        acc = np.zeros(len(base))
        for f in frames:
            acc += f[lab].rank(pct=True).to_numpy()
        base[lab] = acc / len(frames)
    return base


sub_path = os.path.join(WORK, "submission.csv")
sample_path = os.path.join(COMP, "sample_submission.csv")
ref = pd.read_csv(sample_path)

if not all_done:
    print("training incomplete -- skipping inference.")
    print("Attach this notebook's output as input to a new run to resume.")
else:
    # Deliberately NO placeholder file: if anything below raises, Kaggle reports a
    # missing submission (visible), instead of scoring a silent 0.500 (invisible).
    for stale in (sub_path, "/kaggle/working/submission.csv" if ON_KAGGLE else None):
        if stale and os.path.exists(stale):
            os.remove(stale)

    t_inf = time.time()
    test_series_df = scan_series(os.path.join(COMP, "test_series.csv"), TEST_IMG,
                                 os.path.join(WORK, "series_scan_test.csv"))
    test_manifest = build_manifest(test_series_df,
                                   os.path.join(WORK, "manifest_test.csv"))
    all_test = pd.read_csv(os.path.join(COMP, "test.csv")).StudyInstanceUID.tolist()
    with_slots = set(test_manifest.loc[test_manifest.n_slots > 0, "StudyInstanceUID"])
    test_studies = [s for s in all_test if s in with_slots]    # imaged AND has a slot
    coverage = len(test_studies) / max(len(all_test), 1)
    print(f"  test studies: {len(all_test)} listed, {len(test_studies)} imaged "
          f"({coverage:.1%}); scan+manifest {time.time()-t_inf:.0f}s")
    print("  slot fill on test:",
          {s: round(float((test_manifest[s] != '').mean()), 3) for s in SLOTS})
    # Loud failure beats a silent constant submission: a scoring error is visible on
    # the submissions page, a 0.500 looks like a bad model.
    if coverage < 0.9:
        raise SystemExit(f"only {coverage:.1%} of test studies have images under "
                         f"{TEST_IMG} -- refusing to submit constants")

    # ---- decode once PER GEOMETRY GROUP, predict with every member (P-18 / P-21 / P-25) ------
    # A test study is never in the mounted cache, so each member used to re-decode the whole
    # test set (~1.5-2 s/study). Members that share every CACHE key form a group; each group's
    # test arrays are built ONCE with build_study_array -- the cache builder's own function, so
    # a test study is preprocessed exactly like a cached training study -- stored under the
    # system temp dir (NOT WORK: 5-8 MB/study must not become kernel output), registered in
    # CACHE_INDEX[version] so KneeStudyDataset takes the same read branch it takes in training,
    # and deleted once the group's members have predicted (two schemes = two footprints).
    import shutil

    def decode_once(group_cfg, studies, manifest_df):
        version = cache_version_for(group_cfg)
        test_cache_dir = os.path.join(tempfile.gettempdir(), "rsna_test_cache", version)
        os.makedirs(test_cache_dir, exist_ok=True)

        class _BuildOnce(Dataset):
            def __init__(self, manifest, studies):
                self.m = manifest.set_index("StudyInstanceUID")
                self.s = list(studies)

            def __len__(self):
                return len(self.s)

            def __getitem__(self, i):
                study = self.s[i]
                arr, mask = build_study_array(study, self.m.loc[study], TEST_IMG, group_cfg)
                path = os.path.join(test_cache_dir, f"{study}.npy")
                np.save(path, arr)
                return study, path, "".join("1" if v > 0 else "0" for v in mask)

        t_dec = time.time()
        masks, index = {}, {}
        dec_loader = DataLoader(_BuildOnce(manifest_df, studies), batch_size=1, shuffle=False,
                                num_workers=0 if group_cfg.smoke else group_cfg.num_workers,
                                collate_fn=lambda b: b[0])
        for k, (study, path, mk) in enumerate(dec_loader):
            index[study] = path
            masks[study] = mk
            if (k + 1) in (10, 100) or (k + 1) % 500 == 0:
                dt = time.time() - t_dec
                print(f"    decoded {k+1}/{len(studies)} test studies in {dt:.0f}s "
                      f"({dt/(k+1):.2f} s/study) -> ETA {dt/(k+1)*len(studies)/60:.0f} min")
        CACHE_INDEX[version] = index
        n_bytes = sum(os.path.getsize(index[s]) for s in studies[:50]) * len(studies) / max(min(50, len(studies)), 1)
        print(f"  decode-once [{version}]: {len(masks)} test studies -> {test_cache_dir} in "
              f"{(time.time()-t_dec)/60:.1f} min (~{n_bytes/1e9:.1f} GB)")
        # Verify by equality, not by absence of errors (traps 6d/6e): rebuild a few studies on
        # the fly and compare with what every member of the group is about to read.
        _chk = manifest_df.set_index("StudyInstanceUID")
        for study in studies[:3]:
            arr, mask = build_study_array(study, _chk.loc[study], TEST_IMG, group_cfg)
            mk = "".join("1" if v > 0 else "0" for v in mask)
            if not (np.array_equal(arr, np.load(index[study])) and mk == masks[study]):
                raise SystemExit(f"decode-once mismatch on {study}: the stored array or mask "
                                 f"differs from a fresh build -- refusing to predict")
        print(f"  decode-once verified [{version}]: {min(3, len(studies))} studies rebuilt, identical")
        return version, masks, test_cache_dir

    member_list = []                      # (version, fold, path, settings)
    for v, fold, ck in ckpt_members:
        if not ck or not os.path.exists(ck):
            print(f"  {v}/fold{fold}: no checkpoint, skipped")
            continue
        s = infer_settings.get((v, fold))
        if s is None:                     # train mode: this run's own checkpoints
            st0 = torch.load(ck, map_location="cpu", weights_only=False)
            s = member_settings(st0.get("config", {}), v)
            del st0
        member_list.append((v, fold, ck, s))
    geometry_groups = {}
    for item in member_list:
        geometry_groups.setdefault(cache_signature(item[3]), []).append(item)
    print(f"  {len(member_list)} members in {len(geometry_groups)} geometry group(s)")

    frames, member_tags = [], []
    cfg_snapshot = replace(cfg)
    for sig, members in geometry_groups.items():
        apply_settings(cfg, members[0][3], INFER_CACHE_KEYS)
        group_version, tmp_dir = cache_version_for(cfg), None
        if cfg.use_cache and test_studies:
            group_version, masks, tmp_dir = decode_once(cfg, test_studies, test_manifest)
            test_manifest["mask"] = test_manifest.StudyInstanceUID.map(masks).fillna("")
        for v, fold, ck, s in members:
            prev = apply_settings(cfg, s, INFER_MEMBER_KEYS)
            st = torch.load(ck, map_location=device, weights_only=False)
            m = build_model(s, device)
            m.load_state_dict(st["model"])
            t_f = time.time()
            frames.append(predict(m, test_manifest, TEST_IMG, cfg, test_studies, device))
            member_tags.append(f"{v}/fold{fold}")
            dt = time.time() - t_f
            how = (f"windows eval {s['eval_windows'] or 'all'}" if s["window_mode"] == "random"
                   else f"K {s['slices_per_slot']}, tta {s['tta_offsets']}/{s['tta_pool']}, {s['stack_mode']}")
            print(f"  {v}/fold{fold} ({s['backbone']}, {s['head_type']}, {how}, {group_version}): "
                  f"predicted {len(frames[-1])} studies in {dt:.0f}s "
                  f"({dt/max(len(frames[-1]),1)*100:.0f} s per 100 studies) "
                  f"[epoch {st.get('epoch')}, score {st.get('score')}, ema {st.get('ema')}]")
            apply_settings(cfg, prev, INFER_MEMBER_KEYS)
            del m, st
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            CACHE_INDEX.pop(group_version, None)
    apply_settings(cfg, {k: getattr(cfg_snapshot, k) for k in INFER_CACHE_KEYS}, INFER_CACHE_KEYS)

    if not frames:
        raise SystemExit("no checkpoints produced predictions -- refusing to submit "
                         "constants")
    if len(frames) > 1 and len(frames[0]) > 3:
        # Two members that agree perfectly are one model counted twice; print the rank
        # correlation so the blend's diversity is on the record (P-21 measured 0.773 on OOF).
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                rho = float(np.mean([frames[i][l].corr(frames[j][l], method="spearman")
                                     for l in LABELS]))
                print(f"  rank correlation {member_tags[i]} vs {member_tags[j]}: {rho:.3f}")
    if INFER_BLEND == "by_version":
        by_version = {}
        for tag, f in zip(member_tags, frames):
            by_version.setdefault(tag.split("/")[0], []).append(f)
        sub = rank_mean([rank_mean(fs) for fs in by_version.values()])
        print("  blend: by_version -> " + ", ".join(f"{v} ({len(fs)} fold{'s' if len(fs) != 1 else ''})"
                                                  for v, fs in by_version.items()))
    else:
        sub = rank_mean(frames)
        print(f"  blend: flat over {len(frames)} members")

    # Any study we could not image must still appear, or the submission is rejected.
    sub = ref[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    n_filled = int(sub[LABELS[0]].isna().sum())
    for l in LABELS:
        sub[l] = sub[l].fillna(0.5)
    sub = sub[["StudyInstanceUID"] + LABELS]

    assert list(sub.columns) == list(ref.columns), "column mismatch vs sample_submission"
    assert len(sub) == len(ref), f"row count {len(sub)} != {len(ref)}"
    assert (sub.StudyInstanceUID.to_numpy() == ref.StudyInstanceUID.to_numpy()).all(), \
        "row order differs from sample_submission"
    assert np.isfinite(sub[LABELS].to_numpy()).all(), "non-finite predictions"
    n_const = int((sub[LABELS].std(axis=0) < 1e-9).sum())
    if n_const > len(LABELS) // 2 and len(sub) > 3:
        raise SystemExit(f"{n_const}/12 labels are constant across {len(sub)} studies "
                         f"-- model or inputs are broken, refusing to submit")

    sub.to_csv(sub_path, index=False)
    if ON_KAGGLE:
        sub.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"\nwrote {sub_path}  rows={len(sub)}  filled 0.5 for {n_filled}  "
          f"range=[{sub[LABELS].to_numpy().min():.3f}, "
          f"{sub[LABELS].to_numpy().max():.3f}]  constant labels {n_const}  "
          f"inference total {(time.time()-t_inf)/60:.1f} min")
    print(sub.head(3).to_string(index=False))

print(f"\ntotal elapsed {elapsed_h():.2f} h")
