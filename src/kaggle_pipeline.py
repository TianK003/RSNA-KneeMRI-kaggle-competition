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
import time
from dataclasses import dataclass, field, asdict

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
# │       "auto"  = "infer" if such checkpoints are mounted, else "train".   │
# └──────────────────────────────────────────────────────────────────────────┘
MODE = "auto"


@dataclass
class Config:
    smoke: bool = field(default_factory=lambda:
                        (not ON_KAGGLE) if FORCE_SMOKE is None else bool(FORCE_SMOKE))
    version: str = "v02"             # v01 = rank targets, lr 5e-5 uniform (smoke only)

    # data
    img_size: int = 224              # DINOv2 ViT-S/14 patches 14 -> 224 = 16x16 tokens
    slices_per_slot: int = 6         # uniformly sampled centres per slot
    triplet_gap: int = 2             # channels are slices [i-gap, i, i+gap]

    # model
    backbone_dir: str = ""           # filled in below
    dropout: float = 0.1

    # optimisation
    folds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 4
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
    runtime_limit_hours: float = 8.3   # leave headroom under Kaggle's 9 h ceiling
    seed: int = 42
    num_workers: int = 2
    # Smoke only: cap the header scan so a verification run does not spend minutes
    # reading all ~24k series headers before it reaches the training loop.
    smoke_max_studies: int = 24

    def __post_init__(self):
        if self.smoke:
            self.folds = (0,)
            self.epochs = 1
            self.slices_per_slot = 2
            self.img_size = 224
            self.runtime_limit_hours = 0.4
            self.ema_decay = 0.9      # 8 steps of smoke would leave a 0.998 EMA ~= init


cfg = Config()

# Weight locations differ between Kaggle (mounted Model, two possible layouts) and
# local (models/). config.json is the marker that a real HF checkpoint dir is there.
cfg.backbone_dir = resolve_dir([
    "/kaggle/input/dinov2/pytorch/small/1",
    "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1",
    "/kaggle/input/dinov2-small/pytorch/small/1",
    "models/dinov2_small",
], must_contain="config.json")
if cfg.backbone_dir is None:
    raise SystemExit("DINOv2 weights not found -- attach metaresearch/dinov2 "
                     "PyTorch/small/1 as a Model input")
print(f"backbone: {cfg.backbone_dir}")
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
        rows.append({"StudyInstanceUID": study,
                     **{s: slots.get(s, "") for s in SLOTS},
                     "n_slots": len(slots)})
    m = pd.DataFrame(rows)
    m.to_csv(cache, index=False)
    print(f"  manifest -> {cache}; mean slots/study {m.n_slots.mean():.2f}")
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


def ordered_slice_paths(series_dir: str) -> list:
    """Spatially ordered slice paths. NEVER trust filename order."""
    files = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if not files:   # do not assume the hidden test tree keeps the .dcm extension
        files = [f for f in os.listdir(series_dir)
                 if os.path.isfile(os.path.join(series_dir, f))]
    if not files:
        return []
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
        return []
    iop = next((getattr(h, "ImageOrientationPatient", None) for h in heads), None)
    if iop is not None and len(iop) == 6:
        n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        keys, ok = [], True
        for h in heads:
            ipp = getattr(h, "ImagePositionPatient", None)
            if ipp is None:
                ok = False
                break
            keys.append(float(np.dot(np.array(ipp, float), n)))
        if ok:
            return [p for _, p in sorted(zip(keys, paths), key=lambda t: t[0])]
    inst = [getattr(h, "InstanceNumber", None) for h in heads]
    if all(i is not None for i in inst):
        return [p for _, p in sorted(zip(inst, paths), key=lambda t: t[0])]
    print(f"  ! {series_dir}: no usable position/instance headers -- filename order")
    return paths


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


class KneeNet(nn.Module):
    def __init__(self, backbone_dir: str, n_labels=len(LABELS), dropout=0.1):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained(backbone_dir)
        self.dim = self.enc.config.hidden_size
        self.pool = AttnPool(self.dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.dim * len(SLOTS) + len(SLOTS), n_labels)

    def forward(self, imgs, mask):
        # imgs: (B, SLOT, S, 3, H, W)   mask: (B, SLOT)
        B, NS, S = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        flat = imgs.reshape(B * NS * S, *imgs.shape[3:])
        feats = self.enc(pixel_values=flat).last_hidden_state[:, 0]   # CLS token
        feats = feats.reshape(B, NS, S, self.dim)
        pooled = torch.stack([
            torch.stack([self.pool(feats[b, s]) for s in range(NS)])
            for b in range(B)
        ])                                                            # (B, NS, dim)
        pooled = pooled * mask.unsqueeze(-1)      # zero out absent slots
        x = torch.cat([pooled.reshape(B, -1), mask], dim=1)
        return self.head(self.drop(x))


def weighted_bce(logits, y, w):
    """Confidence-weighted soft-target BCE.

    No `pos_weight`: with soft targets it inflates every prediction and the metric
    reads only rank order, so there is nothing to gain and a collapse to overprediction
    to lose.
    """
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1e-6)

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
                       num_workers=nw, drop_last=False),
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


def evaluate(model, loader, device):
    """Validation pass. Returns (metrics, table) where `table` is a DataFrame with the
    per-study predictions, targets, weights and gold flag -- the OOF rows. Per-label
    numbers are kept because the metric charges every label the same, so the label
    stuck at 0.5 is the thing we most need to see."""
    model.eval()
    P, Y, W, G, S = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["imgs"].to(device), b["mask"].to(device))
            P.append(torch.sigmoid(logits).float().cpu().numpy())
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
    n_blocks = model.enc.config.num_hidden_layers
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
        if name.startswith("embeddings."):
            depth = 0
        elif name.startswith("encoder.layer."):
            depth = int(name.split(".")[2]) + 1
        else:                       # final layernorm
            depth = n_blocks + 1
        lr = cfg.lr_backbone * (cfg.llrd_decay ** (n_blocks + 1 - depth))
        add(name, p, lr)
    for mod in (model.pool, model.head):
        for name, p in mod.named_parameters():
            add(name, p, cfg.lr_head)
    out = list(groups.values())
    lrs = sorted({g["lr"] for g in out if g["lr"] < cfg.lr_head})
    print(f"  backbone LR range {lrs[0]:.2e} .. {lrs[-1]:.2e} over {n_blocks} blocks "
          f"(decay {cfg.llrd_decay}); head {cfg.lr_head:.0e}")
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

    model = KneeNet(cfg.backbone_dir, dropout=cfg.dropout).to(device)
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

    start_epoch, best = 0, -1.0
    if os.path.exists(ckpt_last):
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
        print(f"  resumed fold {fold} at epoch {start_epoch} (best {best:.4f})")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        t_epoch = time.time()
        n_studies = 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr_loader):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(b["imgs"].to(device), b["mask"].to(device))
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
                print(f"    {n_studies} studies in {dt:.0f}s = {dt/n_studies:.2f} s/study "
                      f"(slices/slot {cfg.slices_per_slot}, workers "
                      f"{tr_loader.num_workers}) -> epoch ETA "
                      f"{dt/n_studies*len(tr_loader.dataset)/60:.0f} min")
            if out_of_time():
                print("  runtime guard hit mid-epoch")
                break
        train_secs = time.time() - t_epoch

        eval_model = ema.module if ema is not None else model
        t_eval = time.time()
        metrics, oof = evaluate(eval_model, va_loader, device)
        per_label = metrics.pop("per_label", {})
        print(f"  fold {fold} epoch {epoch}: loss {running/max(nb,1):.4f}  {metrics}")
        print(f"    train {train_secs/60:.1f} min ({train_secs/max(n_studies,1):.2f} s/study), "
              f"val {(time.time()-t_eval)/60:.1f} min")
        if per_label:
            print_per_label(per_label)
        if metrics.get("pred_std", 1.0) < 0.01:
            print("  !! prediction spread near zero -- base-rate collapse, not a "
                  "converged model")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    **({"ema": ema.module.state_dict()} if ema is not None else {})},
                   ckpt_last)
        # Which epoch is "the" model? Selecting the best epoch on ~12 gold studies per
        # fold is a coin flip (Hanley-McNeil SE ~0.09), and selecting on the soft
        # targets rewards memorising the teacher. So `{version}_fold{k}_best.pt` is
        # simply the EMA weights after the LAST COMPLETED epoch (fixed-epoch selection,
        # P-03/P-04). The per-epoch score is logged, not used. A NaN score therefore
        # cannot drop a fold from the ensemble either.
        score = metrics.get("auc_gold")
        if score is None or not np.isfinite(score):
            score = metrics.get("auc_soft")
        if score is None or not np.isfinite(score):
            score = -running / max(nb, 1)
        best = max(score, best) if np.isfinite(score) else best
        torch.save({"model": eval_model.state_dict(), "score": score, "epoch": epoch,
                    "ema": ema is not None, "config": asdict(cfg)}, ckpt_best)
        if oof is not None:
            oof.insert(1, "epoch", epoch)
            oof.to_csv(oof_path.replace("_oof.csv", f"_ep{epoch}_oof.csv"), index=False)
            oof.to_csv(oof_path, index=False)        # always the checkpointed epoch
        print(f"    checkpoint = epoch {epoch} EMA (score {score:.4f}) -> "
              f"{os.path.basename(ckpt_best)} + {os.path.basename(oof_path)}")

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
    roots = ["/kaggle/input"] if ON_KAGGLE else ["artifacts/kaggle_out"]
    found = {}
    for root in roots:
        for p in shallow_glob(root, f"{version}_fold*_{kind}.pt", max_depth=3):
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

if mode == "infer" and not mounted_ckpts:
    raise SystemExit(f"MODE=infer but no {cfg.version}_fold*_best.pt is mounted. Attach the "
                     f"training run's output as a kernel input (kernel_sources).")
if mode == "infer":
    # The checkpoint decides the input geometry, not FORCE_SMOKE: a smoke-mode infer
    # would otherwise feed 2 slices/slot to a model trained on 6 and pass every assert.
    st0 = torch.load(mounted_ckpts[min(mounted_ckpts)], map_location="cpu", weights_only=False)
    saved = st0.get("config", {})
    for k in ("slices_per_slot", "triplet_gap", "img_size"):
        if k in saved and getattr(cfg, k) != saved[k]:
            print(f"  infer: {k} {getattr(cfg, k)} -> {saved[k]} (from checkpoint)")
            setattr(cfg, k, saved[k])
    if cfg.smoke and set(mounted_ckpts) != set(cfg.folds):
        print(f"  infer: using all mounted folds {sorted(mounted_ckpts)} (smoke folds ignored)")
        cfg.folds = tuple(sorted(mounted_ckpts))
    del st0
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

results = {}
if mode == "train":
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

    # Studies present as images but absent from targets (local case) get placeholder
    # rows so the smoke test can build batches.
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

    for fold in cfg.folds:
        if out_of_time():
            print(f"skipping fold {fold}: out of time")
            continue
        print(f"\n=== fold {fold} ===")
        _, best, done = train_fold(fold, manifest, targets, TRAIN_IMG, cfg, device)
        results[fold] = {"best": best, "completed": done}
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ckpt_paths = {f: os.path.join(WORK, f"{cfg.version}_fold{f}_best.pt")
                  for f in cfg.folds}
else:
    missing_folds = [f for f in cfg.folds if f not in mounted_ckpts]
    if missing_folds:
        raise SystemExit(f"MODE=infer but no checkpoint mounted for folds "
                         f"{missing_folds} (version {cfg.version}). Attach the "
                         f"training run's output as a kernel input.")
    results = {f: {"best": float("nan"), "completed": True} for f in cfg.folds}
    ckpt_paths = {f: mounted_ckpts[f] for f in cfg.folds}

print("\nfold results:", json.dumps(results, indent=1))
all_done = len(results) == len(cfg.folds) and all(r["completed"] for r in results.values())
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
            logits = model(b["imgs"].to(device), b["mask"].to(device))
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
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

    frames = []
    for fold in cfg.folds:
        ck = ckpt_paths.get(fold)
        if not ck or not os.path.exists(ck):
            print(f"  fold {fold}: no checkpoint, skipped")
            continue
        st = torch.load(ck, map_location=device, weights_only=False)
        m = KneeNet(cfg.backbone_dir, dropout=cfg.dropout).to(device)
        m.load_state_dict(st["model"])
        t_f = time.time()
        frames.append(predict(m, test_manifest, TEST_IMG, cfg, test_studies, device))
        dt = time.time() - t_f
        print(f"  fold {fold}: predicted {len(frames[-1])} studies in {dt:.0f}s "
              f"({dt/max(len(frames[-1]),1)*100:.0f} s per 100 studies) "
              f"[epoch {st.get('epoch')}, score {st.get('score')}, ema {st.get('ema')}]")
        del m
        gc.collect()

    if not frames:
        raise SystemExit("no checkpoints produced predictions -- refusing to submit "
                         "constants")
    sub = rank_mean(frames)

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
