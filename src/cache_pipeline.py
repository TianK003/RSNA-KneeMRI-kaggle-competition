# %% [markdown]
# # RSNA Knee — preprocessing cache (P-01)
#
# Decode every training study **once** into a small uint8 array so that training
# reads one file per study instead of ~90 DICOMs per study per epoch.
#
# Why this exists: the training notebook measured ~4.5 s per study-pass with the
# DICOMs decoded inside the DataLoader (an extrapolation from 8 passes, so the real
# figure is measured here as a by-product). Every RSNA winner 2019–2024 trained
# from pre-extracted arrays; the public knee notebooks report ~16 min to build an
# 11 GB cache and ~100 s per epoch afterwards. Nothing else in `docs/proposals.md`
# is affordable until this exists.
#
# What one cached study is: `[6 slots, S slices, P, P] uint8`, plus a presence
# mask and a manifest row. The 2.5D triplet is formed **at load time** from
# neighbouring cached slices, so the cache stores each slice once.
#
# Three things baked in here that the training notebook must never redo
# differently (same function, same version string — see `CACHE_VERSION`):
#
# 1. **Spatial ordering** by `ImagePositionPatient · normal` — never filename.
# 2. **Per-series intensity normalisation** (1st/99th percentile over the sampled
#    stack) after `RescaleSlope/Intercept` and `MONOCHROME1` inversion.
# 3. **Laterality canonicalisation** to a LEFT knee: right knees have coronal and
#    axial slices mirrored left-right and sagittal stacks reversed. The `Laterality`
#    tag is missing on ~half the corpus, so side is derived from the geometry: the
#    x-coordinate (patient left-right) of the image centre, with a 20 mm dead zone.
#    Tag and geometry are both recorded; conflicts are counted and left untouched.
#
# The header pass also records the **site proxy** (Manufacturer, model, field
# strength, InstitutionName presence, PixelSpacing, TransferSyntax) that P-02 needs
# for site-grouped folds.
#
# **Sharding:** ~4,407 × 6 × 16 × 224² bytes ≈ 21 GB, above Kaggle's per-kernel
# output cap, so studies are split into `N_SHARDS` deterministic shards and this
# notebook is run once per `SHARD` (two kernels: `rsna-knee-cache-a` / `-b`).
#
# **Two cache schemes** (2026-08-30). `c01` is the original: 6 slots × 16 slices at
# 224 px inside a per-plane central band (sag 8–92 %, cor 20–80 %, ax 10–90 %), one
# `.npy` per study. `c02` is the wide-band rebuild: the same 6 slots with **ragged
# slice budgets** 18/12/12/14/8/8 (= 72 slices, order = `SLOTS`), band **2–98 %** for
# every plane, **336 px**, stored FLAT as `(72, 336, 336)` per study with fixed slot
# offsets, and packed into multi-study **blobs** of 64 studies (`blob{shard}_{k}.npy`,
# ~520 MB) with a CSV sidecar each — ~70 files for the corpus instead of 4,407, so it
# can leave Kaggle for a rented GPU without tripping the per-file rate limit. Why the
# rebuild: the 0.936 public notebook's strongest member uses 2–98 % and reports that
# cutting the outer slices "was measurably costing accuracy on the collateral
# ligaments and the lateral meniscus" — exactly our two weakest labels. Four kernels:
# `rsna-knee-cache2-a` … `-d`. The version string now encodes everything that changes
# the stored bytes (band and budgets included — traps 23).

# %%
# ── Section 0: environment ────────────────────────────────────────────────────
import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor


def parallel_map(fn, jobs, workers, chunksize=1):
    """ProcessPool on Linux/Kaggle; serial on Windows, where spawn would re-import
    this script (percent-format, no __main__ guard) inside every worker."""
    if os.name == "nt" or workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(fn, jobs, chunksize=chunksize)

import numpy as np
import pandas as pd
import pydicom

T_START = time.time()
ON_KAGGLE = os.path.exists("/kaggle/input")


def resolve_dir(candidates, must_contain=None):
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
        raise SystemExit("attach the competition to this kernel")
else:
    COMP = "data"
    WORK = "artifacts/cache_local"
os.makedirs(WORK, exist_ok=True)
print(f"ON_KAGGLE={ON_KAGGLE}  COMP={COMP}  WORK={WORK}")

# %% [markdown]
# ## Section 1: configuration
#
# `SMOKE = True` caches 24 studies in a couple of minutes — always the first run of
# an edited notebook. `SHARD` selects which half of the studies this run writes.

# %%
# ── Section 1: configuration ──────────────────────────────────────────────────
SMOKE = None            # True / False / None = auto (smoke locally, real on Kaggle)
SHARD = 0               # which shard this kernel writes (sed'd to 1/2/3 for -b/-c/-d)
N_SHARDS = 4            # c02: 4 shards of ~9 GB (c01 was built as 2 shards of ~10.6 GB)
SCHEME = "c02"          # "c01" = the original 224/16 cache; "c02" = wide-band 336 ragged cache
# Local-only overrides so the sample-DICOM smoke can build one shard of either scheme.
SHARD = int(os.environ.get("RSNA_SHARD", SHARD))
N_SHARDS = int(os.environ.get("RSNA_N_SHARDS", N_SHARDS))
SCHEME = os.environ.get("RSNA_CACHE_SCHEME", SCHEME)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
PLANE_OF_SLOT = {"SAG_FLUID_FS": "Sagittal", "COR_FLUID_FS": "Coronal", "AX_FLUID_FS": "Axial",
                 "SAG_FLUID_NOFS": "Sagittal", "COR_T1": "Coronal", "SAG_T1": "Sagittal"}

# Per-scheme geometry. c02's budgets follow the 0.936 notebook's strongest member (SAG-FS 18,
# SAG 14, COR-FS 12, COR 8, AX 12) mapped onto our six slots: the two T1 slots share the
# notebook's non-fluid coronal budget (8) because they fill only 50-62 % of studies.
SCHEME_DEFAULTS = {
    "c01": dict(px=224, slot_slices=(16, 16, 16, 16, 16, 16),
                band={"Sagittal": (0.08, 0.92), "Axial": (0.10, 0.90), "Coronal": (0.20, 0.80)}),
    "c02": dict(px=336, slot_slices=(18, 12, 12, 14, 8, 8),
                band={"Sagittal": (0.02, 0.98), "Axial": (0.02, 0.98), "Coronal": (0.02, 0.98)}),
}


@dataclass
class CacheConfig:
    smoke: bool = (not ON_KAGGLE) if SMOKE is None else bool(SMOKE)
    scheme: str = SCHEME
    px: int = 0                   # stored resolution; 0 -> scheme default
    slot_slices: tuple = ()       # stored slices per slot, order = SLOTS; () -> scheme default
    crop_mm: float = 130.0        # physical centre crop; 0 disables
    # Central band per plane (fraction of the ordered stack); None -> scheme default.
    band: dict = None
    lat_dead_zone_mm: float = 20.0
    pct_lo: float = 1.0
    pct_hi: float = 99.0
    blob_size: int = 64           # c02: studies per blob file (~520 MB at 72 x 336^2)
    smoke_max_studies: int = 24
    workers: int = 4

    def __post_init__(self):
        if self.scheme not in SCHEME_DEFAULTS:
            raise SystemExit(f"unknown cache scheme {self.scheme!r}; known: {sorted(SCHEME_DEFAULTS)}")
        d = SCHEME_DEFAULTS[self.scheme]
        self.px = self.px or d["px"]
        self.slot_slices = tuple(self.slot_slices) or d["slot_slices"]
        if self.band is None:
            self.band = dict(d["band"])
        if len(self.slot_slices) != len(SLOTS):
            raise SystemExit(f"slot_slices needs {len(SLOTS)} entries, got {self.slot_slices}")

    @property
    def n_slices(self):
        """Uniform slices per slot (c01 only); None when the budgets are ragged."""
        return self.slot_slices[0] if len(set(self.slot_slices)) == 1 else None


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


cfg = CacheConfig()
CACHE_VERSION = cache_version_of(cfg.scheme, cfg.px, cfg.slot_slices, cfg.band,
                                 cfg.crop_mm, cfg.lat_dead_zone_mm)
SLOT_START, N_TOTAL_SLICES = slot_offsets(cfg.slot_slices)
print(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=1))
print(f"cache version: {CACHE_VERSION}  (slot offsets {SLOT_START}, {N_TOTAL_SLICES} slices/study)")

# %% [markdown]
# ## Section 2: series header scan
#
# Same two-tier slot logic as the training notebook (keep in sync), plus the
# geometry and site fields. One header per series here; per-slice headers are read
# in Section 3 when a series is actually cached.

# %%
# ── Section 2: header scan ────────────────────────────────────────────────────
TR_SHORT_MAX = 800.0
TE_LONG_MIN = 60.0
FATSAT_TOKENS = ("fs", "fatsat", "fat_sat", "stir", "spir", "spair", "tirm",
                 "dixon", "chess", "sat", "supp")
FLUID_TOKENS = ("t2", "stir", "pd", "dess", "spair", "spir", "tirm")


def has_token(text, tokens):
    t = text.lower().replace("-", "").replace(" ", "")
    return any(tok.replace("_", "") in t for tok in tokens)


def plane_from_iop(iop):
    if iop is None or len(iop) != 6:
        return "unknown"
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(np.argmax(np.abs(n)))]


def classify_weighting(tr, te, scanning_seq, desc):
    d = desc.lower()
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


def list_slices(d):
    files = [f for f in os.listdir(d) if f.endswith(".dcm")]
    if not files:
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    return sorted(files)


def centre_x_mm(h):
    """Patient-space x (LPS: +x = patient's left) of the image centre, in mm."""
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


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def scan_one_series(args):
    study, series, d = args
    files = list_slices(d)
    if not files:
        return None
    h = None
    for f in files[:5]:
        try:
            h = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
            break
        except Exception:
            continue
    if h is None:
        return None
    desc = " ".join(str(getattr(h, k, "") or "") for k in
                    ("SeriesDescription", "SequenceName", "ScanOptions", "ProtocolName"))
    w = classify_weighting(_f(getattr(h, "RepetitionTime", None)),
                           _f(getattr(h, "EchoTime", None)),
                           str(getattr(h, "ScanningSequence", "") or ""), desc)
    ps = getattr(h, "PixelSpacing", None)
    lat = str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "").upper()
    return {
        "StudyInstanceUID": study, "SeriesInstanceUID": series,
        "n_slices": len(files), "plane_iop": plane_from_iop(getattr(h, "ImageOrientationPatient", None)),
        "weighting": w, "fat_sat": int(has_token(desc, FATSAT_TOKENS)),
        "fluid": int(w in ("T2", "PD") or has_token(desc, FLUID_TOKENS)),
        "laterality_tag": lat if lat in ("L", "R") else "",
        "centre_x_mm": centre_x_mm(h),
        "pixel_spacing": _f(ps[0]) if ps is not None else None,
        "rows": getattr(h, "Rows", None), "cols": getattr(h, "Columns", None),
        "slice_thickness": _f(getattr(h, "SliceThickness", None)),
        "manufacturer": str(getattr(h, "Manufacturer", "") or ""),
        "model": str(getattr(h, "ManufacturerModelName", "") or ""),
        "field_T": _f(getattr(h, "MagneticFieldStrength", None)),
        "institution_present": int(bool(getattr(h, "InstitutionName", None))),
        "transfer_syntax": str(getattr(getattr(h, "file_meta", None), "TransferSyntaxUID", "")),
        "patient_position": str(getattr(h, "PatientPosition", "") or ""),
    }


def scan_series(series_csv, image_root, cache_path, max_studies=0, workers=4):
    if max_studies:                    # a smoke scan must never be mistaken for a full one
        cache_path = cache_path.replace(".csv", f"_smoke{max_studies}.csv")
    if os.path.exists(cache_path):
        print(f"  series cache hit: {cache_path}")
        return pd.read_csv(cache_path)
    meta = pd.read_csv(series_csv)
    if max_studies:
        keep = meta.StudyInstanceUID.drop_duplicates().head(max_studies)
        meta = meta[meta.StudyInstanceUID.isin(set(keep))]
        print(f"  smoke: {len(meta)} series from {len(keep)} studies")
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             os.path.join(image_root, r.StudyInstanceUID, r.SeriesInstanceUID))
            for r in meta.itertuples(index=False)]
    t0 = time.time()
    rows = []
    for i, res in enumerate(parallel_map(scan_one_series, jobs, workers, chunksize=64)):
        if res is not None:
            rows.append(res)
        if (i + 1) % 4000 == 0:
            print(f"    {i+1}/{len(jobs)} series  {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"  scanned 0 series under {image_root} (cache NOT written)")
        return df
    # Trust the shipped plane (100% agreement with IOP on our sample) but fall back.
    shipped = meta.set_index("SeriesInstanceUID").get("Anatomical_Plane")
    if shipped is not None:
        sp = df.SeriesInstanceUID.map(shipped)
        df["plane"] = np.where(sp.isin(["Sagittal", "Coronal", "Axial"]), sp, df.plane_iop)
    else:
        df["plane"] = df.plane_iop
    df.to_csv(cache_path, index=False)
    print(f"  scanned {len(df)} series in {time.time()-t0:.0f}s -> {cache_path}")
    return df


SLOT_SPEC = {
    "SAG_FLUID_FS":   ("Sagittal", lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "COR_FLUID_FS":   ("Coronal",  lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "AX_FLUID_FS":    ("Axial",    lambda r: r.fluid and r.fat_sat, lambda r: r.fluid),
    "SAG_FLUID_NOFS": ("Sagittal", lambda r: r.fluid and not r.fat_sat, lambda r: r.fluid),
    "COR_T1":         ("Coronal",  lambda r: r.weighting == "T1", lambda r: not r.fluid),
    "SAG_T1":         ("Sagittal", lambda r: r.weighting == "T1", lambda r: not r.fluid),
}


def select_slots(sdf):
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


def study_side(sdf, dead_zone_mm):
    """('L'|'R'|'', tag, geometry, conflict) for one study."""
    tags = [t for t in sdf.laterality_tag.tolist() if t in ("L", "R")]
    tag = max(set(tags), key=tags.count) if tags else ""
    xs = sdf.centre_x_mm.dropna().to_numpy(dtype=float)
    geo = ""
    if len(xs):
        med = float(np.median(xs))
        if med > dead_zone_mm:
            geo = "L"
        elif med < -dead_zone_mm:
            geo = "R"
    conflict = int(bool(tag) and bool(geo) and tag != geo)
    side = tag if tag else geo
    if conflict:
        side = ""                       # do not mirror on contradictory evidence
    return side, tag, geo, conflict


def build_manifest(series_df, cfg):
    rows = []
    for study, sdf in series_df.groupby("StudyInstanceUID"):
        slots = select_slots(sdf)
        side, tag, geo, conflict = study_side(sdf, cfg.lat_dead_zone_mm)
        man = sdf.manufacturer.mode().iloc[0] if len(sdf.manufacturer.mode()) else ""
        model = sdf.model.mode().iloc[0] if len(sdf.model.mode()) else ""
        fov = (sdf.pixel_spacing * sdf.cols).median()
        rows.append({"StudyInstanceUID": study, **{s: slots.get(s, "") for s in SLOTS},
                     "n_slots": len(slots), "side": side, "side_tag": tag, "side_geo": geo,
                     "side_conflict": conflict, "manufacturer": man, "model": model,
                     "field_T": sdf.field_T.median(), "fov_mm": fov,
                     "institution_present": int(sdf.institution_present.max()),
                     "transfer_syntaxes": ";".join(sorted(set(sdf.transfer_syntax)))})
    m = pd.DataFrame(rows)
    print(f"  manifest: {len(m)} studies; mean slots {m.n_slots.mean():.2f}; "
          f"side tag present {(m.side_tag != '').mean():.1%}, geometry resolved "
          f"{(m.side_geo != '').mean():.1%}, conflicts {m.side_conflict.sum()}, "
          f"unresolved {(m.side == '').mean():.1%}; FOV median {m.fov_mm.median():.0f} mm, "
          f"< {cfg.crop_mm:.0f} mm in {(m.fov_mm < cfg.crop_mm).mean():.1%}")
    if (m.side_tag != "").any():
        both = m[(m.side_tag != "") & (m.side_geo != "")]
        agree = (both.side_tag == both.side_geo).mean() if len(both) else float("nan")
        print(f"  tag-vs-geometry agreement where both exist: {agree:.3f} (n={len(both)})")
    return m

# %% [markdown]
# ## Section 3: decode one study
#
# Per slot: order the slices spatially, keep the plane's central band, pick
# `n_slices` equidistant slices, read them, apply rescale/inversion, crop to
# `crop_mm` around the image centre, normalise 1/99 over the whole sampled stack of
# that series, resize to `px`, store uint8. Right knees are canonicalised to left.

# %%
# ── Section 3: decode ─────────────────────────────────────────────────────────
def ordered_slice_paths(series_dir, plane=None):
    """Spatially ordered slice paths. NEVER trust filename order.

    The sort key is the slice position along the stack normal. For SAGITTAL stacks the
    normal is taken as **+x in patient space (LPS, towards the patient's left)** rather
    than `cross(row, col)`, whose sign depends on the site's IOP handedness. With a
    fixed sign, "reverse the stack for right knees" canonicalises lateral->medial order
    for every study; with `cross()` it would randomise it between sites.
    """
    files = list_slices(series_dir)
    if not files:
        return [], None
    paths, heads = [], []
    for f in files:
        p = os.path.join(series_dir, f)
        try:
            heads.append(pydicom.dcmread(p, stop_before_pixels=True))
            paths.append(p)
        except Exception:
            continue                    # a stray non-DICOM file must not poison the order
    if not heads:
        return [], None
    first = heads[0]
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        if plane == "Sagittal":
            n = np.array([1.0, 0.0, 0.0])          # fixed sign: patient left = increasing
        elif n[int(np.argmax(np.abs(n)))] < 0:
            n = -n                                 # fixed sign for the other planes too
        keys, ok = [], True
        for h in heads:
            ipp = getattr(h, "ImagePositionPatient", None)
            if ipp is None:
                ok = False
                break
            keys.append(float(np.dot(np.array(ipp, float), n)))
        if ok:
            return [p for _, p in sorted(zip(keys, paths), key=lambda t: t[0])], first
    inst = [getattr(h, "InstanceNumber", None) for h in heads]
    if all(i is not None for i in inst):
        return [p for _, p in sorted(zip(inst, paths), key=lambda t: t[0])], first
    return paths, first


def read_plane(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def centre_crop_mm(arr, pixel_spacing, crop_mm):
    if not crop_mm or pixel_spacing is None or pixel_spacing <= 0:
        return arr
    side_px = int(round(crop_mm / pixel_spacing))
    h, w = arr.shape
    if side_px >= min(h, w):
        return arr                      # FOV smaller than the crop: keep everything
    y0 = (h - side_px) // 2
    x0 = (w - side_px) // 2
    return arr[y0:y0 + side_px, x0:x0 + side_px]


def resize_u8(stack01, px):
    """(S, H, W) float in [0,1] -> (S, px, px) uint8, bilinear, via torch if present."""
    try:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(stack01).unsqueeze(1)
        t = F.interpolate(t, size=(px, px), mode="bilinear", align_corners=False)
        return (t.squeeze(1).clamp_(0, 1) * 255).round().to(torch.uint8).numpy()
    except ImportError:
        from PIL import Image
        return np.stack([np.asarray(Image.fromarray((s * 255).astype(np.uint8))
                                    .resize((px, px), Image.BILINEAR)) for s in stack01])


def cache_series(series_dir, plane, cfg, is_right, n_slices):
    """-> ((n_slices, px, px) uint8, n_failed_slices) or (None, 0).

    `n_slices` is the slot's budget (uniform 16 in c01, ragged in c02); the band comes from
    `cfg.band[plane]`. IDENTICAL to src/kaggle_pipeline.py::cache_series -- keep in sync
    (src/cache_selftest.py checks both schemes bit for bit).

    Canonicalisation to a LEFT knee:
      * sagittal: stack sorted along +x (patient left); reversed for right knees, so
        every cached sagittal stack runs lateral -> medial.
      * coronal / axial: image columns are made to run towards the LATERAL side. The
        column direction in patient space is `iop[0]` (its x-component): columns
        running towards +x (patient left) is lateral for a left knee and medial for a
        right knee. So mirror when (columns run towards +x) XOR (left knee) -- i.e.
        `mirror = (iop[0] > 0) == is_right`.
    """
    ordered, head = ordered_slice_paths(series_dir, plane)
    if not ordered:
        return None, 0
    n = len(ordered)
    lo_f, hi_f = cfg.band.get(plane, (0.0, 1.0))
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
    stack = np.stack([centre_crop_mm(s, ps, cfg.crop_mm) for s in stack])
    lo, hi = np.percentile(stack, [cfg.pct_lo, cfg.pct_hi])   # per SERIES, whole stack
    stack = (np.clip(stack, lo, hi) - lo) / max(hi - lo, 1e-6)
    if mirror:
        stack = stack[:, :, ::-1]
    return resize_u8(np.ascontiguousarray(stack), cfg.px), n_fail


def cache_study(args):
    """c01: one (6, n_slices, px, px) .npy per study."""
    study, row, image_root, cfg, out_dir = args
    t0 = time.time()
    arr = np.zeros((len(SLOTS), cfg.n_slices, cfg.px, cfg.px), np.uint8)
    mask = np.zeros(len(SLOTS), np.uint8)
    is_right = row["side"] == "R"
    fails = 0
    for si, slot in enumerate(SLOTS):
        sid = row[slot]
        if not isinstance(sid, str) or not sid:
            continue
        d = os.path.join(image_root, study, sid)
        if not os.path.isdir(d):
            fails += 1
            continue
        plane = PLANE_OF_SLOT[slot]
        a, n_fail = cache_series(d, plane, cfg, is_right, cfg.n_slices)
        fails += n_fail
        if a is None:
            fails += 1
            continue
        arr[si] = a
        mask[si] = 1
    if mask.sum() == 0:
        return {"StudyInstanceUID": study, "cached": 0, "decode_fails": fails,
                "seconds": time.time() - t0}
    np.save(os.path.join(out_dir, f"{study}.npy"), arr)
    return {"StudyInstanceUID": study, "cached": 1, "n_slots_cached": int(mask.sum()),
            "mask": "".join(map(str, mask.tolist())), "decode_fails": fails,
            "seconds": time.time() - t0}


def build_study_flat(args):
    """c02: one study as a FLAT (sum(slot_slices), px, px) uint8 array; slot `si` occupies rows
    `SLOT_START[si] : SLOT_START[si] + slot_slices[si]`. Returns the array (None when no slot
    could be read) so the parent can pack it into a blob. Mirrors
    src/kaggle_pipeline.py::build_study_array for scheme c02 -- keep in sync."""
    study, row, image_root, cfg = args
    t0 = time.time()
    starts, total = slot_offsets(cfg.slot_slices)
    arr = np.zeros((total, cfg.px, cfg.px), np.uint8)
    mask = np.zeros(len(SLOTS), np.uint8)
    is_right = row["side"] == "R"
    fails = 0
    for si, slot in enumerate(SLOTS):
        sid = row[slot]
        if not isinstance(sid, str) or not sid:
            continue
        d = os.path.join(image_root, study, sid)
        if not os.path.isdir(d):
            fails += 1
            continue
        a, n_fail = cache_series(d, PLANE_OF_SLOT[slot], cfg, is_right, cfg.slot_slices[si])
        fails += n_fail
        if a is None:
            fails += 1
            continue
        arr[starts[si]:starts[si] + cfg.slot_slices[si]] = a
        mask[si] = 1
    meta = {"StudyInstanceUID": study, "cached": int(mask.sum() > 0),
            "n_slots_cached": int(mask.sum()), "mask": "".join(map(str, mask.tolist())),
            "decode_fails": fails, "seconds": time.time() - t0}
    return (arr if mask.sum() else None), meta


def blob_names(shard, k):
    return f"blob{shard:02d}_{k:03d}.npy", f"blob{shard:02d}_{k:03d}.csv"


def npy_header(path):
    """(shape, dtype, header_bytes) of a .npy file, public numpy API only. The training loader
    uses header_bytes to read ONE study out of a blob with a single seek+read instead of
    mmap-ing 520 MB inside every DataLoader worker."""
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


def blob_is_complete(out_dir, shard, k, n_expected):
    """A blob counts as done when the array AND its sidecar exist and both hold n_expected
    rows (a crash between the two writes, or a short partial write, is rebuilt)."""
    npy, csv = blob_names(shard, k)
    p_npy, p_csv = os.path.join(out_dir, npy), os.path.join(out_dir, csv)
    if not (os.path.exists(p_npy) and os.path.exists(p_csv)):
        return False
    try:
        shape, dtype, hdr = npy_header(p_npy)
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return (shape[0] == n_expected and os.path.getsize(p_npy) == hdr + n_bytes
                and len(pd.read_csv(p_csv)) == n_expected)
    except Exception:
        return False


def write_blob(out_dir, shard, k, arr, rows):
    """Atomic write of blob k: array + CSV sidecar go to *.tmp first, then os.replace, so a
    killed kernel never leaves a half-written blob that looks complete."""
    npy, csv = blob_names(shard, k)
    p_npy, p_csv = os.path.join(out_dir, npy), os.path.join(out_dir, csv)
    tmp_npy = p_npy + ".tmp.npy"
    np.save(tmp_npy, arr)
    os.replace(tmp_npy, p_npy)
    tmp_csv = p_csv + ".tmp"
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, p_csv)

# %% [markdown]
# ## Section 4: run
#
# Studies are assigned to shards by a hash of their UID, so the split is stable
# across kernels and independent of fold assignment.

# %%
# ── Section 4: run ────────────────────────────────────────────────────────────
if os.environ.get("RSNA_DEFS_ONLY"):
    raise SystemExit(0)          # src/cache_selftest.py imports the definitions above and stops here

TRAIN_IMG = os.path.join(COMP, "train_series")
series_csv = os.path.join(COMP, "train_series.csv")
if not os.path.isdir(TRAIN_IMG) and os.path.isdir(os.path.join(COMP, "sample_dicom", "test_series")):
    TRAIN_IMG = os.path.join(COMP, "sample_dicom", "test_series")     # local sample
    series_csv = os.path.join(COMP, "test_series.csv")
print(f"images: {TRAIN_IMG}")

series_df = scan_series(series_csv, TRAIN_IMG, os.path.join(WORK, "series_meta.csv"),
                        max_studies=cfg.smoke_max_studies if cfg.smoke else 0,
                        workers=cfg.workers)
if len(series_df) == 0:
    raise SystemExit("no series found -- wrong image root")
manifest = build_manifest(series_df, cfg)


def shard_of(uid):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % N_SHARDS


manifest["shard"] = manifest.StudyInstanceUID.map(shard_of)
manifest["cache_version"] = CACHE_VERSION
manifest.to_csv(os.path.join(WORK, "manifest.csv"), index=False)
todo = manifest[manifest.shard == SHARD]
print(f"shard {SHARD}/{N_SHARDS}: {len(todo)} of {len(manifest)} studies")

out_dir = os.path.join(WORK, CACHE_VERSION)
os.makedirs(out_dir, exist_ok=True)
# c01 keeps its historical file names (manifest_shard{k}.csv next to the version dir); c02 adds
# the scheme suffix so a local run of both schemes into one WORK never overwrites the other's
# manifest. The loader globs `manifest_shard*.csv` and reads the `cache_version` column.
suffix = "" if cfg.scheme == "c01" else f"_{cfg.scheme}"
manifest_shard_path = os.path.join(WORK, f"manifest_shard{SHARD}{suffix}.csv")
log_path = os.path.join(WORK, f"cache_log_shard{SHARD}{suffix}.csv")
t0 = time.time()

if cfg.scheme == "c01":
    jobs = [(r.StudyInstanceUID, r.to_dict(), TRAIN_IMG, cfg, out_dir)
            for _, r in todo.iterrows()
            if not os.path.exists(os.path.join(out_dir, f"{r.StudyInstanceUID}.npy"))]
    print(f"  {len(todo) - len(jobs)} already cached, {len(jobs)} to do")
    logs = []
    for i, res in enumerate(parallel_map(cache_study, jobs, cfg.workers)):
        logs.append(res)
        if (i + 1) in (10, 50) or (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"    {i+1}/{len(jobs)} studies  {dt:.0f}s  {dt/(i+1):.2f} s/study  "
                  f"ETA {dt/(i+1)*(len(jobs)-i-1)/60:.0f} min")
    log_df = pd.DataFrame(logs)
    if len(log_df):
        log_df.to_csv(log_path, index=False)
        # The presence mask lives in the manifest too, so the loader needs one file per shard
        # and never has to infer presence from "is the slot all zeros".
        m2 = manifest.merge(log_df[["StudyInstanceUID", "cached", "mask", "decode_fails"]]
                            if "mask" in log_df else log_df[["StudyInstanceUID", "cached", "decode_fails"]],
                            on="StudyInstanceUID", how="left")
        m2.to_csv(manifest_shard_path, index=False)
        n_ok = int(log_df.cached.sum())
        print(f"\ncached {n_ok}/{len(log_df)} studies in {(time.time()-t0)/60:.1f} min; "
              f"decode failures {int(log_df.decode_fails.sum())}; "
              f"mean {log_df.seconds.mean():.2f} s/study")
        if n_ok == 0:
            raise SystemExit("cached nothing -- refusing to leave an empty cache")
else:
    # c02: deterministic blobs. Studies of this shard sorted by UID and chunked by blob_size;
    # EVERY study of a group gets a row (an unreadable one stays zeros with cached=0), so
    # `row` is derivable from the manifest alone and never shifts on a rebuild. Resume is per
    # blob: a complete blob (array + sidecar, both with the group's row count) is skipped.
    uids = sorted(todo.StudyInstanceUID.tolist())
    rows_by_uid = {r.StudyInstanceUID: r.to_dict() for _, r in todo.iterrows()}
    groups = [uids[i:i + cfg.blob_size] for i in range(0, len(uids), cfg.blob_size)]
    done = [k for k, g in enumerate(groups) if blob_is_complete(out_dir, SHARD, k, len(g))]
    n_todo = sum(len(g) for k, g in enumerate(groups) if k not in done)
    print(f"  {len(groups)} blobs of <= {cfg.blob_size} studies; {len(done)} complete, "
          f"{len(groups) - len(done)} to build ({n_todo} studies)")
    n_done_studies = 0
    for k, grp in enumerate(groups):
        if k in done:
            continue
        arr = np.zeros((len(grp), N_TOTAL_SLICES, cfg.px, cfg.px), np.uint8)
        rows = []
        jobs = [(u, rows_by_uid[u], TRAIN_IMG, cfg) for u in grp]
        for i, (a, meta) in enumerate(parallel_map(build_study_flat, jobs, cfg.workers)):
            if a is not None:
                arr[i] = a
            rows.append({**meta, "row": i, "blob": blob_names(SHARD, k)[0]})
            n_done_studies += 1
            if n_done_studies in (10, 50) or n_done_studies % 200 == 0:
                dt = time.time() - t0
                print(f"    {n_done_studies}/{n_todo} studies  {dt:.0f}s  {dt/n_done_studies:.2f} s/study  "
                      f"ETA {dt/n_done_studies*(n_todo-n_done_studies)/60:.0f} min")
        write_blob(out_dir, SHARD, k, arr, rows)
        del arr
        if time.time() - T_START > 11.5 * 3600:
            print("!! approaching the 12 h CPU-kernel limit -- stopping after a complete blob; "
                  "re-run this kernel with its own output attached to resume")
            break
    # Sidecars are the log; the shard manifest is rebuilt from ALL of them every run, so a
    # resume-only run (nothing new to build) still leaves a complete manifest_shard file.
    sidecars = sorted(f for f in os.listdir(out_dir)
                      if f.startswith(f"blob{SHARD:02d}_") and f.endswith(".csv"))
    log_df = pd.concat([pd.read_csv(os.path.join(out_dir, f), dtype={"mask": str}) for f in sidecars],
                       ignore_index=True) if sidecars else pd.DataFrame()
    if len(log_df) == 0 or int(log_df.cached.sum()) == 0:
        raise SystemExit("cached nothing -- refusing to leave an empty cache")
    log_df.to_csv(log_path, index=False)
    m2 = manifest.merge(log_df[["StudyInstanceUID", "cached", "mask", "decode_fails", "blob", "row"]],
                        on="StudyInstanceUID", how="inner")
    m2.to_csv(manifest_shard_path, index=False)
    n_ok = int(log_df.cached.sum())
    print(f"\ncached {n_ok}/{len(log_df)} studies of shard {SHARD} ({len(sidecars)} blobs) in "
          f"{(time.time()-t0)/60:.1f} min this run; decode failures {int(log_df.decode_fails.sum())}; "
          f"mean {log_df.seconds.mean():.2f} s/study")
    if len(m2) < len(todo):
        print(f"!! manifest_shard covers {len(m2)}/{len(todo)} studies -- shard incomplete, resume needed")

size_gb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e9
print(f"cache dir {out_dir}: {len(os.listdir(out_dir))} files, {size_gb:.2f} GB -> {manifest_shard_path}")
if len(manifest) and not cfg.smoke:
    n_cached_here = int(pd.read_csv(manifest_shard_path).cached.sum()) if os.path.exists(manifest_shard_path) else 0
    est = size_gb / max(n_cached_here, 1) * len(manifest) / N_SHARDS
    print(f"  projected full shard size {est:.1f} GB (Kaggle output cap ~20 GB)")
print(f"total elapsed {(time.time()-T_START)/60:.1f} min")
