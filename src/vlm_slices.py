# %% [markdown]
# # RSNA Knee — slice grids for a multimodal-LLM labelling pilot (gold-58)
#
# For each selected study, every usable series becomes ONE 3×3 grid image of nine slices spread over the
# series' central band, in a fixed, stated anatomical order, plus a JSON sidecar that says what the grid
# shows (plane, sequence text, knee side, tile order, display convention). A labeller (a multimodal LLM)
# reads the grids together with the report.
#
# Same rules as the cache (`src/cache_pipeline.py`): slices are ordered by `ImagePositionPatient · normal`,
# never by filename; rescale / MONOCHROME1 handled; per-series 0.5 / 99.5 percentile window; knee side from
# the Laterality tag, else from the geometry (20 mm dead zone). Unlike the cache, images are NOT mirrored to
# a canonical left knee: they are put in the standard radiological display, and the sidecar tells the reader
# which image side is medial.
#
# Local run (smoke): the 3 sample test studies in `data/sample_dicom/test_series` → `artifacts/vlm_slices_smoke/`.
# Kaggle run: the 58 gold training studies → `/kaggle/working/vlm_gold/`.

# %%
import glob
import json
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

ON_KAGGLE = os.path.exists("/kaggle/input")
N_TILES = 9                  # 3 x 3 grid
TILE_PX = 384                # grid = 1152 x 1152
BAND = (0.08, 0.92)          # central band of each stack the nine slices are spread over
MAX_SERIES = 8
MIN_SLICES = 5               # localisers / screenshots are skipped
DEAD_ZONE_MM = 20.0
FLUID_TOKENS = ("t2", "stir", "pd", "dess", "spair", "spir", "tirm")
FATSAT_TOKENS = ("fs", "fatsat", "fat_sat", "stir", "spir", "spair", "tirm", "dixon", "water", "fsat")


def resolve_dir(candidates, must_contain=None):
    for c in candidates:
        if c and os.path.isdir(c) and (not must_contain or os.path.exists(os.path.join(c, must_contain))):
            return c
    return None


if ON_KAGGLE:
    COMP = resolve_dir(["/kaggle/input/rsna-knee-abnormality-detection",
                        "/kaggle/input/competitions/rsna-knee-abnormality-detection"], must_contain="train.csv")
    if COMP is None:
        hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        COMP = os.path.dirname(hits[0]) if hits else None
    if COMP is None:
        raise SystemExit("attach the competition to this kernel")
    IMAGE_ROOT = os.path.join(COMP, "train_series")
    SERIES_CSV = os.path.join(COMP, "train_series.csv")
    tr = pd.read_csv(os.path.join(COMP, "train.csv"))
    LABELS = [c for c in tr.columns if c not in ("StudyInstanceUID", "Report")]
    STUDIES = tr.loc[tr[LABELS].notna().all(axis=1), "StudyInstanceUID"].tolist()   # the 58 gold studies
    OUT = "/kaggle/working/vlm_gold"
else:
    COMP = "data"
    IMAGE_ROOT = "data/sample_dicom/test_series"
    SERIES_CSV = "data/test_series.csv"
    STUDIES = sorted(os.listdir(IMAGE_ROOT))
    OUT = "artifacts/vlm_slices_smoke"
os.makedirs(OUT, exist_ok=True)
print(f"ON_KAGGLE={ON_KAGGLE} COMP={COMP} IMAGE_ROOT={IMAGE_ROOT} studies={len(STUDIES)} OUT={OUT}")


# %%
def has_token(text, tokens):
    t = text.lower().replace("-", "").replace(" ", "")
    return any(tok.replace("_", "") in t for tok in tokens)


def list_slices(d):
    files = [f for f in os.listdir(d) if f.endswith(".dcm")]
    return files or [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]


def plane_from_iop(iop):
    if iop is None or len(iop) != 6:
        return "unknown"
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(np.argmax(np.abs(n)))]


def centre_x_mm(h):
    ipp, iop, ps = (getattr(h, k, None) for k in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"))
    rows, cols = getattr(h, "Rows", None), getattr(h, "Columns", None)
    if None in (ipp, iop, ps, rows, cols) or len(iop) != 6:
        return None
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    return float((np.array(ipp, float) + r * cols / 2 * float(ps[1]) + c * rows / 2 * float(ps[0]))[0])


def read_series(d):
    """-> (headers, paths) in the order read; non-DICOM files skipped."""
    heads, paths = [], []
    for f in list_slices(d):
        p = os.path.join(d, f)
        try:
            heads.append(pydicom.dcmread(p, stop_before_pixels=True))
            paths.append(p)
        except Exception:
            continue
    return heads, paths


def spatial_order(heads, paths, plane):
    """Paths sorted by position along a FIXED-SIGN axis: +x (patient left) for sagittal, +y (posterior)
    for coronal, +z (superior) for axial. Falls back to InstanceNumber. Never filename."""
    axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[plane]
    if all(getattr(h, "ImagePositionPatient", None) is not None for h in heads):
        keys = [float(h.ImagePositionPatient[axis]) for h in heads]
        return [p for _, p in sorted(zip(keys, paths), key=lambda t: t[0])], "position"
    inst = [getattr(h, "InstanceNumber", None) for h in heads]
    if all(i is not None for i in inst):
        return [p for _, p in sorted(zip(inst, paths), key=lambda t: t[0])], "instance"
    return paths, "unordered"


def read_plane(path):
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    if a.ndim == 3:
        a = a[a.shape[0] // 2]
    a = a * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    return a


def display_flips(plane, iop):
    """(flip_lr, flip_ud) that put a slice in the standard display. Columns run along iop[:3], rows along iop[3:].
    Sagittal: anterior on the image left, superior up. Coronal / axial: radiological (patient's right on the
    image left), superior up (coronal) / anterior up (axial)."""
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    if plane == "Sagittal":
        return r[1] < 0, c[2] > 0
    if plane == "Coronal":
        return r[0] < 0, c[2] > 0
    return r[0] < 0, c[1] < 0


def tile(a, px):
    h, w = a.shape
    s = px / max(h, w)
    im = Image.fromarray(a).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    canvas = Image.new("L", (px, px), 0)
    canvas.paste(im, ((px - im.width) // 2, (px - im.height) // 2))
    return canvas


# %%
series_df = pd.read_csv(SERIES_CSV)
summary = []
for n_study, study in enumerate(STUDIES):
    sdir = os.path.join(IMAGE_ROOT, study)
    if not os.path.isdir(sdir):
        print(f"MISSING study dir {study}")
        summary.append({"StudyInstanceUID": study, "n_series": 0, "error": "missing dir"})
        continue
    series = []
    for sid in sorted(os.listdir(sdir)):
        heads, paths = read_series(os.path.join(sdir, sid))
        if len(heads) < MIN_SLICES:
            continue
        h = heads[0]
        plane = plane_from_iop(getattr(h, "ImageOrientationPatient", None))
        if plane == "unknown":
            continue
        desc = " ".join(str(getattr(h, k, "") or "") for k in ("SeriesDescription", "ProtocolName")).strip()
        lat = str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "").upper()
        series.append({"sid": sid, "heads": heads, "paths": paths, "plane": plane, "desc": desc,
                       "fluid": int(has_token(desc, FLUID_TOKENS)), "fat_sat": int(has_token(desc, FATSAT_TOKENS)),
                       "lat": lat if lat in ("L", "R") else "", "cx": centre_x_mm(h)})
    # knee side: majority tag, else geometry (LPS +x = patient's left), as in the cache
    tags = [s["lat"] for s in series if s["lat"]]
    tag = max(set(tags), key=tags.count) if tags else ""
    xs = [s["cx"] for s in series if s["cx"] is not None]
    geo = ""
    if xs:
        med = float(np.median(xs))
        geo = "L" if med > DEAD_ZONE_MM else ("R" if med < -DEAD_ZONE_MM else "")
    side = "" if (tag and geo and tag != geo) else (tag or geo)
    side_src = "tag" if tag and side else ("geometry" if side else ("conflict" if tag and geo else "unknown"))
    order = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    series.sort(key=lambda s: (order[s["plane"]], -s["fat_sat"], -s["fluid"], s["desc"]))
    series = series[:MAX_SERIES]

    out_dir = os.path.join(OUT, study)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"StudyInstanceUID": study, "knee_side": {"L": "LEFT", "R": "RIGHT"}.get(side, "UNKNOWN"),
            "knee_side_source": side_src, "series": []}
    n_fail = 0
    for k, s in enumerate(series):
        paths, how = spatial_order(s["heads"], s["paths"], s["plane"])
        n = len(paths)
        lo, hi = int(round(BAND[0] * (n - 1))), int(round(BAND[1] * (n - 1)))
        idx = np.linspace(lo, hi, N_TILES).round().astype(int)
        # tile order in plain anatomy
        if s["plane"] == "Sagittal":
            if side == "L":            # left knee: medial side towards the patient's right = smaller x
                order_txt = "tile 1 = most MEDIAL slice ... tile 9 = most LATERAL slice"
            elif side == "R":          # right knee: medial side towards the patient's left = larger x
                idx = idx[::-1]
                order_txt = "tile 1 = most MEDIAL slice ... tile 9 = most LATERAL slice"
            else:
                order_txt = "tile 1 = towards the patient's RIGHT ... tile 9 = towards the patient's LEFT (knee side unknown)"
            disp_txt = "each tile: anterior on the image LEFT, superior UP"
        elif s["plane"] == "Coronal":
            order_txt = "tile 1 = most ANTERIOR slice ... tile 9 = most POSTERIOR slice"
            med = {"L": "image LEFT", "R": "image RIGHT"}.get(side, "unknown")
            disp_txt = f"each tile: radiological display (patient's right on the image left), superior UP; medial side of this knee = {med}"
        else:
            idx = idx[::-1]            # +z = superior: show superior -> inferior
            order_txt = "tile 1 = most SUPERIOR slice ... tile 9 = most INFERIOR slice"
            med = {"L": "image LEFT", "R": "image RIGHT"}.get(side, "unknown")
            disp_txt = f"each tile: radiological display (patient's right on the image left), anterior UP; medial side of this knee = {med}"
        iop = getattr(s["heads"][0], "ImageOrientationPatient", None)
        flr, fud = display_flips(s["plane"], iop) if iop is not None and len(iop) == 6 else (False, False)
        planes = []
        for i in idx:
            try:
                planes.append(read_plane(paths[int(i)]))
            except Exception:
                planes.append(None)
                n_fail += 1
        good = [a for a in planes if a is not None]
        if not good:
            continue
        stack = np.concatenate([a.ravel() for a in good])
        p_lo, p_hi = np.percentile(stack, [0.5, 99.5])
        grid = Image.new("L", (3 * TILE_PX, 3 * TILE_PX), 0)
        draw = ImageDraw.Draw(grid)
        for t, a in enumerate(planes):
            if a is None:
                continue
            u8 = (np.clip((a - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            if flr:
                u8 = u8[:, ::-1]
            if fud:
                u8 = u8[::-1, :]
            x0, y0 = (t % 3) * TILE_PX, (t // 3) * TILE_PX
            grid.paste(tile(np.ascontiguousarray(u8), TILE_PX), (x0, y0))
            draw.rectangle([x0, y0, x0 + 22, y0 + 14], fill=0)
            draw.text((x0 + 4, y0 + 2), str(t + 1), fill=255)
        fn = f"series_{k + 1}.jpg"
        grid.convert("RGB").save(os.path.join(out_dir, fn), quality=88)
        meta["series"].append({"file": fn, "plane": s["plane"], "description": s["desc"],
                               "fat_suppressed_by_name": s["fat_sat"], "fluid_sensitive_by_name": s["fluid"],
                               "n_slices_in_series": n, "slice_order_source": how,
                               "tile_order": order_txt, "display": disp_txt})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    summary.append({"StudyInstanceUID": study, "n_series": len(meta["series"]), "side": side, "side_source": side_src,
                    "failed_slices": n_fail, "error": ""})
    print(f"[{n_study + 1}/{len(STUDIES)}] {len(meta['series'])} grids, side {side or '?'} ({side_src}), failed slices {n_fail}")

S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(f"DONE: {len(S)} studies, {int(S.n_series.sum())} grids, studies with 0 grids {(S.n_series == 0).sum()}, "
      f"failed slices {int(S.failed_slices.fillna(0).sum())}, side unknown {(S.get('side', pd.Series(dtype=str)).fillna('') == '').sum()}")
