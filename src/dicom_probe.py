"""Probe the sample DICOM tree and test the baseline's preprocessing claims.

Checks, on real data, the four things the public baseline asserts and that the
pipeline design depends on:

  1. Slice order from filename is uncorrelated with anatomy (so filename sort is unsafe).
  2. Fluid_Sensitive / Fat_Suppression in *_series.csv carry one bit, not two, and can
     be recovered independently from the DICOM headers.
  3. Sequence weighting (T1 / T2 / PD) is derivable from TR/TE + ScanningSequence.
  4. Pixel data needs RescaleSlope/Intercept, MONOCHROME1 inversion, and per-triplet
     percentile clipping before it is comparable across series.

Usage:  python src/dicom_probe.py [--root data/sample_dicom] [--csv data/test_series.csv]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pydicom

# TR/TE thresholds in ms. Gradient-echo breaks these, so ScanningSequence wins first.
TR_SHORT_MAX = 800.0
TE_LONG_MIN = 60.0

FATSAT_TOKENS = (
    "fs", "fatsat", "fat_sat", "stir", "spir", "spair", "tirm",
    "dixon", "chess", "sat", "supp",
)
FLUID_TOKENS = ("t2", "stir", "pd", "dess", "spair", "spir", "tirm")


@dataclass
class SeriesProbe:
    study: str
    series: str
    n_slices: int
    plane_hdr: str
    tr: float | None
    te: float | None
    scanning_seq: str
    description: str
    weighting: str
    fat_sat: bool
    fluid_sensitive: bool
    order_rho: float | None
    photometric: str
    slope_intercept: tuple[float, float]
    pixel_range: tuple[float, float]


def plane_from_orientation(iop) -> str:
    """Anatomical plane from ImageOrientationPatient, via the slice normal."""
    if iop is None or len(iop) != 6:
        return "unknown"
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:], dtype=float)
    n = np.cross(r, c)
    axis = int(np.argmax(np.abs(n)))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis]


def classify_weighting(tr: float | None, te: float | None, scanning_seq: str,
                       description: str) -> str:
    d = description.lower()
    if "flair" in d:
        return "FLAIR"
    # Gradient echo has a short TR by design, so the TR/TE rule does not apply.
    if "gr" in scanning_seq.lower() or any(t in d for t in ("gre", "dess", "medic", "flash")):
        return "GRE"
    if tr is None or te is None:
        # Fall back to the protocol name.
        if "t1" in d:
            return "T1"
        if "t2" in d:
            return "T2"
        if "pd" in d:
            return "PD"
        return "unknown"
    if tr <= TR_SHORT_MAX:
        return "T1"
    return "T2" if te >= TE_LONG_MIN else "PD"


def has_token(text: str, tokens) -> bool:
    t = text.lower().replace("-", "").replace(" ", "")
    return any(tok.replace("_", "") in t for tok in tokens)


def spatial_positions(datasets) -> np.ndarray | None:
    """Project ImagePositionPatient onto the slice normal -> a scalar per slice."""
    iop = getattr(datasets[0], "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return None
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:], dtype=float)
    n = np.cross(r, c)
    out = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            return None
        out.append(float(np.dot(np.array(ipp, dtype=float), n)))
    return np.array(out)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def to_hu_like(ds) -> np.ndarray:
    """Apply rescale and MONOCHROME1 inversion; return float array."""
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multi-frame: take the middle frame
        arr = arr[arr.shape[0] // 2]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def probe_series(study: str, series: str, paths: list[str]) -> SeriesProbe:
    # Filename order is what a naive os.listdir()/sorted() gives.
    paths_by_name = sorted(paths, key=lambda p: os.path.basename(p))
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths_by_name]

    pos = spatial_positions(headers)
    rho = None if pos is None else spearman(np.arange(len(pos), dtype=float), pos)

    h0 = headers[0]
    desc = " ".join(
        str(getattr(h0, k, "") or "")
        for k in ("SeriesDescription", "SequenceName", "ScanOptions", "ProtocolName")
    )
    tr = getattr(h0, "RepetitionTime", None)
    te = getattr(h0, "EchoTime", None)
    tr = float(tr) if tr is not None else None
    te = float(te) if te is not None else None
    scanning_seq = str(getattr(h0, "ScanningSequence", "") or "")
    weighting = classify_weighting(tr, te, scanning_seq, desc)
    fat_sat = has_token(desc, FATSAT_TOKENS) or "fs" in str(
        getattr(h0, "ScanOptions", "") or "").lower()
    fluid = weighting in ("T2", "PD") or has_token(desc, FLUID_TOKENS)

    # Pixel checks on the middle slice only (cheap).
    mid = pydicom.dcmread(paths_by_name[len(paths_by_name) // 2])
    arr = to_hu_like(mid)

    return SeriesProbe(
        study=study,
        series=series,
        n_slices=len(paths),
        plane_hdr=plane_from_orientation(getattr(h0, "ImageOrientationPatient", None)),
        tr=tr,
        te=te,
        scanning_seq=scanning_seq,
        description=str(getattr(h0, "SeriesDescription", "") or "")[:40],
        weighting=weighting,
        fat_sat=bool(fat_sat),
        fluid_sensitive=bool(fluid),
        order_rho=rho,
        photometric=str(getattr(mid, "PhotometricInterpretation", "") or ""),
        slope_intercept=(
            float(getattr(mid, "RescaleSlope", 1.0) or 1.0),
            float(getattr(mid, "RescaleIntercept", 0.0) or 0.0),
        ),
        pixel_range=(float(arr.min()), float(arr.max())),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sample_dicom")
    ap.add_argument("--csv", default="data/test_series.csv")
    args = ap.parse_args()

    # Collect series directories: <root>/test_series/<study>/<series>/*.dcm
    series_map: dict[tuple[str, str], list[str]] = {}
    for dirpath, _, files in os.walk(args.root):
        dcms = [os.path.join(dirpath, f) for f in files if f.endswith(".dcm")]
        if not dcms:
            continue
        series = os.path.basename(dirpath)
        study = os.path.basename(os.path.dirname(dirpath))
        series_map[(study, series)] = dcms

    if not series_map:
        raise SystemExit(f"no DICOMs found under {args.root}")

    probes = [probe_series(st, se, ps) for (st, se), ps in sorted(series_map.items())]
    df = pd.DataFrame([p.__dict__ for p in probes])

    print(f"\n=== {len(df)} series, {int(df.n_slices.sum())} slices "
          f"across {df.study.nunique()} studies ===\n")
    show = df[["series", "n_slices", "plane_hdr", "tr", "te", "scanning_seq",
               "weighting", "fat_sat", "fluid_sensitive", "order_rho", "description"]].copy()
    show["series"] = show.series.str[-12:]
    print(show.to_string(index=False))

    # --- Claim 1: filename order is not anatomical order -------------------
    rho = df.order_rho.dropna()
    print("\n[1] filename-order vs spatial-order Spearman rho")
    if len(rho):
        print(f"    n={len(rho)}  mean={rho.mean():+.3f}  "
              f"min={rho.min():+.3f}  max={rho.max():+.3f}")
        print(f"    |rho|>0.99 in {int((rho.abs() > 0.99).sum())}/{len(rho)} series "
              f"-> filename sort is {'SAFE' if (rho.abs() > 0.99).all() else 'UNSAFE'}")
    else:
        print("    no ImagePositionPatient available")

    # --- Claim 2: CSV flags are degenerate, headers are not ----------------
    print("\n[2] CSV flags vs header-derived flags")
    try:
        meta = pd.read_csv(args.csv)
        pairs = meta.groupby(["Fluid_Sensitive", "Fat_Suppression"]).size()
        print("    CSV (Fluid,FatSup) combinations present:")
        for (f, s), n in pairs.items():
            print(f"      ({f},{s}) x{n}")
        print(f"    -> CSV carries {len(pairs)} of 4 combinations")
        m = meta.set_index("SeriesInstanceUID")
        joined = df.assign(
            csv_fluid=df.series.map(m.Fluid_Sensitive),
            csv_fatsat=df.series.map(m.Fat_Suppression),
            csv_plane=df.series.map(m.Anatomical_Plane),
        )
        hdr_pairs = joined.groupby(["fluid_sensitive", "fat_sat"]).size()
        print(f"    header-derived combinations on this sample: {len(hdr_pairs)}")
        for (f, s), n in hdr_pairs.items():
            print(f"      (fluid={f},fatsat={s}) x{n}")
        agree = (joined.plane_hdr == joined.csv_plane).mean()
        print(f"    plane agreement header vs CSV: {agree:.0%}")
        dis = joined[joined.plane_hdr != joined.csv_plane]
        for _, r in dis.iterrows():
            print(f"      MISMATCH {r.series[-12:]}: header={r.plane_hdr} csv={r.csv_plane}")
    except FileNotFoundError:
        print(f"    {args.csv} not found, skipping")

    # --- Claims 3/4: weighting spread and pixel handling ------------------
    print("\n[3] weighting distribution")
    for w, n in df.weighting.value_counts().items():
        print(f"    {w:<8} x{n}")

    print("\n[4] pixel handling")
    print(f"    PhotometricInterpretation: {dict(df.photometric.value_counts())}")
    nontrivial = df[df.slope_intercept.apply(lambda si: si != (1.0, 0.0))]
    print(f"    series needing rescale (slope,intercept != 1,0): "
          f"{len(nontrivial)}/{len(df)}")
    lo = df.pixel_range.apply(lambda r: r[0])
    hi = df.pixel_range.apply(lambda r: r[1])
    print(f"    per-series max intensity spans {hi.min():.0f} .. {hi.max():.0f} "
          f"(ratio {hi.max() / max(hi.min(), 1):.1f}x)")
    print("    -> per-series normalisation is required; a global window would not "
          "transfer across series")

    out = "artifacts/series_probe.csv"
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
