"""P-81: OAI baseline knees -> a c03-format cache shard that the training loader indexes beside the competition shards.

The loader (`load_cache_manifests` in src/kaggle_pipeline.py) globs every `manifest_shard*.csv` under its root and keys the
studies by the manifest's `cache_version`. This script writes `manifest_shard90_oai.csv` + `<version>/blob90_KKK.npy|csv`
with the c03 version string, so an arm with `oai=True` reads the OAI knees exactly like competition studies. The arrays
are built by src/cache_pipeline.py's own `build_study_flat` / `cache_series` (loaded with the c03 settings), so the
preprocessing is byte-for-byte the competition cache's; only the series choice is OAI-specific:

    slot SAG_FLUID_FS <- SAG_IW_TSE  (sagittal intermediate-weighted TSE, fat-suppressed)
    slot COR_FLUID_FS <- COR_MPR     (coronal 1.5 mm reformat of the water-excited 3D DESS)
    slot AX_FLUID_FS  <- AX_MPR      (axial reformat of the same)
    SAG_FLUID_NOFS, COR_T1, SAG_T1 empty (the presence mask)

That is what our own slot rules pick once the headers are read correctly (a DESS-WE reformat is fluid-bright and
fat-suppressed). Two OAI quirks are handled here, both checked on 6 sample knees on 2026-10-08:
  * each MPR series starts with ONE reference image in another orientation (file 001; the real slices are 002.., 1.5 mm
    apart). The cache builder takes the plane and the stack normal from the first header, so that image would misfile
    the series. It is deleted before the build: every file whose plane differs from the series' majority plane.
  * OAI headers carry no Laterality tag. The side comes from the series name (_RIGHT / _LEFT); the header geometry agreed
    on all 6 sample knees and is checked again per knee (a disagreement is logged and the knee keeps the name's side).
A knee-series scanned twice keeps the file that sorts last (the later archive id).

Modes (run from the repo root; data/oai/ is gitignored and the OAI terms forbid redistributing anything below):
  --links  OUT.txt      write the S3 links of the chosen series for `downloadcmd -t` (scripts/nda_run.py)
  --build  TAR_ROOT     build the shard from the downloaded tarballs (TAR_ROOT holds image03/00m/... as downloadcmd lays
                        them out) into --out (default artifacts/cache_local/oai, which the local smoke indexes);
                        --delete-tars removes each knee's tarballs once it is built (peak disk on a pod)
Common: --targets artifacts/oai/oai_targets.csv (the knee list), --image03 data/oai/nda_pkg_meta/image03.txt,
        --limit N (first N knees, for a smoke), --workers N.
"""
import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES_SLOT = {"SAG_IW_TSE": "SAG_FLUID_FS", "COR_MPR": "COR_FLUID_FS", "AX_MPR": "AX_FLUID_FS"}
SERIES_PLANE = {"SAG_IW_TSE": "Sagittal", "COR_MPR": "Coronal", "AX_MPR": "Axial"}
SHARD = 90
C03_ENV = {"RSNA_CACHE_SCHEME": "c02", "RSNA_SLOT_SLICES": "24,24,24,14,8,8", "RSNA_CROP_MM": "150"}
C03_VERSION = "c02_p336_b24-24-24-14-8-8_band2-98_crop150_lat20"

_DEFS = None


def defs():
    """src/cache_pipeline.py's definitions with the c03 settings (its run section is skipped: RSNA_DEFS_ONLY)."""
    global _DEFS
    if _DEFS is None:
        os.environ.update(C03_ENV)
        os.environ["RSNA_DEFS_ONLY"] = "1"
        path = os.path.join(ROOT, "src", "cache_pipeline.py")
        ns = {"__name__": "defs_cache_pipeline", "__file__": path}
        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, "exec")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(code, ns)
            except SystemExit as e:
                if e.code not in (0, None):
                    raise
        if ns["CACHE_VERSION"] != C03_VERSION:
            raise SystemExit(f"cache version {ns['CACHE_VERSION']} != c03 {C03_VERSION}: the c03 settings did not apply")
        _DEFS = ns
    return _DEFS


def choose_series(image03_path, targets_path, limit=0):
    """One row per (knee, series): StudyInstanceUID, series, slot, side, s3 url, download alias."""
    knees = pd.read_csv(targets_path, usecols=["StudyInstanceUID"]).StudyInstanceUID
    if limit:
        knees = knees.head(limit)
    knees = set(knees)
    im = pd.read_csv(image03_path, sep="\t", skiprows=[1], low_memory=False, dtype=str)
    desc = im["image_description"].fillna("")
    im["series"] = desc.str.rsplit("_", n=1).str[0]
    im["side"] = desc.str.rsplit("_", n=1).str[-1].map({"RIGHT": "R", "LEFT": "L"})
    im["StudyInstanceUID"] = "OAI_" + im["src_subject_id"].str.strip() + "_" + im["side"].fillna("?")
    sel = im[im.StudyInstanceUID.isin(knees) & im.series.isin(SERIES_SLOT)].copy()
    sel = sel.sort_values(["StudyInstanceUID", "series", "image_file"]).groupby(["StudyInstanceUID", "series"]).tail(1)
    m = sel["image_file"].str.extract(r"^s3://[^/]+/submission_\d+/(.+)$")[0]
    if m.isna().any():
        raise SystemExit(f"unexpected S3 url layout, e.g. {sel.loc[m.isna(), 'image_file'].iloc[0]}")
    sel["alias"] = "image03/" + m
    sel["slot"] = sel.series.map(SERIES_SLOT)
    out = sel[["StudyInstanceUID", "series", "slot", "side", "image_file", "alias"]].reset_index(drop=True)
    have = out.groupby("StudyInstanceUID").series.nunique()
    print(f"chosen: {len(out)} series for {have.size} of {len(knees)} knees; all 3 series for {(have == 3).sum()}")
    return out


def plane_of(iop):
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[int(np.argmax(np.abs(n)))]


def clean_series(d, expected_plane):
    """Delete every file whose plane differs from the series' majority plane. -> (majority plane, n_removed)."""
    import pydicom
    planes = {}
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            h = pydicom.dcmread(p, stop_before_pixels=True)
            planes[p] = plane_of(h.ImageOrientationPatient)
        except Exception:
            planes[p] = None
    if not planes:
        return None, 0
    major = Counter(v for v in planes.values() if v).most_common(1)
    major = major[0][0] if major else None
    removed = 0
    for p, v in planes.items():
        if v != major:
            os.remove(p)
            removed += 1
    return major, removed


def geometry_side(d):
    """'R' / 'L' from the image centre's patient x (LPS: + = patient left), '' when unclear (|x| < 20 mm)."""
    import pydicom
    C = defs()
    f = sorted(os.listdir(d))[0]
    x = C["centre_x_mm"](pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True))
    if x is None or abs(x) < 20.0:
        return ""
    return "L" if x > 0 else "R"


def build_knee(args):
    """Extract one knee's tarballs to a temp dir, clean them, build the flat c03 array. -> (array | None, meta dict)."""
    study, rows, tar_root, tmp_root, delete_tars = args
    C = defs()
    t0 = time.time()
    kdir = os.path.join(tmp_root, study)
    os.makedirs(kdir, exist_ok=True)
    slots = {s: "" for s in C["SLOTS"]}
    side = rows[0]["side"]
    notes = []
    try:
        for r in rows:
            tar = os.path.join(tar_root, r["alias"])
            sdir = os.path.join(kdir, r["series"])
            if not os.path.exists(tar):
                notes.append(f"{r['series']}:missing_tar")
                continue
            os.makedirs(sdir, exist_ok=True)
            with tarfile.open(tar) as tf:
                tf.extractall(sdir, filter="data")
            for sub, _, files in list(os.walk(sdir)):        # flatten: the members sit under "./"
                for f in files:
                    src = os.path.join(sub, f)
                    if os.path.dirname(src) != sdir:
                        shutil.move(src, os.path.join(sdir, f"{os.path.basename(sub)}_{f}"))
            major, removed = clean_series(sdir, SERIES_PLANE[r["series"]])
            if removed:
                notes.append(f"{r['series']}:-{removed}")
            if major != SERIES_PLANE[r["series"]]:
                notes.append(f"{r['series']}:plane_{major}")
                continue
            if r["series"] == "SAG_IW_TSE":
                g = geometry_side(sdir)
                if g and g != side:
                    notes.append(f"side_geo_{g}")
            slots[r["slot"]] = r["series"]
        row = {**slots, "side": side}
        arr, meta = C["build_study_flat"]((study, row, tmp_root, C["cfg"]))     # reads <tmp_root>/<study>/<series>/
    except Exception as e:                   # one unreadable knee must not end a pod job: it stays cached=0
        arr = None
        notes.append(f"error_{type(e).__name__}")
        meta = {"StudyInstanceUID": study, "cached": 0, "n_slots_cached": 0, "mask": "0" * len(C["SLOTS"]),
                "decode_fails": -1}
        slots = {s: "" for s in C["SLOTS"]}
    finally:
        shutil.rmtree(kdir, ignore_errors=True)
    if delete_tars and arr is not None:      # the knee is in the array now: free the disk (pod: ~58 GB of tarballs)
        for r in rows:
            tar = os.path.join(tar_root, r["alias"])
            if os.path.exists(tar):
                os.remove(tar)
    meta.update({**slots, "side": side, "n_slots": sum(bool(v) for v in slots.values()), "notes": ";".join(notes),
                 "seconds": time.time() - t0})
    return arr, meta


def build(chosen, tar_root, out_root, workers, delete_tars=False):
    C = defs()
    cfg = C["cfg"]
    out_dir = os.path.join(out_root, C03_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="oai_build_", dir=out_root)
    by_knee = {k: g.to_dict("records") for k, g in chosen.groupby("StudyInstanceUID")}
    uids = sorted(by_knee)
    groups = [uids[i:i + cfg.blob_size] for i in range(0, len(uids), cfg.blob_size)]
    done = [k for k, g in enumerate(groups) if C["blob_is_complete"](out_dir, SHARD, k, len(g))]
    print(f"{len(uids)} knees -> {len(groups)} blobs of <= {cfg.blob_size}; {len(done)} complete; workers {workers}")
    t0, n = time.time(), 0
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        for k, grp in enumerate(groups):
            if k in done:
                continue
            arr = np.zeros((len(grp), C["N_TOTAL_SLICES"], cfg.px, cfg.px), np.uint8)
            jobs = [(u, by_knee[u], tar_root, tmp_root, delete_tars) for u in grp]
            results = pool.map(build_knee, jobs) if pool else map(build_knee, jobs)
            rows = []
            for i, (a, meta) in enumerate(results):
                if a is not None:
                    arr[i] = a
                rows.append({**meta, "row": i, "blob": C["blob_names"](SHARD, k)[0]})
                n += 1
            C["write_blob"](out_dir, SHARD, k, arr, rows)
            dt = time.time() - t0
            print(f"  blob {k + 1}/{len(groups)}: {n} knees, {dt / max(n, 1):.2f} s/knee", flush=True)
    finally:
        if pool:
            pool.shutdown()
        shutil.rmtree(tmp_root, ignore_errors=True)
    side_cols = ["StudyInstanceUID", *C["SLOTS"], "n_slots", "side", "cached", "mask", "decode_fails", "notes",
                 "blob", "row"]
    sidecars = sorted(f for f in os.listdir(out_dir) if f.startswith(f"blob{SHARD:02d}_") and f.endswith(".csv"))
    log = pd.concat([pd.read_csv(os.path.join(out_dir, f), dtype={"mask": str}, keep_default_na=False)
                     for f in sidecars], ignore_index=True)
    man = log[[c for c in side_cols if c in log.columns]].copy()
    man["cache_version"] = C03_VERSION
    man["shard"] = SHARD
    path = os.path.join(out_root, f"manifest_shard{SHARD}_oai.csv")
    man.to_csv(path, index=False)
    print(f"cached {int(man.cached.sum())}/{len(man)} knees; slots {man.n_slots.value_counts().to_dict()}; "
          f"masks {man['mask'].value_counts().head(4).to_dict()}; notes "
          f"{Counter(x for s in man.notes for x in str(s).split(';') if x).most_common(6)} -> {path}")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="artifacts/oai/oai_targets.csv")
    ap.add_argument("--image03", default="data/oai/nda_pkg_meta/image03.txt")
    ap.add_argument("--links")
    ap.add_argument("--build")
    ap.add_argument("--out", default="artifacts/cache_local/oai")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-present", action="store_true", help="--build: keep only knees whose 3 tarballs exist")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--delete-tars", action="store_true", help="--build: delete a knee's tarballs once it is in a blob")
    a = ap.parse_args()
    os.chdir(ROOT)
    chosen = choose_series(a.image03, a.targets, a.limit)
    if a.links:
        with open(a.links, "w", encoding="utf-8") as f:
            f.write("\n".join(chosen.image_file) + "\n")
        print(f"{len(chosen)} links -> {a.links}")
    if a.build:
        if a.only_present:
            ok = chosen.alias.map(lambda p: os.path.exists(os.path.join(a.build, p)))
            full = ok.groupby(chosen.StudyInstanceUID).transform("all")
            chosen = chosen[full]
            print(f"--only-present: {chosen.StudyInstanceUID.nunique()} knees with every tarball on disk")
        build(chosen, a.build, a.out, a.workers, a.delete_tars)
    if not (a.links or a.build):
        sys.exit("nothing to do: give --links and/or --build")


if __name__ == "__main__":
    main()
