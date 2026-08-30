"""Keep-in-sync check for the two copies of the preprocessing code (no GPU, ~1 min).

`src/cache_pipeline.py` builds the training cache; `src/kaggle_pipeline.py` carries hand-copied
versions of the same functions to build TEST studies on the fly. Until 2026-08-30 the only guard
was a "keep in sync" comment and a two-study manual check, and two latent divergences existed
(which header supplied IOP/PixelSpacing; a hard-coded [1, 99]). This script executes both files
up to their run sections (RSNA_DEFS_ONLY=1) and, for BOTH cache schemes, builds every sample
study through both code paths and asserts bit-for-bit equality of arrays, masks, slot choices,
side, version strings and slot offsets; for c02 it also checks that a study read back from the
local blob equals the fresh build. Exit status is the verdict.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/cache_selftest.py
"""
import os
import sys
import tempfile

import numpy as np

os.environ["RSNA_DEFS_ONLY"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def load_defs(path):
    ns = {"__name__": f"defs_{os.path.basename(path)}", "__file__": path}
    with open(path, encoding="utf-8") as f:
        code = compile(f.read(), path, "exec")
    try:
        exec(code, ns)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    return ns


def main():
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print("loading definitions ...")
    C = load_defs(os.path.join("src", "cache_pipeline.py"))
    K = load_defs(os.path.join("src", "kaggle_pipeline.py"))

    image_root = os.path.join("data", "sample_dicom", "test_series")
    series_csv = os.path.join("data", "test_series.csv")
    if not os.path.isdir(image_root):
        raise SystemExit(f"no sample DICOMs under {image_root}")
    tmp = tempfile.mkdtemp(prefix="rsna_selftest_")

    # header scan + manifest through both modules
    series_c = C["scan_series"](series_csv, image_root, os.path.join(tmp, "meta_c.csv"), workers=1)
    series_k = K["scan_series"](series_csv, image_root, os.path.join(tmp, "meta_k.csv"))
    check(len(series_c) == len(series_k) > 0, f"both scans see {len(series_c)} series")

    for scheme in ("c01", "c02"):
        print(f"\n== scheme {scheme}")
        ccfg = C["CacheConfig"](scheme=scheme, workers=1)
        kcfg = K["Config"](cache_scheme=scheme)
        v_c = C["cache_version_of"](ccfg.scheme, ccfg.px, ccfg.slot_slices, ccfg.band,
                                    ccfg.crop_mm, ccfg.lat_dead_zone_mm)
        v_k = K["cache_version_for"](kcfg)
        check(v_c == v_k, f"version string {v_c}")
        if scheme == "c01":
            check(v_c == "c01_p224_s16_crop130_lat20", "c01 string unchanged from the built cache")
        else:
            check(v_c == "c02_p336_b18-12-12-14-8-8_band2-98_crop130_lat20", "c02 string as planned")
        off_c = C["slot_offsets"](ccfg.slot_slices)
        off_k = K["slot_offsets"](K["cache_geom"](kcfg)[2])
        check(off_c == off_k, f"slot offsets {off_c}")
        g = K["cache_geom"](kcfg)
        check((g[1], tuple(g[2])) == (ccfg.px, tuple(ccfg.slot_slices)), f"px/budgets {g[1]} {g[2]}")
        check(all(tuple(g[3][p]) == tuple(ccfg.band[p]) for p in ccfg.band), f"bands {g[3]}")

        man_c = C["build_manifest"](series_c, ccfg)
        man_k = K["build_manifest"](series_k, os.path.join(tmp, f"man_k_{scheme}.csv"))
        cols = list(C["SLOTS"]) + ["side"]
        a = man_c.set_index("StudyInstanceUID")[cols].fillna("").astype(str)
        b = man_k.set_index("StudyInstanceUID")[cols].fillna("").astype(str).loc[a.index]
        check(a.equals(b), f"manifest slots + side agree on {len(a)} studies")

        out_dir = os.path.join(tmp, v_c)
        os.makedirs(out_dir, exist_ok=True)
        for study, row in man_c.set_index("StudyInstanceUID").iterrows():
            row = row.to_dict()
            row["StudyInstanceUID"] = study
            if scheme == "c01":
                meta = C["cache_study"]((study, row, image_root, ccfg, out_dir))
                arr_c = np.load(os.path.join(out_dir, f"{study}.npy")) if meta["cached"] else None
            else:
                arr_c, meta = C["build_study_flat"]((study, row, image_root, ccfg))
            arr_k, mask_k = K["build_study_array"](study, row, image_root, kcfg)
            mk = "".join("1" if v > 0 else "0" for v in mask_k)
            same = arr_c is not None and np.array_equal(arr_c, arr_k) and meta.get("mask", "") == mk
            check(same, f"{study[-12:]}: array {getattr(arr_c, 'shape', None)} bit-identical, mask {mk}")
            if scheme == "c02":
                # window enumeration must respect the ragged budgets
                cen, sid = K["valid_windows"](mask_k, kcfg)
                exp = sum(n - 2 for n, m in zip(g[2], mask_k) if m > 0)
                check(len(cen) == exp, f"{study[-12:]}: {len(cen)} valid windows (expected {exp})")

        if scheme == "c02":
            # a study read out of the LOCAL blob (built by the real Section 4 run) == fresh build
            local = os.path.join("artifacts", "cache_local", v_c)
            side = [f for f in sorted(os.listdir(local)) if f.endswith(".csv")] if os.path.isdir(local) else []
            if not side:
                check(False, f"no local c02 blob under {local} -- run cache_pipeline.py locally first")
            else:
                import pandas as pd
                sc = pd.read_csv(os.path.join(local, side[0]), dtype={"mask": str})
                for _, r in sc.iterrows():
                    if not int(r.cached):
                        continue
                    loc = (os.path.join(local, r.blob), int(r.row))
                    arr_blob = K["read_cached"](loc)
                    row = man_c.set_index("StudyInstanceUID").loc[r.StudyInstanceUID].to_dict()
                    row["StudyInstanceUID"] = r.StudyInstanceUID
                    arr_k, _ = K["build_study_array"](r.StudyInstanceUID, row, image_root, kcfg)
                    check(np.array_equal(arr_blob, arr_k),
                          f"blob row {int(r.row)} == fresh build for {r.StudyInstanceUID[-12:]}")
                    shape, dtype, hdr = K["npy_header"](loc[0])
                    check(shape[1:] == arr_k.shape and str(dtype) == "uint8", f"blob header {shape} {dtype}")

    print("\n" + ("SELFTEST PASSED" if not fails else f"SELFTEST FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
