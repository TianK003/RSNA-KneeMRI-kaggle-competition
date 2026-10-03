"""Merge raptor_teacher_shard*.npz / raptor_teacher_partial*.npz (Task 7 kernel outputs) into
artifacts/teacher/raptor_teacher.csv: UID + 12 view-weighted mean probabilities.

    .venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s*/raptor_teacher_shard*.npz

A partial file (guard-stopped run) carries only study_uids + raw_probabilities: it borrows the
view_names/view_weights from a sibling input that has them (R18). A `.tmp.npz` name is a
mid-flush leftover from a crash and is always skipped, never opened.

P-45, the D4 teacher (src/build_d4_teacher_pass.py; one view, view_weights [1.0]):

    .venv/Scripts/python.exe src/merge_teacher.py --teacher d4 "artifacts/kaggle_out/d4_s*/d4_teacher_shard*.npz"
    # -> artifacts/teacher/d4_teacher.csv

`--teacher d4` only sets the default --out and refuses any input not named d4_teacher_*; the D4 files
carry `pass_mode`, and a "gold" pass (the 58 gold studies) is refused whatever the flag. The default
(`--teacher raptor`) behaves exactly as before.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TEACHERS = {"raptor": {"out": "artifacts/teacher/raptor_teacher.csv", "prefix": None},
            "d4": {"out": "artifacts/teacher/d4_teacher.csv", "prefix": "d4_teacher_"}}


def merge_teacher(npz_paths, out_csv, expect_n=4349, allow_partial=False, prefix=None):
    paths = []
    for p in npz_paths:
        if p.endswith(".tmp.npz"):
            print(f"  skipping leftover: {os.path.basename(p)}")
            continue
        if prefix and not os.path.basename(p).startswith(prefix):
            raise SystemExit(f"{p}: not a {prefix}* file -- one teacher per merge")
        paths.append(p)
    for p in paths:   # a file that says it is a gold pass (D4 writes pass_mode; Raptor files carry none) never teaches
        with np.load(p, allow_pickle=False) as z:
            if "pass_mode" in z.files and str(z["pass_mode"]) != "train":
                raise SystemExit(f"{p}: pass_mode {str(z['pass_mode'])!r} -- the gold studies never enter a training table")

    # first pass: resolve the shared view_weights / view_names from whichever inputs carry them
    weights, names = None, None
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            if "view_weights" in z.files:
                w = np.asarray(z["view_weights"], float); w = w / w.sum()
                if weights is not None and not np.allclose(w, weights):
                    raise SystemExit(f"{p}: view weights differ from an earlier input")
                weights = w
            if "view_names" in z.files:
                n = [str(x) for x in z["view_names"]]
                if names is not None and n != names:
                    raise SystemExit(f"{p}: view names differ from an earlier input")
                names = n
    if weights is None:
        raise SystemExit("no input file carries view_weights")

    uids, means = [], []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            raw = z["raw_probabilities"].astype(float)
            ok = np.isfinite(raw).all(axis=(0, 2))
            print(f"  {os.path.basename(p)}: {int(ok.sum())}/{len(ok)} studies complete")
            means.append(np.tensordot(weights, np.clip(raw[:, ok, :], 0, 1), axes=(0, 0)))
            uids += [str(u) for u in z["study_uids"][ok]]

    out = pd.DataFrame(np.concatenate(means), columns=LABELS); out.insert(0, "StudyInstanceUID", uids)
    if out.StudyInstanceUID.duplicated().any():
        raise SystemExit("duplicate StudyInstanceUID across shards")
    if not allow_partial and len(out) != expect_n:
        raise SystemExit(f"expected {expect_n} studies, got {len(out)} (use --allow-partial for a spike)")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True); out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(out)} studies; per-label mean " + ", ".join(f"{l} {out[l].mean():.2f}" for l in LABELS[:4]) + " …")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("npz", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--expect-n", type=int, default=4349); ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), default="raptor",
                    help="raptor (default, unchanged) or d4: default --out and the d4_teacher_* name check")
    a = ap.parse_args(); paths = sorted(sum([glob.glob(p) for p in a.npz], []))
    t = TEACHERS[a.teacher]
    merge_teacher(paths, a.out or t["out"], a.expect_n, a.allow_partial, prefix=t["prefix"])
