"""Merge raptor_teacher_shard*.npz / raptor_teacher_partial*.npz (Task 7 kernel outputs) into
artifacts/teacher/raptor_teacher.csv: UID + 12 view-weighted mean probabilities.

    .venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s*/raptor_teacher_shard*.npz

A partial file (guard-stopped run) carries only study_uids + raw_probabilities: it borrows the
view_names/view_weights from a sibling input that has them (R18). A `.tmp.npz` name is a
mid-flush leftover from a crash and is always skipped, never opened.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def merge_teacher(npz_paths, out_csv, expect_n=4349, allow_partial=False):
    paths = []
    for p in npz_paths:
        if p.endswith(".tmp.npz"):
            print(f"  skipping leftover: {os.path.basename(p)}")
            continue
        paths.append(p)

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
    ap = argparse.ArgumentParser(); ap.add_argument("npz", nargs="+"); ap.add_argument("--out", default="artifacts/teacher/raptor_teacher.csv")
    ap.add_argument("--expect-n", type=int, default=4349); ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(); paths = sorted(sum([glob.glob(p) for p in a.npz], []))
    merge_teacher(paths, a.out, a.expect_n, a.allow_partial)
