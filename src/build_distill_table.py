"""Self-distillation teacher table (P-17 / P-38): rank-mean of complete 5-fold OOF sets.

Every prediction is out-of-fold, so no study is taught by a model that saw it. Values are rank
percentiles per label (mean over sets); build_targets.quantile_match() puts them on the LLM scale.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_distill_table.py            # -> artifacts/teacher/selfdistill_v1.csv
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# character class, not fold* -- fold*_oof.csv would also match the per-epoch fold0_ep7_oof.csv
# files that live in the same output folders and trip the duplicate-UID guard below.
DEFAULT_SETS = ["artifacts/kaggle_out/pod_v09h_5fold/v09h_fold[0-9]_oof.csv",   # pooled OOF 0.8625
                "artifacts/kaggle_out/folds_v4/v05g_fold[0-9]_oof.csv"]         # pooled OOF 0.8467


def _load_set(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no OOF csvs match {pattern}")
    d = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in files], ignore_index=True)
    if d.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{pattern}: a study appears in more than one fold's OOF")
    return d.set_index("StudyInstanceUID")[[f"pred__{l}" for l in LABELS]]


def build_distill_table(oof_globs: list[str], out_csv: str) -> pd.DataFrame:
    sets = [_load_set(p) for p in oof_globs]
    ids = set(sets[0].index)
    for s, pat in zip(sets[1:], oof_globs[1:]):
        if set(s.index) != ids:
            raise SystemExit(f"OOF sets cover different studies ({pat}: {len(set(s.index) ^ ids)} differ)")
    index = sorted(ids)
    acc = np.zeros((len(index), len(LABELS)))
    for s in sets:
        s = s.loc[index]
        for j, l in enumerate(LABELS):
            acc[:, j] += pd.Series(s[f"pred__{l}"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
    acc /= len(sets)
    out = pd.DataFrame(acc, columns=LABELS)
    out.insert(0, "StudyInstanceUID", index)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(out)} studies from {len(sets)} OOF sets")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--out", default="artifacts/teacher/selfdistill_v1.csv")
    a = ap.parse_args()
    build_distill_table(a.sets, a.out)
