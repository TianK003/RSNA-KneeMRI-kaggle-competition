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


def _load_set(pattern: str, per_fold_rank: bool = False, expect_folds: int | None = None,
              expect_rows: int | None = None) -> pd.DataFrame:
    """One complete OOF set = one csv per fold. per_fold_rank (P-54, 2026-09-28): each fold's predictions become rank
    percentiles WITHIN that fold before pooling, so a fold model that is calibrated higher or lower than its siblings does
    not shift its fifth of the table (the pooled rank then compares studies across folds on equal terms)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no OOF csvs match {pattern}")
    # P-68 (2026-10-05, the critic): a partial k-fold set would build a table that silently leaves rows uncovered, and
    # mix_teacher falls back to the LLM value on uncovered rows -- refuse instead.
    if expect_folds is not None and len(files) != expect_folds:
        raise SystemExit(f"{pattern}: {len(files)} fold csvs, expected {expect_folds}: {files}")
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype={"StudyInstanceUID": str})
        if per_fold_rank:
            for l in LABELS:
                d[f"pred__{l}"] = d[f"pred__{l}"].rank(pct=True)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    if d.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{pattern}: a study appears in more than one fold's OOF")
    if expect_rows is not None and len(d) != expect_rows:
        raise SystemExit(f"{pattern}: {len(d)} studies, expected {expect_rows} (an incomplete OOF set)")
    return d.set_index("StudyInstanceUID")[[f"pred__{l}" for l in LABELS]]


def build_distill_table(oof_globs: list[str], out_csv: str, per_fold_rank: bool = False,
                        expect_folds: int | None = None, expect_rows: int | None = None) -> pd.DataFrame:
    sets = [_load_set(p, per_fold_rank, expect_folds, expect_rows) for p in oof_globs]
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
    print(f"wrote {out_csv}: {len(out)} studies from {len(sets)} OOF sets"
          + (" (ranked within each fold first)" if per_fold_rank else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--out", default="artifacts/teacher/selfdistill_v1.csv")
    ap.add_argument("--per-fold-rank", action="store_true",
                    help="rank each fold's predictions within the fold before pooling (P-54 cross-fit table)")
    ap.add_argument("--expect-folds", type=int, default=None, help="refuse a set with a different number of fold csvs")
    ap.add_argument("--expect-rows", type=int, default=None, help="refuse a set covering a different number of studies")
    a = ap.parse_args()
    build_distill_table(a.sets, a.out, per_fold_rank=a.per_fold_rank, expect_folds=a.expect_folds,
                        expect_rows=a.expect_rows)
