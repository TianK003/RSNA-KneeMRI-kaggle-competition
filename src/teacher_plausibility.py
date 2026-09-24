"""P-39 plausibility read of a Raptor teacher table against the hard LLM teacher.

    python src/teacher_plausibility.py artifacts/teacher/raptor_partial_s01.csv
    python src/teacher_plausibility.py artifacts/teacher/raptor_teacher.csv --targets artifacts/targets.csv

The table is what `src/merge_teacher.py` writes (StudyInstanceUID + the 12 label columns, view-weighted mean
probabilities). The reference is `artifacts/targets.csv` from `src/build_targets.py`: its 12 plain label columns are
the LLM blend (soft, in [0, 1]) for report-only rows and the hard 0/1 labels for the 58 gold rows. This script reports,
per label, the AUC of the Raptor probability against `LLM > 0.5`, the Spearman correlation, and both operating points
(mean Raptor probability vs LLM positive rate), plus coverage checks. It is a *plausibility* read, not a verdict: the
spike (100 studies, 2026-09-23) gave macro AUC 0.914 with MCL lowest (0.77) and a Raptor operating point more positive
than the LLM blend on Fracture / MCL -- quantile matching in the kernel removes that. A label near 0.5 or a macro far
below 0.9 means a view / label-order problem in the pass (stop and inspect the npz), not a finding about the teacher.
Never read a teacher table against OOF of a model trained on the same teacher (traps 39).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="raptor teacher csv (merge_teacher.py output)")
    ap.add_argument("--targets", default="artifacts/targets.csv")
    ap.add_argument("--min-macro", type=float, default=0.85, help="warn below this macro AUC vs the hard LLM teacher")
    a = ap.parse_args()

    t = pd.read_csv(a.table, dtype={"StudyInstanceUID": str})
    ref = pd.read_csv(a.targets, dtype={"StudyInstanceUID": str})
    missing = [l for l in LABELS if l not in t.columns]
    if missing:
        print(f"!! table lacks label columns {missing}; columns = {list(t.columns)[:14]}")
        return 2
    if not t.StudyInstanceUID.is_unique:
        print(f"!! duplicate StudyInstanceUID in the table ({t.StudyInstanceUID.duplicated().sum()})")
        return 2
    vals = t[LABELS].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        print(f"!! non-finite or out-of-range probabilities: finite {np.isfinite(vals).all()}, min {vals.min():.3f}, max {vals.max():.3f}")
        return 2
    j = t.merge(ref, on="StudyInstanceUID", suffixes=("", "__llm"), how="inner")
    n_gold = int(j["is_gold"].sum()) if "is_gold" in j.columns else -1
    print(f"table rows {len(t)}, joined with targets {len(j)}, gold rows in the table {n_gold} (expected 0: shards are gold-free)")
    if len(j) < len(t):
        print(f"!! {len(t) - len(j)} table rows have no targets row")
    rep = j[j["is_gold"] == 0] if "is_gold" in j.columns else j
    print(f"scoring on {len(rep)} report-only rows; hard LLM label = blend > 0.5\n")
    print(f"{'label':<18}{'AUC':>7}{'rho':>7}{'raptor mean':>13}{'LLM pos rate':>14}{'LLM mean':>10}")
    aucs = []
    for l in LABELS:
        y_soft = rep[f"{l}__llm"].to_numpy(dtype=float)
        ok = np.isfinite(y_soft)
        y = (y_soft[ok] > 0.5).astype(int)
        p = rep[l].to_numpy(dtype=float)[ok]
        if y.min() == y.max():
            auc = float("nan")
        else:
            auc = roc_auc_score(y, p)
        rho = spearmanr(p, y_soft[ok]).correlation
        aucs.append(auc)
        flag = "  <-- check" if (np.isfinite(auc) and auc < 0.75) else ""
        print(f"{l:<18}{auc:>7.3f}{rho:>7.2f}{p.mean():>13.3f}{y.mean():>14.3f}{y_soft[ok].mean():>10.3f}{flag}")
    macro = float(np.nanmean(aucs))
    print(f"\nmacro AUC vs the hard LLM teacher: {macro:.4f}  (spike, 100 studies: 0.914)")
    if macro < a.min_macro:
        print(f"!! macro {macro:.3f} < {a.min_macro}: inspect the npz (view order, label order) before using this table")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
