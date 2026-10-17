"""A composite teacher table: per label, w_base x pct_rank(base) + (1 - w_base) x pct_rank(other), over the report-only studies.

This is the `claude_rap_v1` construction of P-65 (2/3 Raptor rank + 1/3 Claude rank, rounded to 6 decimals); at TEACHER_MIX
0.75 the training target is then ~ 0.25 LLM + 0.25 other + 0.5 Raptor in rank terms. The self-test rebuilds `claude_rap_v1`
from `raptor_teacher` + `claude_v1` and must match it to the rounding (5e-7) before anything is written.

P-86 (2026-10-17): `jevf_rap_v1` = 2/3 Raptor + 1/3 OpenJev (riadmohamed42 `jev-knee-labels` FINAL2, CC0), its gold rows dropped
(the table must never speak on a gold row; build_targets.mix_teacher masks them too).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_rank_composite.py \
        --other artifacts/research_1003/dl/riadmohamed42_jev-knee-labels/FINAL2_labels_scores.csv --out jevf_rap_v1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_targets import LABELS, TEACHER_ROOT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    missing = [l for l in LABELS if l not in d.columns]
    if missing or d.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{path}: missing labels {missing} or duplicate StudyInstanceUID")
    return d.set_index("StudyInstanceUID")[LABELS].astype(float)


def composite(base, other, w_base, index):
    if not index.isin(base.index).all() or not index.isin(other.index).all():
        raise SystemExit(f"a source misses {int((~index.isin(base.index)).sum())} / {int((~index.isin(other.index)).sum())} "
                         f"of the {len(index)} studies")
    b, o = base.loc[index], other.loc[index]
    if not (np.isfinite(b.to_numpy()).all() and np.isfinite(o.to_numpy()).all()):
        raise SystemExit("a source has non-finite values on the report-only studies")
    c = w_base * b.rank(method="average", pct=True) + (1.0 - w_base) * o.rank(method="average", pct=True)
    return c.round(6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(TEACHER_ROOT, "raptor_teacher.csv"))
    ap.add_argument("--other", required=True, help="a table with StudyInstanceUID + the 12 label columns")
    ap.add_argument("--w-base", type=float, default=2.0 / 3.0)
    ap.add_argument("--out", required=True, help="table name: writes <TEACHER_ROOT>/<out>.csv")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    base = load(a.base)
    report_only = base.index                     # raptor_teacher covers exactly the 4,349 report-only studies
    tr = pd.read_csv("data/train.csv", dtype={"StudyInstanceUID": str})
    gold = set(tr.dropna(subset=["ACL"]).StudyInstanceUID)
    if len(report_only) != 4349 or gold & set(report_only):
        raise SystemExit(f"base table: {len(report_only)} rows, {len(gold & set(report_only))} gold rows (want 4349, 0)")
    # self-test: claude_rap_v1 = 2/3 raptor_teacher + 1/3 claude_v1
    ref = load(os.path.join(TEACHER_ROOT, "claude_rap_v1.csv"))
    rebuilt = composite(base, load(os.path.join(TEACHER_ROOT, "claude_v1.csv")), 2.0 / 3.0, ref.index)
    err = float((rebuilt - ref).abs().max().max())
    print(f"self-test: claude_rap_v1 rebuilt to max |diff| {err:.1e}")
    if err > 1e-6:
        raise SystemExit("the construction does not reproduce claude_rap_v1")
    other = load(a.other)
    print(f"other: {len(other)} rows, {len(gold & set(other.index))} gold rows (dropped)")
    out = composite(base, other, a.w_base, report_only)
    out.index.name = "StudyInstanceUID"
    for l in LABELS:
        rb = out[l].corr(base.loc[report_only, l], method="spearman")
        ro = out[l].corr(other.loc[report_only, l], method="spearman")
        print(f"  {l:17s} rho vs base {rb:.3f}  vs other {ro:.3f}")
    path = os.path.join(TEACHER_ROOT, a.out + ".csv")
    out.to_csv(path)
    print(f"wrote {path}: {out.shape}, values {float(out.min().min()):.6f} .. {float(out.max().max()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
