"""P-23 acceptance check for a candidate blend member, on fold-0 OOF csvs. No GPU.

    python src/blend_check.py --base v05a=artifacts/kaggle_out/v13/v05a_fold0_oof.csv \
                                     v05b=artifacts/kaggle_out/v13/v05b_fold0_oof.csv \
                              --cand v06c=artifacts/kaggle_out/v15/v06c_fold0_oof.csv \
                                     v07s=artifacts/kaggle_out/stack_v2/v07s_fold0_oof.csv

Every csv is a `{version}_fold0_oof.csv` written by kaggle_pipeline.py at the checkpointed epoch
(columns StudyInstanceUID, epoch, is_gold, pred__<label>, y__<label>, w__<label>). All members must
be fold 0, so they score the same 882 held-out studies; the script aligns on the intersection and
refuses if it shrinks by more than 1%.

Rule (proposals.md P-23), applied to each candidate against the *base blend*:
  (a) own OOF macro >= best base single - 0.02
  (b) mean Spearman rho vs the base blend < 0.80
  (c) blend OOF gain (base + candidate, one vote per version) > 0.008
Accepted candidates are then blended together with the base and the final table is printed.
`--json` appends a machine-readable line for the overnight automation.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FLOOR_OOF = 0.008
RULE_OWN_SLACK = 0.02
RULE_RHO = 0.80


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro(Y, P, rows=None):
    vals = []
    for i in range(len(LABELS)):
        yy, pp = (Y[:, i], P[:, i]) if rows is None else (Y[rows, i], P[rows, i])
        vals.append(auc(yy, pp))
    return float(np.nanmean(vals)), vals


def rank_cols(P):
    return pd.DataFrame(P).rank(pct=True).to_numpy()


def parse_members(items):
    out = []
    for it in items or []:
        if "=" not in it:
            sys.exit(f"member must be version=path, got {it!r}")
        v, p = it.split("=", 1)
        out.append((v, p))
    return out


def load(path):
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    need = [f"pred__{l}" for l in LABELS] + [f"y__{l}" for l in LABELS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing[:3]}...")
    return df.set_index("StudyInstanceUID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="version=oof.csv of the current blend")
    ap.add_argument("--cand", nargs="*", default=[], help="version=oof.csv candidates to test")
    ap.add_argument("--json", default=None, help="append a JSON verdict line to this file")
    a = ap.parse_args()

    base, cand = parse_members(a.base), parse_members(a.cand)
    frames = {v: load(p) for v, p in base + cand}
    ids = None
    for df in frames.values():
        ids = set(df.index) if ids is None else ids & set(df.index)
    n_max = max(len(df) for df in frames.values())
    ids = sorted(ids)
    if len(ids) < 0.99 * n_max:
        sys.exit(f"members overlap on only {len(ids)} of {n_max} studies -- not the same fold?")
    ref = frames[base[0][0]].loc[ids]
    Y = (ref[[f"y__{l}" for l in LABELS]].to_numpy() > 0.5).astype(int)
    gold = ref["is_gold"].to_numpy() > 0.5 if "is_gold" in ref else np.zeros(len(ids), bool)
    P = {v: frames[v].loc[ids, [f"pred__{l}" for l in LABELS]].to_numpy(float) for v in frames}
    R = {v: rank_cols(P[v]) for v in P}
    print(f"{len(ids)} shared studies, {int(gold.sum())} gold; labels scored: {len(LABELS)}")

    def by_version_blend(versions):
        # one vote per version (INFER_BLEND="by_version"); each member here is a single fold
        return rank_cols(np.mean([R[v] for v in versions], axis=0))

    def rho(A, B):
        return float(np.mean([pd.Series(A[:, i]).corr(pd.Series(B[:, i]), method="spearman")
                              for i in range(len(LABELS))]))

    # singles
    print("\nSingles (fold 0 OOF macro vs teacher; gold n is tiny -- direction only)")
    single = {}
    for v in frames:
        m, per = macro(Y, P[v])
        g, _ = macro(Y, P[v], gold) if gold.sum() >= 4 else (float("nan"), None)
        single[v] = m
        print(f"  {v:<6} OOF {m:.4f}   gold {g:.3f}   {'(base)' if v in dict(base) else '(cand)'}")
    best_base = max(single[v] for v, _ in base)

    # pairwise rho
    names = list(frames)
    print("\nMean Spearman rho over labels")
    print("        " + "".join(f"{n:>8}" for n in names))
    for i in names:
        print(f"  {i:<6}" + "".join(f"{rho(P[i], P[j]):8.3f}" if i != j else "       -" for j in names))

    base_versions = [v for v, _ in base]
    B = by_version_blend(base_versions)
    base_macro, base_per = macro(Y, B)
    print(f"\nBase blend {'+'.join(base_versions)}: OOF {base_macro:.4f}")

    verdicts = {}
    for v, _ in cand:
        own = single[v]
        r = rho(P[v], B)
        C = by_version_blend(base_versions + [v])
        cm, cper = macro(Y, C)
        gain = cm - base_macro
        ok_own, ok_rho, ok_gain = own >= best_base - RULE_OWN_SLACK, r < RULE_RHO, gain > FLOOR_OOF
        accept = ok_own and ok_rho and ok_gain
        verdicts[v] = {"own": own, "rho_vs_base_blend": r, "blend": cm, "gain": gain,
                       "own_ok": ok_own, "rho_ok": ok_rho, "gain_ok": ok_gain, "accept": accept}
        print(f"\nCandidate {v}: own {own:.4f} ({'ok' if ok_own else 'FAIL'} vs {best_base:.4f}-{RULE_OWN_SLACK}) "
              f"| rho vs base blend {r:.3f} ({'ok' if ok_rho else 'FAIL'} < {RULE_RHO}) "
              f"| blend {cm:.4f} = {gain:+.4f} ({'ok' if ok_gain else 'FAIL'} > {FLOOR_OOF})"
              f"  ->  {'ACCEPT' if accept else 'REJECT'}")
        print(f"  {'label':<18}{'base':>8}{'+cand':>8}{'delta':>8}{'cand':>8}")
        _, vper = macro(Y, P[v])
        for i, l in enumerate(LABELS):
            print(f"  {l:<18}{base_per[i]:8.3f}{cper[i]:8.3f}{cper[i]-base_per[i]:+8.3f}{vper[i]:8.3f}")

    accepted = [v for v in verdicts if verdicts[v]["accept"]]
    final_versions = base_versions + accepted
    F = by_version_blend(final_versions)
    fm, _ = macro(Y, F)
    print(f"\nFinal blend {'+'.join(final_versions)}: OOF {fm:.4f} ({fm - base_macro:+.4f} vs base)")
    print("Accepted:", accepted or "none")
    if a.json:
        with open(a.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"base": base_versions, "base_oof": base_macro, "candidates": verdicts,
                                 "accepted": accepted, "final": final_versions, "final_oof": fm}) + "\n")


if __name__ == "__main__":
    main()
