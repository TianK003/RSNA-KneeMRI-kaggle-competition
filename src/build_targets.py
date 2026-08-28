"""Build training targets and leak-safe folds.

Why this file exists at all: only 58 of 4,407 studies carry official labels, so the
training signal has to come from the radiology reports. Reports exist in train.csv and
NOT in test.csv, so they can only ever be a source of *targets* -- never a model input.

Three jobs:

  1. Merge the public LLM-read report labels into one soft target per (study, label),
     and weight each study by how confidently its report could be read.
  2. Overwrite those soft targets with the 58 official labels where they exist, and give
     them a much larger sample weight -- they are the only ground truth.
  3. Split into folds that cannot leak. Two leak channels are handled:
       - the same report text shared by several studies (49 texts cover 183 studies,
         largest group 37) -> group them
       - the gold studies -> spread evenly so each fold can be scored at all

Outputs: artifacts/targets.csv, artifacts/folds.csv, artifacts/label_report.txt

Usage:  python src/build_targets.py [--folds 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Measured gold macro-AUC of each source (n=58, so differences under ~0.02 are noise).
# Ordered best-first. Targets are the MEAN OF PROBABILITIES (P-00); ranks are used only
# to score and to ensemble predictions, never to build a BCE target.
SOURCES = [
    ("hans_v4", "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"),      # 0.893
    ("pilkwang", "rsna-knee-llm-labels/report_labels_v2.csv"),               # 0.870
    ("sol56", "rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"),  # 0.835
]

GOLD_WEIGHT = 8.0        # official labels are worth many weak ones
WEAK_WEIGHT_FLOOR = 0.15  # a study whose report says nothing still pulls a little


# --------------------------------------------------------------------- metric

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC. Hand-rolled to avoid a sklearn dependency."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_se(a: float, npos: int, nneg: int) -> float:
    """Hanley-McNeil standard error -- the reason gold-only metrics are near-useless here."""
    if not np.isfinite(a) or npos < 1 or nneg < 1:
        return float("nan")
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
    return float(np.sqrt(max(var, 0.0)))


# ----------------------------------------------------------------- label load

def load_sources(root: str, index: pd.Index) -> dict[str, pd.DataFrame]:
    out = {}
    for name, rel in SOURCES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  ! missing {rel}, skipping")
            continue
        d = pd.read_csv(path).set_index("StudyInstanceUID")
        if not set(LABELS) <= set(d.columns):
            print(f"  ! {rel} lacks label columns, skipping")
            continue
        out[name] = d.reindex(index)
        print(f"  loaded {name:<9} {d.shape[0]} rows")
    if not out:
        raise SystemExit("no LLM label sources found under " + root)
    return out


def rank_blend(sources: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Per label, average the percentile rank of each source that has a value.

    KEPT ONLY AS A DIAGNOSTIC (see P-00 in docs/proposals.md). Rank space is the right
    place to *score* and to *ensemble predictions*, but it is the wrong place to build
    a *training target*: `rank(pct=True)` gives tied values their average rank, so on a
    label where most sources say exactly 0 (MCL, Lateral OA, Baker's, ...) every
    confidently-negative study lands at ~0.3-0.4 instead of ~0, while the 58 gold rows
    sit at a hard 0/1. BCE fits the value, not the order, so the network was being
    taught "definitely absent" = 0.31. Measured on 2026-08-28: only 1% of studies had a
    target < 0.1 on any label.
    """
    out = pd.DataFrame(index=index)
    for lab in LABELS:
        acc = np.zeros(len(index))
        cnt = np.zeros(len(index))
        for d in sources.values():
            s = d[lab].to_numpy(dtype=float)
            r = pd.Series(s).rank(pct=True).to_numpy()
            m = np.isfinite(s)
            acc[m] += r[m]
            cnt[m] += 1
        out[lab] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


def prob_blend(sources: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Per label, mean of the sources' probabilities (each already in [0, 1]).

    This is the training target (P-00). It keeps the LLMs' own 0/1 semantics -- a
    report that clearly says "no tear" becomes ~0, a clear positive ~1, and an
    unaddressed finding stays near the source's uncertainty value -- so the soft
    targets live on the same scale as the gold 0/1 labels that override them.
    Sources disagreeing about scale is handled by `confidence_weights`, not by
    ranking.
    """
    out = pd.DataFrame(index=index)
    for lab in LABELS:
        arr = np.vstack([d[lab].to_numpy(dtype=float) for d in sources.values()])
        with np.errstate(invalid="ignore"):
            out[lab] = np.nanmean(arr, axis=0)
    return out


def target_quantiles(soft: pd.DataFrame) -> list[str]:
    """One line per label: where the soft-target mass sits. Used to show the P-00
    before/after in label_report.txt."""
    rows = []
    for lab in LABELS:
        s = soft[lab].dropna().to_numpy()
        if len(s) == 0:
            rows.append(f"  {lab:<18} (empty)")
            continue
        q = np.percentile(s, [5, 25, 50, 75, 95])
        rows.append(f"  {lab:<18} p5 {q[0]:.2f} p25 {q[1]:.2f} p50 {q[2]:.2f} "
                    f"p75 {q[3]:.2f} p95 {q[4]:.2f}  <0.1: {(s < 0.1).mean():4.0%}  "
                    f">0.9: {(s > 0.9).mean():4.0%}")
    return rows


def confidence_weights(sources: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Per (study, label) weight in [floor, 1].

    Two ingredients:
      - agreement: sources that disagree about a study mean the report is ambiguous
      - decisiveness: a soft label near 0.5 carries less information than one near 0/1
    Caveat measured 2026-08-28 (label_audit.py): silence is NOT down-weighted much. A
    report that never mentions synovitis blends to ~0.18 (a confident-looking negative), so
    the `decisive` term rewards it: mean weight 0.69 on unaddressed rows vs 0.80 on
    addressed ones. Gating on pilkwang's UNK verdict is an open card (P-07/P-16).
    """
    w = pd.DataFrame(index=index)
    for lab in LABELS:
        cols = [d[lab].to_numpy(dtype=float) for d in sources.values()]
        arr = np.vstack(cols)                       # (n_src, n_studies)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(arr, axis=0)         # 0 = perfect agreement
            mean = np.nanmean(arr, axis=0)
        agree = 1.0 - np.nan_to_num(spread, nan=0.5) * 2.0        # std 0.5 -> 0
        decisive = np.abs(np.nan_to_num(mean, nan=0.5) - 0.5) * 2  # 0.5 -> 0, 0/1 -> 1
        raw = 0.5 * np.clip(agree, 0, 1) + 0.5 * np.clip(decisive, 0, 1)
        w[lab] = np.clip(raw, WEAK_WEIGHT_FLOOR, 1.0)
    return w


# ------------------------------------------------------------------- folds

def report_group(reports: pd.Series) -> pd.Series:
    """Stable group id per distinct report text.

    Studies sharing a report share a target vector, so they must never straddle a fold
    boundary -- otherwise the model can memorise the text-derived answer in training and
    be scored on it in validation.
    """
    norm = reports.fillna("").str.strip().str.lower()
    return norm.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()[:16])


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Grouped, gold-stratified fold assignment.

    Greedy: place the largest groups first into whichever fold is currently smallest,
    handling gold groups first so the 58 gold studies spread across folds instead of
    landing in one. Greedy-largest-first keeps folds near-equal even with a 37-study group.
    """
    rng = np.random.default_rng(seed)
    g = df.groupby("report_group").agg(n=("StudyInstanceUID", "size"),
                                       gold=("is_gold", "sum"))
    # Shuffle before sorting so ties break randomly but reproducibly.
    g = g.sample(frac=1.0, random_state=seed)
    g = g.sort_values(["gold", "n"], ascending=False)

    sizes = np.zeros(n_folds)
    golds = np.zeros(n_folds)
    out: dict[str, int] = {}
    for gid, row in g.iterrows():
        if row.gold > 0:
            # balance gold first, then total size
            k = int(np.lexsort((sizes, golds))[0])
        else:
            k = int(np.argmin(sizes))
        out[gid] = k
        sizes[k] += row.n
        golds[k] += row.gold
    _ = rng  # seed already consumed via sample/lexsort determinism
    return df.report_group.map(out)


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--llm-root", default="data/llm_labels")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    tr = pd.read_csv(args.train)
    idx = pd.Index(tr.StudyInstanceUID, name="StudyInstanceUID")
    is_gold = tr[LABELS].notna().all(axis=1)
    log(f"train.csv: {len(tr)} studies, {int(is_gold.sum())} with official labels")

    log("\nloading LLM report-label sources:")
    sources = load_sources(args.llm_root, idx)

    soft = prob_blend(sources, idx)          # training target (P-00)
    soft_rank = rank_blend(sources, idx)     # diagnostic only
    weights = confidence_weights(sources, idx)

    gold = tr.set_index("StudyInstanceUID")[LABELS]
    gold_idx = idx[is_gold.to_numpy()]

    # ---- evaluate each source and the blend on the 58 gold studies -----
    log("\ngold macro-AUC by source (n=58 -- treat gaps under ~0.02 as noise):")
    gy = gold.loc[gold_idx].astype(float)
    for name, d in sources.items():
        a = [auc(gy[l].to_numpy(), d.loc[gold_idx, l].to_numpy()) for l in LABELS]
        log(f"  {name:<9} {np.nanmean(a):.4f}")
    blend_aucs = [auc(gy[l].to_numpy(), soft.loc[gold_idx, l].to_numpy()) for l in LABELS]
    rank_aucs = [auc(gy[l].to_numpy(), soft_rank.loc[gold_idx, l].to_numpy()) for l in LABELS]
    log(f"  {'BLEND':<9} {np.nanmean(blend_aucs):.4f}  (mean of probabilities -- the target)")
    log(f"  {'rank':<9} {np.nanmean(rank_aucs):.4f}  (old rank blend, diagnostic only)")

    log("\nP-00 target scale -- where the soft-target mass sits (before gold override):")
    log("  old rank blend:")
    for r in target_quantiles(soft_rank):
        log(r)
    log("  probability blend (used):")
    for r in target_quantiles(soft):
        log(r)
    log("  -> confidently-negative reports must land near 0, not ~0.3; BCE fits the value.")

    log("\nper-label blend AUC with Hanley-McNeil SE:")
    for lab, a in zip(LABELS, blend_aucs):
        y = gy[lab].to_numpy()
        se = auc_se(a, int((y == 1).sum()), int((y == 0).sum()))
        log(f"  {lab:<18} {a:.3f} +/- {se:.3f}")
    log("  -> a +/-0.09 SE is wider than most differences you will try to measure;")
    log("     judge label changes on coverage and OOF, not on this number alone.")

    # ---- assemble targets ---------------------------------------------
    targets = soft.copy()
    w = weights.copy()
    # Official labels override the weak ones and carry much more weight.
    for lab in LABELS:
        gvals = gold[lab].reindex(idx)
        have = gvals.notna().to_numpy()
        targets.loc[have, lab] = gvals[have].to_numpy()
        w.loc[have, lab] = GOLD_WEIGHT

    n_nan = int(targets[LABELS].isna().sum().sum())
    if n_nan:
        log(f"\n! {n_nan} target cells still NaN; filling with the per-label mean "
            f"and dropping their weight to the floor")
        for lab in LABELS:
            m = targets[lab].isna()
            targets.loc[m, lab] = float(targets[lab].mean())
            w.loc[m, lab] = WEAK_WEIGHT_FLOOR

    # ---- folds --------------------------------------------------------
    meta = pd.DataFrame({
        "StudyInstanceUID": tr.StudyInstanceUID,
        "is_gold": is_gold.astype(int),
        "report_group": report_group(tr.Report),
    })
    ng = meta.report_group.nunique()
    shared = meta.report_group.value_counts()
    shared = shared[shared > 1]
    log(f"\nreport groups: {ng} distinct texts over {len(meta)} studies")
    log(f"  {len(shared)} texts shared by >1 study, covering {int(shared.sum())} studies "
        f"(largest group {int(shared.max())})")
    log("  -> grouping these is not optional: a 37-study group split across folds "
        "leaks its target vector")

    meta["fold"] = assign_folds(meta, args.folds, args.seed)
    log("\nfold sizes (studies / gold):")
    for k, grp in meta.groupby("fold"):
        log(f"  fold {k}: {len(grp):5d} / {int(grp.is_gold.sum()):3d}")
    # sanity: no group spans two folds
    spans = meta.groupby("report_group").fold.nunique().max()
    log(f"  max folds touched by any single report group: {spans} "
        f"({'OK' if spans == 1 else 'LEAK'})")

    # ---- write --------------------------------------------------------
    tgt = targets.reset_index().rename(columns={"index": "StudyInstanceUID"})
    tgt.columns = ["StudyInstanceUID"] + LABELS
    wide_w = w.reset_index(drop=True)
    wide_w.columns = [f"w__{c}" for c in LABELS]
    out = pd.concat([tgt.reset_index(drop=True), wide_w], axis=1)
    out = out.merge(meta[["StudyInstanceUID", "is_gold", "fold", "report_group"]],
                    on="StudyInstanceUID")

    tpath = os.path.join(args.out_dir, "targets.csv")
    fpath = os.path.join(args.out_dir, "folds.csv")
    rpath = os.path.join(args.out_dir, "label_report.txt")
    out.to_csv(tpath, index=False)
    meta.to_csv(fpath, index=False)
    with open(rpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\nwrote {tpath}  ({out.shape[0]} rows x {out.shape[1]} cols)")
    log(f"wrote {fpath}")
    log(f"wrote {rpath}")


if __name__ == "__main__":
    main()
