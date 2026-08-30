"""Per-fold and POOLED out-of-fold summary of a k-fold version, from its `{version}_fold{k}_oof.csv`
files. No GPU. The numbers the 5-fold `v05g` entry in docs/experiments.md reports (per-fold OOF, mean
of folds, pooled over all 4,407 studies, gold over all 58, per-label pooled) -- so a new 5-fold run
(`v09h`) is read the same way, and the fold-ensemble base is compared like for like.

    python src/fold_oof_summary.py v05g artifacts/kaggle_out/folds_v4
    python src/fold_oof_summary.py v09h artifacts/kaggle_out/pod_v09h_5fold --json artifacts/kaggle_out/fold_summaries.jsonl

Reads `{dir}/{version}_fold[0-9]_oof.csv` (the checkpointed epoch; the per-epoch `_ep*_oof.csv`
files are ignored). Pooling: each fold's predictions are percentile-ranked WITHIN the fold before the
folds are concatenated (fold-rank-normalised), because the five checkpoints are five different
models whose raw logits are not on one scale; the raw-pooled number is printed next to it as a
check -- for `v05g` the two agree to 4 dp (0.8467). Targets are the teacher labels thresholded at
0.5, as in training's `auc_soft`; `is_gold` rows give the gold AUC (n is tiny: direction only).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from blend_check import LABELS, load, macro, rank_cols


def find_folds(version, folder):
    files = {}
    for p in glob.glob(os.path.join(folder, f"{version}_fold[0-9]_oof.csv")):
        m = re.search(rf"{re.escape(version)}_fold(\d)_oof\.csv$", os.path.basename(p))
        if m:
            files[int(m.group(1))] = p
    if not files:
        sys.exit(f"no {version}_fold[0-9]_oof.csv under {folder}")
    return dict(sorted(files.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("folder", help="directory holding {version}_fold{k}_oof.csv")
    ap.add_argument("--json", default=None, help="append a JSON summary line to this file")
    a = ap.parse_args()

    files = find_folds(a.version, a.folder)
    print(f"{a.version}: {len(files)} fold(s) under {a.folder}: {sorted(files)}")
    frames = {k: load(p) for k, p in files.items()}

    # a study in two folds is a split leak, not a fold ensemble
    seen = {}
    for k, df in frames.items():
        for sid in df.index:
            if sid in seen:
                sys.exit(f"study {sid} appears in fold {seen[sid]} and fold {k} -- fold leak, refusing")
            seen[sid] = k

    pred_cols = [f"pred__{l}" for l in LABELS]
    y_cols = [f"y__{l}" for l in LABELS]
    per_fold, Ys, Rs, Ps, golds = {}, [], [], [], []
    print(f"\n  {'fold':<6}{'n':>6}{'OOF':>9}{'gold':>8}{'(n)':>5}{'epoch':>7}{'pred_std':>10}")
    for k, df in frames.items():
        Y = (df[y_cols].to_numpy() > 0.5).astype(int)
        P = df[pred_cols].to_numpy(float)
        gold = df["is_gold"].to_numpy() > 0.5 if "is_gold" in df else np.zeros(len(df), bool)
        m, _ = macro(Y, P)
        g = macro(Y, P, gold)[0] if gold.sum() >= 4 else float("nan")
        ep = int(df["epoch"].iloc[0]) if "epoch" in df else -1
        std = float(np.mean(P.std(axis=0)))
        per_fold[k] = {"n": int(len(df)), "oof": m, "gold": g, "n_gold": int(gold.sum()), "epoch": ep}
        print(f"  {k:<6}{len(df):>6}{m:>9.4f}{g:>8.3f}{int(gold.sum()):>5}{ep:>7}{std:>10.3f}")
        Ys.append(Y); Ps.append(P); Rs.append(rank_cols(P)); golds.append(gold)

    Y = np.concatenate(Ys); P = np.concatenate(Ps); R = np.concatenate(Rs); gold = np.concatenate(golds)
    oofs = [v["oof"] for v in per_fold.values()]
    pooled_rank, per_label = macro(Y, R)
    pooled_raw, _ = macro(Y, P)
    gold_all = macro(Y, R, gold)[0] if gold.sum() >= 4 else float("nan")
    gold_raw = macro(Y, P, gold)[0] if gold.sum() >= 4 else float("nan")
    print(f"\n  mean of folds {np.mean(oofs):.4f}   spread {min(oofs):.4f}-{max(oofs):.4f} "
          f"(range {max(oofs) - min(oofs):.4f}; the OOF floor is 0.008)")
    print(f"  POOLED over {len(Y)} studies: {pooled_rank:.4f} fold-rank-normalised "
          f"(raw {pooled_raw:.4f})   gold over all {int(gold.sum())}: {gold_all:.4f} "
          f"(raw {gold_raw:.4f}; SE ~0.05 at n=58 -- direction only)")
    print("\n  per-label pooled AUC (fold-rank-normalised), best to worst:")
    order = np.argsort(-np.asarray(per_label))
    for i in order:
        print(f"    {LABELS[i]:<18}{per_label[i]:.3f}")

    if a.json:
        with open(a.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"version": a.version, "folds": per_fold, "mean_of_folds": float(np.mean(oofs)),
                                 "pooled_rank": pooled_rank, "pooled_raw": pooled_raw, "gold_all": gold_all,
                                 "n_gold": int(gold.sum()), "n": int(len(Y)),
                                 "per_label": dict(zip(LABELS, map(float, per_label)))}) + "\n")


if __name__ == "__main__":
    main()
