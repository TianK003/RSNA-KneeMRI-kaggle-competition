"""P-22: checkpoint selection on OOF-vs-teacher instead of fixed last epoch.

Reads the per-epoch OOF csvs a training kernel writes (`{version}_fold{k}_ep{e}_oof.csv`),
recomputes the kernel's own metric, and answers three questions per arm:

1. How much does fixed-last-epoch leave on the table versus the best epoch (in-sample)?
2. What is that gain *honestly*, when the epoch is chosen on one half of the validation
   studies and scored on the other half? Picking the max of N noisy numbers on the same set
   is optimistically biased; the card's 0.008 floor needs the unbiased version.
3. Does selecting on the teacher chase the teacher -- OOF-vs-teacher rising while gold falls?

Then it re-reads P-09 (v05a vs v05b), P-04 (v05b vs v04d) and the P-21 blend under best-epoch
selection.

Usage (repo root):
    set PYTHONUTF8=1
    python src/oof_epoch_analysis.py [dir ...]        # default: artifacts/kaggle_out/{v13,v6,v8}
Writes artifacts/oof_epoch_analysis.md and prints the same.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Logged numbers the script must reproduce before any new number is trusted
# (docs/experiments.md, kernel v13 table). version -> {epoch: auc_soft}
EXPECTED = {
    "v05a": {7: 0.8574, 6: 0.8576, 4: 0.8536},
    "v05b": {7: 0.8471, 4: 0.8600, 3: 0.8590},
    "v02": {3: 0.821},
}
V04D_LAST = 0.8528      # concat, 4 epochs, kernel v11 -- its OOF csv is not on disk (traps 12e)
FLOOR = 0.008           # measured OOF macro noise floor (v04a vs v04base)
N_SPLITS = 200


def auc_score(y, s) -> float:
    """Copied verbatim from src/kaggle_pipeline.py so the numbers match the kernel log."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_auc(P: np.ndarray, H: np.ndarray, rows=None) -> float:
    """Kernel `evaluate()` semantics: per-label AUC on hard = y > 0.5, mean of finite labels."""
    if rows is not None:
        P, H = P[rows], H[rows]
    vals = [auc_score(H[:, i], P[:, i]) for i in range(P.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def per_label_auc(P, H, rows=None):
    if rows is not None:
        P, H = P[rows], H[rows]
    return {lab: auc_score(H[:, i], P[:, i]) for i, lab in enumerate(LABELS)}


def load_arm(files):
    """-> dict epoch -> (P, H, G, studies); asserts every epoch covers the same studies."""
    out, ref = {}, None
    for e, f in sorted(files.items()):
        d = pd.read_csv(f).sort_values("StudyInstanceUID").reset_index(drop=True)
        if ref is None:
            ref = d.StudyInstanceUID.to_numpy()
        elif not np.array_equal(ref, d.StudyInstanceUID.to_numpy()):
            raise SystemExit(f"{f}: study set differs from epoch {min(files)} -- not one arm")
        P = d[[f"pred__{l}" for l in LABELS]].to_numpy(float)
        Y = d[[f"y__{l}" for l in LABELS]].to_numpy(float)
        H = (Y > 0.5).astype(int)
        G = d["is_gold"].to_numpy() > 0.5
        out[e] = (P, H, G)
    return out, ref


def rank_mean(Ps):
    acc = np.zeros_like(Ps[0])
    for P in Ps:
        acc += pd.DataFrame(P).rank(pct=True).to_numpy()
    return acc / len(Ps)


def split_half(arm, last_e, rng, n_splits=N_SPLITS):
    """Choose the epoch on half A, score on half B. Returns per-split arrays:
    (score_B of chosen epoch, score_B of last epoch, chosen epoch, oracle best on B)."""
    epochs = sorted(arm)
    n = arm[last_e][0].shape[0]
    chosen_b, last_b, chosen_e, oracle_b = [], [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        sa = {e: macro_auc(arm[e][0], arm[e][1], a) for e in epochs}
        best_e = max(epochs, key=lambda e: sa[e])
        sb = {e: macro_auc(arm[e][0], arm[e][1], b) for e in epochs}
        chosen_b.append(sb[best_e]); last_b.append(sb[last_e])
        chosen_e.append(best_e); oracle_b.append(max(sb.values()))
    return np.array(chosen_b), np.array(last_b), np.array(chosen_e), np.array(oracle_b)


def main(dirs):
    files = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*_fold*_ep*_oof.csv")):
            m = re.search(r"(v\w+?)_fold(\d+)_ep(\d+)_oof\.csv$", os.path.basename(f))
            if m:
                files.setdefault((m.group(1), int(m.group(2))), {})[int(m.group(3))] = f
    if not files:
        raise SystemExit(f"no *_fold*_ep*_oof.csv under {dirs}")

    L = []
    pr = L.append
    pr("# P-22 — best-epoch vs fixed-last-epoch checkpoint selection\n")
    pr(f"Inputs: {', '.join(dirs)}. Metric = the kernel's `evaluate()` (hard = y > 0.5, "
       f"macro over finite labels); gold = same on `is_gold` rows. Noise floor {FLOOR} macro.\n")

    arms = {}
    rng = np.random.default_rng(0)
    for (ver, fold), fs in sorted(files.items()):
        arm, studies = load_arm(fs)
        arms[(ver, fold)] = arm
        epochs = sorted(arm)
        last_e = epochs[-1]
        curve = {e: macro_auc(arm[e][0], arm[e][1]) for e in epochs}
        gold = {e: macro_auc(arm[e][0], arm[e][1], arm[e][2]) for e in epochs}
        n_gold = int(arm[last_e][2].sum())
        best_e = max(epochs, key=lambda e: curve[e])

        # -- reproduce the log before believing anything else
        for e, want in EXPECTED.get(ver, {}).items():
            if e in curve and abs(round(curve[e], 4) - want) > 5e-4:
                raise SystemExit(f"{ver} epoch {e}: recomputed {curve[e]:.4f} != logged {want} "
                                 f"-- metric replication is wrong, stop")

        pr(f"\n## {ver} fold {fold} — {len(studies)} val studies, {n_gold} gold, epochs {epochs[0]}–{last_e}\n")
        pr("| epoch | OOF vs teacher | gold (n=%d) |" % n_gold)
        pr("|---|---|---|")
        for e in epochs:
            mark = " ← best" if e == best_e else (" ← checkpointed" if e == last_e else "")
            pr(f"| {e} | {curve[e]:.4f}{mark} | {gold[e]:.4f} |")
        d_in = curve[best_e] - curve[last_e]
        pr(f"\n- in-sample: best epoch {best_e} = {curve[best_e]:.4f}, last = {curve[last_e]:.4f}, "
           f"**Δ = {d_in:+.4f}** ({d_in / FLOOR:.1f}× floor) — biased upward (max of {len(epochs)} noisy values)")

        # -- honest estimate
        cb, lb, ce, ob = split_half(arm, last_e, rng)
        d_honest = float((cb - lb).mean())
        pr(f"- split-half ({N_SPLITS} splits, choose on half A, score on half B): chosen-epoch − last "
           f"= **{d_honest:+.4f}** (sd {float((cb - lb).std()):.4f}; chosen > last in "
           f"{float((cb > lb).mean()):.0%} of splits; oracle-on-B − last = {float((ob - lb).mean()):+.4f})")
        ce_counts = pd.Series(ce).value_counts().sort_index()
        pr(f"- epochs chosen on half A: {dict(ce_counts)}")

        # -- teacher-chasing check
        g_best, g_last = gold[best_e], gold[last_e]
        gpeak = max(epochs, key=lambda e: gold[e])
        pr(f"- gold at best-OOF epoch {best_e}: {g_best:.4f}; at last: {g_last:.4f}; gold's own peak: "
           f"epoch {gpeak} = {gold[gpeak]:.4f}. Gold SE ≈ 0.09 at n≈{n_gold} — read direction only.")
        after = [e for e in epochs if e > best_e]
        if after:
            trend_oof = curve[last_e] - curve[best_e]
            trend_gold = gold[last_e] - gold[best_e]
            pr(f"- after the OOF peak: OOF {trend_oof:+.4f}, gold {trend_gold:+.4f} → "
               + ("both fall together — ordinary overfitting, not teacher-chasing"
                  if trend_oof < 0 and trend_gold < 0 else
                  "gold rises while OOF falls — the last epoch may be *under*-read by the teacher"
                  if trend_gold > 0 and trend_oof < 0 else "mixed"))
        pl_b = per_label_auc(arm[best_e][0], arm[best_e][1])
        pl_l = per_label_auc(arm[last_e][0], arm[last_e][1])
        diffs = {l: pl_b[l] - pl_l[l] for l in LABELS}
        pr("- per-label (best − last): " + ", ".join(f"{l} {v:+.3f}" for l, v in
                                                     sorted(diffs.items(), key=lambda kv: -abs(kv[1]))))
        arms[(ver, fold)] = (arm, curve, gold, best_e, last_e, d_honest)

    # -- verdicts that depend on the policy
    pr("\n## Re-reading closed verdicts under best-epoch selection\n")
    if ("v05a", 0) in arms and ("v05b", 0) in arms:
        a, b = arms[("v05a", 0)], arms[("v05b", 0)]
        pr(f"- **P-09** attn − concat: last-epoch {a[1][a[4]] - b[1][b[4]]:+.4f} (logged +0.0103); "
           f"best-epoch {a[1][a[3]] - b[1][b[3]]:+.4f} (attn ep{a[3]} vs concat ep{b[3]}).")
        pr(f"- **P-04** concat 8 ep vs `v04d` concat 4 ep ({V04D_LAST}): last-epoch "
           f"{b[1][b[4]] - V04D_LAST:+.4f}; best-epoch {b[1][b[3]] - V04D_LAST:+.4f} "
           f"(in-sample, biased; honest gain for this arm was {b[5]:+.4f}).")
        # P-21 blend under both policies
        Pa_l, Pb_l = a[0][a[4]][0], b[0][b[4]][0]
        Pa_b, Pb_b = a[0][a[3]][0], b[0][b[3]][0]
        H = a[0][a[4]][1]
        bl_last = macro_auc(rank_mean([Pa_l, Pb_l]), H)
        bl_best = macro_auc(rank_mean([Pa_b, Pb_b]), H)
        pr(f"- **P-21** rank-mean attn+concat: last-epoch checkpoints {bl_last:.4f} (logged 0.8670); "
           f"best-epoch checkpoints {bl_best:.4f}.")
        # blend across epochs of one arm (snapshot ensemble) -- free, worth one line
        for name, arm_t in (("v05a", a), ("v05b", b)):
            arm = arm_t[0]
            eps = sorted(arm)[-3:]
            snap = macro_auc(rank_mean([arm[e][0] for e in eps]), arm[eps[-1]][1])
            pr(f"- snapshot rank-mean of {name} epochs {eps}: {snap:.4f} "
               f"(last alone {arm_t[1][arm_t[4]]:.4f})")

    pr("\n## Verdict rule (from the card)\n")
    pr(f"Switch to best-OOF checkpointing only if the **split-half** gain exceeds {FLOOR} for the "
       "arms that matter **and** gold does not move against it. In-sample Δ is not evidence.")
    text = "\n".join(L)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/oof_epoch_analysis.md", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["artifacts/kaggle_out/v13", "artifacts/kaggle_out/v6",
                            "artifacts/kaggle_out/v8"]
    main(dirs)
