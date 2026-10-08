"""P-81: OAI baseline MOAKS readings -> masked soft training targets for the 12 competition labels.

One row per OAI knee with a baseline MOAKS reading (kMRI_SQ_MOAKS_BICL00: 2,400 knees, projects 22 / 30 / 63 / 65).
StudyInstanceUID = "OAI_<participant>_<R|L>" -- the id the OAI cache shard (src/build_oai_cache.py) stores the knee under.

Columns, in the layout src/kaggle_pipeline.py appends to `targets` for an arm with `oai=True`:
    StudyInstanceUID, oai_id, side, yt__<label> (soft target in [0, 1]), w__<label> (1.0 supervised, 0.0 masked)
A masked cell has w = 0, so the per-study weighted BCE (`weighted_bce`) averages over the supervised cells only.

Supervised labels (Tian, 2026-10-08, after the critic: replicate the 0.949 author's three = our three weakest gold-58 labels;
Lateral Meniscus, our fourth weakest, is computed but MASKED in this first table -- `--with-lateral-meniscus` adds it);
every other label is masked:
    Synovitis        Hoffa-synovitis grade 0-3 (V00MSYIC) / 3                                        (1,660 knees)
    PF OA            max over patella M/L + trochlea M/L of cartilage (area grade + full-thickness grade), / 4, clip 1
    Lateral OA       the same over the lateral femur (central, posterior) + lateral tibia (anterior, central, posterior)
    (Lateral Meniscus, optional: 1 for a tear or maceration in any lateral horn (codes 2-8) or a lateral root tear,
                     0.2 for signal abnormality only (code 1), 0 for normal)
A knee read in several projects takes, per variable, the first non-missing value in project order 22, 30, 65, 63*.

Reads data/oai/ (gitignored; OAI terms forbid redistribution). Writes artifacts/oai/oai_targets.csv (gitignored).
Usage: python src/build_oai_targets.py [--check]       (--check: rebuild in memory and compare with the file on disk)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SUPERVISED = ("Synovitis", "PF OA", "Lateral OA")
OPTIONAL = ("Lateral Meniscus",)                     # --with-lateral-meniscus
MOAKS = "data/oai/MR Image Assessment_ASCII/Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL00.txt"
OUT = "artifacts/oai/oai_targets.csv"
PROJECT_ORDER = {"22": 0, "30": 1, "65": 2}           # every 63A-F after these
PF_CART = ["V00MCMPM", "V00MCMPL", "V00MCMFMA", "V00MCMFLA"]
LAT_CART = ["V00MCMFLC", "V00MCMFLP", "V00MCMTLA", "V00MCMTLC", "V00MCMTLP"]
LAT_MEN = ["V00MMTLA", "V00MMTLB", "V00MMTLP"]


def code(series):
    """'2.1: 10-75% area, 1-10% full thickness' -> 2.1 ; '.: Missing Form/...' -> NaN."""
    return pd.to_numeric(series.astype(str).str.split(":").str[0].str.strip(), errors="coerce")


def cartilage_severity(v):
    """MOAKS cartilage 'a.f' (a = area grade 0-3, f = full-thickness grade 0-3) -> a + f in 0..6; NaN stays NaN."""
    a = np.floor(v + 1e-9)
    f = np.round((v - a) * 10)
    return a + f


def knee_table(moaks_path=MOAKS):
    m = pd.read_csv(moaks_path, sep="|", low_memory=False, dtype=str)
    m["side"] = m["SIDE"].str[0].map({"1": "R", "2": "L"})
    if m["side"].isna().any():
        raise SystemExit(f"unexpected SIDE values: {sorted(m.loc[m.side.isna(), 'SIDE'].unique())}")
    m["prio"] = m["READPRJ"].astype(str).map(lambda p: PROJECT_ORDER.get(p, 3))
    feats = ["V00MSYIC", "V00MMRTL", *PF_CART, *LAT_CART, *LAT_MEN]
    for c in feats:
        m[c] = code(m[c])
    m = m.sort_values(["ID", "side", "prio", "READPRJ"])
    return m.groupby(["ID", "side"])[feats].first()    # first non-missing per variable, in project order


def build(moaks_path=MOAKS, supervised=SUPERVISED):
    k = knee_table(moaks_path)
    out = pd.DataFrame(index=k.index)
    syn = k["V00MSYIC"]
    out["yt__Synovitis"] = (syn / 3.0).clip(0, 1)
    for name, cols in (("PF OA", PF_CART), ("Lateral OA", LAT_CART)):
        sev = pd.concat([cartilage_severity(k[c]) for c in cols], axis=1).max(axis=1, skipna=True)
        sev[k[cols].isna().all(axis=1)] = np.nan
        out[f"yt__{name}"] = (sev / 4.0).clip(0, 1)
    men = k[LAT_MEN]
    tear = men.isin([2, 3, 4, 5, 6, 7, 8]).any(axis=1) | (k["V00MMRTL"] == 1)
    signal = (men == 1).any(axis=1)
    lm = np.where(tear, 1.0, np.where(signal, 0.2, 0.0))
    out["yt__Lateral Meniscus"] = np.where(men.isna().all(axis=1), np.nan, lm)
    for lab in LABELS:
        col = f"yt__{lab}"
        if lab in supervised:
            out[f"w__{lab}"] = out[col].notna().astype(float)
            out[col] = out[col].fillna(0.5)
        else:
            out[col] = 0.5
            out[f"w__{lab}"] = 0.0
    out = out.reset_index().rename(columns={"ID": "oai_id"})
    out.insert(0, "StudyInstanceUID", "OAI_" + out.oai_id.astype(str) + "_" + out.side)
    cols = ["StudyInstanceUID", "oai_id", "side", *[f"yt__{l}" for l in LABELS], *[f"w__{l}" for l in LABELS]]
    out = out[cols].sort_values("StudyInstanceUID").reset_index(drop=True)
    empty = out[[f"w__{l}" for l in LABELS]].sum(axis=1) == 0
    if empty.any():
        print(f"  dropped {int(empty.sum())} knee(s) with no supervised cell (every supervised MOAKS variable missing)")
        out = out[~empty].reset_index(drop=True)
    validate(out, supervised)
    return out


def validate(t, supervised=SUPERVISED):
    if t.StudyInstanceUID.duplicated().any():
        raise SystemExit("duplicate knee ids")
    for lab in LABELS:
        y, w = t[f"yt__{lab}"], t[f"w__{lab}"]
        if y.isna().any() or ((y < 0) | (y > 1)).any():
            raise SystemExit(f"{lab}: soft target outside [0, 1] or NaN")
        if not set(w.unique()) <= {0.0, 1.0}:
            raise SystemExit(f"{lab}: weights must be 0 or 1, got {sorted(w.unique())}")
        if lab not in supervised and w.any():
            raise SystemExit(f"{lab} is not supervised but has weight")
    if (t[[f"w__{l}" for l in LABELS]].sum(axis=1) == 0).any():
        raise SystemExit("a knee with no supervised cell would divide by zero in weighted_bce")


def summary(t, supervised=SUPERVISED):
    print(f"OAI targets: {len(t)} knees ({(t.side == 'R').sum()} right, {(t.side == 'L').sum()} left)")
    for lab in supervised:
        w = t[f"w__{lab}"] == 1
        y = t.loc[w, f"yt__{lab}"]
        print(f"  {lab:18s} supervised {int(w.sum()):5d}  mean {y.mean():.3f}  >=0.5 {(y >= 0.5).mean():.2f}  "
              f"values {sorted(y.round(2).unique())[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--moaks", default=MOAKS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--with-lateral-meniscus", action="store_true")
    a = ap.parse_args()
    sup = SUPERVISED + (OPTIONAL if a.with_lateral_meniscus else ())
    t = build(a.moaks, sup)
    summary(t, sup)
    if a.check:
        old = pd.read_csv(a.out)
        same = old.shape == t.shape and np.allclose(old.drop(columns=["StudyInstanceUID", "oai_id", "side"]).to_numpy(float),
                                                     t.drop(columns=["StudyInstanceUID", "oai_id", "side"]).to_numpy(float))
        print("check:", "identical" if same else "DIFFERS")
        sys.exit(0 if same else 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t.to_csv(a.out, index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
