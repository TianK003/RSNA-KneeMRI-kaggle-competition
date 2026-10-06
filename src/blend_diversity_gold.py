"""Gold-58: what a flat rank-mean gains by pair type, and what one more family would add to B6 (research.md 2.7.8).

Reads each member's 58 gold predictions (the `train_all` `_fold0_oof.csv` = SWA predictions on the gold rows) and the gold
labels in data/train.csv. Prints (1) the gain of every pair's rank-mean over its two members' mean macro-AUC, grouped as
same recipe / same family other architecture / cross-family, (2) a fit gain ~ (1 - mean ρ) + ln(n) over random subsets, and
(3) the fitted marginal value of a sixth member added to B6 by its solo quality and its correlation to B6's members.
Gold-58 is direction only (traps 39). Run from the repo root:  PYTHONUTF8=1 .venv/Scripts/python.exe src/blend_diversity_gold.py
"""
import itertools
import random

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
FILES = {  # member -> artifacts/kaggle_out/<path>_fold0_oof.csv
    "v11a": "c03_D/v11a", "v11b": "c03_D/v11b", "v11n": "p60_part1/v11n", "v11n2": "p60_part1/v11n2",
    "v11d": "sB/v11d", "v11p": "sA/v11p",
    "v13h": "sA/v13h", "v13r": "sC/v13r", "v13rs": "sD/v13rs",
    "v13e": "sC/v13e", "v13e2": "pod_v13e2/v13e2", "v13b3": "pod_v13b3/v13b3", "v13es": "sD/v13es",
    "v13ecp": "pod_v13ecp/v13ecp", "v13ec": "pod_v13ec/v13ec",
}
FAMILY = {a: "CoAtNet" if a.startswith("v11") else "ResNet" if a in ("v13h", "v13r", "v13rs") else "EffNet" for a in FILES}
SAME_RECIPE = [{"v11a", "v11b", "v11n", "v11n2", "v11d", "v11p"}, {"v13r", "v13rs"},
               {"v13e", "v13e2", "v13es", "v13ecp", "v13ec"}]   # seed twins and target / regularisation variants
B6 = ["v11a", "v13r", "v13e", "v13b3", "v13e2"]

gold = pd.read_csv("data/train.csv").dropna(subset=LABELS).set_index("StudyInstanceUID")
Y = gold[LABELS].values.astype(int)
P = {a: pd.read_csv(f"artifacts/kaggle_out/{f}_fold0_oof.csv").set_index("StudyInstanceUID")
     .loc[gold.index, [f"pred__{c}" for c in LABELS]].values for a, f in FILES.items()}


def macro(p):
    return float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(len(LABELS))]))


def rankmean(ps):
    return np.mean([np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])]) for p in ps], axis=0)


def rho(p, q):
    return float(np.mean([spearmanr(p[:, j], q[:, j]).correlation for j in range(p.shape[1])]))


solo = {a: macro(p) for a, p in P.items()}
names = list(P)
R = {frozenset((a, b)): rho(P[a], P[b]) for a, b in itertools.combinations(names, 2)}

rows = []
for a, b in itertools.combinations(names, 2):
    kind = ("same recipe" if any({a, b} <= s for s in SAME_RECIPE)
            else "same family, other arch" if FAMILY[a] == FAMILY[b] else "cross-family")
    rows.append(dict(kind=kind, rho=R[frozenset((a, b))],
                     gain=macro(rankmean([P[a], P[b]])) - (solo[a] + solo[b]) / 2))
df = pd.DataFrame(rows)
print("(1) pair gain over the two members' mean, gold-58")
print(df.groupby("kind").agg(pairs=("gain", "size"), mean_rho=("rho", "mean"), gain=("gain", "mean"),
                             gain_sd=("gain", "std")).round(4).to_string())
print(f"    r(rho, gain) over all {len(df)} pairs = {np.corrcoef(df.rho, df.gain)[0, 1]:.2f}")


def mean_rho(m):
    return float(np.mean([R[frozenset(p)] for p in itertools.combinations(m, 2)]))


random.seed(0)
sub = []
for _ in range(1500):
    m = random.sample(names, random.randint(2, 7))
    sub.append((len(m), mean_rho(m), macro(rankmean([P[a] for a in m])) - np.mean([solo[a] for a in m])))
n, r, g = np.array(sub).T
X = np.column_stack([np.ones_like(n), 1 - r, np.log(n)])
coef = np.linalg.lstsq(X, g, rcond=None)[0]
r2 = 1 - ((g - X @ coef) ** 2).sum() / ((g - g.mean()) ** 2).sum()
print(f"\n(2) gain over members' mean ≈ {coef[0]:+.4f} + {coef[1]:.3f}·(1 - mean ρ) + {coef[2]:.4f}·ln(n)   R² {r2:.2f}")

m5, r5 = float(np.mean([solo[a] for a in B6])), mean_rho(B6)
g5 = coef @ [1, 1 - r5, np.log(5)]
print(f"\n(3) B6: members' mean {m5:.4f}, mean ρ {r5:.3f}, observed gain {macro(rankmean([P[a] for a in B6])) - m5:+.4f}, "
      f"model {g5:+.4f}. Δ(B6 → B6 + X) on gold-58 by X's solo and X's mean ρ to the five:")
print("    X solo    ρ 0.955 (seed)  ρ 0.944 (sibling)  ρ 0.922 (new family)  ρ 0.90")
for sx in (m5 - 0.010, m5 - 0.005, m5, m5 + 0.005):
    cells = []
    for rx in (0.955, 0.944, 0.922, 0.90):
        g6 = coef @ [1, 1 - (r5 * 10 + rx * 5) / 15, np.log(6)]
        cells.append(f"{(sx - m5) / 6 + g6 - g5:+.4f}")
    print(f"    {sx:.4f}    " + "          ".join(cells))
