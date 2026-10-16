"""Public vs private LB: a binormal Monte-Carlo of the hidden test (no competition data used).

Model: N = 1,284 studies (nartaa's script), public 30 % / private 70 % (forum, unverified), 12 independent labels at the gold-58
prevalences. Scores per label are binormal with unit-variance noise (AUC = Phi(mu / sqrt 2)). heliosli's public blend is two halves,
A (the CoAtNet "Raptor" half) and B (the R6 half), each ~0.950, rank-blended 0.5 / 0.5; their error correlation rho is calibrated so
the blend reads 0.957.

  1. noise:     SD of macro AUC on public / private, and of public - private, for one fixed submission.
  2. siblings:  swap B for a sibling with the same true AUC and the same correlation with A (corr(B, B') = rho_sib): the SD of the
                public difference, its correlation with the private difference, and the winner's-curse optimism of the best of K.
  3. weights:   heliosli-style per-finding weights. Per label, read the public AUC of w * rank(A) + (1 - w) * rank(B) on a grid
                (exactly, or through a per-class probe whose LB rounding gives 12 x 0.0005 = +-0.006 of class AUC), keep the argmax
                (ties -> 0.5); report the macro gain on public and on private, for three levels of true per-label difference.
  4. outsiders: an independent pipeline with a true lead over the plateau blend: how often the lead flips on private.

Run: .venv/Scripts/python.exe src/pubpriv_sim.py [--reps 300]   (≈ 5 min on a laptop CPU)
"""
import argparse

import numpy as np
from scipy.stats import norm, rankdata

PREV = np.array([0.41, 0.16, 0.45, 0.40, 0.26, 0.19, 0.36, 0.60, 0.47, 0.21, 0.33, 0.31])  # CLAUDE.md "Verified data facts"
N, PUB = 1284, 0.30
GRID = np.array([0.20, 0.35, 0.50, 0.65, 0.80])
rng = np.random.default_rng(20261016)


def auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mu_for(a):
    return np.sqrt(2) * norm.ppf(a)


def macro(y, s, idx):
    return np.mean([auc(y[idx, c], s[idx, c]) for c in range(12)])


def split():
    idx = rng.permutation(N)
    k = int(round(PUB * N))
    return idx[:k], idx[k:]


def labels():
    return (rng.random((N, 12)) < PREV).astype(int)


def blend(sa, sb, w):
    return w * rankdata(sa, axis=0) + (1 - w) * rankdata(sb, axis=0)


def calibrate_rho(target=0.957, half=0.950):
    return (mu_for(half) / norm.ppf(target)) ** 2 - 1  # Phi(mu / sqrt(2 * (0.5 + 0.5 rho))) = target


def halves(y, auc_a, auc_b, rho):
    z1, z2 = rng.standard_normal((N, 12)), rng.standard_normal((N, 12))
    eb = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    return mu_for(auc_a) * y + z1, mu_for(auc_b) * y + eb, z1, z2


def exp_noise_and_siblings(reps, rho):
    print("1. one fixed submission (the 0.5 / 0.5 blend of two 0.950 halves):")
    pubs, privs = [], []
    sib = {rs: ([], []) for rs in (0.98, 0.95, 0.90)}
    m = mu_for(0.950)
    for _ in range(reps):
        y = labels()
        sa, sb, z1, z2 = halves(y, 0.950, 0.950, rho)
        pub, priv = split()
        s0 = blend(sa, sb, 0.5)
        p0, q0 = macro(y, s0, pub), macro(y, s0, priv)
        pubs.append(p0)
        privs.append(q0)
        for rs, (dp, dq) in sib.items():
            r = (rs - rho ** 2) / (1 - rho ** 2)  # keeps corr(B', A) = rho and gives corr(B', B) = rs
            z4 = rng.standard_normal((N, 12))
            sb2 = m * y + rho * z1 + np.sqrt(1 - rho ** 2) * (r * z2 + np.sqrt(1 - r ** 2) * z4)
            s1 = blend(sa, sb2, 0.5)
            dp.append(macro(y, s1, pub) - p0)
            dq.append(macro(y, s1, priv) - q0)
    pubs, privs = np.array(pubs), np.array(privs)
    print(f"   public sd {pubs.std():.4f} | private sd {privs.std():.4f} | public - private sd {(pubs - privs).std():.4f}")
    print("2. sibling swap with no true difference (R7 -> R6, TTA on / off):")
    for rs, (dp, dq) in sib.items():
        dp, dq = np.array(dp), np.array(dq)
        curse = ", ".join(f"K={k} +{e * dp.std():.4f}" for k, e in ((3, 0.846), (5, 1.163), (8, 1.424)))
        print(f"   corr(B, B') {rs}: public diff sd {dp.std():.4f}, private diff sd {dq.std():.4f}, "
              f"corr(public, private diff) {np.corrcoef(dp, dq)[0, 1]:+.2f}; best of K on public: {curse}")


def exp_weights(reps, rho, het_sd, resolution):
    gp, gq, nchg = [], [], []
    for _ in range(reps):
        a_c = np.clip(0.950 + rng.normal(0, het_sd, 12), 0.88, 0.995)
        b_c = np.clip(0.950 + rng.normal(0, het_sd, 12), 0.88, 0.995)
        y = labels()
        sa, sb, _, _ = halves(y, a_c, b_c, rho)
        pub, priv = split()
        ra, rb = rankdata(sa, axis=0), rankdata(sb, axis=0)
        g_p = g_q = 0.0
        changed = 0
        for c in range(12):
            vals = np.array([auc(y[pub, c], (w * ra + (1 - w) * rb)[pub, c]) for w in GRID])
            obs = vals if resolution == 0 else np.round(vals / resolution) * resolution
            cand = np.where(obs == obs.max())[0]
            j = 2 if 2 in cand else cand[np.argmin(np.abs(GRID[cand] - 0.5))]
            if j == 2:
                continue
            changed += 1
            sw = (GRID[j] * ra + (1 - GRID[j]) * rb)[:, c]
            s5 = (0.5 * ra + 0.5 * rb)[:, c]
            g_p += auc(y[pub, c], sw[pub]) - auc(y[pub, c], s5[pub])
            g_q += auc(y[priv, c], sw[priv]) - auc(y[priv, c], s5[priv])
        gp.append(g_p / 12)
        gq.append(g_q / 12)
        nchg.append(changed)
    gp, gq = np.array(gp), np.array(gq)
    kept = gq.mean() / gp.mean() if gp.mean() > 0 else float("nan")
    print(f"   per-label AUC sd {het_sd} per half, {'exact readout' if resolution == 0 else 'probe +-0.006'}: "
          f"{np.mean(nchg):.1f} labels changed | public {gp.mean():+.4f} | private {gq.mean():+.4f} "
          f"(P(< 0) {np.mean(gq < 0):.2f}) | kept {kept:.0%}")


def exp_outsiders(reps):
    print("4. an independent pipeline with a true lead over a 0.958 plateau blend:")
    for rho in (0.7, 0.8, 0.9):
        for lead in (0.002, 0.004):
            dp, dq = [], []
            for _ in range(reps):
                y = labels()
                z1, z2 = rng.standard_normal((N, 12)), rng.standard_normal((N, 12))
                sx = mu_for(0.958) * y + z1
                sy = mu_for(0.958 + lead) * y + rho * z1 + np.sqrt(1 - rho ** 2) * z2
                pub, priv = split()
                dp.append(macro(y, sy, pub) - macro(y, sx, pub))
                dq.append(macro(y, sy, priv) - macro(y, sx, priv))
            dp, dq = np.array(dp), np.array(dq)
            print(f"   error corr {rho}, true lead {lead}: public lead sd {dp.std():.4f} | P(private lead < 0) {np.mean(dq < 0):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    reps = ap.parse_args().reps
    rho = calibrate_rho()
    print(f"calibrated error correlation of the two halves: {rho:.3f}")
    exp_noise_and_siblings(reps, rho)
    print("3. per-finding weights picked on public (grid 0.20 .. 0.80):")
    for het in (0.0, 0.0075, 0.015):
        for res in (0.0, 0.012):
            exp_weights(reps, rho, het, res)
    exp_outsiders(reps)
