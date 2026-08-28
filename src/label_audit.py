"""Audit the public LLM report labels that are our entire training signal.

No LLM is called. This reads the three mounted label tables plus train.csv and asks,
per label and per language, where the teacher is weak and why -- so that any
re-labelling effort (P-16) is aimed, and the cheap Synovitis back-fill (P-07) is
measured before it is adopted.

Questions answered (numbers land in artifacts/label_audit.md; per-study rows with
UIDs in artifacts/label_audit.csv, which is gitignored):

  1. Language mix of the 4,407 reports, and of the 58 gold studies.
  2. Per label: how often each source says "unaddressed", the source positive rate vs
     the gold positive rate (prevalence mismatch), and how each source encodes silence.
  3. Per label: pairwise error correlation (phi) of the sources on gold -- do three
     LLMs make one vote or three?
  4. Per language x label: pilkwang's UNK rate (coverage), and source agreement.
  5. Synovitis <- Effusion back-fill: coverage gained per language, gold AUC before /
     after with a paired bootstrap CI, and the weight recomputation it needs.
  6. Gold and weak label co-occurrence (phi), for the per-label-head risk in P-09.

Usage:  python src/label_audit.py            (run from the repo root, PYTHONUTF8=1)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_targets import (  # noqa: E402
    LABELS, WEAK_WEIGHT_FLOOR, auc, auc_se, confidence_weights, load_sources, prob_blend,
)

WEAK = ["Synovitis", "Lateral OA", "Fracture", "Contusion", "Lateral Meniscus", "Effusion"]
OUT_DIR = "artifacts"


# ------------------------------------------------------------------ helpers

def detect_language(reports: pd.Series) -> pd.Series:
    """langdetect if installed (seeded, deterministic), else a stopword heuristic."""
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0

        def one(t):
            t = (t or "").strip()
            if len(t) < 20:
                return "unk"
            try:
                return detect(t[:2000])
            except Exception:
                return "unk"
        return reports.fillna("").map(one)
    except ImportError:
        print("  langdetect not installed -- falling back to stopword heuristic")
        stop = {
            "en": {" the ", " and ", " with ", " no ", " of "},
            "de": {" und ", " der ", " kein ", " mit ", " nicht "},
            "nl": {" en ", " van ", " geen ", " het ", " met "},
            "fr": {" et ", " le ", " pas ", " avec ", " des "},
            "es": {" y ", " el ", " sin ", " con ", " del "},
            "it": {" e ", " il ", " non ", " con ", " del "},
            "pt": {" e ", " do ", " sem ", " com ", " da "},
            "ru": {" и ", " не ", " в ", " на ", " без "},
            "el": {" και ", " της ", " με ", " χωρίς ", " του "},
        }

        def one(t):
            t = f" {(t or '').lower()} "
            best, score = "unk", 0
            for lang, words in stop.items():
                s = sum(t.count(w) for w in words)
                if s > score:
                    best, score = lang, s
            return best
        return reports.map(one)


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient between two binary vectors (Pearson r on 0/1)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def paired_bootstrap_delta(y, s_new, s_old, n_boot=4000, seed=0):
    """Bootstrap the *difference* in AUC on the same resampled studies. Returns
    (delta, lo, hi) -- the interval a 58-study comparison honestly supports."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        a_new = auc(y[ix], s_new[ix])
        a_old = auc(y[ix], s_old[ix])
        if np.isfinite(a_new) and np.isfinite(a_old):
            deltas.append(a_new - a_old)
    if not deltas:
        return (float("nan"),) * 3
    return (auc(y, s_new) - auc(y, s_old),
            float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def fmt_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join([d.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in d.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------- main

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    md: list[str] = ["# Label audit", "",
                     "Generated by `src/label_audit.py`. Aggregates only -- no UIDs, no "
                     "report text. Per-study rows are in `artifacts/label_audit.csv` "
                     "(gitignored).", ""]

    def log(s: str = "") -> None:
        print(s)
        md.append(s)

    tr = pd.read_csv("data/train.csv")
    idx = pd.Index(tr.StudyInstanceUID, name="StudyInstanceUID")
    is_gold = tr[LABELS].notna().all(axis=1).to_numpy()
    gold = tr.set_index("StudyInstanceUID")[LABELS].astype(float)
    gy = gold.loc[idx[is_gold]]

    print("loading sources:")
    sources = load_sources("data/llm_labels", idx)
    # pilkwang ships a verdict column per label: YES / NO / UNK (= not addressed).
    pk_raw = pd.read_csv("data/llm_labels/rsna-knee-llm-labels/report_labels_v2.csv"
                         ).set_index("StudyInstanceUID").reindex(idx)
    verdict = {l: pk_raw[f"{l}__verdict"] for l in LABELS if f"{l}__verdict" in pk_raw}

    soft = prob_blend(sources, idx)
    weights = confidence_weights(sources, idx)

    # ---------------------------------------------------------------- 1. language
    log("## 1. Languages")
    log("")
    lang = detect_language(tr.Report)
    lang_all = lang.value_counts()
    lang_gold = lang[is_gold].value_counts()
    t = pd.DataFrame({"reports": lang_all, "share": lang_all / len(tr),
                      "gold": lang_gold}).fillna({"gold": 0})
    t["gold"] = t["gold"].astype(int)
    t.index.name = "lang"
    log(fmt_table(t.head(15)))
    log("")
    log(f"{lang_all.size} languages detected; top {min(15, lang_all.size)} shown. "
        f"Gold covers {lang_gold.size} of them.")
    log("")

    # ------------------------------------------------- 2. silence and prevalence
    log("## 2. How each source encodes silence, and prevalence vs gold")
    log("")
    log("`unaddressed` = pilkwang verdict UNK (the only source that flags silence). "
        "`hans on UNK` = mean hans_v4 value on those rows; `sol56==0` cannot separate "
        "'reported absent' from 'not mentioned'.")
    log("")
    rows = []
    for l in LABELS:
        r = {"gold pos rate": float(gy[l].mean())}
        if l in verdict:
            v = verdict[l]
            r["pilk UNK"] = float((v == "UNK").mean())
            r["pilk YES"] = float((v == "YES").mean())
        if l in verdict:
            unk_rows = (verdict[l] == "UNK").to_numpy()
            r["hans on UNK"] = float(sources["hans_v4"][l].to_numpy()[unk_rows].mean()) if unk_rows.any() else np.nan
            r["blend on UNK"] = float(soft[l].to_numpy()[unk_rows].mean()) if unk_rows.any() else np.nan
            r["w on UNK"] = float(weights[l].to_numpy()[unk_rows].mean()) if unk_rows.any() else np.nan
            r["w on addressed"] = float(weights[l].to_numpy()[~unk_rows].mean()) if (~unk_rows).any() else np.nan
        r["hans>=0.5"] = float((sources["hans_v4"][l] >= 0.5).mean())
        r["sol56>=0.5"] = float((sources["sol56"][l] >= 0.5).mean())
        r["blend>=0.5"] = float((soft[l] >= 0.5).mean())
        r["w at floor"] = float((weights[l] <= WEAK_WEIGHT_FLOOR + 1e-9).mean())
        rows.append(pd.Series(r, name=l))
    prev = pd.DataFrame(rows)
    prev.index.name = "label"
    log(fmt_table(prev))
    log("")
    hi_silence = prev.sort_values("pilk UNK", ascending=False).head(4)
    log("Most-silent labels (pilkwang UNK): " + ", ".join(
        f"{l} {v:.0%}" for l, v in hi_silence["pilk UNK"].items()))
    log("Largest gold-vs-blend positive-rate gaps: " + ", ".join(
        f"{l} gold {g:.0%} vs blend {b:.0%}" for l, g, b in
        (prev["gold pos rate"] - prev["blend>=0.5"]).abs().sort_values(ascending=False)
        .head(4).index.map(lambda l: (l, prev.loc[l, "gold pos rate"], prev.loc[l, "blend>=0.5"]))))
    log("")

    # -------------------------------------------- 3. error correlation on gold
    log("## 3. Do three LLM sources make three votes or one? (error phi on gold)")
    log("")
    log("Each source binarised at 0.5; error = disagrees with gold. phi near 0 = "
        "independent errors (blending helps); phi > 0.4 = one effective vote. phi = 1.0 "
        "means identical decisions at the 0.5 cut, which is consistent with (not proof of) "
        "one table containing the other; raw values still differ.")
    log("")
    names = list(sources)
    rows = []
    for l in LABELS:
        errs = {n: ((sources[n][l].loc[gy.index] >= 0.5).astype(int) != gy[l]).astype(int).to_numpy()
                for n in names}
        r = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r[f"{names[i]}~{names[j]}"] = phi(errs[names[i]], errs[names[j]])
        for n in names:
            r[f"err {n}"] = float(errs[n].mean())
        rows.append(pd.Series(r, name=l))
    corr = pd.DataFrame(rows)
    corr.index.name = "label"
    log(fmt_table(corr))
    pair_cols = [c for c in corr.columns if "~" in c]
    log("")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agree = np.mean([((sources[names[i]][l] >= 0.5) == (sources[names[j]][l] >= 0.5)).mean()
                             for l in LABELS])
            log(f"Agreement at the 0.5 cut over all 4,407 studies, {names[i]}~{names[j]}: {agree:.4f}")
    log(f"Mean pairwise error phi over labels: {np.nanmean(corr[pair_cols].to_numpy()):.2f} "
        f"(literature: frontier-LLM panels ~0.39, ~2.2 effective votes).")
    log("")

    # ------------------------------------- 4. coverage and agreement by language
    log("## 4. Coverage by language (pilkwang UNK rate) for the weak labels")
    log("")
    top_langs = lang_all.index[:8].tolist()
    rows = []
    for lg in top_langs:
        m = (lang == lg).to_numpy()
        r = {"n": int(m.sum())}
        for l in WEAK:
            if l in verdict:
                r[l] = float((verdict[l][m] == "UNK").mean())
        rows.append(pd.Series(r, name=lg))
    cov = pd.DataFrame(rows)
    cov.index.name = "lang"
    log(fmt_table(cov))
    log("")
    log("## 4b. Source agreement by language (mean Spearman hans_v4~pilkwang, all labels; "
        "sol56 excluded as a near-duplicate of hans_v4)")
    log("")
    rows = []
    for lg in top_langs:
        m = (lang == lg).to_numpy()
        vals = []
        pair_names = [n for n in ("hans_v4", "pilkwang") if n in sources]
        for l in LABELS:
            for i in range(len(pair_names)):
                for j in range(i + 1, len(pair_names)):
                    a = sources[pair_names[i]][l][m]
                    b = sources[pair_names[j]][l][m]
                    ok = a.notna() & b.notna()
                    if ok.sum() > 20 and a[ok].std() > 0 and b[ok].std() > 0:
                        # Spearman without scipy: Pearson on ranks.
                        vals.append(float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1]))
        rows.append(pd.Series({"n": int(m.sum()), "mean spearman": float(np.mean(vals)) if vals else np.nan},
                              name=lg))
    agr = pd.DataFrame(rows)
    agr.index.name = "lang"
    log(fmt_table(agr))
    log("")

    # -------------------------------------------- 5. Synovitis <- Effusion
    log("## 5. P-07: Synovitis back-filled from Effusion where Synovitis is unaddressed")
    log("")
    if "Synovitis" in verdict:
        unk = (verdict["Synovitis"] == "UNK").to_numpy()
        new_syn = soft["Synovitis"].to_numpy().copy()
        new_syn[unk] = soft["Effusion"].to_numpy()[unk]
        # weights should be recomputed from the back-filled values: the original
        # Synovitis sources on these rows are ~0.28/0.14/0 (a de-facto confident negative,
        # mean weight ~0.69), not the Effusion evidence the new target now carries.
        new_w = weights["Synovitis"].to_numpy().copy()
        new_w[unk] = weights["Effusion"].to_numpy()[unk]
        y = gy["Synovitis"].to_numpy()
        old_s = soft.loc[gy.index, "Synovitis"].to_numpy()
        new_s = pd.Series(new_syn, index=idx).loc[gy.index].to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, new_s, old_s)
        a_old, a_new = auc(y, old_s), auc(y, new_s)
        se = auc_se(a_old, int(y.sum()), int((1 - y).sum()))
        log(f"- Unaddressed Synovitis: {unk.mean():.1%} of studies ({int(unk.sum())}); "
            f"of the 58 gold, {int(unk[is_gold].sum())} are unaddressed and "
            f"{int(y[unk[is_gold]].sum())} of those are gold-positive.")
        log(f"- Gold AUC Synovitis: current blend {a_old:.3f} (HM SE {se:.3f}) -> "
            f"back-filled {a_new:.3f}; paired-bootstrap delta {d:+.3f} "
            f"[95% CI {lo:+.3f}, {hi:+.3f}].")
        log(f"- Mean Synovitis weight on unaddressed rows: {weights['Synovitis'].to_numpy()[unk].mean():.2f} "
            f"-> {new_w[unk].mean():.2f} after recomputing from Effusion.")
        cov_lang = pd.DataFrame({
            "unaddressed (eligible)": pd.Series(unk, index=tr.index).groupby(lang).mean(),
            "n": lang.value_counts()}).loc[top_langs]
        cov_lang.index.name = "lang"
        log("")
        log("Unaddressed Synovitis per language (rows eligible for back-fill):")
        log("")
        log(fmt_table(cov_lang))
        log("")
        verdict_note = ("CI excludes 0 -> direction is supported on gold"
                        if lo > 0 else
                        "CI includes 0 -> INCONCLUSIVE on gold, as expected at n=58; "
                        "judge on coverage + student OOF")
        log(f"Verdict on gold: {verdict_note}.")
        log("")
        log("Contusion <- bone-marrow-oedema synonyms: no source exposes the matched "
            "terms, so this cannot be audited from the tables; it is an open question "
            "for P-16 (native-language open-weights re-read with a synonym list).")
    log("")

    # -------------------------------------------------- 6. co-occurrence
    log("## 6. Label co-occurrence (phi) -- gold above the diagonal, weak blend>=0.5 below")
    log("")
    co = pd.DataFrame(index=LABELS, columns=LABELS, dtype=float)
    hard_weak = (soft >= 0.5).astype(int)
    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            if i < j:
                co.loc[a, b] = phi(gy[a], gy[b])
            elif i > j:
                co.loc[a, b] = phi(hard_weak[a], hard_weak[b])
    co.index.name = "label"
    short = {l: l.replace("Medial ", "M").replace("Lateral ", "L").replace("Meniscus", "Men")
             for l in LABELS}
    co2 = co.rename(index=short, columns=short)
    co2.index.name = "label"
    log(fmt_table(co2, "{:.2f}"))
    pairs = [("Effusion", "Synovitis"), ("Medial OA", "Medial Meniscus"),
             ("Contusion", "Fracture"), ("Medial OA", "Lateral OA")]
    log("")
    def gold_weak(a, b):
        i, j = LABELS.index(a), LABELS.index(b)
        lo_, hi_ = min(i, j), max(i, j)
        return co.iloc[lo_, hi_], co.iloc[hi_, lo_]      # gold above, weak below diagonal
    log("Clinically expected pairs -- gold phi / weak phi (n=58 gold, SE of phi ~0.13): "
        + "; ".join(f"{a}~{b} {gold_weak(a, b)[0]:.2f} / {gold_weak(a, b)[1]:.2f}"
                    for a, b in pairs))
    log("")

    # ------------------------------------------------------------- write
    per_study = pd.DataFrame({"StudyInstanceUID": idx, "lang": lang.to_numpy(),
                              "is_gold": is_gold.astype(int)})
    for l in LABELS:
        if l in verdict:
            per_study[f"unk__{l}"] = (verdict[l] == "UNK").astype(int).to_numpy()
        per_study[f"blend__{l}"] = soft[l].to_numpy()
    per_study.to_csv(os.path.join(OUT_DIR, "label_audit.csv"), index=False)
    with open(os.path.join(OUT_DIR, "label_audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nwrote {OUT_DIR}/label_audit.md and {OUT_DIR}/label_audit.csv")


if __name__ == "__main__":
    main()
