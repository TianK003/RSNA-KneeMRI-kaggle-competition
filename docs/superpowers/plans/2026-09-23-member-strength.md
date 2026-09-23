# Member strength (teacher upgrade + last recipe knobs) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give our members a stronger teacher (the public Raptor CoAtNet's predictions on our training studies, mixed 0.5/0.5 with the LLM blend) and measure the last untested recipe knobs (16 ep × 3e-5, `pos_weight`, self-distillation) on fold 0, so that a member of ours can reach ≥ 0.92 solo on the LB.

**Architecture:** (1) `build_targets` (local script and kernel copy) learns *prediction-table sources*: CSVs of per-study predictions that are quantile-matched per label onto the LLM blend and mixed into **training** targets `yt__*`, while the evaluation targets `y__*` stay the unchanged 3-source teacher. (2) A new builder extracts the public 0.942 notebook's Raptor branch verbatim into its own Kaggle kernel that runs over chunks of *training* studies (redirected through `RSNA_COMP_ROOT`) and saves raw per-view probabilities; a merge script turns the shards into one teacher table. (3) Three fold-0 arms (`v09e`, `v09f`, `v09s`) run on a RunPod 4090 through the existing bootstrap; `pos_weight` is one Config field and one loss argument.

**Tech Stack:** Python 3.11 (`.venv/Scripts/python.exe`), pandas / numpy / torch (CPU locally), the percent-format pipeline `src/kaggle_pipeline.py` + `src/nbgen.py`, Kaggle CLI (`.venv/Scripts/kaggle.exe`), RunPod via `scripts/runpod_bootstrap.sh`. No pytest in the venv: checks are `src/*_test.py` scripts that print `ok`/`FAIL` and exit non-zero on failure (the `window_head_test.py` pattern).

**Spec:** `docs/superpowers/specs/2026-09-23-member-strength-design.md`

## Global Constraints

- Every file under `docs/`, `CLAUDE.md` and `src/kaggle_pipeline.py` is **CRLF**; `docs/traps.md` and new files may be LF. Scripted edits must match the file's line endings (a `\n`-only multi-line pattern matches nothing in a CRLF file).
- `FORCE_SMOKE = True` on the first Kaggle push after any edit; the real run is a sed'd copy in `artifacts/`, never an edit of `src/kaggle_pipeline.py`'s switches. Edit `src/kaggle_pipeline.py`, never a generated `.ipynb`.
- `"machine_shape": "NvidiaTeslaT4"` in every `kernel-metadata.json` (never P100). `enable_internet: false`.
- After any `kaggle kernels push` error that is not a clear 4xx, run `kernels status` **before** retrying (traps 36: the version may already exist). Wrap CLI calls in `timeout 60 …` (Git Bash) / `Start-Process … WaitForExit` (PowerShell).
- Never download the competition images in bulk (the teacher pass runs on Kaggle, where `train_images` is mounted).
- The default teacher must stay byte-identical: `python src/build_targets.py` prints `BLEND 0.8948` and `artifacts/targets.csv` keeps md5 `29f641ed…` (check with `--check-md5`, Task 1).
- Evaluation targets `y__*` and every OOF csv column keep their meaning; only the loss may read `yt__*`.
- Verdicts by the noise floors: fold-0 OOF 0.008 macro / 0.03 per label; gold-58 0.05; public LB 0.005. Untried ideas are cards in `docs/proposals.md` *before* they run; results go to `docs/experiments.md` with a verdict (the `/update` skill).
- Commit messages carry no AI attribution (managed policy). Real Kaggle runs and submissions happen only where this plan says Tian's go exists (design approval 2026-09-23 covers: the LIMIT-6 smoke and LIMIT-100 spike of the teacher kernel, the RunPod arms, the baseline and distilled solo submissions, the full teacher pass after the quota reset).
- Bash tool: `export PATH="/usr/bin:/bin:$PATH"` at the top of every call (no `sed`/`grep` otherwise; no `curl`); `git` through PowerShell.

## Review Focus

1. **A teacher table that lists a study twice or lacks a label column** must fail loudly in `load_teacher_tables`, never silently take the first row — Task 1 `test_duplicate_uid_rejected` / `test_missing_label_rejected`.
2. **A listed `TEACHER_TABLES` name whose CSV is not mounted on Kaggle** must abort the kernel (like an `INFER_MEMBERS` version without a checkpoint), never train on the plain teacher under a distilled version name — Task 3 check `unmounted teacher table is fatal`.
3. **All-NaN or constant prediction column** in quantile matching (e.g. a label the teacher never predicts): output must stay finite and equal the reference median where ranks tie, not raise or emit NaN into `yt` — Task 1 `test_constant_column`.
4. **`pos_weight_max` with a label whose training positive rate is 0 or 1** must clip to a finite weight, never divide by zero — Task 4 `pos_weight extremes clip`.
5. **A chunk preamble that includes a gold study or duplicates a UID across shards** would leak validation into the teacher or double-count a study in the merge — Task 7 `test_chunk_disjoint_and_gold_free`; Task 8 asserts 4,349 unique UIDs.

---

### Task 1: Prediction-table sources with quantile matching in `src/build_targets.py`

**Files:**
- Modify: `src/build_targets.py` (add constants + 3 functions after `confidence_weights`, ~line 188; extend `main()` ~lines 234–330)
- Create: `src/targets_test.py`

**Interfaces:**
- Produces: `TEACHER_ROOT = "artifacts/teacher"`; `load_teacher_tables(root: str, index: pd.Index, names: list[str]) -> dict[str, pd.DataFrame]` (each frame reindexed to `index`, NaN where the table has no row; raises `SystemExit` on a duplicate UID, a missing label column, or a value outside [0, 1]); `quantile_match(pred: np.ndarray, ref: np.ndarray) -> np.ndarray` (NaN in `pred` stays NaN; finite output otherwise); `mix_teacher(soft: pd.DataFrame, tables: dict[str, pd.DataFrame], mix: float, is_gold: np.ndarray) -> pd.DataFrame` (columns `LABELS`, the training target before the gold override); `main()` accepts `--teacher-tables a,b` and `--teacher-mix 0.5`, writes `yt__{L}` columns, and with tables writes `artifacts/targets_teacher_<names>.csv` instead of `artifacts/targets.csv`; `--check-md5` prints the md5 of the written file.
- Consumes: `LABELS`, `prob_blend`, `confidence_weights`, `WEAK_WEIGHT_FLOOR`, `GOLD_WEIGHT` (existing).

- [ ] **Step 1: Write the failing checks**

```python
# src/targets_test.py
"""Unit checks for the prediction-table teacher path in src/build_targets.py (CPU, seconds).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/targets_test.py
"""
import hashlib, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import build_targets as bt  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def test_quantile_match_preserves_ranks_and_scale():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.3, 0.9, 500)            # narrow, high scale
    ref = np.r_[np.zeros(400), np.ones(100)]      # LLM-like: mostly 0, some 1
    out = bt.quantile_match(pred, ref)
    check(np.isfinite(out).all(), "quantile_match: finite output")
    order_in = np.argsort(pred); order_out = np.argsort(out, kind="stable")
    check(np.all(np.diff(out[order_in]) >= 0), "quantile_match: monotone in the prediction's ranks")
    check(abs(np.mean(out < 0.5) - 0.8) < 0.03, "quantile_match: adopts the reference distribution (~80 % below 0.5)")


def test_quantile_match_nan_passthrough():
    pred = np.array([0.2, np.nan, 0.9, 0.5]); ref = np.linspace(0, 1, 101)
    out = bt.quantile_match(pred, ref)
    check(np.isnan(out[1]) and np.isfinite(out[[0, 2, 3]]).all(), "quantile_match: NaN stays NaN, others finite")


def test_constant_column():
    pred = np.full(50, 0.7); ref = np.linspace(0, 1, 101)
    out = bt.quantile_match(pred, ref)
    check(np.isfinite(out).all() and abs(out[0] - 0.5) < 0.02, "quantile_match: constant column -> reference median, finite")


def _fake_sources(idx):
    rng = np.random.default_rng(1)
    return {"s1": pd.DataFrame({l: rng.uniform(0, 1, len(idx)) for l in bt.LABELS}, index=idx)}


def test_mix_teacher_rows_and_gold():
    idx = pd.Index([f"u{i}" for i in range(200)], name="StudyInstanceUID")
    soft = bt.prob_blend(_fake_sources(idx), idx)
    is_gold = np.zeros(200, bool); is_gold[:10] = True
    rng = np.random.default_rng(2)
    table = pd.DataFrame({l: rng.uniform(0, 1, 200) for l in bt.LABELS}, index=idx)
    table.iloc[100:] = np.nan                     # table covers half the studies
    yt = bt.mix_teacher(soft, {"t": table}, 0.5, is_gold)
    check(list(yt.columns) == bt.LABELS and len(yt) == 200, "mix_teacher: shape")
    check(np.allclose(yt.iloc[100:].to_numpy(), soft.iloc[100:].to_numpy()), "mix_teacher: uncovered rows keep the LLM value")
    check(np.allclose(yt.iloc[:10].to_numpy(), soft.iloc[:10].to_numpy()), "mix_teacher: gold rows untouched before the override")
    covered = yt.iloc[10:100].to_numpy(); base = soft.iloc[10:100].to_numpy()
    check(not np.allclose(covered, base) and np.isfinite(covered).all(), "mix_teacher: covered rows move and stay finite")
    check(np.all((covered >= 0) & (covered <= 1)), "mix_teacher: values in [0, 1]")


def test_load_tables_validation():
    idx = pd.Index(["a", "b", "c"], name="StudyInstanceUID")
    with tempfile.TemporaryDirectory() as d:
        good = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{l: [0.1, 0.9] for l in bt.LABELS}})
        good.to_csv(os.path.join(d, "good.csv"), index=False)
        t = bt.load_teacher_tables(d, idx, ["good"])
        check(set(t) == {"good"} and np.isnan(t["good"].loc["c", "ACL"]), "load_teacher_tables: reindexed, NaN for absent studies")
        dup = pd.concat([good, good.iloc[:1]]); dup.to_csv(os.path.join(d, "dup.csv"), index=False)
        try:
            bt.load_teacher_tables(d, idx, ["dup"]); check(False, "duplicate UID rejected")
        except SystemExit:
            check(True, "duplicate UID rejected")
        bad = good.drop(columns=["Fracture"]); bad.to_csv(os.path.join(d, "bad.csv"), index=False)
        try:
            bt.load_teacher_tables(d, idx, ["bad"]); check(False, "missing label column rejected")
        except SystemExit:
            check(True, "missing label column rejected")
        rng = good.copy(); rng["ACL"] = [1.5, 0.2]; rng.to_csv(os.path.join(d, "rng.csv"), index=False)
        try:
            bt.load_teacher_tables(d, idx, ["rng"]); check(False, "value outside [0,1] rejected")
        except SystemExit:
            check(True, "value outside [0,1] rejected")


def test_default_teacher_unchanged():
    p = os.path.join(ROOT, "artifacts", "targets.csv")
    if not os.path.exists(p):
        print("  skip default md5 (artifacts/targets.csv absent; run build_targets.py first)"); return
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    check(md5.startswith("29f641ed"), f"artifacts/targets.csv md5 {md5[:8]} == 29f641ed (default teacher byte-identical)")


if __name__ == "__main__":
    for fn in [test_quantile_match_preserves_ranks_and_scale, test_quantile_match_nan_passthrough, test_constant_column,
               test_mix_teacher_rows_and_gold, test_load_tables_validation, test_default_teacher_unchanged]:
        print(fn.__name__); fn()
    print("\n" + ("TARGET CHECKS PASSED" if not fails else f"TARGET CHECKS FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `export PYTHONUTF8=1; .venv/Scripts/python.exe src/targets_test.py`
Expected: `AttributeError: module 'build_targets' has no attribute 'quantile_match'` (exit ≠ 0).

- [ ] **Step 3: Implement the three functions and the constants** (insert after `confidence_weights`, before `# ---- folds`):

```python
# ------------------------------------------------ prediction-table teachers (2026-09-23)

TEACHER_ROOT = "artifacts/teacher"     # <name>.csv: StudyInstanceUID + the 12 label columns in [0, 1]


def load_teacher_tables(root: str, index: pd.Index, names: list[str]) -> dict[str, pd.DataFrame]:
    """Model-prediction tables (self-distillation OOF, the public Raptor pass) as extra *teachers*.

    Same schema as an LLM table. A table may cover a subset of studies (the Raptor pass excludes the 58
    gold rows); absent studies are NaN after reindexing. Anything malformed is fatal: a duplicate UID
    would silently pick one row, a missing label would silently teach the LLM value under a distilled
    version name, a value outside [0, 1] is not a probability."""
    out = {}
    for name in names:
        p = os.path.join(root, f"{name}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"teacher table {name!r} not found at {p}")
        d = pd.read_csv(p, dtype={"StudyInstanceUID": str})
        missing = [l for l in LABELS if l not in d.columns]
        if missing:
            raise SystemExit(f"teacher table {name!r}: missing label columns {missing}")
        if d.StudyInstanceUID.duplicated().any():
            raise SystemExit(f"teacher table {name!r}: duplicate StudyInstanceUID")
        vals = d[LABELS].to_numpy(dtype=float)
        if np.nanmin(vals) < 0 or np.nanmax(vals) > 1:
            raise SystemExit(f"teacher table {name!r}: values outside [0, 1]")
        out[name] = d.set_index("StudyInstanceUID")[LABELS].reindex(index)
        print(f"  teacher table {name}: {int(np.isfinite(vals[:, 0]).sum())} studies covered, from {p}")
    return out


def quantile_match(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map `pred` onto the value distribution of `ref`, keeping `pred`'s ranks.

    Mid-rank quantiles (rank - 0.5) / n so ties and constant columns land on the reference median
    rather than its extremes; NaN in `pred` stays NaN. The result lives on the LLM blend's scale, so
    the 0.5-centred confidence weights keep their meaning when the two are averaged."""
    pred = np.asarray(pred, dtype=float)
    out = np.full(pred.shape, np.nan)
    m = np.isfinite(pred)
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if m.sum() == 0 or len(ref) == 0:
        return out
    r = pd.Series(pred[m]).rank(method="average").to_numpy()
    q = (r - 0.5) / m.sum()
    out[m] = np.quantile(ref, np.clip(q, 0.0, 1.0))
    return out


def mix_teacher(soft: pd.DataFrame, tables: dict[str, pd.DataFrame], mix: float,
                is_gold: np.ndarray) -> pd.DataFrame:
    """Training target = (1 - mix) * LLM blend + mix * mean of the quantile-matched tables, on the
    report-only rows a table covers; every other row (uncovered, gold) keeps the LLM value. Called
    BEFORE the gold override, which then applies to this frame exactly as to `soft`."""
    if not 0.0 <= mix <= 1.0:
        raise SystemExit(f"teacher mix must be in [0, 1], got {mix}")
    yt = soft.copy()
    weak = ~np.asarray(is_gold, dtype=bool)
    for lab in LABELS:
        ref = soft[lab].to_numpy(dtype=float)[weak]
        matched = []
        for d in tables.values():
            col = d[lab].to_numpy(dtype=float)
            col = np.where(weak, col, np.nan)          # never let a table speak on a gold row
            matched.append(quantile_match(col, ref))
        stack = np.vstack(matched)
        with np.errstate(invalid="ignore"):
            mean_matched = np.nanmean(stack, axis=0)
        covered = np.isfinite(mean_matched)
        base = soft[lab].to_numpy(dtype=float)
        yt[lab] = np.where(covered, (1.0 - mix) * base + mix * mean_matched, base)
    return yt
```

- [ ] **Step 4: Extend `main()`** — add the arguments after `--sources`:

```python
    ap.add_argument("--teacher-tables", default="",
                    help="comma-separated prediction tables under artifacts/teacher/ mixed into the TRAINING targets (yt__*)")
    ap.add_argument("--teacher-mix", type=float, default=0.5)
    ap.add_argument("--teacher-root", default=TEACHER_ROOT)
    ap.add_argument("--check-md5", action="store_true", help="print the md5 of the written targets file")
```

after `weights = confidence_weights(sources, idx)`:

```python
    teacher_names = [x.strip() for x in args.teacher_tables.split(",") if x.strip()]
    tables = load_teacher_tables(args.teacher_root, idx, teacher_names) if teacher_names else {}
    yt = mix_teacher(soft, tables, args.teacher_mix, is_gold.to_numpy()) if tables else None
    if tables:
        log(f"\nteacher tables {teacher_names} mixed at {args.teacher_mix} into the TRAINING targets (yt__*); "
            f"y__* stays the LLM teacher   ! NON-DEFAULT targets -- written beside artifacts/targets.csv, not over it")
        covered_gold = [n for n, d in tables.items() if np.isfinite(d.loc[gold_idx, LABELS].to_numpy()).all()]
        for n in covered_gold:
            a = [auc(gy[l].to_numpy(), tables[n].loc[gold_idx, l].to_numpy()) for l in LABELS]
            log(f"  {n:<16} gold macro-AUC {np.nanmean(a):.4f}  (the table alone, n=58 -- direction only)")
```

in the gold-override loop, mirror the override onto `yt`:

```python
    for lab in LABELS:
        gvals = gold[lab].reindex(idx)
        have = gvals.notna().to_numpy()
        targets.loc[have, lab] = gvals[have].to_numpy()
        w.loc[have, lab] = GOLD_WEIGHT
        if yt is not None:
            yt.loc[have, lab] = gvals[have].to_numpy()
```

where the output frame is assembled (find `out = pd.concat(` near the end of `main()`), add the `yt__` columns and pick the file name:

```python
    if yt is not None:
        ytdf = yt.reset_index(drop=True)
        ytdf.columns = [f"yt__{c}" for c in LABELS]
        out = pd.concat([out, ytdf], axis=1)
    out_name = "targets.csv" if yt is None else f"targets_teacher_{'_'.join(teacher_names)}.csv"
    out_path = os.path.join(args.out_dir, out_name)
    out.to_csv(out_path, index=False)
    if args.check_md5:
        log(f"md5 {hashlib.md5(open(out_path, 'rb').read()).hexdigest()}  {out_path}")
```

(Adapt the existing `to_csv` call rather than adding a second one: keep exactly one write of the targets file. `hashlib` is already imported by `report_group`.)

- [ ] **Step 5: Run the checks and the default build**

Run: `export PYTHONUTF8=1; .venv/Scripts/python.exe src/targets_test.py`
Expected: every line `ok`, `TARGET CHECKS PASSED`.
Run: `.venv/Scripts/python.exe src/build_targets.py --check-md5 | tail -3`
Expected: `BLEND 0.8948` earlier in the log and `md5 29f641ed…  artifacts/targets.csv`.

- [ ] **Step 6: Commit**

```powershell
git add src/build_targets.py src/targets_test.py
git commit -m "build_targets: prediction-table teachers (quantile-matched, mixed into yt__ training targets; y__ unchanged) + targets_test.py"
```

---

### Task 2: Self-distillation table from the two 5-fold OOF sets

**Files:**
- Create: `src/build_distill_table.py`
- Output: `artifacts/teacher/selfdistill_v1.csv` (gitignored under `artifacts/`)
- Test: extend `src/targets_test.py` with `test_distill_table_builder`

**Interfaces:**
- Produces: `build_distill_table(oof_globs: list[str], out_csv: str) -> pd.DataFrame` — per label, the mean over OOF sets of the study's rank-percentile (each set = the concatenation of its `fold*_oof.csv`, one row per study); output columns `StudyInstanceUID` + `LABELS`, values in (0, 1]. Consumed by Task 1's `load_teacher_tables` under the name `selfdistill_v1`.

- [ ] **Step 1: Write the failing check** (append to `src/targets_test.py`, register in the `__main__` list):

```python
def test_distill_table_builder():
    import build_distill_table as bd
    with tempfile.TemporaryDirectory() as d:
        ids = [f"u{i}" for i in range(6)]
        rng = np.random.default_rng(3)
        for setname in ("a", "b"):
            for k in (0, 1):
                rows = ids[k * 3:(k + 1) * 3]
                df = pd.DataFrame({"StudyInstanceUID": rows, "epoch": 7, "is_gold": 0})
                for l in bt.LABELS:
                    df[f"pred__{l}"] = rng.uniform(0, 1, 3); df[f"y__{l}"] = 0.5; df[f"w__{l}"] = 1.0
                df.to_csv(os.path.join(d, f"{setname}_fold{k}_oof.csv"), index=False)
        out = bd.build_distill_table([os.path.join(d, "a_fold*_oof.csv"), os.path.join(d, "b_fold*_oof.csv")],
                                     os.path.join(d, "t.csv"))
        check(len(out) == 6 and list(out.columns) == ["StudyInstanceUID", *bt.LABELS], "distill table: one row per study, schema")
        v = out[bt.LABELS].to_numpy()
        check(np.all((v > 0) & (v <= 1)), "distill table: rank percentiles in (0, 1]")
        check(os.path.exists(os.path.join(d, "t.csv")), "distill table: csv written")
        try:
            bd.build_distill_table([os.path.join(d, "a_fold0_oof.csv")], os.path.join(d, "t2.csv"))
            check(False, "distill table: an OOF set that does not cover every study of another set is rejected")
        except SystemExit:
            check(True, "distill table: partial coverage rejected")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe src/targets_test.py`
Expected: `ModuleNotFoundError: No module named 'build_distill_table'`.

- [ ] **Step 3: Implement**

```python
# src/build_distill_table.py
"""Self-distillation teacher table (P-17 / P-38): rank-mean of complete 5-fold OOF sets.

Every prediction is out-of-fold, so no study is taught by a model that saw it. Values are rank
percentiles per label (mean over sets); build_targets.quantile_match() puts them on the LLM scale.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_distill_table.py            # -> artifacts/teacher/selfdistill_v1.csv
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DEFAULT_SETS = ["artifacts/kaggle_out/pod_v09h_5fold/v09h_fold*_oof.csv",      # pooled OOF 0.8625
                "artifacts/kaggle_out/folds_v4/v05g_fold*_oof.csv"]            # pooled OOF 0.8467


def _load_set(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no OOF csvs match {pattern}")
    d = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in files], ignore_index=True)
    if d.StudyInstanceUID.duplicated().any():
        raise SystemExit(f"{pattern}: a study appears in more than one fold's OOF")
    return d.set_index("StudyInstanceUID")[[f"pred__{l}" for l in LABELS]]


def build_distill_table(oof_globs: list[str], out_csv: str) -> pd.DataFrame:
    sets = [_load_set(p) for p in oof_globs]
    ids = set(sets[0].index)
    for s, pat in zip(sets[1:], oof_globs[1:]):
        if set(s.index) != ids:
            raise SystemExit(f"OOF sets cover different studies ({pat}: {len(set(s.index) ^ ids)} differ)")
    index = sorted(ids)
    acc = np.zeros((len(index), len(LABELS)))
    for s in sets:
        s = s.loc[index]
        for j, l in enumerate(LABELS):
            acc[:, j] += pd.Series(s[f"pred__{l}"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
    acc /= len(sets)
    out = pd.DataFrame(acc, columns=LABELS)
    out.insert(0, "StudyInstanceUID", index)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(out)} studies from {len(sets)} OOF sets")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--out", default="artifacts/teacher/selfdistill_v1.csv")
    a = ap.parse_args()
    build_distill_table(a.sets, a.out)
```

- [ ] **Step 4: Run the checks and build the real table**

Run: `.venv/Scripts/python.exe src/targets_test.py` → `TARGET CHECKS PASSED`.
Run: `.venv/Scripts/python.exe src/build_distill_table.py` → `wrote artifacts/teacher/selfdistill_v1.csv: 4407 studies from 2 OOF sets`.
Run: `.venv/Scripts/python.exe src/build_targets.py --teacher-tables selfdistill_v1 | tail -12` → prints `teacher table selfdistill_v1: 4407 studies covered`, `selfdistill_v1 gold macro-AUC 0.8…` and writes `artifacts/targets_teacher_selfdistill_v1.csv`; then `.venv/Scripts/python.exe src/build_targets.py --check-md5 | tail -1` still shows `29f641ed`.

- [ ] **Step 5: Commit**

```powershell
git add src/build_distill_table.py src/targets_test.py
git commit -m "build_distill_table: self-distillation teacher (rank-mean of the v09h + v05g 5-fold OOF sets) -> artifacts/teacher/selfdistill_v1.csv"
```

---

### Task 3: Teacher tables in the kernel — `TEACHER_TABLES`, `yt` through Dataset / collate / loss

**Files:**
- Modify: `src/kaggle_pipeline.py` — config cell (after `PARALLEL_ARMS` block, before `@dataclass class Config`), `Config` (supervision block ~line 444), `build_targets` (~739–845), `KneeStudyDataset.__getitem__` (two sites ~1538–1543 and ~1573–1578), `collate_windows` (~1951), training loop loss call (~2345)
- Modify: `src/window_head_test.py` (new checks at the end of `main()`)

**Interfaces:**
- Produces: module constants `TEACHER_TABLES: tuple[str, ...] = ()`, `TEACHER_MIX = 0.5`, `TEACHER_PATHS: dict[str, list[str]]`; `Config.teacher_tables: tuple = ()`, `Config.teacher_mix: float = 0.5` (recorded in every checkpoint's `config`); targets frame gains `yt__{L}` columns only when `TEACHER_TABLES` is non-empty; batches gain `b["yt"]` (shape `(B, 12)`) under the same condition; `weighted_bce` receives `yt` when present, `y` otherwise. `evaluate()` is untouched.
- Consumes: the same `quantile_match` / mixing rules as Task 1 (copied, not imported — the kernel is a single file).

- [ ] **Step 1: Write the failing checks** (append inside `main()` of `src/window_head_test.py`, before the final print):

```python
    # ---- teacher tables (2026-09-23, spec section 2) ------------------------------------------------
    qm = K["quantile_match"]
    pred = np.array([0.9, 0.1, np.nan, 0.5]); ref = np.r_[np.zeros(80), np.ones(20)]
    out = qm(pred, ref)
    check(np.isnan(out[2]) and np.isfinite(out[[0, 1, 3]]).all() and out[0] >= out[3] >= out[1],
          "kernel quantile_match: rank-preserving, NaN passthrough")
    import pandas as pd
    idx = pd.Index([f"s{i}" for i in range(6)])
    soft = pd.DataFrame({l: np.linspace(0.1, 0.9, 6) for l in LABELS}, index=idx)
    table = pd.DataFrame({l: np.linspace(0.9, 0.1, 6) for l in LABELS}, index=idx)     # reversed ranks
    yt = K["mix_teacher"](soft, {"t": table}, 0.5, np.array([True] + [False] * 5))
    check(np.allclose(yt.iloc[0].to_numpy(), soft.iloc[0].to_numpy()), "kernel mix_teacher: gold row keeps the LLM value")
    check(not np.allclose(yt.iloc[1:].to_numpy(), soft.iloc[1:].to_numpy()), "kernel mix_teacher: covered rows move")
    # collate carries yt when present, and the loss reads it
    items = [{"study": "a", "arr": torch.zeros(2, 4, 4, dtype=torch.uint8), "centres": torch.zeros(3, dtype=torch.long),
              "slot_id": torch.zeros(3, dtype=torch.long), "mask": torch.ones(6, dtype=torch.bool),
              "y": torch.full((12,), 0.2), "yt": torch.full((12,), 0.8), "w": torch.ones(12), "is_gold": torch.tensor(0.)}]
    b = K["collate_windows"](items)
    check("yt" in b and b["yt"].shape == (1, 12), "collate_windows: stacks yt")
    logits = torch.zeros(1, 12)
    l_y = K["weighted_bce"](logits, b["y"], b["w"]); l_yt = K["weighted_bce"](logits, b["yt"], b["w"])
    check(abs(float(l_y) - float(l_yt)) < 1e-6, "weighted_bce at logit 0 is symmetric in the target (sanity)")
    check(K["TEACHER_TABLES"] == () and K["Config"]().teacher_tables == (), "TEACHER_TABLES default () (no teacher mixing unless sed'd)")
    check("selfdistill_v1" in K["TEACHER_PATHS"] and "raptor_teacher" in K["TEACHER_PATHS"], "TEACHER_PATHS lists both tables")
```

- [ ] **Step 2: Run to verify they fail**

Run: `export PYTHONUTF8=1; .venv/Scripts/python.exe src/window_head_test.py 2>&1 | tail -4`
Expected: `KeyError: 'quantile_match'`.

- [ ] **Step 3: Config-cell constants** (insert directly above `@dataclass` / `class Config:`):

```python
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ TEACHER_TABLES (2026-09-23, spec docs/superpowers/specs/2026-09-23-…):     │
# │ prediction tables (StudyInstanceUID + 12 label columns in [0, 1]) mixed  │
# │ into the TRAINING targets `yt__*` after per-label quantile matching onto │
# │ the LLM blend: yt = (1 - MIX) * llm + MIX * matched on the report-only   │
# │ rows a table covers; gold rows stay hard 0/1. `y__*` (evaluation, OOF)   │
# │ stays the 3-source LLM teacher, so OOF-vs-teacher remains comparable.    │
# │ Sed'd per kernel session like ARM_ONLY, e.g.                             │
# │   sed 's/^TEACHER_TABLES = ()/TEACHER_TABLES = ("selfdistill_v1",)/' …   │
# │ A listed table that is not mounted is FATAL (never silently train on the │
# │ plain teacher under a distilled version name).                           │
# └──────────────────────────────────────────────────────────────────────────┘
TEACHER_TABLES = ()
TEACHER_MIX = 0.5
TEACHER_PATHS = {
    "selfdistill_v1": ["/kaggle/input/rsna-knee-teacher-tables/selfdistill_v1.csv", "artifacts/teacher/selfdistill_v1.csv"],
    "raptor_teacher": ["/kaggle/input/rsna-knee-teacher-tables/raptor_teacher.csv", "artifacts/teacher/raptor_teacher.csv"],
}
```

Config fields (supervision block, after `weak_weight_floor`):

```python
    teacher_tables: tuple = TEACHER_TABLES   # recorded in the checkpoint; training-only (not an INFER_MEMBER_KEY)
    teacher_mix: float = TEACHER_MIX
```

- [ ] **Step 4: `quantile_match` / `mix_teacher` in the kernel** (module level, just above `def build_targets`), identical to Task 1's bodies except `raise SystemExit` messages may be shortened; then inside `build_targets`, after the `soft` / `wt` block and **before** `gold = tr.set_index(...)`:

```python
    yt = None
    if TEACHER_TABLES:
        tables = {}
        for name in TEACHER_TABLES:
            if name not in TEACHER_PATHS:
                raise SystemExit(f"unknown teacher table {name!r}; known: {sorted(TEACHER_PATHS)}")
            p = first_existing(TEACHER_PATHS[name])
            if p is None:
                raise SystemExit(f"teacher table {name!r} is listed but not mounted -- refusing to train on the plain teacher")
            d = pd.read_csv(p, dtype={"StudyInstanceUID": str})
            if any(l not in d.columns for l in LABELS) or d.StudyInstanceUID.duplicated().any():
                raise SystemExit(f"teacher table {name!r}: bad schema or duplicate UID ({p})")
            tables[name] = d.set_index("StudyInstanceUID")[LABELS].reindex(idx)
            print(f"  teacher table {name}: {int(tables[name][LABELS[0]].notna().sum())} studies from {p}")
        yt = mix_teacher(soft, tables, TEACHER_MIX, is_gold.to_numpy())
        print(f"  training targets = (1 - {TEACHER_MIX}) * LLM + {TEACHER_MIX} * quantile-matched {list(TEACHER_TABLES)}; evaluation targets unchanged")
```

in the gold override loop add `if yt is not None: yt.loc[have, lab] = g[have].to_numpy()`; in the NaN fill add `if yt is not None: yt.loc[m, lab] = soft.loc[m, lab]`; when assembling `out`:

```python
    if yt is not None:
        ytdf = yt.reset_index(drop=True)
        ytdf.columns = [f"yt__{c}" for c in LABELS]
        out = pd.concat([out, ytdf], axis=1)
```

- [ ] **Step 5: Dataset, collate, loss.** In **both** `__getitem__` target blocks (search `out["is_gold"] = torch.tensor(float(r["is_gold"]))`, two occurrences) add after that line:

```python
                    if f"yt__{LABELS[0]}" in self.t.columns:
                        out["yt"] = torch.tensor([float(r[f"yt__{l}"]) for l in LABELS])
```

(indentation per site). In `collate_windows` change `for k in ("y", "w", "is_gold"):` to `for k in ("y", "yt", "w", "is_gold"):`. In the training loop replace

```python
                loss = weighted_bce(logits, b["y"].to(device), b["w"].to(device))
```
with
```python
                y_train = b["yt"] if "yt" in b else b["y"]           # teacher-mixed targets train; y stays the OOF target
                loss = weighted_bce(logits, y_train.to(device), b["w"].to(device))
```

The fixed-window path collates with the default collate (dict of tensors) — `yt` rides along automatically there.

- [ ] **Step 6: Run the unit checks, then the local smoke with the table sed'd in**

Run: `.venv/Scripts/python.exe src/window_head_test.py 2>&1 | tail -6` → `UNIT CHECKS PASSED`.
Run (Git Bash, from the repo root, `export PATH="/usr/bin:/bin:$PATH" PYTHONUTF8=1 PYTHONPATH=src`):
```bash
sed -e 's/^MODE = "auto"/MODE = "train"/' -e 's/^TEACHER_TABLES = ()/TEACHER_TABLES = ("selfdistill_v1",)/' src/kaggle_pipeline.py > artifacts/smoke_teacher.py
.venv/Scripts/python.exe artifacts/smoke_teacher.py 2>&1 | grep -E "teacher table|training targets|arm v0|-> v0|Traceback|wrote .*submission" 
```
Expected: `teacher table selfdistill_v1: 4407 studies …`, `training targets = (1 - 0.5) * LLM + 0.5 * …`, both arms train, `wrote … submission.csv`. Then the plain smoke (`MODE = "train"` only) must print neither teacher line.

- [ ] **Step 7: Commit**

```powershell
git add src/kaggle_pipeline.py src/window_head_test.py
git commit -m "kernel: TEACHER_TABLES / TEACHER_MIX -> quantile-matched teacher mixing into yt__ training targets (y__ unchanged); Dataset/collate carry yt; loss reads it; unit checks"
```

---

### Task 4: `pos_weight` as a Config knob

**Files:**
- Modify: `src/kaggle_pipeline.py` — `Config` (optimisation block, after `grad_accum`), `weighted_bce` (~1909), `train_fold` (after `tr_loader, va_loader = make_loaders(...)`, ~2288), the loss call (Task 3's line)
- Modify: `src/window_head_test.py`

**Interfaces:**
- Produces: `Config.pos_weight_max: float = 0.0` (0 = off, loss byte-identical); `weighted_bce(logits, y, w, pos_weight=None)`; `label_pos_weight(targets: pd.DataFrame, study_ids: list[str], max_w: float) -> np.ndarray` (12 floats in [1, max_w]).

- [ ] **Step 1: Write the failing checks** (append in `main()` of `window_head_test.py`):

```python
    # ---- pos_weight (P-37) --------------------------------------------------------------------------
    lpw = K["label_pos_weight"]
    tg = pd.DataFrame({"StudyInstanceUID": [f"s{i}" for i in range(10)]})
    for l in LABELS:
        tg[l] = 0.0
    tg.loc[:0, "ACL"] = 1.0            # 10 % positive -> (1-p)/p = 9
    tg["MCL"] = 1.0                    # all positive -> clip to 1
    tg["Fracture"] = 0.0               # no positive -> clip to max
    pw = lpw(tg, tg.StudyInstanceUID.tolist(), 10.0)
    check(pw.shape == (12,) and np.isfinite(pw).all(), "label_pos_weight: 12 finite weights")
    check(abs(pw[LABELS.index("ACL")] - 9.0) < 1e-6, "label_pos_weight: 10 % positive -> 9")
    check(pw[LABELS.index("MCL")] == 1.0 and pw[LABELS.index("Fracture")] == 10.0, "pos_weight extremes clip to [1, max]")
    lg = torch.randn(2, 12); yy = torch.rand(2, 12); ww = torch.ones(2, 12)
    check(torch.allclose(K["weighted_bce"](lg, yy, ww), K["weighted_bce"](lg, yy, ww, pos_weight=None)), "weighted_bce: pos_weight=None is the old loss")
    check(float(K["weighted_bce"](lg, yy, ww, pos_weight=torch.full((12,), 3.0))) > float(K["weighted_bce"](lg, yy, ww)), "weighted_bce: pos_weight > 1 raises the loss")
    check(K["Config"]().pos_weight_max == 0.0, "Config.pos_weight_max defaults to 0 (off)")
```

- [ ] **Step 2: Run to verify they fail** → `KeyError: 'label_pos_weight'`.

- [ ] **Step 3: Implement**

Config (optimisation block):
```python
    # P-37 (2026-09-23): per-label pos_weight = clip((1 - p) / p, 1, pos_weight_max), p = positive rate of the
    # training targets (yt if present, else y) at the 0.5 cut, computed once per fold. 0 = off (byte-identical loss).
    # The public 0.924 member trains with [1, 10]. Rejected earlier as "AUC ignores calibration" -- this measures
    # its effect on training dynamics, not on calibration.
    pos_weight_max: float = 0.0
```

`weighted_bce`:
```python
def weighted_bce(logits, y, w, pos_weight=None):
    """... (existing docstring) ...
    `pos_weight` (P-37): per-label multiplier of the positive term, or None (the loss through 2026-09-22)."""
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    per_study = (loss * w).sum(1) / w.sum(1).clamp_min(1e-6)
    return per_study.mean()
```

module-level helper (next to `split_studies`):
```python
def label_pos_weight(targets, study_ids, max_w):
    """P-37: clip((1 - p) / p, 1, max_w) per label from the training rows' hard targets (yt if present)."""
    t = targets.set_index("StudyInstanceUID").loc[study_ids]
    cols = [f"yt__{l}" if f"yt__{l}" in t.columns else l for l in LABELS]
    p = (t[cols].to_numpy(dtype=float) > 0.5).mean(0)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip((1.0 - p) / p, 1.0, float(max_w))
```

`train_fold`, after the loaders:
```python
    pos_w = None
    if cfg.pos_weight_max > 0:
        tr_ids, _ = split_studies(targets, fold, cfg)
        pw = label_pos_weight(targets, [s for s in tr_ids if s in set(tr_loader.dataset.studies)], cfg.pos_weight_max)
        pos_w = torch.tensor(pw, dtype=torch.float32, device=device)
        print("    pos_weight [1, %g]: " % cfg.pos_weight_max + ", ".join(f"{l} {v:.1f}" for l, v in zip(LABELS, pw)))
```
and the loss call becomes `weighted_bce(logits, y_train.to(device), b["w"].to(device), pos_weight=pos_w)`.

- [ ] **Step 4: Run the checks** → `UNIT CHECKS PASSED`; local smoke of an arm with `pos_weight_max` (Task 5's `v09f`) prints the `pos_weight [1, 10]:` line.

- [ ] **Step 5: Commit**

```powershell
git add src/kaggle_pipeline.py src/window_head_test.py
git commit -m "P-37: Config.pos_weight_max -> per-label pos_weight clip((1-p)/p, 1, max) in weighted_bce (0 = off, byte-identical)"
```

---

### Task 5: Arms `v09e` / `v09f` / `v09s`, cards P-36…P-38, RunPod bootstrap, local smoke

**Files:**
- Modify: `src/kaggle_pipeline.py` `ARMS` (after the `v08c` entry)
- Modify: `docs/proposals.md` (three cards before `## Rejected without testing`, three index rows after P-35's, amend the `pos_weight` rejected row)
- Modify: `scripts/runpod_bootstrap.sh` (`LABELS=(…)` array + a `teacher tables` pull in `setup`; usage comment)
- Modify: `kaggle/rsna-knee-train/kernel-metadata.json`, `kaggle/rsna-knee-folds/kernel-metadata.json` (`dataset_sources` += `tiankljucanin/rsna-knee-teacher-tables`)

**Interfaces:**
- Consumes: Task 3 `TEACHER_TABLES` (sed'd, not an arm key — `v09s` is documented as "run with `TEACHER_TABLES=("selfdistill_v1",)` sed'd"), Task 4 `pos_weight_max`.
- Produces: arm names `v09e`, `v09f`, `v09s` for `RSNA_ARM` / `PARALLEL_ARMS`.

- [ ] **Step 1: Arms** (append inside `ARMS`, after `("v08c", …)`):

```python
    # 2026-09-23 (P-36 / P-37 / P-38, spec docs/superpowers/specs/2026-09-23-member-strength-design.md): fold-0
    # arms vs v09c 0.8730, floor 0.008, run on RunPod (~1 h each on a 4090). v09e = the public schedule length at
    # the public backbone LR (P-29 over-trained 16 epochs at 1e-4); v09f = the public member's pos_weight [1, 10];
    # v09s = v09c on the self-distilled targets -- run with TEACHER_TABLES=("selfdistill_v1",) sed'd in (the arm
    # dict cannot carry it: targets are built once per session).
    ("v09e", {**C02, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 3e-5,
              "batch_studies": 2, "grad_accum": 2, "aug": "light", "epochs": 16}),
    ("v09f", {**C02, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 1e-4,
              "batch_studies": 2, "grad_accum": 2, "aug": "light", "pos_weight_max": 10.0}),
    ("v09s", {**C02, "backbone": "timm:coatnet_rmlp_1_rw_224", "img_size": 224, "lr_backbone": 1e-4,
              "batch_studies": 2, "grad_accum": 2, "aug": "light"}),
```

Add a guard in the config cell (after the arm filter `_only` block): `if any(a[0] == "v09s" for a in ARMS) and (ARM_ONLY == "v09s" or os.environ.get("RSNA_ARM") == "v09s") and not TEACHER_TABLES: raise SystemExit("v09s is the self-distillation arm: sed TEACHER_TABLES = (\"selfdistill_v1\",) into the copy you run")`.

- [ ] **Step 2: Cards** (template of the file; Status 🔧 implemented, effect pending; Measure = fold-0 OOF vs `v09c` 0.8730; Noise floor 0.008: ≥ 0.881 ✅ / 0.865–0.881 🔁 / < 0.865 harmful; ≥ 9/12 labels up; Cost ≈ 1 h 4090 each (`v09e` ≈ 1.7 h); If it works → the knob joins the CoAtNet PROD recipe; If it fails → P-36: the LR/schedule is not the gap; P-37: `pos_weight` stays 0 and the rejected row is confirmed by measurement; P-38: self-distillation from a 0.86 OOF teacher does not denoise — wait for the Raptor teacher (P-39). Depends on: P-32/P-33 (`v09c`), P-31/P-24 (pod)). Index rows in the same format as P-34/P-35. Amend the "Rejected without testing" row `| Calibration, Platt scaling, thresholds, `pos_weight`, label smoothing on soft targets |` → move `pos_weight` out of it with the note "`pos_weight` re-opened as a *training-dynamics* A/B (P-37, 2026-09-23)".

- [ ] **Step 3: RunPod bootstrap** — in `scripts/runpod_bootstrap.sh` change

```bash
LABELS=(pilkwang/rsna-knee-llm-labels stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56)
```
to
```bash
LABELS=(pilkwang/rsna-knee-llm-labels stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56
        tiankljucanin/rsna-knee-teacher-tables)   # 2026-09-23: selfdistill_v1.csv / raptor_teacher.csv (Task 6 publishes it)
```
and in `train)` add before the sed: `TEACHER="${RSNA_TEACHER_TABLES:-}"` and extend the sed with `-e "s/^TEACHER_TABLES = ()/TEACHER_TABLES = ${TEACHER:-()}/"` so `RSNA_TEACHER_TABLES='("selfdistill_v1",)' bash scripts/runpod_bootstrap.sh train v09s` selects the table (document in the usage header). Kernel metadata: add `"tiankljucanin/rsna-knee-teacher-tables"` to both training kernels' `dataset_sources` (JSON edit; keep key order).

- [ ] **Step 4: Local smokes** (Git Bash, repo root):

```bash
for a in v09e v09f; do sed -e 's/^MODE = "auto"/MODE = "train"/' -e "s/^ARM_ONLY = \"\"/ARM_ONLY = \"$a\"/" src/kaggle_pipeline.py > artifacts/smoke_$a.py; .venv/Scripts/python.exe artifacts/smoke_$a.py 2>&1 | grep -E "########## arm|pos_weight|epochs 1|-> v0|Traceback" | head -6; done
sed -e 's/^MODE = "auto"/MODE = "train"/' -e 's/^ARM_ONLY = ""/ARM_ONLY = "v09s"/' -e 's/^TEACHER_TABLES = ()/TEACHER_TABLES = ("selfdistill_v1",)/' src/kaggle_pipeline.py > artifacts/smoke_v09s.py; .venv/Scripts/python.exe artifacts/smoke_v09s.py 2>&1 | grep -E "teacher table|########## arm|-> v0|Traceback" | head -6
sed -e 's/^MODE = "auto"/MODE = "train"/' -e 's/^ARM_ONLY = ""/ARM_ONLY = "v09s"/' src/kaggle_pipeline.py > artifacts/smoke_v09s_bad.py; .venv/Scripts/python.exe artifacts/smoke_v09s_bad.py 2>&1 | grep -E "self-distillation arm" 
```
Expected: `v09e` banner shows `'lr_backbone': 3e-05 … 'epochs': 16` (smoke clamps to 1 epoch), `v09f` prints `pos_weight [1, 10]: ACL …`, `v09s` prints the teacher-table lines, and the last command prints the guard message.

- [ ] **Step 5: Commit**

```powershell
git add src/kaggle_pipeline.py docs/proposals.md scripts/runpod_bootstrap.sh kaggle/rsna-knee-train/kernel-metadata.json kaggle/rsna-knee-folds/kernel-metadata.json
git commit -m "P-36/P-37/P-38: arms v09e (16 ep x 3e-5), v09f (pos_weight 10), v09s (self-distill); cards + index; runpod bootstrap pulls the teacher-tables Dataset and seds TEACHER_TABLES"
```

---

### Task 6: Publish the teacher-tables Dataset (self-distill table now, Raptor table later)

**Files:**
- Create: `artifacts/ship_teacher/dataset-metadata.json` (gitignored dir; the command is what matters)

- [ ] **Step 1: Stage and create**

```bash
export PATH="/usr/bin:/bin:$PATH"; R=/c/Users/Tian/Desktop/RSNA-KneeMRI-kaggle-competition; K=$R/.venv/Scripts/kaggle.exe
mkdir -p $R/artifacts/ship_teacher && cp $R/artifacts/teacher/selfdistill_v1.csv $R/artifacts/ship_teacher/
printf '%s\n' '{"title": "RSNA knee teacher tables", "id": "tiankljucanin/rsna-knee-teacher-tables", "licenses": [{"name": "other"}]}' > $R/artifacts/ship_teacher/dataset-metadata.json
( cd $R/artifacts/ship_teacher && timeout 600 "$K" datasets create -p . )
for i in 1 2 3 4 5 6; do s=$(timeout 60 "$K" datasets status tiankljucanin/rsna-knee-teacher-tables | tail -1); echo "$s"; case "$s" in *ready*) break;; esac; sleep 20; done
timeout 60 "$K" datasets files tiankljucanin/rsna-knee-teacher-tables
```
Expected: `ready`, one file `selfdistill_v1.csv` (~1 MB). (Title ≤ 50 chars, run from inside the dir — traps 21.) Later versions: `kaggle datasets version -p . -m "add raptor_teacher.csv"` from the same dir (Task 8).

- [ ] **Step 2: Kaggle smoke of the teacher path** (both T4s, ≈ 5 min): build `artifacts/train_teacher_smoke.py` = `src/kaggle_pipeline.py` with `PARALLEL_ARMS = ("v09s", "v09f")` and `TEACHER_TABLES = ("selfdistill_v1",)` sed'd, `FORCE_SMOKE = True`; nbgen into `kaggle/rsna-knee-train/rsna-knee-train.ipynb`; assert the flags with grep; push; `kernels status` (traps 36); on COMPLETE pull `--file-pattern "\.log$"` and confirm `teacher table selfdistill_v1: 4407 studies`, `pos_weight [1, 10]` in `v09f.log`, both `_best.pt`. Commit the notebook: `git commit -m "Kaggle smoke of the teacher-table path (train vNN): v09s || v09f, both T4s"`.

---

### Task 7: `src/build_teacher_pass.py` — the Raptor branch as its own Kaggle kernel

**Files:**
- Create: `src/build_teacher_pass.py`
- Create (generated): `kaggle/rsna-knee-teacher/rsna-knee-teacher.ipynb`, `kaggle/rsna-knee-teacher/kernel-metadata.json`
- Create: `src/teacher_pass_test.py`

**Interfaces:**
- Consumes: `notebook_score_0.942.ipynb` (committed, byte-identical), `src/nbgen.py` (`build(src, dst)` from a percent-format `.py`).
- Produces: `render_teacher_py(nb: dict, shard: int, n_shards: int, limit: int) -> str` (the percent-format source), `write_kernel(out_dir: str, shard: int, n_shards: int, limit: int) -> None`; CLI `--shard 0 --n-shards 1 --limit 0 [--check]`. Kernel outputs: `/kaggle/working/raptor_teacher_shard{SHARD}.npz` (`study_uids`, `raw_probabilities` 4×N×12, `view_names`, `view_weights`, `checkpoint_sha256`), `raptor_teacher_shard{SHARD}.csv` (UID + 12 view-weighted means), `teacher_receipt.json` (`studies`, `sec_per_study`, `failed_uids`, `elapsed_s`), `raptor_teacher_partial.npz` (flushed every 5 min).

The generated `.py` has six cells, in this order:

1. **Preamble (ours)** — constants `SHARD`, `N_SHARDS`, `LIMIT` (module top so the cache-shard `sed` pattern works), the chunk builder, the resume skip-list, `RSNA_COMP_ROOT`, and stubs the extracted code expects (`RUN` with `raptor_view_w` and `raptor_k_eval` copied from cell 2; `coat_w = None`; `TIME_BUDGET = 8.0 * 3600`; `T0 = time.time()`; `_RSNA_TEST_IDS` set to the chunk ids):

```python
# %%
SHARD = 0                 # sed'd at build: shard index
N_SHARDS = 1              # sed'd at build: number of shards over the 4,349 report-labelled studies
LIMIT = 0                 # sed'd at build: > 0 = first N studies of the shard (6 = smoke, 100 = the timing spike)
import os, json, time, shutil
import numpy as np, pandas as pd
from pathlib import Path
COMP_IN = "/kaggle/input/rsna-knee-abnormality-detection"
CHUNK = Path("/kaggle/working/chunk"); CHUNK.mkdir(parents=True, exist_ok=True)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def chunk_study_ids(train_csv, shard, n_shards, limit):
    """Report-labelled studies only (the 58 gold rows are validation and never teach), sorted by UID,
    shard `shard` of `n_shards` by position, then the first `limit` (> 0)."""
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    is_gold = tr[LABELS].notna().all(axis=1)
    ids = sorted(tr.loc[~is_gold, "StudyInstanceUID"].tolist())
    assert len(ids) == 4349, len(ids)
    ids = ids[shard::n_shards]
    return ids[:limit] if limit > 0 else ids

def done_uids():
    """Resume: UIDs already in any mounted teacher npz (a previous, guard-stopped run of this shard)."""
    done = set()
    for p in list(Path("/kaggle/input").glob("**/raptor_teacher_shard*.npz"))[:50] + list(Path("/kaggle/input").glob("**/raptor_teacher_partial.npz"))[:50]:
        try:
            z = np.load(p, allow_pickle=False)
            fin = np.isfinite(z["raw_probabilities"]).all(axis=(0, 2))
            done |= set(map(str, z["study_uids"][fin]))
        except Exception as e:
            print("  ! unreadable", p, e)
    return done

ids = chunk_study_ids(f"{COMP_IN}/train.csv", SHARD, N_SHARDS, LIMIT)
skip = done_uids() & set(ids)
ids = [u for u in ids if u not in skip]
print(f"chunk: shard {SHARD}/{N_SHARDS}, limit {LIMIT} -> {len(ids)} studies ({len(skip)} already done)")
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(CHUNK / "test.csv", index=False)
ser = pd.read_csv(f"{COMP_IN}/train_series.csv", dtype=str)
ser[ser.StudyInstanceUID.isin(ids)].to_csv(CHUNK / "test_series.csv", index=False)
sub = pd.DataFrame({"StudyInstanceUID": ids, **{l: 0.5 for l in LABELS}}); sub.to_csv(CHUNK / "sample_submission.csv", index=False)
if not (CHUNK / "test_images").exists():
    os.symlink(f"{COMP_IN}/train_images", CHUNK / "test_images")
os.environ["RSNA_COMP_ROOT"] = str(CHUNK)
RUN = {"raptor_view_w": {"maxspan-v5": 0.60, "native384dense-v10": 0.10, "maxspan-v5-reverse": 0.10, "native384-v8": 0.20},
       "raptor_k_eval": 94}
coat_w = None
TIME_BUDGET = 8.0 * 3600
T0 = time.time()
_RSNA_TEST_IDS = list(ids)
```

2. **Cell 16 helpers (verbatim slice)** — lines from `# Runtime integrity: no partial ensemble` (cell 16 line 104) through the end of `rsna_deadline` (line 195), *without* the `TIME_BUDGET = …` / `T0 = …` lines (the preamble owns them). The builder asserts the slice starts with `# Runtime integrity` and contains `def rsna_strict_load(`.
3. **Cell 12 asset finder (verbatim slice)** — from `ASSET_ROOTS = [` (line 123) through the end of `_asset_find_asset` (line 165).
4. **Cell 14 dense sampler (verbatim, whole cell)**.
5. **Cell 45 Raptor branch (verbatim slice with two token patches)** — lines 2–8 (imports) and 14–635, **skipping lines 10–12** (`_ke_primary` / `_ke_ours` / `_KE_LAB`) and **removing lines 629–635** (the `raptor_input_before_coat.csv` copy and summary that reference the transformer stack). Patches, each asserted to match exactly once: (a) `    outputs = [` (line 491) → `    outputs = globals().setdefault("_KE_TEACHER_OUTPUTS", [`, with the matching `    ]` two lines later → `    ])`; (b) `    test_ids = test['StudyInstanceUID'].tolist()` (line 478) → the same line followed by `    globals()["_KE_TEACHER_IDS"] = list(test_ids)`. The two patches make the running arrays visible to the flush thread. `RUN['raptor_k_eval']` (94 windows) is honoured as in the 0.942 run.
6. **Flush thread + final writer (ours)**, placed *before* cell 5's `_ke_run_raptor_arms()` call would run — so the builder inserts this cell between cell 4 and cell 5 with the thread started at import time, and puts the final writer as cell 6 after cell 5:

```python
# %%
import threading
def _teacher_snapshot(path):
    outs = globals().get("_KE_TEACHER_OUTPUTS"); ids = globals().get("_KE_TEACHER_IDS")
    if not outs or not ids:
        return 0
    raw = np.stack(outs)                                  # (4, N, 12), NaN where not yet predicted
    np.savez_compressed(path, study_uids=np.asarray(ids, dtype=str), raw_probabilities=raw)
    return int(np.isfinite(raw).all(axis=(0, 2)).sum())
def _teacher_flusher():
    while not globals().get("_TEACHER_DONE"):
        time.sleep(300)
        try:
            n = _teacher_snapshot("/kaggle/working/raptor_teacher_partial.npz")
            print(f"[teacher] partial flush: {n} studies complete, {(time.time() - T0) / 3600:.2f} h", flush=True)
        except Exception as e:
            print("[teacher] flush failed:", e, flush=True)
threading.Thread(target=_teacher_flusher, daemon=True).start()
```

```python
# %%
globals()["_TEACHER_DONE"] = True
raw = np.load("/kaggle/working/raptor_raw.npz", allow_pickle=False)
probs = raw["raw_probabilities"]; uids = [str(u) for u in raw["study_uids"]]
names = [a["name"] for a in _KE_NS["ARMS"]]; weights = np.asarray([float(a["w"]) for a in _KE_NS["ARMS"]]); weights /= weights.sum()
ok = np.isfinite(probs).all(axis=(0, 2))
mean = np.tensordot(weights, np.clip(np.nan_to_num(probs, nan=0.5), 0, 1), axes=(0, 0))
np.savez_compressed(f"/kaggle/working/raptor_teacher_shard{SHARD}.npz", study_uids=np.asarray(uids, dtype=str),
                    raw_probabilities=probs, view_names=np.asarray(names, dtype=str), view_weights=weights,
                    checkpoint_sha256=np.asarray([e.get("sha256", "") for e in _RSNA_AUDIT["events"] if e.get("kind") == "raptor_checkpoint"], dtype=str))
csv = pd.DataFrame(mean[ok], columns=LABELS); csv.insert(0, "StudyInstanceUID", np.asarray(uids)[ok]); csv.to_csv(f"/kaggle/working/raptor_teacher_shard{SHARD}.csv", index=False)
elapsed = time.time() - T0
json.dump({"shard": SHARD, "n_shards": N_SHARDS, "limit": LIMIT, "studies": int(ok.sum()), "failed_uids": [u for u, f in zip(uids, ok) if not f],
           "elapsed_s": elapsed, "sec_per_study": elapsed / max(1, int(ok.sum())), "k_eval": int(RUN["raptor_k_eval"])},
          open("/kaggle/working/teacher_receipt.json", "w"), indent=1)
print(f"[teacher] {int(ok.sum())} studies, {elapsed / max(1, int(ok.sum())):.1f} s/study, {len(uids) - int(ok.sum())} failed")
for p in ("/kaggle/working/chunk",):
    shutil.rmtree(p, ignore_errors=True)         # never publish the symlink into train_images as an output
```

`kernel-metadata.json`: id `tiankljucanin/rsna-knee-teacher`, title `RSNA Knee Teacher`, `code_file` `rsna-knee-teacher.ipynb`, GPU on, TPU off, internet off, `dataset_sources` = the three `dreaddevelopment/raptor-knee-*`, `kernel_sources` = `[]` (a resume adds the previous `tiankljucanin/rsna-knee-teacher` here), `competition_sources` = the competition, `model_sources` `[]`, `"machine_shape": "NvidiaTeslaT4"`.

- [ ] **Step 1: Write the failing checks**

```python
# src/teacher_pass_test.py
"""Checks for src/build_teacher_pass.py: the extracted Raptor branch compiles, the token patches applied exactly once,
the chunk preamble excludes gold and shards disjointly. CPU, seconds."""
import json, os, sys, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, "src")
import build_teacher_pass as tp  # noqa: E402
fails = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m); fails.append(m) if not c else None

nb = tp.load_notebook("notebook_score_0.942.ipynb")
src = tp.render_teacher_py(nb, shard=0, n_shards=1, limit=0)
cells = [c for c in src.split("\n# %%") if c.strip()]
check(len(cells) == 7, f"seven cells rendered ({len(cells)})")
for i, c in enumerate(cells):
    try:
        compile(c.replace("# %%", "", 1), f"cell{i}", "exec"); check(True, f"cell {i} compiles")
    except SyntaxError as e:
        check(False, f"cell {i} compiles: {e}")
check(src.count('globals().setdefault("_KE_TEACHER_OUTPUTS"') == 1 and src.count('globals()["_KE_TEACHER_IDS"]') == 1, "both token patches applied exactly once")
check("_ke_primary" not in src and "_pipeline_stage.csv" not in src and "raptor_input_before_coat" not in src, "transformer-stack dependencies removed")
check("def _ke_run_raptor_arms" in src and "def rsna_strict_load" in src and "def _asset_find_asset" in src and "def _dense_allocate" in src, "runner, helpers, asset finder, dense sampler present")
check("_ke_run_raptor_arms()" in src and src.index("def _teacher_flusher") < src.index("_ke_run_raptor_arms()"), "flusher defined before the runner call")

# chunking on the real train.csv
ns = {}
exec(tp.PREAMBLE_FUNCS, ns)
a = ns["chunk_study_ids"]("data/train.csv", 0, 3, 0); b = ns["chunk_study_ids"]("data/train.csv", 1, 3, 0); c = ns["chunk_study_ids"]("data/train.csv", 2, 3, 0)
check(len(a) + len(b) + len(c) == 4349 and not (set(a) & set(b)) and not (set(b) & set(c)), "3 shards: disjoint, 4,349 in total")
tr = pd.read_csv("data/train.csv", dtype={"StudyInstanceUID": str}); gold = set(tr.loc[tr[tp.LABELS].notna().all(axis=1), "StudyInstanceUID"])
check(not (set(a) | set(b) | set(c)) & gold, "no gold study in any shard")
check(ns["chunk_study_ids"]("data/train.csv", 0, 1, 100) == sorted(set(a) | set(b) | set(c))[:100], "LIMIT takes the first N of the sorted shard")

with tempfile.TemporaryDirectory() as d:
    tp.write_kernel(d, shard=1, n_shards=4, limit=6)
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    check(meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and not meta["enable_internet"], "metadata: T4x2, GPU, no internet")
    check(set(meta["dataset_sources"]) == {"dreaddevelopment/raptor-knee-maxspan", "dreaddevelopment/raptor-knee-native384", "dreaddevelopment/raptor-knee-native384dense"}, "metadata: the three CC0 Raptor datasets")
    ipynb = json.load(open(os.path.join(d, "rsna-knee-teacher.ipynb")))
    first = "".join(ipynb["cells"][0]["source"])
    check("SHARD = 1" in first and "N_SHARDS = 4" in first and "LIMIT = 6" in first, "shard constants baked into the notebook")
print("\n" + ("TEACHER PASS CHECKS PASSED" if not fails else f"TEACHER PASS CHECKS FAILED ({len(fails)})")); sys.exit(1 if fails else 0)
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError: No module named 'build_teacher_pass'`.

- [ ] **Step 3: Implement `src/build_teacher_pass.py`**: `load_notebook(path)` (json), `cell_lines(nb, i)`; `slice_lines(lines, start_pred, end_pred, label)` that finds the unique start line matching a predicate and the end line, asserting uniqueness; `patch_once(text, old, new, label)`; `PREAMBLE_FUNCS` (the `chunk_study_ids` / `done_uids` source as a string, so the test can exec it) and `PREAMBLE_TEMPLATE` with `__SHARD__ / __N_SHARDS__ / __LIMIT__`; `FLUSHER_CELL`, `FINAL_CELL`; `render_teacher_py(nb, shard, n_shards, limit)` concatenating `# %%`-prefixed cells in the order above; `write_kernel(out_dir, shard, n_shards, limit)` writing `rsna-knee-teacher.py` next to the notebook (percent format, committed for review), calling `nbgen.build(py_path, ipynb_path)` and writing `kernel-metadata.json` with `json.dump(indent=2)`; `--check` re-renders and diffs against disk (like `build_fork.py`). Exact slices: cell 16 lines `104..195` (assert `lines[104].startswith("# Runtime integrity")` and `lines[195] == ""` after `rsna_deadline`'s body; drop the two lines that start with `TIME_BUDGET = ` and `T0 = `); cell 12 lines `123..165`; cell 14 all lines; cell 45 lines `2..8` + `14..628` with the two patches. Name the cell-45 slice's `import` of `_RSNA_TEST_IDS`: the preamble defines it, so `len(_RSNA_TEST_IDS)` in line 632 is gone with the removed tail — assert `_RSNA_TEST_IDS` still appears (line 603's use is in the kept part? it is not; assert only that the kept text compiles).

- [ ] **Step 4: Run the checks** → `TEACHER PASS CHECKS PASSED`; then `.venv/Scripts/python.exe src/build_teacher_pass.py --limit 6 && .venv/Scripts/python.exe src/build_teacher_pass.py --limit 6 --check` → `check: ok` (deterministic).

- [ ] **Step 5: Commit**

```powershell
git add src/build_teacher_pass.py src/teacher_pass_test.py kaggle/rsna-knee-teacher/
git commit -m "build_teacher_pass: the 0.942 notebook's Raptor branch (verbatim, 2 token patches) as kaggle/rsna-knee-teacher over training-study chunks via RSNA_COMP_ROOT; raw per-view probabilities, partial flush, resume; checks"
```

---

### Task 8: `src/merge_teacher.py` — shards → one teacher table

**Files:**
- Create: `src/merge_teacher.py`
- Test: extend `src/teacher_pass_test.py` (`test_merge`)

**Interfaces:**
- Produces: `merge_teacher(npz_paths: list[str], out_csv: str, expect_n: int = 4349) -> pd.DataFrame` — concatenates shards, keeps studies finite in all four views, asserts unique UIDs and `expect_n` rows (`--allow-partial` skips the count assert for the spike), writes UID + 12 columns = view-weighted mean probability; prints per-label mean and the fraction of failed studies. Output `artifacts/teacher/raptor_teacher.csv` → published as a new version of `rsna-knee-teacher-tables` (Task 6's dir).

- [ ] **Step 1: Write the failing check** (append to `src/teacher_pass_test.py` before the final print):

```python
import merge_teacher as mt
with tempfile.TemporaryDirectory() as d:
    rng = np.random.default_rng(0)
    for k in range(2):
        uids = np.asarray([f"u{k}_{i}" for i in range(5)], dtype=str); raw = rng.uniform(0, 1, (4, 5, 12)).astype(np.float32)
        if k == 1: raw[2, 4, :] = np.nan                                 # one failed study in one view
        np.savez_compressed(os.path.join(d, f"raptor_teacher_shard{k}.npz"), study_uids=uids, raw_probabilities=raw,
                            view_names=np.asarray(["a", "b", "c", "d"]), view_weights=np.asarray([.6, .1, .1, .2]))
    out = mt.merge_teacher([os.path.join(d, f"raptor_teacher_shard{k}.npz") for k in range(2)], os.path.join(d, "t.csv"), expect_n=9)
    check(len(out) == 9 and list(out.columns) == ["StudyInstanceUID", *tp.LABELS], "merge: failed study dropped, schema")
    check(np.all((out[tp.LABELS].to_numpy() >= 0) & (out[tp.LABELS].to_numpy() <= 1)), "merge: probabilities in [0, 1]")
    try:
        mt.merge_teacher([os.path.join(d, "raptor_teacher_shard0.npz")] * 2, os.path.join(d, "t2.csv"), expect_n=10); check(False, "merge: duplicate UID across shards rejected")
    except SystemExit:
        check(True, "merge: duplicate UID across shards rejected")
```

- [ ] **Step 2: Run → fails** (`No module named 'merge_teacher'`).

- [ ] **Step 3: Implement**

```python
# src/merge_teacher.py
"""Merge raptor_teacher_shard*.npz (Task 7 kernel outputs) into artifacts/teacher/raptor_teacher.csv.

    .venv/Scripts/python.exe src/merge_teacher.py artifacts/kaggle_out/teacher_s*/raptor_teacher_shard*.npz
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def merge_teacher(npz_paths, out_csv, expect_n=4349, allow_partial=False):
    uids, means, weights_seen = [], [], None
    for p in npz_paths:
        z = np.load(p, allow_pickle=False)
        raw = z["raw_probabilities"].astype(float); w = np.asarray(z["view_weights"], float); w = w / w.sum()
        if weights_seen is not None and not np.allclose(w, weights_seen):
            raise SystemExit(f"{p}: view weights differ from the first shard")
        weights_seen = w
        ok = np.isfinite(raw).all(axis=(0, 2))
        print(f"  {os.path.basename(p)}: {int(ok.sum())}/{len(ok)} studies complete")
        means.append(np.tensordot(w, np.clip(raw[:, ok, :], 0, 1), axes=(0, 0))); uids += [str(u) for u in z["study_uids"][ok]]
    out = pd.DataFrame(np.concatenate(means), columns=LABELS); out.insert(0, "StudyInstanceUID", uids)
    if out.StudyInstanceUID.duplicated().any():
        raise SystemExit("duplicate StudyInstanceUID across shards")
    if not allow_partial and len(out) != expect_n:
        raise SystemExit(f"expected {expect_n} studies, got {len(out)} (use --allow-partial for a spike)")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True); out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(out)} studies; per-label mean " + ", ".join(f"{l} {out[l].mean():.2f}" for l in LABELS[:4]) + " …")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("npz", nargs="+"); ap.add_argument("--out", default="artifacts/teacher/raptor_teacher.csv")
    ap.add_argument("--expect-n", type=int, default=4349); ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(); paths = sorted(sum([glob.glob(p) for p in a.npz], []))
    merge_teacher(paths, a.out, a.expect_n, a.allow_partial)
```

- [ ] **Step 4: Run the checks** → `TEACHER PASS CHECKS PASSED`. **Step 5: Commit** `git commit -m "merge_teacher: shard npz -> raptor_teacher.csv (view-weighted mean probabilities, failed studies dropped, UID uniqueness)"`.

---

### Task 9: Teacher kernel smoke (LIMIT 6) and spike (LIMIT 100) on Kaggle — card P-39

**Files:**
- Modify: `docs/proposals.md` (card **P-39 Raptor teacher pass** + index row: Hypothesis = a member of ours trained on `0.5 · LLM + 0.5 · quantile-matched Raptor` reaches gold-58 ≥ S2 `v09a` 0.8922 + 0.05 (direction) and a solo LB ≥ baseline + 0.005; Measure/Noise floor per spec §4; Cost = spike 1 h + pass ≈ 10–12 GPU-h + retrain 2.7 h; If it works / fails per spec), `docs/experiments.md` (Scoreboard ⏳ rows), `CLAUDE.md` layout table (the three new scripts + `kaggle/rsna-knee-teacher/`)

- [ ] **Step 1: Smoke** — `python src/build_teacher_pass.py --limit 6` → grep the `.py` for `^LIMIT = 6`, `^SHARD = 0`, `^N_SHARDS = 1`; `kaggle kernels push -p kaggle/rsna-knee-teacher`; `kernels status` until COMPLETE (≈ 8–10 min: 3 × 293 MB checkpoints load + 6 studies × 4 views); pull `--file-pattern "(teacher_receipt|raptor_teacher_shard0\.csv|\.log)$"` into `artifacts/kaggle_out/teacher_smoke/`. Green: receipt `studies: 6`, `failed_uids: []`, the log shows `[raptor-fast] 6 studies; balanced arm groups`, no `raptor_study_failed` event, csv values in (0, 1) and not all 0.5. Red on a validator (`Raptor study order drift`, `missing/extra/duplicate UID`, `expected one pinned asset`) → fix the **preamble** (schema of the chunk files / dataset mount slugs), never their code; re-smoke.
- [ ] **Step 2: Spike** — `--limit 100`, push, wait (≈ 1 h), pull the receipt: `sec_per_study` → `N_SHARDS = ceil(4349 * sec_per_study / (7.5 * 3600))` (7.5 h of the 8 h budget; expect 2–3). Run `merge_teacher.py … --allow-partial --out artifacts/teacher/raptor_spike100.csv` and print its per-label mean vs the LLM blend's positive rates (`build_targets.py` log) as a plausibility read (their operating point is *less positive* on Effusion / MCL; that is fine, quantile matching removes it). Log both runs (Scoreboard rows) and the shard plan in the P-39 card via `/update`; commit.
- [ ] **Step 3 (after Saturday's quota reset, Tian's go from the design):** build shards `for k in 0..N-1: build_teacher_pass.py --shard k --n-shards N` into `kaggle/rsna-knee-teacher/` **one at a time** (the slug is one kernel; each push = one version; run two versions concurrently at most — the second GPU slot), pull each `raptor_teacher_shard{k}.npz` into `artifacts/kaggle_out/teacher_s{k}/`, `merge_teacher.py` → `artifacts/teacher/raptor_teacher.csv` (4,349 rows), copy into `artifacts/ship_teacher/`, `kaggle datasets version -p . -m "raptor_teacher.csv (4 views, 94 windows, 4,349 studies)"`, `datasets status` ready. A guard-stopped shard: re-push the same shard with the previous version added to `kernel_sources` (resume by `done_uids`).

---

### Task 10: RunPod run of `v09e` / `v09f` / `v09s` and the read-out

**Files:**
- Read: `scripts/runpod_bootstrap.sh`; outputs `artifacts/kaggle_out/pod_v09{e,f,s}/`
- Modify (after reading): `docs/experiments.md`, `docs/proposals.md` (P-36…P-38 → measured), via `/update`

- [ ] **Step 1 (needs Tian's pod + SSH):** on the pod, from a fresh clone at the commit of Task 6: `export KAGGLE_USERNAME=… KAGGLE_KEY=…` (Tian's; never printed), `bash scripts/runpod_bootstrap.sh setup` (≈ 40 min; verify `layout under /kaggle/input` lists the four cache shards, the three label tables, `rsna-knee-teacher-tables`, the two timm weight dirs, dinov2), then sequentially `bash scripts/runpod_bootstrap.sh train v09f`, `RSNA_TEACHER_TABLES='("selfdistill_v1",)' bash scripts/runpod_bootstrap.sh train v09s`, `bash scripts/runpod_bootstrap.sh train v09e` (16 epochs, ≈ 1.7 h), each followed by `bash scripts/runpod_bootstrap.sh ship <arm>`. If P-34's `v09d` (3e-5 at 8 epochs) has read `< 0.865` by then, skip `v09e`.
- [ ] **Step 2: Read-out** — `kaggle datasets download -d tiankljucanin/rsna-knee-ckpt-<arm> -f <arm>_fold0_oof.csv -p artifacts/kaggle_out/pod_<arm>/`; per-label table with the S1 script pattern (all 882 rows, `hard = y__ > 0.5` — the **unchanged** teacher, which is why `y__` had to stay) vs `v09c` 0.8730 (`artifacts/kaggle_out/train_v23/v09c_fold0_oof.csv`): ≥ 0.881 ✅ / 0.865–0.881 🔁 / < 0.865 harmful; ≥ 9/12 labels up as support. `/update`: experiments entry, Scoreboard rows, card statuses, index; commit.

---

### Task 11: Baseline solo submission of the S2 `v09a` (Tian's go from the design)

**Files:**
- Modify: `kaggle/rsna-knee-infer/kernel-metadata.json` (`dataset_sources` += `tiankljucanin/rsna-knee-ckpt-v09a`)
- Generate: `kaggle/rsna-knee-infer/rsna-knee-infer.ipynb` from a sed'd copy

- [ ] **Step 1:** `sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "infer"/' -e 's/^INFER_MEMBERS = \[.*\]/INFER_MEMBERS = ["v09a"]/' src/kaggle_pipeline.py > artifacts/infer_solo_v09a.py`; grep the three lines; `nbgen` into `kaggle/rsna-knee-infer/rsna-knee-infer.ipynb`; add the ckpt dataset to the metadata; push; `kernels status`; on COMPLETE pull the log (`--file-pattern "no_match"`) and confirm `infer members (1): v09a/fold0 … score 0.8922`, `wrote … submission.csv … constant labels 0`.
- [ ] **Step 2:** `kaggle competitions submit rsna-knee-abnormality-detection -k tiankljucanin/rsna-knee-infer -v <version> -f submission.csv -m "Solo baseline: S2 v09a alone (CoAtNet-1 c02 window_attn, all-data 8 ep SWA, batch 2 + light aug; gold-58 0.8922) -- the reference for the distilled member"`; Submissions + Scoreboard rows (⏳, read ≈ 2 h later; expected ≈ 0.90–0.91 by the offset). Commit the infer tree.

---

### Task 12: Distilled production retrain and the second solo read (after Task 9 step 3)

- [ ] **Step 1:** `TEACHER_TABLES = ("raptor_teacher",)`, `PARALLEL_ARMS = ("v09a", "v08a")` sed'd (both members on the same teacher), `FORCE_SMOKE = True` → smoke on Kaggle (`teacher table raptor_teacher: 4349 studies` in both children's logs) → real (≈ 2.7 h) → read gold-58 SWA vs 0.8922 / 0.8850 (direction only) → ship as **new Dataset slugs** `rsna-knee-ckpt-v09a-rt` / `-v08a-rt` (so the S2 members stay mounted for the fork; `ship` dir with the renamed metadata id) — note: the checkpoints keep `version v09a`; the fork's `--member v09a=…-rt` flag points at the new slug.
- [ ] **Step 2:** solo submission of the distilled `v09a` exactly as Task 11 with the `-rt` dataset; read vs the baseline: ≥ +0.005 ✅ → fork at β 0.10 with the `-rt` members and `/update` P-39 ✅; ≤ +0.004 🔁; a drop ❌ (track closed, production members unchanged).

---

## Self-review

- **Spec coverage:** §1 → Tasks 7, 8, 9 (builder, merge, spike/pass, licence noted in P-39); §2 → Tasks 1, 2, 3, 6; §3 → Tasks 4, 5, 10; §4 → Tasks 9–12 (measurement rules, baseline + distilled solo reads, order, budget). Risks: validator rejections → Task 9 step 1 rule; rank-vs-probability → Task 8 merges raw probabilities; quota guard → Task 7 flush + `done_uids`; teacher-copies-errors → Task 12 verdict rule; CLI hangs → Global Constraints.
- **Placeholders:** none; every code step carries its code; the P-36…P-39 cards follow the template fields listed in Task 5/9.
- **Type consistency:** `quantile_match(pred, ref)`, `mix_teacher(soft, tables, mix, is_gold)`, `load_teacher_tables(root, index, names)` identical in Tasks 1 and 3; `yt__{L}` / `b["yt"]` / `TEACHER_TABLES` / `TEACHER_MIX` / `TEACHER_PATHS` named the same in Tasks 3, 5, 6, 10, 12; `label_pos_weight(targets, study_ids, max_w)` and `weighted_bce(..., pos_weight=None)` in Task 4 and its loss call; `render_teacher_py` / `write_kernel` / `PREAMBLE_FUNCS` in Tasks 7's test and implementation; `merge_teacher(npz_paths, out_csv, expect_n, allow_partial)` in Task 8.
- **Review Focus:** items 1–3 pinned in Task 1's checks (and 2 additionally in Task 3's fatal-if-unmounted branch, exercised by the smoke with a wrong name if desired), 4 in Task 4, 5 in Task 7 + Task 8.
