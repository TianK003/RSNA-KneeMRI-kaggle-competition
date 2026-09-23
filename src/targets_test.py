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
            bd.build_distill_table([os.path.join(d, "a_fold0_oof.csv"), os.path.join(d, "b_fold*_oof.csv")],
                                    os.path.join(d, "t2.csv"))
            check(False, "distill table: an OOF set that does not cover every study of another set is rejected")
        except SystemExit:
            check(True, "distill table: partial coverage rejected")


if __name__ == "__main__":
    for fn in [test_quantile_match_preserves_ranks_and_scale, test_quantile_match_nan_passthrough, test_constant_column,
               test_mix_teacher_rows_and_gold, test_load_tables_validation, test_default_teacher_unchanged,
               test_distill_table_builder]:
        print(fn.__name__); fn()
    print("\n" + ("TARGET CHECKS PASSED" if not fails else f"TARGET CHECKS FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)
