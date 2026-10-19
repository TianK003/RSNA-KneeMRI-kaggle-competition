"""Mock tests of C17 (src/build_anchor_copy.py --anchor 09586s6fx) on CPU, no Kaggle data.

A. The run cell (cell 12): V2's own cell 8 (stage 3) and cell 9 (Part B, its inference subprocess mocked) on fake data, then C17's
   run cell with a fake s6fall script.  Checks SELF_CHECK, the re-blend against an independent recompute, every fallback branch,
   and that /kaggle/working/submission.csv exists at every point (traps 74 P1: the fake s6fall checks it while it runs).
B. The s6fall script's C17 patches (traps 74 P3): run_b with a fake worker group (deadline cuts under / over the 50 % cap, a retry,
   a retry that reaches its deadline) and the hang guard taken from K1_GUARD_S.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/c17_mock_test.py [rsna-knee-anchor9586s6fxw]
"""
import glob as globmod
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = sys.argv[1] if len(sys.argv) > 1 else "rsna-knee-anchor9586s6fx"   # any C17-based render (C18 / C19 variants too)
NB = os.path.join(ROOT, "kaggle", SLUG, SLUG + ".ipynb")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
W = lambda p: str(p).replace("\\", "/")   # noqa: E731


def cells():
    nb = json.load(open(NB, encoding="utf-8"))
    src = lambda i: (lambda x: "".join(x) if isinstance(x, list) else x)(nb["cells"][i]["source"])   # noqa: E731
    c8, c9, script, run = src(8), src(9), src(11), src(12)
    assert run.startswith("# Anchor9586s6fx: stage S"), run[:60]
    assert script.startswith("%%writefile /tmp/s6fall_run.py\n")
    return c8, c9, script.split("\n", 1)[1], run


CELL8, CELL9, SCRIPT, RUN = cells()


# ---------------------------------------------------------------------------------------------------------------------------- A
def scenario(name, n=40, s6_mode="ok", partb=True, r6_fail_rows=(), s6_fail_rows=(), dup384=(), dup224=(), seed=0, noise_s6=False,
             no_receipt=False):
    rng = np.random.default_rng(seed)
    root = Path(tempfile.mkdtemp(prefix="c17_"))
    work, data, pred, kw, s6root = root / "working", root / "data", root / "pred", root / "kw", root / "s6xroot"
    for d in (work, data, pred / "worker0", kw):
        d.mkdir(parents=True)
    ids = ["1.2.%d" % i for i in range(n)]
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(data / "test.csv", index=False)
    pd.DataFrame([[ids[0]] + [0.5] * 12], columns=["StudyInstanceUID"] + LABELS).to_csv(data / "sample_submission.csv", index=False)

    def frame(vals):
        df = pd.DataFrame(vals, columns=LABELS)
        df.insert(0, "StudyInstanceUID", ids)
        return df
    base = rng.random((n, 12))
    ours_v = np.clip(base + 0.3 * rng.random((n, 12)), 0, 1)
    for r in r6_fail_rows:
        ours_v[r] = ours_v.mean(axis=0)
    frame(ours_v).to_csv(work / "ours_sub.csv", index=False)
    r384 = base + 0.2 * rng.random((n, 12))
    for r in dup384[1:]:
        r384[r] = r384[dup384[0]]              # a blank-volume block: rows that exactly repeat each other
    r224 = base + 0.25 * rng.random((n, 12))
    for r in dup224[1:]:
        r224[r] = r224[dup224[0]]
    frame(r384).to_csv(work / "raptor_raw.csv", index=False)
    frame(r384).to_csv(work / "raptor_sub.csv", index=False)
    frame(r224).to_csv(work / "raptor224_raw.csv", index=False)
    frame(r224).to_csv(work / "raptor224_sub.csv", index=False)
    frame(ours_v).drop(index=list(r6_fail_rows)).to_csv(pred / "worker0" / "predictions.csv", index=False)
    g = {"__name__": "__main__", "DATA": data, "WORK": work, "pred": pred, "time": time, "json": json,
         "RAPTOR224_SECONDS": 1.0, "RAPTOR_SECONDS": 1.0, "NB_T0": time.time(), "display": lambda *a, **k: None}
    exec(compile(CELL8, "cell8", "exec"), g)
    c9 = CELL9.replace("/kaggle/working", W(work))
    runner = c9[c9.index("# ---- goodpjw2008"):]
    own_v = base + 0.5 * rng.random((n, 12))
    real_run, real_glob = subprocess.run, globmod.glob

    def fake_run(args, *a, **k):
        if any(str(x).endswith("infer.py") for x in args):
            if not partb:
                raise RuntimeError("mock: Part B infer fails")
            frame(own_v).sample(frac=1, random_state=1).to_csv(args[4], index=False)
        return subprocess.CompletedProcess(args, 0)

    def fake_glob(pat, *a, **k):
        if pat.endswith("/") and "/kaggle/input" in pat:
            return [W(kw) + "/"] if pat.count("*") == 1 else []
        if "cnxt_v0_fold" in pat:
            return ["ck0.pt"]
        return real_glob(pat, *a, **k)
    (kw / "knee.py").write_text("#")
    (kw / "test_series.csv").write_text("x")
    subprocess.run, globmod.glob = fake_run, fake_glob
    try:
        exec(compile(runner, "cell9", "exec"), g)
    finally:
        subprocess.run, globmod.glob = real_run, real_glob
    v2_final = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})

    # the fake s6fall: deletes ITS OWN root's submission.csv (as nartaa's does at start), checks V2's is still in place, writes its
    # outputs under K1_KAGGLE_ROOT, records the env it got
    s6_v = rng.random((n, 12)) if noise_s6 else base + 0.4 * rng.random((n, 12))
    frame(s6_v).to_csv(root / "s6.csv", index=False)
    fake = root / "s6fall_run.py"
    real_sub = W(work / "submission.csv")
    head = ("import os, sys, json, time, pandas as pd\nR = os.environ['K1_KAGGLE_ROOT']\nos.makedirs(R + '/working/k1', exist_ok=True)\n"
            "json.dump({'root': R, 'guard': os.environ.get('K1_GUARD_S')}, open(%r, 'w'))\n"
            "if os.path.exists(R + '/working/submission.csv'): os.remove(R + '/working/submission.csv')\n"
            "assert os.path.isfile(%r), 'V2 submission.csv absent while s6fall runs'\nopen(%r, 'w').write('seen')\n"
            % (W(root / "s6env.json"), real_sub, W(root / "v2_seen")))
    if s6_mode == "ok":
        body = head + ("pd.read_csv(%r).to_csv(R + '/working/submission.csv', index=False)\n" % W(root / "s6.csv"))
        if not no_receipt:
            body += ("json.dump({'b_failed_rank_A_only': %r}, open(R + '/working/k1/blend_receipt.json', 'w'))\n"
                     "json.dump({'c17': {'crops': 'mock'}}, open(R + '/working/k1/k1_receipt.json', 'w'))\n"
                     % {ids[r]: "x" for r in s6_fail_rows})
    elif s6_mode == "fail":
        body = head + "sys.exit(3)\n"
    elif s6_mode == "hang":
        body = head + "time.sleep(60)\n"
    elif s6_mode == "late":
        body = "raise SystemExit(9)\n"           # must never be started
    fake.write_text(body)
    rc = (RUN.replace("/tmp/s6xroot", W(s6root)).replace("/kaggle/working", W(work)).replace("/tmp/s6fall_run.py", W(fake))
          .replace("_s6x_os.symlink('/kaggle/input', _s6x_root + '/input')", "_s6x_os.makedirs(_s6x_root + '/input')")
          .replace("['pkill', '-9', '-f', '/tmp/k1']", "['true']"))
    if s6_mode == "hang":
        rc = rc.replace("_S6X_END_H = 8.4", "_S6X_END_H = %r" % ((time.time() - g["NB_T0"]) / 3600 + 3 / 3600))
    if s6_mode == "late":
        rc = rc.replace("_S6X_LATEST_START_H = 6.0", "_S6X_LATEST_START_H = 0.0")
    if os.name == "nt":
        rc = rc.replace("start_new_session=True", "start_new_session=False").replace(
            "lambda: _s6x_os.killpg(p.pid, _s6x_sig.SIGKILL), ", "")
    exec(compile(rc, "run_cell", "exec"), g)
    info = json.load(open(work / "s6x_receipt.json"))
    assert (work / "submission.csv").is_file(), "no submission.csv after the run cell"
    assert not (work / ".submission_s6x.tmp").exists(), "temporary file left behind"
    final = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})
    if s6_mode != "late":
        assert (root / "v2_seen").is_file(), "the fake s6fall did not see V2's submission.csv"
        env = json.load(open(root / "s6env.json"))
        assert env["root"] == W(s6root) and int(env["guard"]) >= 600, env
        info["guard_s"] = int(env["guard"])
    if info["status"] == "ok":                  # independent expected value
        pr = lambda v, ok: pd.Series(np.where(ok, v, np.nan)).rank(pct=True).values   # noqa: E731
        r6ok = np.ones(n, bool)
        r6ok[list(r6_fail_rows)] = False
        s6ok = np.ones(n, bool)
        s6ok[list(s6_fail_rows)] = False
        f384 = np.zeros(n, bool)
        f384[list(dup384)] = True
        f224 = np.zeros(n, bool)
        f224[list(dup224)] = True
        frap = f384 & f224
        oursdf = pd.read_csv(work / "ours_sub.csv").set_index("StudyInstanceUID").loc[ids]
        exp = {}
        for j, c in enumerate(LABELS):
            a, b = pr(oursdf[c].values, r6ok), pr(s6_v[:, j], s6ok)
            ro = np.where(r6ok & s6ok, 0.6 * a + 0.4 * b, np.where(r6ok, a, np.where(s6ok, b, np.nan)))
            q384, q224 = pr(r384[:, j], ~f384), pr(r224[:, j], ~f224)
            rr = np.where(f384, q224, np.where(f224, q384, 0.65 * q384 + 0.35 * q224))
            w = g["W_RAPTOR_C"].get(c, 0.5)
            v = w * rr + (1 - w) * ro
            v = np.where(frap, ro, v)
            v = np.where(np.isnan(ro), rr, v)
            v = np.where(frap & np.isnan(ro), 0.5, v)
            exp[c] = v.astype(np.float32)
        st = pd.DataFrame(exp)
        st.insert(0, "StudyInstanceUID", ids)
        if info.get("part_b_applied"):
            own = pd.DataFrame(own_v, columns=LABELS)
            rs, rown = st[LABELS].rank(pct=True), own.rank(pct=True)
            for c in LABELS:
                w = g["OWN_W_PER_TARGET"][c]
                st[c] = (1 - w) * rs[c] + w * rown[c]
            st[LABELS] = st[LABELS].rank(pct=True)
        err = float(np.abs(st[LABELS].values - final[LABELS].values).max())
        assert err < 1e-6, ("independent recompute differs", err)
        info["independent_check_err"] = err
    else:
        assert final.equals(v2_final), "fallback did not restore V2"
    print("  %-34s -> %s" % (name, {k: info.get(k) for k in ("status", "part_b_applied", "selfcheck_max_abs_diff",
                                                                "raptor_duplicate_rows", "s6f_failed", "rho_mean",
                                                                "independent_check_err", "error") if info.get(k) is not None}))
    shutil.rmtree(root, ignore_errors=True)
    return info


def part_a():
    print("A. run cell")
    assert scenario("ok + Part B")["status"] == "ok"
    assert scenario("ok, Part B failed", partb=False, seed=1)["status"] == "ok"
    assert scenario("ok, R6 / s6fall / both failed", r6_fail_rows=(2, 5), s6_fail_rows=(5, 7), seed=2)["status"] == "ok"
    assert scenario("3 studies", n=3, seed=3)["status"] == "ok"
    i = scenario("partial s6fall (40 % failed), n 200", n=200, s6_fail_rows=tuple(range(0, 200, 5)) + tuple(range(1, 200, 5)), seed=4)
    assert i["status"] == "ok" and i["s6f_failed"] == 80
    i = scenario("Raptor duplicate rows (P2)", dup384=(3, 9, 11), dup224=(9, 20), seed=5)
    assert i["status"] == "ok" and i["raptor_duplicate_rows"] == {"384": 3, "224": 2}
    assert scenario("rho floor: s6fall noise, n 150", n=150, noise_s6=True, seed=6)["status"] == "combine_failed"
    assert scenario("blend_receipt.json missing", no_receipt=True, seed=7)["status"] == "combine_failed"
    assert scenario("s6fall fails", s6_mode="fail", seed=8)["status"] == "failed"
    assert scenario("s6fall late", s6_mode="late", seed=9)["status"] == "skipped_late"
    assert scenario("s6fall hangs", s6_mode="hang", seed=10)["status"] == "timeout"


# ---------------------------------------------------------------------------------------------------------------------------- B
def load_script(tmp, guard=None):
    p = tmp / "s6fall_c17.py"
    p.write_text(SCRIPT, encoding="utf-8")
    old = os.environ.pop("K1_GUARD_S", None)
    os.environ.pop("K1_CROPS", None)
    if guard is not None:
        os.environ["K1_GUARD_S"] = str(guard)
    os.environ["K1_KAGGLE_ROOT"] = W(tmp / "kroot")
    os.environ["K1_TMP"] = W(tmp / "k1tmp")
    try:
        spec = importlib.util.spec_from_file_location("s6f_c17_%d" % (guard or 0), p)
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
    finally:
        os.environ.pop("K1_GUARD_S", None)
        if old is not None:
            os.environ["K1_GUARD_S"] = old
    return S


def fake_group(S, plan):
    """run_group stand-in: each spec's worker 'scores' its assignment per plan(tag, ids) -> (cut ids, failed ids)."""
    calls = []

    def run_group(specs, timeout_s, what):
        for s in specs:
            cmd = s["cmd"]
            ids = json.loads(Path(cmd[cmd.index("--assignment") + 1]).read_text())
            out = Path(cmd[cmd.index("--output") + 1])
            cut, bad = plan(s["name"], ids)
            fb = {u: "not reached before the deadline" for u in cut}
            fb.update({u: "decode error" for u in bad})
            ok = [u for u in ids if u not in fb]
            z = {u: [list(np.linspace(-1, 1, 12) + (hash(u) % 97) / 97.0)] for u in ok}
            pd.DataFrame([[u] + list(1 / (1 + np.exp(-np.asarray(z[u][0])))) for u in ok],
                         columns=["StudyInstanceUID"] + S.LAB).to_csv(out / "predictions.csv", index=False)
            (out / "logits.json").write_text(json.dumps(z))
            rec = {"status": "PASS", "studies": len(ids), "scored": len(ok), "fallback": fb, "timed_out": bool(cut),
                   "amp": S.B_AMP, "models": [S.B_MODEL["file"]], "spatial_models": 1, "anchor_weights_sha256": S.ANATOMY_SHA256,
                   "sha256": {k: S.SOURCE_SHA[k] for k in S.B_RECEIPT_SOURCES}, "anchor_methods": {"m": len(ok)}}
            (out / "receipt.json").write_text(json.dumps(rec))
            (out / "log.txt").write_text("WORKER_DONE\n")
            calls.append((s["name"], len(ids), len(cut), len(bad)))
        return [(s["name"], 0, 0.0, "") for s in specs]
    S.run_group = run_group
    return calls


def part_b():
    print("B. s6fall script (C17 patches)")
    tmp = Path(tempfile.mkdtemp(prefix="c17s_"))
    S0 = load_script(tmp)
    assert S0.GUARD_S == S0.GUARD_MAX_S == 480 * 60 and S0.B_FAIL_FRAC == 0.5
    assert os.environ["K1_CROPS"] == "16,16;0,0;0,32;32,0;32,32" == S0._RECEIPT["c17"]["crops"]
    S = load_script(tmp, guard=7740)
    assert S.GUARD_S == 7740 and S.GUARD_MAX_S == 480 * 60, (S.GUARD_S, S.GUARD_MAX_S)
    print("  guard: default %ds, K1_GUARD_S 7740 -> %ds; crops %s" % (S0.GUARD_S, S.GUARD_S, S._RECEIPT["c17"]["crops"]))
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.TMP.mkdir(parents=True, exist_ok=True)
    S._RECEIPT["env"] = {"n_gpu": 2}
    I = S.Inputs()
    I.launcher = tmp / "launcher.py"
    for k in ("comp", "code", "models", "anatomy"):
        setattr(I, k, tmp)
    ids = ["9.9.%d" % i for i in range(200)]
    I.cost = {u: 1 + i % 7 for i, u in enumerate(ids)}

    def run(plan, n=200):
        calls = fake_group(S, plan)
        P, Z, info, failed = S.run_b(I, "B", ids[:n], "test", 3600)
        return calls, P, info, failed

    # a deadline cut keeps the scored studies: 30 % cut -> failed, no retry, no error
    calls, P, info, failed = run(lambda nm, u: (u[int(0.7 * len(u)):], []))
    assert len(failed) == sum(len(p) - int(0.7 * len(p)) for p in S.split_parts(ids, I.cost, 2)), len(failed)
    assert info["retry"] is None and len(calls) == 2 and np.isfinite(P[[u not in failed for u in ids]]).all()
    print("  deadline cut 30 %%: %d failed, %d scored, no retry" % (len(failed), 200 - len(failed)))
    # a cut beyond the cap (50 %) is still a hard error
    try:
        run(lambda nm, u: (u[int(0.3 * len(u)):], []))
        raise AssertionError("a 70 % cut must exceed the cap")
    except S.K1Error as e:
        print("  deadline cut 70 %%: K1Error (%s...)" % str(e)[:60])
    # no deadline, 6 failures: one retry that scores 4 of them; 2 stay failed
    first = {}

    def plan_retry(nm, u):
        if nm.endswith("retry"):
            return [], u[:2]
        first[nm] = u[:3]
        return [], u[:3]
    calls, P, info, failed = run(plan_retry)
    assert info["retry"] is not None and len(failed) == 2 and calls[-1][0].endswith("retry"), (calls, failed)
    print("  6 failures -> retry -> %d failed" % len(failed))
    # a retry that reaches its deadline keeps what it scored (C17: logged, not a hard error)
    calls, P, info, failed = run(lambda nm, u: (u[:1], u[1:2]) if nm.endswith("retry") else ([], u[:3]))
    assert len(failed) == 2, failed
    print("  retry reaches its deadline -> %d failed, no error" % len(failed))
    # the 3-study placeholder: any failure is still a hard error (cap 0 below 100 studies)
    try:
        run(lambda nm, u: ([], u[:1]) if not nm.endswith("retry") else ([], u[:1]), n=3)
        raise AssertionError("a failure on 3 studies must be a hard error")
    except S.K1Error:
        print("  3 studies, 1 failure after the retry: K1Error (cap 0)")
    calls, P, info, failed = run(lambda nm, u: ([], []), n=3)
    assert not failed
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    part_b()
    part_a()
    print("ALL C17 MOCK SCENARIOS PASS")
