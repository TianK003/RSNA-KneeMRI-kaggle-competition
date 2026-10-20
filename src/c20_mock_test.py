"""Mock tests of C20 (src/build_anchor_copy.py --anchor 09586s6fr7) on CPU, no Kaggle data.

V2's own cell 8 (stage 3) and cell 9 (Part B, its inference subprocess mocked) run on fake data, then C14's s6fall cell (cell 12) with
a fake s6fall script, then C20's stage R (cell 14) with its payload (V2's cells 6-7 patched to R7) swapped for a fake controller that
writes R7's predictions the way V2's cell 7 does (pred/worker*/predictions.csv + WORK/ours_sub.csv).  Checks the SELF_CHECK, the
three-member re-blend against an independent recompute, every fallback branch (C14's own file restored), and that submission.csv
and r7x_receipt.json exist at the end of every scenario.  The real payload is checked separately: it decodes to the render's own
patched cells 6-7 (sha256), and those differ from C14's cells only by build_anchor_copy.R7X_CELL6_PATCHES.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/c20_mock_test.py
"""
import base64
import glob as globmod
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from build_anchor_copy import R7X_CELL6_PATCHES  # noqa: E402

SLUG = "rsna-knee-anchor9586s6fr7"
NB = os.path.join(ROOT, "kaggle", SLUG, SLUG + ".ipynb")
C14 = os.path.join(ROOT, "kaggle", "rsna-knee-anchor9586s6f", "rsna-knee-anchor9586s6f.ipynb")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
W = lambda p: str(p).replace("\\", "/")   # noqa: E731
TXT = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]   # noqa: E731


def cells():
    nb = json.load(open(NB, encoding="utf-8"))
    c8, c9, script, s6x, r7x = (TXT(nb["cells"][i]) for i in (8, 9, 11, 12, 14))
    assert s6x.startswith("# Anchor9586s6f: stage S"), s6x[:60]
    assert r7x.startswith("# Anchor9586s6fr7 (C20): stage R"), r7x[:60]
    assert script.startswith("%%writefile /tmp/s6fall_run.py\n")
    return nb, c8, c9, s6x, r7x


NBJ, CELL8, CELL9, S6X, R7X = cells()
PAYLOAD_RE = re.compile(r"_R7X_CODE = \(\('cell6', '[A-Za-z0-9+/=]+', '[0-9a-f]{64}'\),\n\s+\('cell7', '[A-Za-z0-9+/=]+', '[0-9a-f]{64}'\)\)")


def payload(c6, c7, corrupt=False):
    enc = lambda s: base64.b64encode(zlib.compress(s.encode("utf-8"), 9)).decode("ascii")   # noqa: E731
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()   # noqa: E731
    return ("_R7X_CODE = (('cell6', %r, %r),\n             ('cell7', %r, %r))"
            % (enc(c6), ("0" * 64) if corrupt else sha(c6), enc(c7), sha(c7)))


def check_real_payload():
    """The render's payload = its own cells 6-7 with R7X_CELL6_PATCHES applied, and those cells = C14's (the R7 stage starts from
    exactly what #101 ran)."""
    c14 = json.load(open(C14, encoding="utf-8"))
    for i in range(10):                                   # V2's cells: byte-identical to C14's
        assert TXT(NBJ["cells"][i]) == TXT(c14["cells"][i]), ("V2 cell differs from C14's", i)
    for i in (11, 12):                                    # C14's s6fall script and run cell
        assert TXT(NBJ["cells"][i]) == TXT(c14["cells"][i]), ("s6fall cell differs from C14's", i)
    blobs = dict((k, (b, h)) for k, b, h in re.findall(r"\('(cell[67])', '([A-Za-z0-9+/=]+)', '([0-9a-f]{64})'\)", R7X))
    dec = {k: zlib.decompress(base64.b64decode(b)).decode("utf-8") for k, (b, h) in blobs.items()}
    for k, (b, h) in blobs.items():
        assert hashlib.sha256(dec[k].encode("utf-8")).hexdigest() == h, k
    exp6 = TXT(c14["cells"][6])
    for old, new in R7X_CELL6_PATCHES:
        assert exp6.count(old) == 1, old
        exp6 = exp6.replace(old, new)
    assert dec["cell6"] == exp6, "payload cell 6 != C14's cell 6 + the R7 patches"
    assert dec["cell7"] == TXT(c14["cells"][7]), "payload cell 7 != C14's cell 7"
    for needle in ("C_SLUG = 'rsna-knee-compact-r7cnxfsss'", "BUNDLE_ID = '8c7b666e27dc0e8f'", "FLIP_TTA = False",
                   "TMP = TMP / 'probe_r7'", "WORK = Path('/kaggle/working/r7')", "DEADLINE_HOURS = 8.3\n"):
        assert dec["cell6"].count(needle) == 1, needle
    meta = json.load(open(os.path.join(ROOT, "kaggle", SLUG, "kernel-metadata.json")))
    assert "heliosli/rsna-knee-compact-r7cnxfsss" in meta["dataset_sources"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_internet"] is False and meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
    print("  real payload: cells 6-7 = C14's + the %d R7 patches; metadata mounts R7, T4, no internet" % len(R7X_CELL6_PATCHES))


def scenario(name, n=40, r7_mode="ok", r7_fail_rows=(), r6_fail_rows=(), s6_fail_rows=(), s6_mode="ok", partb=True, seed=0,
             r7_noise=False, late=False, corrupt=False, expect="ok"):
    rng = np.random.default_rng(seed)
    root = Path(tempfile.mkdtemp(prefix="c20_"))
    work, data, pred, kw = root / "working", root / "data", root / "pred", root / "kw"
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
    r384, r224 = base + 0.2 * rng.random((n, 12)), base + 0.25 * rng.random((n, 12))
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

    # C14's s6fall cell with a fake s6fall (writes its submission.csv and k1/blend_receipt.json as nartaa's script does)
    s6_v = base + 0.4 * rng.random((n, 12))
    frame(s6_v).to_csv(root / "s6.csv", index=False)
    fake = root / "s6fall_run.py"
    if s6_mode == "ok":
        fake.write_text("import json, os, pandas as pd\nos.makedirs(%r, exist_ok=True)\npd.read_csv(%r).to_csv(%r, index=False)\n"
                        "json.dump({'b_failed_rank_A_only': %r}, open(%r, 'w'))\n"
                        % (W(work / "k1"), W(root / "s6.csv"), W(work / "submission.csv"), {ids[r]: "x" for r in s6_fail_rows},
                           W(work / "k1" / "blend_receipt.json")))
    else:
        fake.write_text("import sys\nsys.exit(3)\n")
    s6x = S6X.replace("/kaggle/working", W(work)).replace("/tmp/s6fall_run.py", W(fake))
    if os.name == "nt":
        s6x = s6x.replace("start_new_session=True", "start_new_session=False").replace(
            "lambda: _s6x_os.killpg(p.pid, _s6x_sig.SIGKILL), ", "")
    exec(compile(s6x, "s6x_cell", "exec"), g)
    c14_final = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})

    # stage R with a fake controller payload (V2's cell-7 file layout: pred/worker*/predictions.csv + WORK/ours_sub.csv)
    r7_v = rng.random((n, 12)) if r7_noise else np.clip(ours_v + 0.15 * rng.random((n, 12)) - 0.07, 0, 1)
    np.save(root / "r7.npy", r7_v)
    fake6 = ("from pathlib import Path\nimport numpy as np, pandas as pd\nWORK = Path(%r)\nWORK.mkdir(parents=True, exist_ok=True)\n"
             "TMP = Path(%r)\nT0 = NB_T0\nDEADLINE_HOURS = 8.3\n" % (W(work / "r7"), W(root / "probe_r7")))
    fake7 = ("pred = TMP / 'prediction'\n(pred / 'worker0').mkdir(parents=True, exist_ok=True)\n"
             "if %r == 'fail':\n    raise RuntimeError('controller failed with exit code 3')\n"
             "ids, L, v = %r, %r, np.load(%r)\nfail = set(%r)\n"
             "df = pd.DataFrame(v, columns=L); df.insert(0, 'StudyInstanceUID', ids)\n"
             "df.drop(index=sorted(fail)).to_csv(pred / 'worker0' / 'predictions.csv', index=False)\n"
             "fill = df.copy()\nfor r in fail:\n    fill.loc[r, L] = df[L].mean().values\n"
             "fill.to_csv(WORK / 'ours_sub.csv', index=False)\ndisplay(fill.head())\n"
             % (r7_mode, ids, LABELS, W(root / "r7.npy"), list(r7_fail_rows)))
    rc = PAYLOAD_RE.sub(lambda m: payload(fake6, fake7, corrupt), R7X, count=1)
    assert rc != R7X, "payload not swapped"
    rc = rc.replace("/kaggle/working", W(work))
    if late:
        rc = rc.replace("_R7X_LATEST_START_H = 6.3", "_R7X_LATEST_START_H = 0.0")
    exec(compile(rc, "r7x_cell", "exec"), g)
    info = json.load(open(work / "r7x_receipt.json"))
    assert (work / "submission.csv").is_file(), "no submission.csv after stage R"
    assert not (work / "submission.csv.r7x.tmp").exists(), "temporary file left behind"
    final = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})
    assert info["status"] == expect, (name, info)
    if info["status"] == "ok":                    # independent expected value
        pr = lambda v, ok: pd.Series(np.where(ok, v, np.nan)).rank(pct=True).values   # noqa: E731
        ok6, oks, ok7 = (np.ones(n, bool) for _ in range(3))
        ok6[list(r6_fail_rows)] = False
        oks[list(s6_fail_rows)] = False
        ok7[list(r7_fail_rows)] = False
        oursdf = pd.read_csv(work / "ours_sub.csv").set_index("StudyInstanceUID").loc[ids]
        exp = {}
        for j, c in enumerate(LABELS):
            mem = [(0.36, pr(oursdf[c].values, ok6), ok6), (0.40, pr(s6_v[:, j], oks), oks), (0.24, pr(r7_v[:, j], ok7), ok7)]
            num = sum(np.where(ok, w * r, 0.0) for w, r, ok in mem)
            den = sum(np.where(ok, w, 0.0) for w, r, ok in mem)
            ro = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
            q384, q224 = pr(r384[:, j], np.ones(n, bool)), pr(r224[:, j], np.ones(n, bool))
            rr = 0.65 * q384 + 0.35 * q224
            w = g["W_RAPTOR_C"].get(c, 0.5)
            exp[c] = np.where(np.isnan(ro), rr, w * rr + (1 - w) * ro).astype(np.float32)
        st = pd.DataFrame(exp)
        st.insert(0, "StudyInstanceUID", ids)
        if g.get("_s6x_use_b"):
            own = pd.DataFrame(own_v, columns=LABELS)
            rs, rown = st[LABELS].rank(pct=True), own.rank(pct=True)
            for c in LABELS:
                w = g["OWN_W_PER_TARGET"][c]
                st[c] = (1 - w) * rs[c] + w * rown[c]
            st[LABELS] = st[LABELS].rank(pct=True)
        err = float(np.abs(st[LABELS].values - final[LABELS].values).max())
        assert err < 1e-6, ("independent recompute differs", err)
        assert float(np.abs(final[LABELS].values - c14_final[LABELS].values).max()) > 0, "R7 changed nothing"
        info["independent_check_err"] = err
    else:
        assert final.equals(c14_final), "fallback did not restore C14's file"
    print("  %-38s -> %s" % (name, {k: info.get(k) for k in ("status", "selfcheck_max_abs_diff", "r7_scored", "rho_mean",
                                                                "independent_check_err", "error") if info.get(k) is not None}))
    shutil.rmtree(root, ignore_errors=True)
    return info


def main():
    print("C20 stage R mock tests (%s)" % SLUG)
    check_real_payload()
    scenario("ok + Part B")
    scenario("ok, Part B failed", partb=False, seed=1)
    scenario("ok, R6 / s6fall / R7 partial failures", r6_fail_rows=(3,), s6_fail_rows=(5, 9), r7_fail_rows=(7, 11, 20), seed=2)
    scenario("ok, a study none of the three scored", r6_fail_rows=(4,), s6_fail_rows=(4,), r7_fail_rows=(4,), seed=3)
    scenario("R7 covers < 50 %", r7_fail_rows=tuple(range(25)), seed=4, expect="combine_failed")
    scenario("R7 controller fails", r7_mode="fail", seed=5, expect="r7_failed")
    scenario("R7 output uncorrelated with R6", r7_noise=True, seed=6, expect="combine_failed")
    scenario("payload sha256 mismatch", corrupt=True, seed=7, expect="r7_failed")
    scenario("too late to start R7", late=True, seed=8, expect="skipped_late")
    scenario("s6fall failed (C14 = V2)", s6_mode="fail", seed=9, expect="selfcheck_failed")
    print("all C20 mock scenarios passed")


if __name__ == "__main__":
    main()
