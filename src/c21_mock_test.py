"""Mock tests of C21m's stage V (src/build_anchor_copy.py --anchor 09586s6fwvj / 09586s6fr7wvj) on CPU, no Kaggle data.

The render's last code cell runs on a fake base submission with its worker payload swapped for a fake worker that writes its
interleaved shard of studies the way the real one does (a partial CSV every 10 studies).  Checks the blend against an independent
recompute, every fallback branch (the base's own file restored exactly), a timeout that keeps the scored studies, and that
submission.csv and vjx_receipt.json exist at the end of every scenario.  The real payload is checked separately: it decodes to
build_anchor_copy.VJX_WORKER (sha256), and the render = its base's cells + the two stage-V cells.

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/c21_mock_test.py          # C19 + V-JEPA
    .venv/Scripts/python.exe src/c21_mock_test.py r7w      # C20 on C19 + V-JEPA
"""
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from build_anchor_copy import VJX_WORKER  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
assert VARIANT in ("", "r7w"), VARIANT
BASE_SLUG = "rsna-knee-anchor9586s6fr7w" if VARIANT else "rsna-knee-anchor9586s6fw"
SLUG = BASE_SLUG + "vj"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
W = lambda p: str(p).replace("\\", "/")   # noqa: E731
TXT = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]   # noqa: E731
NBJ = json.load(open(os.path.join(ROOT, "kaggle", SLUG, SLUG + ".ipynb"), encoding="utf-8"))
BASE = json.load(open(os.path.join(ROOT, "kaggle", BASE_SLUG, BASE_SLUG + ".ipynb"), encoding="utf-8"))
VJX = TXT(NBJ["cells"][-1])
WORKER_RE = re.compile(r"_VJX_WORKER = \('[A-Za-z0-9+/=]+', '[0-9a-f]{64}'\)")

FAKE_WORKER = r'''import os, sys, time
import numpy as np, pandas as pd
SHARD, NSHARD, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MODE = os.environ["FAKE_MODE"].split(",")[SHARD]
v = np.load(os.environ["FAKE_NPY"])
ids = [l.strip() for l in open(os.environ["FAKE_IDS"])]
idx = list(range(len(ids)))[SHARD::NSHARD]
if MODE == "fail":
    sys.exit(3)
rows = []
for k, i in enumerate(idx):
    if MODE == "partial" and k == len(idx) // 2:
        time.sleep(120)
    if MODE == "skipsome" and k % 4 == 0:
        continue
    rows.append([ids[i]] + list(v[i]))
    if (k + 1) % 10 == 0 or k + 1 == len(idx):
        pd.DataFrame(rows, columns=["StudyInstanceUID"] + __LABELS__).to_csv(OUT + ".tmp", index=False)
        os.replace(OUT + ".tmp", OUT)
'''.replace("__LABELS__", repr(LABELS))


def payload(src, corrupt=False):
    b = base64.b64encode(zlib.compress(src.encode("utf-8"), 9)).decode("ascii")
    return "_VJX_WORKER = (%r, %r)" % (b, ("0" * 64) if corrupt else hashlib.sha256(src.encode("utf-8")).hexdigest())


def check_real():
    """The render = the base's cells + the two stage-V cells; the payload decodes to VJX_WORKER."""
    assert len(NBJ["cells"]) == len(BASE["cells"]) + 2, "render is not base + 2 cells"
    tag = "09586s6fr7w" if VARIANT else "09586s6fw"
    for i, c in enumerate(BASE["cells"]):                    # the s6fall intro names the build tag: the base's + "vj"
        exp = TXT(c).replace("--anchor %s`" % tag, "--anchor %svj`" % tag) if i == 10 else TXT(c)
        assert TXT(NBJ["cells"][i]) == exp, ("cell differs from the base's", i)
    m = re.search(r"_VJX_WORKER = \('([A-Za-z0-9+/=]+)', '([0-9a-f]{64})'\)", VJX)
    dec = zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8")
    assert dec == VJX_WORKER and hashlib.sha256(dec.encode("utf-8")).hexdigest() == m.group(2), "payload != VJX_WORKER"
    meta = json.load(open(os.path.join(ROOT, "kaggle", SLUG, "kernel-metadata.json")))
    base_meta = json.load(open(os.path.join(ROOT, "kaggle", BASE_SLUG, "kernel-metadata.json")))
    assert meta["dataset_sources"] == base_meta["dataset_sources"] + ["nurikw3/rsna-knee-vjepa21-checkpoints"], "mounts"
    for k in ("kernel_sources", "docker_image", "machine_shape", "enable_gpu", "enable_internet", "competition_sources"):
        assert meta[k] == base_meta[k], k
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_internet"] is False
    print("  real payload = VJX_WORKER (sha ok); render = %s's %d cells + 2; metadata = base + V-JEPA, T4, no internet"
          % (BASE_SLUG, len(BASE["cells"])))


def scenario(name, n=60, modes="ok,ok", seed=0, garbage=False, late=False, corrupt=False, end_h=None, expect="ok"):
    rng = np.random.default_rng(seed)
    root = Path(tempfile.mkdtemp(prefix="c21_"))
    work = root / "working"
    work.mkdir()
    ids = ["1.2.%d" % i for i in range(n)]
    truth = rng.random((n, 12))
    base = pd.DataFrame(truth + 0.3 * rng.random((n, 12)), columns=LABELS).rank(pct=True)
    base.insert(0, "StudyInstanceUID", ids)
    base = base.sample(frac=1, random_state=seed).reset_index(drop=True)        # the base's own row order, not test_ids'
    base.to_csv(work / "submission.csv", index=False)
    base_file = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})   # the base file as the stage reads it
    vj = rng.random((n, 12)) if garbage else np.clip(truth + 0.4 * rng.random((n, 12)) - 0.2, 0, 1)
    np.save(root / "vj.npy", vj)
    (root / "ids.txt").write_text("\n".join(ids))
    os.environ.update(FAKE_MODE=modes, FAKE_NPY=W(root / "vj.npy"), FAKE_IDS=W(root / "ids.txt"))
    rc = WORKER_RE.sub(lambda m: payload(FAKE_WORKER, corrupt), VJX, count=1)
    assert rc != VJX, "payload not swapped"
    rc = rc.replace("/kaggle/working", W(work)).replace("/tmp/vjx_run.py", W(root / "vjx_run.py"))
    rc = re.sub(r"_VJX_NSHARD = max\(1, min\(\d+, _vjx_torch\.cuda\.device_count\(\)\)\)", "_VJX_NSHARD = 2", rc)
    if late:
        rc = re.sub(r"_VJX_LATEST_START_H, _VJX_END_H = [0-9.]+, [0-9.]+", "_VJX_LATEST_START_H, _VJX_END_H = 0.0, 8.5", rc)
    if end_h is not None:
        rc = re.sub(r"_VJX_LATEST_START_H, _VJX_END_H = [0-9.]+, [0-9.]+", "_VJX_LATEST_START_H, _VJX_END_H = 7.0, %r" % end_h, rc)
    if os.name == "nt":
        rc = rc.replace("start_new_session=True", "start_new_session=False").replace(
            "lambda: _vjx_os.killpg(p.pid, _vjx_sig.SIGKILL), ", "")
    g = {"__name__": "__main__", "test_ids": ids, "labels": LABELS, "NB_T0": time.time()}
    exec(compile(rc, "vjx_cell", "exec"), g)
    info = json.load(open(work / "vjx_receipt.json"))
    final = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})
    assert not (work / "submission.csv.vjx.tmp").exists(), "temporary file left behind"
    assert info["status"] == expect, (name, info)
    assert final["StudyInstanceUID"].tolist() == base["StudyInstanceUID"].tolist(), "row order changed"
    if info["status"] == "ok":                                              # independent expected value
        got = {}
        for part in sorted((work / "vjx").glob("shard*.csv")):
            for _, r in pd.read_csv(part, dtype={"StudyInstanceUID": str}).iterrows():
                got[r.StudyInstanceUID] = r[LABELS].values.astype(float)
        ok = np.array([u in got for u in base.StudyInstanceUID])
        err = 0.0
        for c in LABELS:
            a = base[c].rank(pct=True).values
            v = np.full(n, np.nan)
            v[ok] = pd.Series([got[u][LABELS.index(c)] for u in base.StudyInstanceUID[ok]]).rank(pct=True).values
            m = np.where(ok, 0.85 * a + 0.15 * v, a)
            err = max(err, float(np.abs(pd.Series(m).rank(pct=True).values - final[c].values).max()))
        assert err < 1e-9, ("independent recompute differs", err)
        assert float(np.abs(final[LABELS].values - base[LABELS].values).max()) > 0, "V-JEPA changed nothing"
        info["independent_check_err"] = err
    else:
        assert final.equals(base_file), "fallback did not restore the base's file"
    print("  %-42s -> %s" % (name, {k: info.get(k) for k in ("status", "scored", "rho_mean", "worker_rc", "independent_check_err",
                                                           "error") if info.get(k) is not None}))
    shutil.rmtree(root, ignore_errors=True)
    return info


def main():
    print("C21m stage V mock tests (%s; base %s)" % (SLUG, BASE_SLUG))
    check_real()
    scenario("ok, both workers")
    scenario("ok, one worker skips a quarter of its studies", modes="ok,skipsome", seed=1)
    scenario("one worker fails (exactly 50 % coverage)", modes="ok,fail", seed=2)
    scenario("timeout: both stop halfway (scored kept)", modes="partial,partial", end_h=0.003, seed=3, expect="combine_failed")
    scenario("timeout: one stops halfway, one completes", modes="ok,partial", end_h=0.003, seed=4)
    scenario("both workers fail", modes="fail,fail", seed=5, expect="combine_failed")
    scenario("V-JEPA output uncorrelated with the base", garbage=True, seed=6, expect="combine_failed")
    scenario("payload sha256 mismatch", corrupt=True, seed=7, expect="vjx_failed")
    scenario("too late to start", late=True, seed=8, expect="skipped_late")
    print("all C21m mock scenarios passed")


if __name__ == "__main__":
    main()
