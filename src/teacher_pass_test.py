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

# --- fix round 1, item 2: sibling slugs for resumes / the second concurrent slot (traps 31)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    tp.write_kernel(d, shard=1, n_shards=3, limit=0, slug="tiankljucanin/rsna-knee-teacher-b",
                    kernel_sources=["tiankljucanin/rsna-knee-teacher"])
    with open(os.path.join(d, "kernel-metadata.json")) as f:
        meta = json.load(f)
    check(meta["id"] == "tiankljucanin/rsna-knee-teacher-b" and meta["kernel_sources"] == ["tiankljucanin/rsna-knee-teacher"]
          and meta["code_file"] == "rsna-knee-teacher-b.ipynb" and meta["title"] == "RSNA Knee Teacher B"
          and os.path.isfile(os.path.join(d, "rsna-knee-teacher-b.ipynb")) and meta["machine_shape"] == "NvidiaTeslaT4",
          "--slug / --kernel-source: sibling id, title, code_file, kernel_sources")
try:
    tp.build_metadata("tiankljucanin/rsna-knee-teacher", ["tiankljucanin/rsna-knee-teacher"])
    check(False, "a kernel mounting its own output is refused")
except SystemExit:
    check(True, "a kernel mounting its own output is refused")

# --- fix round 1, items 1 + 3: prior rows accumulate; no flush once raptor_raw.npz exists (their in-place neutral fill)
import shutil, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402
def _npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    d = d.replace("\\", "/"); work, inp = f"{d}/working", f"{d}/input"
    for sub in ("working", "input/rsna-knee-teacher", "input/rsna-knee-abnormality-detection"):
        os.makedirs(f"{d}/{sub}")
    prior = np.full((4, 3, 12), 0.7, np.float32); prior[1, 2, 0] = np.nan                  # p0, p1 complete; p2 not
    np.savez_compressed(f"{inp}/rsna-knee-teacher/raptor_teacher_partial.npz", study_uids=np.asarray(["p0", "p1", "p2"]), raw_probabilities=prior)
    np.savez_compressed(f"{inp}/rsna-knee-abnormality-detection/raptor_teacher_shard0.npz",   # competition tree: never read
                        study_uids=np.asarray(["c0"]), raw_probabilities=np.zeros((4, 1, 12), np.float32))
    pns = {}
    exec(tp.PREAMBLE_FUNCS, pns)
    pu, pr = pns["load_prior"](inp, keep={"p0", "p1", "p2", "c0", "u9"})
    check(pu == ["p0", "p1"] and pr.shape == (4, 2, 12) and pns["done_uids"](inp) == {"p0", "p1"},
          "load_prior: complete rows only, competition tree skipped")
    cells_ = src.split("\n# %%")
    ns = dict(np=np, pd=pd, os=os, time=time, json=json, shutil=shutil, T0=time.time() - 60, ids=["u0", "u1", "u2"],
              SHARD=0, N_SHARDS=1, LIMIT=0, LABELS=tp.LABELS, RUN={"raptor_k_eval": 94}, CHUNK=Path(f"{d}/chunk"),
              _TEACHER_DONE=True, _TEACHER_PRIOR_UIDS=pu, _TEACHER_PRIOR_RAW=pr)   # DONE: the thread exits at once
    exec(compile(cells_[4].replace("/kaggle/working", work), "flusher", "exec"), ns)
    outs = [np.full((3, 12), np.nan, np.float32) for _ in range(4)]
    for o in outs:
        o[:2] = 0.2                                                                       # u0, u1 done; u2 failed
    ns["_KE_TEACHER_OUTPUTS"], ns["_KE_TEACHER_IDS"] = outs, ["u0", "u1", "u2"]
    part = f"{work}/raptor_teacher_partial.npz"
    n = ns["_teacher_snapshot"](part); z = _npz(part)
    check(n == 4 and z["study_uids"].tolist() == ["p0", "p1", "u0", "u1", "u2"] and z["raw_probabilities"].shape == (4, 5, 12),
          "partial flush: prior complete rows first, then this run's")
    np.savez_compressed(f"{work}/raptor_raw.npz", study_uids=np.asarray(["u0", "u1", "u2"]), raw_probabilities=np.stack(outs))
    for o in outs:
        o[2] = 0.5                                                                        # their neutral fill, in place
    check(ns["_teacher_snapshot"](part) == -1 and np.isnan(_npz(part)["raw_probabilities"][:, 4]).all(),
          "no flush once raptor_raw.npz exists: neutral-filled rows are never published")
    ns["_KE_NS"] = {"ARMS": [{"name": nm, "w": w} for nm, w in (("maxspan-v5", .6), ("native384dense-v10", .1),
                                                               ("maxspan-v5-reverse", .1), ("native384-v8", .2))]}
    ns["_RSNA_AUDIT"] = {"events": [{"kind": "raptor_checkpoint", "sha256": s} for s in "abc"]}
    exec(compile(cells_[6].replace("/kaggle/working", work), "final", "exec"), ns)
    z = _npz(f"{work}/raptor_teacher_shard0.npz"); csv = pd.read_csv(f"{work}/raptor_teacher_shard0.csv", dtype={"StudyInstanceUID": str})
    with open(f"{work}/teacher_receipt.json") as f:
        rc = json.load(f)
    check(z["study_uids"].tolist() == ["p0", "p1", "u0", "u1", "u2"] and z["raw_probabilities"].shape == (4, 5, 12)
          and np.isnan(z["raw_probabilities"][:, 4]).all() and np.allclose(z["raw_probabilities"][:, :2], 0.7)
          and csv.StudyInstanceUID.tolist() == ["p0", "p1", "u0", "u1"] and len(z["checkpoint_sha256"]) == 3,
          "final shard npz / csv: prior rows + this run's, the failed study stays NaN and out of the csv")
    check(rc["studies"] == 2 and rc["prior_studies"] == 2 and rc["total_studies"] == 4 and rc["failed_uids"] == ["u2"],
          "receipt: this run's studies counted apart from prior_studies")
    ns["_TEACHER_PRIOR_UIDS"] = ["u0"]; ns["_TEACHER_PRIOR_RAW"] = np.zeros((4, 1, 12), np.float32)
    try:
        ns["_teacher_combine"](["u0"], np.zeros((4, 1, 12), np.float32)); check(False, "a UID twice is refused")
    except RuntimeError:
        check(True, "a UID twice across prior and this run is refused")
print("\n" + ("TEACHER PASS CHECKS PASSED" if not fails else f"TEACHER PASS CHECKS FAILED ({len(fails)})")); sys.exit(1 if fails else 0)
