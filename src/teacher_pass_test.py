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
