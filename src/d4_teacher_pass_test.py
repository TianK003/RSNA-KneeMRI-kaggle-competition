# src/d4_teacher_pass_test.py
"""Checks for src/build_d4_teacher_pass.py (P-45) and the D4 side of src/merge_teacher.py: the extracted D4 code is the
0.942 notebook's text but for the listed token patches (each applied exactly once), every generated cell compiles, the
chunker is gold-free / disjoint / complete, gold mode is the 58 sorted gold studies, the resume reads 1-view files of the
same grid only, and the driver + final writer -- run against a fake `_asset_run_d4` -- flush partials, drop a failing
study, bisect an unattributed failure, honour the time guard, pad a single study, run both grids in gold mode and score
them against a reference. CPU, seconds."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time  # noqa: E401
import numpy as np, pandas as pd
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, "src")
import build_teacher_pass as tp  # noqa: E402
import build_d4_teacher_pass as d4  # noqa: E402
fails = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m); fails.append(m) if not c else None
def refused(fn, m):
    try:
        fn(); check(False, m)
    except SystemExit:
        check(True, m)

nb = tp.load_notebook(tp.NOTEBOOK)
src = d4.render_teacher_py(nb, shard=0, n_shards=2, limit=0)
gsrc = d4.render_teacher_py(nb, gold=True)
cells = src.split("\n# %%")
check(len(cells) == 6, f"six cells rendered ({len(cells)})")

# --- verbatim + patches
blocks, table = d4.extract(nb)
for label, old, new, _ in d4.PATCHES:
    check(src.count(new) == 1 and old not in src, f"patch {label}: applied exactly once, anchor gone")
    refused(lambda: tp.patch_once(blocks["c12"], old, new, label), f"patch {label}: a second application fails loudly")
c12 = blocks["c12"]
for label, old, new, _ in d4.PATCHES:
    c12 = c12.replace(new, old)
check(c12 == "\n".join(tp.cell_lines(nb, 12)).rstrip("\n"), "cell 12 = the notebook's cell 12 once the 3 patches are undone")
check(blocks["c14"] == "\n".join(tp.cell_lines(nb, 14)).rstrip("\n"), "cell 14 = the notebook's cell 14, whole")
l45 = tp.cell_lines(nb, 45); pos = 0; ok45 = True
for ln in (x for x in blocks["c45"].split("\n") if x.strip()):
    hit = next((i for i in range(pos, len(l45)) if l45[i].strip() == ln.strip() and l45[i].endswith(ln)), None)
    ok45 = ok45 and hit is not None; pos = (hit or pos) + 1
check(ok45, "cell-45 block: every line is a line of _coat_substitute, re-indented, in order")
refused(lambda: tp.patch_once("abc", "x", "y", "absent"), "patch_once: an absent anchor fails loudly")
refused(lambda: tp.patch_once("xx", "x", "y", "twice"), "patch_once: a doubled anchor fails loudly")
check("def _asset_run_d4(" in src and "def _d4_check_outputs(" in src and "def _dense_stack(" in src
      and "envd = _P('/kaggle/working/_coat_env')" in src and "d4_env = _P('/kaggle/working/_d4_env')" in src
      and "raise RuntimeError('D4 timm wheel changed')" in src, "D4 wrapper, dense grid, both pinned wheels present")
check(not any(b in src for b in ("_ke_run_raptor_arms", "_ke_primary", "_KE_LAB", "rsna_phase(", "raptor_raw.npz",
                                 "_pipeline_stage.csv", "MAN_SHA")), "no Raptor / transformer-stack / resgated code")

# --- the generated notebook: every code cell compiles; metadata
with tempfile.TemporaryDirectory() as d:
    d4.write_kernel(d, 0, 1, 0, gold=True)
    ipy = json.load(open(os.path.join(d, "rsna-knee-teacher-d4.ipynb"), encoding="utf-8"))
    codes = ["".join(c["source"]) for c in ipy["cells"] if c["cell_type"] == "code"]
    good = 0
    for i, c in enumerate(codes):
        try:
            compile(c, f"ipynb cell {i}", "exec"); good += 1
        except SyntaxError as e:
            print("   ", i, e)
    check(len(codes) == 6 and good == 6, f"generated .ipynb: {good}/{len(codes)} code cells compile")
    check("GOLD = True" in codes[0] and 'D4_GRID = "notebook"' in codes[0] and d4.GOLD_CALL in codes[0], "gold flags baked in")
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    check(meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and not meta["enable_internet"]
          and meta["dataset_sources"] == d4.DATASETS and meta["competition_sources"] == [tp.COMPETITION]
          and meta["id"] == "tiankljucanin/rsna-knee-teacher-d4" and meta["title"] == "RSNA Knee Teacher D4",
          "metadata: T4x2, GPU, no internet, the D4 Dataset + the opencv wheel, competition")
with tempfile.TemporaryDirectory() as d:
    d4.write_kernel(d, 1, 2, 0, slug="tiankljucanin/rsna-knee-teacher-d4-c", kernel_sources=["tiankljucanin/rsna-knee-teacher-d4"],
                    grid="original", sub_chunk=300)
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    first = open(os.path.join(d, "rsna-knee-teacher-d4-c.py"), encoding="utf-8").read().split("\n# %%")[0]
    check(meta["kernel_sources"] == ["tiankljucanin/rsna-knee-teacher-d4"] and meta["code_file"] == "rsna-knee-teacher-d4-c.ipynb"
          and "SHARD = 1 " in first and "N_SHARDS = 2 " in first and 'D4_GRID = "original"' in first and "SUB_CHUNK = 300 " in first
          and "GOLD = False" in first and d4.CHUNK_CALL in first, "sibling slug: kernel_sources, shard, grid, sub-chunk baked in")
refused(lambda: d4.build_metadata("tiankljucanin/rsna-knee-teacher-d4", ["tiankljucanin/rsna-knee-teacher-d4"]),
        "a kernel mounting its own output is refused")
refused(lambda: d4.render_teacher_py(nb, 1, 2, 0, gold=True), "--gold with a shard is refused")
refused(lambda: d4.render_teacher_py(nb, grid="dense"), "an unknown --grid is refused")
refused(lambda: d4.render_teacher_py(nb, sub_chunk=3), "--sub-chunk < 4 is refused")
check(d4.GOLD_CALL in gsrc and d4.CHUNK_CALL not in gsrc and d4.CHUNK_CALL in src and d4.GOLD_CALL not in src,
      "gold render calls gold_study_ids, shard render the chunker")

# --- chunking on the real train.csv (the Raptor chunker, verbatim)
ns = {}
exec(d4.PREAMBLE_FUNCS, ns)
check(d4.SHARED_HEAD in tp.PREAMBLE_FUNCS and d4.SHARED_TAIL in tp.PREAMBLE_FUNCS, "chunker / root finder: the Raptor text, verbatim")
a, b = ns["chunk_study_ids"]("data/train.csv", 0, 2, 0), ns["chunk_study_ids"]("data/train.csv", 1, 2, 0)
tr = pd.read_csv("data/train.csv", dtype={"StudyInstanceUID": str}); gold = set(tr.loc[tr[tp.LABELS].notna().all(axis=1), "StudyInstanceUID"])
check(len(a) + len(b) == 4349 and not set(a) & set(b) and not (set(a) | set(b)) & gold, "2 shards: disjoint, 4,349 in total, gold-free")
g = ns["gold_study_ids"]("data/train.csv")
check(len(g) == 58 and g == sorted(g) and set(g) == gold, "gold mode: the 58 gold studies, sorted")
check(ns["chunk_study_ids"]("data/train.csv", 0, 1, 6) == sorted(set(a) | set(b))[:6], "LIMIT: the first N of the sorted shard")
sub = ns["d4_subchunks"](a, 400)
check([u for s in sub for u in s] == a and max(map(len, sub)) <= 400 and min(map(len, sub)) >= 2 and len(sub) == 6,
      f"sub-chunks: consecutive, <= 400, >= 2 ({[len(s) for s in sub]})")
check(all(min(map(len, ns["d4_subchunks"](list(range(n)), 4))) >= 2 for n in range(2, 60)), "sub-chunks of size 4: never one study")
try:
    ns["d4_subchunks"](a, 3); check(False, "sub-chunk size < 4 refused")
except ValueError:
    check(True, "sub-chunk size < 4 refused")
log = ("[0:3] preparation failed for 1.2.3; fallback is forbidden\n"
       "[3:6] packed inference failed for studies [1, 2]; fallback is forbidden\n")
check(ns["d4_failed_uids"](log, ["1.2.0", "1.2.1", "1.2.2", "1.2.3", "u4", "u5"]).keys() == {"1.2.3", "u4", "u5"}
      and ns["d4_failed_uids"]("RuntimeError: CUDA", ["a", "b"]) == {}, "failed-study attribution: both log forms; none when unnamed")
ps = ns["d4_progress_seconds"]("[0:30] studies 10/30 | elapsed 50.0s\n[0:30] studies 20/30 | elapsed 70.0s\n"
                               "[30:60] studies 10/30 | elapsed 40.0s\n[30:60] studies 30/30 | elapsed 100.0s\n")
check(len(ps) == 30 and np.allclose(sorted(set(ps.round(6))), [2.0, 3.0]), "per-study seconds from their progress lines")

# --- resume: 1-view files of the same grid only
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    d = d.replace("\\", "/"); inp = f"{d}/input"
    for s in ("notebooks/me/run1", "notebooks/me/run2", "competitions/rsna-knee-abnormality-detection"):
        os.makedirs(f"{inp}/{s}")
    pr = np.full((1, 3, 12), 0.7, np.float32); pr[0, 2, 0] = np.nan
    ns["d4_save"](f"{inp}/notebooks/me/run1/d4_teacher_partial.npz", ["p0", "p1", "p2"], pr, "d4-swa3-notebook", ["s"], "train")
    ns["d4_save"](f"{inp}/notebooks/me/run2/d4_teacher_shard0.npz", ["p1", "q0"], np.full((1, 2, 12), .1, np.float32), "d4-swa3-notebook", ["s"], "train")
    ns["d4_save"](f"{inp}/notebooks/me/run2/d4_teacher_shard1.npz", ["o0"], np.full((1, 1, 12), .1, np.float32), "d4-swa3-original", ["s"], "train")
    ns["d4_save"](f"{inp}/notebooks/me/run2/d4_teacher_shard7.npz", ["g0"], np.full((1, 1, 12), .1, np.float32), "d4-swa3-notebook", ["s"], "gold")
    np.savez_compressed(f"{inp}/notebooks/me/run2/d4_teacher_shard9.npz", study_uids=np.asarray(["r0"]), raw_probabilities=np.zeros((4, 1, 12), np.float32))
    np.savez_compressed(f"{inp}/notebooks/me/run2/raptor_teacher_shard0.npz", study_uids=np.asarray(["r1"]), raw_probabilities=np.zeros((4, 1, 12), np.float32))
    ns["d4_save"](f"{inp}/competitions/rsna-knee-abnormality-detection/d4_teacher_shard0.npz", ["c0"], np.zeros((1, 1, 12), np.float32), "d4-swa3-notebook", ["s"], "train")
    pu, praw = ns["load_prior"](inp, keep={"p0", "p1", "p2", "q0", "o0", "g0", "r0", "r1", "c0"}, view="d4-swa3-notebook")
    got = dict(zip(pu, praw[0, :, 0].astype(float).round(3).tolist()))
    check(pu == ["p1", "q0", "p0"] and praw.shape == (1, 3, 12) and got == {"p1": .1, "q0": .1, "p0": .7},
          "load_prior: complete rows of this grid (shard files before partials, first wins); other grid, gold pass, "
          "4-view, raptor-named, competition tree skipped")
    check(ns["load_prior"](inp, keep={"q0"}, view="d4-swa3-notebook")[0] == ["q0"]
          and ns["done_uids"](inp, view="d4-swa3-original") == {"o0"}, "load_prior: keep restricts; the original grid sees its own rows")
    ref = Path(f"{inp}/notebooks/me/run1/x.bin"); ref.write_bytes(b"ref")
    check(ns["find_input_file"]("x.bin", hashlib.sha256(b"ref").hexdigest(), inp) == ref, "find_input_file: glob + sha256")
    try:
        ns["find_input_file"]("x.bin", "0" * 64, inp); check(False, "find_input_file: wrong sha refused")
    except FileNotFoundError:
        check(True, "find_input_file: wrong sha refused")

# --- the real preamble cell against a /kaggle/input-shaped temp tree (gold and a shard with a mounted prior)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    d = d.replace("\\", "/"); comp = f"{d}/input/competitions/rsna-knee-abnormality-detection"
    os.makedirs(f"{comp}/train_series"); os.makedirs(f"{d}/input/notebooks/me/run1")
    shutil.copy("data/train.csv", comp); shutil.copy("data/train_series.csv", comp)
    for gold_, kw in ((True, {}), (False, dict(shard=1, n_shards=2))):
        pre = d4.render_teacher_py(nb, gold=gold_, **kw).split("\n# %%")[0]
        pre = pre.replace("/kaggle/input", f"{d}/input").replace("/kaggle/working", f"{d}/working").replace("/tmp/", f"{d}/tmp/")
        if not gold_:
            ns["d4_save"](f"{d}/input/notebooks/me/run1/d4_teacher_partial.npz", b[:3], np.full((1, 3, 12), .5, np.float32),
                          "d4-swa3-notebook", ["s"], "train")
        pg = {}
        exec(compile(pre, "preamble", "exec"), pg)
        if gold_:
            check(pg["ids"] == g and pg["COMP_IN"] == comp and pg["TRAIN_TREE"] == "train_series" and not pg["_TEACHER_PRIOR_UIDS"]
                  and set(pg["SERIES"].StudyInstanceUID) >= set(g), "preamble (gold): root found, 58 gold ids, no prior, SERIES read")
        else:
            check(pg["ALL_IDS"] == b and pg["_TEACHER_PRIOR_UIDS"] == b[:3] and pg["ids"] == b[3:],
                  "preamble (shard 1/2): the mounted partial's rows are prior, the rest is left to run")

# --- the driver + final writer against a fake _asset_run_d4
def value(u, suffix):
    v = np.random.default_rng(int(hashlib.md5(u.encode()).hexdigest()[:8], 16)).uniform(.05, .9, 12)
    return v + (0.01 if suffix == "" else 0.0)                     # the notebook grid moves every probability by 0.01
def run_cells(d, uids, *, prior=((), None), gold=False, grid="notebook", sub_chunk=4, bad=(), flaky=0, budget_s=1e9,
              truth=None):
    d = d.replace("\\", "/"); work, inp = f"{d}/working", f"{d}/input"
    os.makedirs(work, exist_ok=True); os.makedirs(f"{inp}/datasets/m/d4", exist_ok=True)
    pred = "coatnet_d4_depthzone_swa3_predictions.npz"
    refp = f"{inp}/datasets/m/d4/d4_gold58_reference.npz"
    if gold:
        np.savez_compressed(refp, study_uids=np.asarray(uids), truth=truth,
                            probability_mean=np.stack([value(u, "orig") for u in uids]).astype(np.float32))
    man = Path(f"{inp}/datasets/m/d4/coatnet_pairfilm_manifest.json")
    man.write_text(json.dumps({"predictions_name": pred, "checkpoint": {"sha256": "P" * 64}, "slice_depth": {"adapter_sha256": "A" * 64},
                               "files": {"d4_gold58_reference.npz": hashlib.sha256(open(refp, "rb").read()).hexdigest() if gold else ""}}))
    calls, state = [], {"flaky": flaky}
    def fake_run(artifact, environment_dir, competition, output, grid_suffix=""):
        run = pd.read_csv(Path(competition) / "test.csv", dtype=str).StudyInstanceUID.tolist()
        calls.append((grid_suffix, run)); output = Path(output); lg = output.with_suffix(".log")
        hit = [u for u in run if u in bad]
        if hit:
            lg.write_text(f"[0:{len(run)}] preparation failed for {hit[0]}; fallback is forbidden\n")
            raise RuntimeError("Required model branch failed (1)")
        if state["flaky"]:
            state["flaky"] -= 1; lg.write_text("RuntimeError: CUDA error: unspecified launch failure\n")
            raise RuntimeError("Required model branch failed (1)")
        np.savez_compressed(output.parent / pred, study_uids=np.asarray(run),
                            raw_probabilities=np.stack([value(u, grid_suffix) for u in run])[None].astype(np.float32))
        lg.write_text("[0:9] studies 2/9 | elapsed 9.0s\n[0:9] studies 4/9 | elapsed 13.0s\n")
        return {"shards": [{"elapsed_seconds": 20.0, "model_load_seconds": 3.0, "peak_reserved_bytes": 2 * 2 ** 30,
                            "preparation_warnings": 1}]}
    g = {}
    exec(d4.PREAMBLE_FUNCS, g)
    pu, praw = prior
    g.update(np=np, pd=pd, os=os, re=__import__("re"), json=json, time=time, shutil=shutil, subprocess=subprocess, Path=Path,
             T0=time.time(), TIME_BUDGET=budget_s, INPUT_ROOT=inp, GOLD=gold, D4_GRID=grid, D4_VIEW=f"d4-swa3-{grid}",
             SUB_CHUNK=sub_chunk, SHARD=0, N_SHARDS=1, LIMIT=0, ALL_IDS=list(uids), ids=[u for u in uids if u not in set(pu)],
             _TEACHER_PRIOR_UIDS=list(pu), _TEACHER_PRIOR_RAW=praw if praw is not None else np.zeros((1, 0, 12), np.float32),
             CHUNK_ROOT=Path(f"{d}/chunk"), WORK=Path(f"{work}/d4_sub"), IMAGE_TREE=None,
             SERIES=pd.DataFrame({"StudyInstanceUID": list(uids), "SeriesInstanceUID": [f"s{i}" for i in range(len(uids))]}),
             d4_manifest=man, d4_env=Path(f"{work}/_d4_env"), _asset_run_d4=fake_run)
    cs = d4.render_teacher_py(nb, gold=gold, grid=grid).split("\n# %%")
    for i in (4, 5):
        exec(compile(cs[i].replace("/kaggle/working", work), f"cell{i}", "exec"), g)
    out = {"g": g, "calls": calls, "work": work}
    for f in os.listdir(work):
        if f.endswith(".npz"):
            with np.load(f"{work}/{f}", allow_pickle=False) as z:
                out[f] = {k: z[k] for k in z.files}
    out["receipt"] = json.load(open(f"{work}/d4_teacher_receipt.json"))
    return out

ids10 = [f"u{i:02d}" for i in range(10)]
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    prior = (["p0", "p1"], np.full((1, 2, 12), .3, np.float32))
    r = run_cells(d, ["p0", "p1"] + ids10, prior=prior, bad={"u05"})
    z, rc = r["d4_teacher_shard0.npz"], r["receipt"]
    csv = pd.read_csv(f"{r['work']}/d4_teacher_shard0.csv", dtype={"StudyInstanceUID": str})
    check([s for s, _ in r["calls"]] == [""] * 4 and r["calls"][1][1] == ["u03", "u04", "u05", "u06"] and r["calls"][2][1] == ["u03", "u04", "u06"],
          f"train: 3 sub-chunks; the one holding a failing study re-runs without it ({[len(c) for _, c in r['calls']]})")
    check(z["study_uids"].tolist() == ["p0", "p1"] + ids10 and z["raw_probabilities"].shape == (1, 12, 12)
          and np.isnan(z["raw_probabilities"][0, 7]).all() and np.allclose(z["raw_probabilities"][0, :2], .3)
          and np.allclose(z["raw_probabilities"][0, 2], value("u00", "")),
          "shard npz: prior rows first, then this run's raw probabilities; the failed study stays NaN")
    check(z["view_names"].tolist() == ["d4-swa3-notebook"] and z["view_weights"].tolist() == [1.0] and str(z["pass_mode"]) == "train"
          and z["checkpoint_sha256"].tolist() == ["P" * 64, "A" * 64], "shard npz: one view, weight 1.0, pass_mode train, both checkpoint hashes")
    check(len(csv) == 11 and "u05" not in set(csv.StudyInstanceUID) and list(csv.columns) == ["StudyInstanceUID", *tp.LABELS],
          "shard csv: the 11 complete rows, schema")
    p = r["d4_teacher_partial.npz"]
    check(p["study_uids"].tolist() == ["p0", "p1"] + ids10 and str(p["pass_mode"]) == "train", "partial flush after the last sub-chunk: prior + attempted")
    check(rc["studies"] == 9 and rc["prior_studies"] == 2 and rc["total_studies"] == 11 and list(rc["failed_uids"]) == ["u05"]
          and rc["not_attempted"] == 0 and rc["stopped"] is None and rc["timing"]["runs"] == 3
          and rc["timing"]["preparation_warnings"] == 3 and rc["timing"]["worker_s_per_study"]["n"] == 6,
          "receipt: studies vs prior, failed uid, timing (runs, warnings, per-study seconds)")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    r = run_cells(d, ids10, flaky=1, sub_chunk=10)
    check([len(c) for _, c in r["calls"]] == [10, 5, 5] and r["receipt"]["total_studies"] == 10,
          "an unattributed failure is bisected and both halves complete")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    r = run_cells(d, ids10, flaky=99, sub_chunk=5)
    check(r["receipt"]["total_studies"] == 0 and "produced no study" in (r["receipt"]["stopped"] or "")
          and [len(c) for _, c in r["calls"]] == [5, 2, 2, 2, 3, 2, 2],   # 5 -> 2 -> 1,1 -> 3 -> 1,2 (singles padded); 7 = 1 + budget 6
          f"a sub-chunk that never succeeds spends its retry budget, then the pass stops ({len(r['calls'])} runs)")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    r = run_cells(d, ["p0"] + ids10, prior=(["p0"], np.full((1, 1, 12), .3, np.float32)), budget_s=30)
    check(not r["calls"] and "time guard" in r["receipt"]["stopped"] and r["receipt"]["not_attempted"] == 10
          and r["d4_teacher_shard0.npz"]["study_uids"].tolist() == ["p0"], "time guard: no child started, outputs still written (prior rows)")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    r = run_cells(d, ["v0", "v1"], prior=(["v1"], np.full((1, 1, 12), .3, np.float32)))
    check(r["calls"][0][1] == ["v0", "v1"] and r["d4_teacher_shard0.npz"]["study_uids"].tolist() == ["v1", "v0"],
          "a single study runs with a companion; only its own row is kept")
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    gids = [f"g{i:02d}" for i in range(8)]
    truth = (np.random.default_rng(1).uniform(size=(8, 12)) > .5).astype(np.uint8); truth[0], truth[1] = 1, 0
    r = run_cells(d, gids, gold=True, truth=truth)
    gr, orig = r["receipt"]["gold_report"], r["g"]["D4_ORIGINAL_GRID"]
    check([s for s, _ in r["calls"]] == ["", "", orig, orig] and "def _dense_stack" in orig,
          "gold: both grids run (notebook first, then the original-grid suffix)")
    check(gr["original"]["strict_pass"] and gr["original"]["max_abs"] < 1e-6 and not gr["notebook"]["strict_pass"]
          and abs(gr["notebook"]["max_abs"] - 0.01) < 1e-5 and gr["notebook"]["compared"] == 8,
          "gold report: PASS when equal to the reference, STRICT FAIL at 0.01 off")
    check(str(r["d4_teacher_shard0.npz"]["pass_mode"]) == "gold" and "d4_teacher_gold_original.npz" in r
          and r["d4_teacher_gold_original.npz"]["view_names"].tolist() == ["d4-swa3-original"]
          and os.path.isfile(f"{r['work']}/d4_teacher_gold_original.csv"), "gold outputs: pass_mode gold, the second grid in its own files")

# --- merge_teacher: 1-view D4 shards
import merge_teacher as mt  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    ns["d4_save"](f"{d}/d4_teacher_shard0.npz", ["a0", "a1", "a2"], np.full((1, 3, 12), .4, np.float32), "d4-swa3-notebook", ["s"], "train")
    raw1 = np.full((1, 2, 12), .6, np.float32); raw1[0, 1, 3] = np.nan
    ns["d4_save"](f"{d}/d4_teacher_shard1.npz", ["b0", "b1"], raw1, "d4-swa3-notebook", ["s"], "train")
    out = mt.merge_teacher([f"{d}/d4_teacher_shard0.npz", f"{d}/d4_teacher_shard1.npz"], f"{d}/t.csv", expect_n=4, prefix="d4_teacher_")
    check(len(out) == 4 and list(out.columns) == ["StudyInstanceUID", *tp.LABELS] and np.allclose(out[tp.LABELS].to_numpy()[:3], .4),
          "merge (--teacher d4): 1-view shards, weight 1.0, failed study dropped")
    ns["d4_save"](f"{d}/d4_teacher_shard2.npz", ["c0"], np.full((1, 1, 12), .4, np.float32), "d4-swa3-original", ["s"], "train")
    refused(lambda: mt.merge_teacher([f"{d}/d4_teacher_shard0.npz", f"{d}/d4_teacher_shard2.npz"], f"{d}/t2.csv", expect_n=4, prefix="d4_teacher_"),
            "merge: two input grids (view names) refused")
    ns["d4_save"](f"{d}/d4_teacher_shard3.npz", ["g0"], np.full((1, 1, 12), .4, np.float32), "d4-swa3-notebook", ["s"], "gold")
    refused(lambda: mt.merge_teacher([f"{d}/d4_teacher_shard3.npz"], f"{d}/t3.csv", allow_partial=True), "merge: a gold pass refused (any --teacher)")
    np.savez_compressed(f"{d}/raptor_teacher_shard0.npz", study_uids=np.asarray(["r0"]), raw_probabilities=np.zeros((4, 1, 12), np.float32),
                        view_names=np.asarray(list("abcd")), view_weights=np.asarray([.25] * 4))
    refused(lambda: mt.merge_teacher([f"{d}/d4_teacher_shard0.npz", f"{d}/raptor_teacher_shard0.npz"], f"{d}/t4.csv", allow_partial=True,
                                     prefix="d4_teacher_"), "merge --teacher d4: a raptor file refused")
    check(mt.TEACHERS["raptor"] == {"out": "artifacts/teacher/raptor_teacher.csv", "prefix": None}, "merge: the Raptor default is unchanged")
print("\n" + ("D4 TEACHER PASS CHECKS PASSED" if not fails else f"D4 TEACHER PASS CHECKS FAILED ({len(fails)})")); sys.exit(1 if fails else 0)
