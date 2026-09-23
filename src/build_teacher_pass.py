"""Build kaggle/rsna-knee-teacher/ -- the public 0.942 notebook's Raptor branch over chunks of TRAINING studies.

`notebook_score_0.942.ipynb` (committed, byte-identical to the public "DINOsaur V5" run) scores the hidden test set
with, among others, a "Raptor" branch (cell 45): three public CC0 CoAtNet-2 checkpoints
(`dreaddevelopment/raptor-knee-maxspan` v5 swa, `-native384` v8 swa, `-native384dense` v10), four views, two T4s,
raw per-view probabilities saved to `/kaggle/working/raptor_raw.npz` (4 x N x 12) before any fill or ranking.

This builder extracts that branch VERBATIM (plus the helper slices it names from cells 12, 14 and 16), prepends our
chunk preamble -- which writes a fake competition root (`test.csv`, `test_series.csv`, `sample_submission.csv`,
`test_images -> train_images`) holding one shard of the 4,349 report-labelled training studies (the 58 gold rows are
validation and never teach) and points `RSNA_COMP_ROOT` at it -- and appends a 5-minute partial-flush thread and a
final writer. The Raptor code then believes the chunk is the test set. Nothing else of the 0.942 graph (DINO x20,
A5, RadImageNet, calibrator, CoAt family, FineSpacing) is included.

The generated percent-format `.py` has seven cells:
    1 preamble (ours)   2 cell-16 helpers   3 cell-12 asset finder   4 cell-14 dense sampler
    5 flush thread (ours)   6 cell-45 Raptor branch (2 token patches)   7 final writer (ours)

Kernel outputs: raptor_teacher_shard{SHARD}.npz (study_uids, raw_probabilities 4xNx12, view_names, view_weights,
checkpoint_sha256), raptor_teacher_shard{SHARD}.csv (UID + 12 view-weighted means), teacher_receipt.json,
raptor_teacher_partial.npz (every 5 min), raptor_raw.npz (theirs).

Deterministic: the same notebook gives byte-identical files (`--check`).

    export PYTHONUTF8=1
    .venv/Scripts/python.exe src/build_teacher_pass.py --limit 6              # smoke: 6 studies of shard 0/1
    .venv/Scripts/python.exe src/build_teacher_pass.py --limit 100            # the timing spike
    .venv/Scripts/python.exe src/build_teacher_pass.py --shard 1 --n-shards 3 # one production chunk
    .venv/Scripts/python.exe src/build_teacher_pass.py --limit 6 --check      # rebuild in memory, diff vs disk
"""
import argparse
import difflib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import nbgen  # noqa: E402

NOTEBOOK = "notebook_score_0.942.ipynb"
OUT_DIR = os.path.join("kaggle", "rsna-knee-teacher")
KERNEL_ID = "tiankljucanin/rsna-knee-teacher"
KERNEL_TITLE = "RSNA Knee Teacher"
CODE_FILE = "rsna-knee-teacher.ipynb"
PY_FILE = "rsna-knee-teacher.py"
COMPETITION = "rsna-knee-abnormality-detection"
DATASETS = ["dreaddevelopment/raptor-knee-maxspan", "dreaddevelopment/raptor-knee-native384",
            "dreaddevelopment/raptor-knee-native384dense"]
N_CELLS = 53
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_notebook(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_lines(nb, i):
    src = nb["cells"][i]["source"]
    return (src if isinstance(src, str) else "".join(src)).split("\n")


def _pred(p):
    return p if callable(p) else (lambda line: line == p)


def slice_lines(lines, start_pred, end_pred, label):
    """lines[start..end] inclusive: `start` is the UNIQUE line matching start_pred, `end` the first line at or after
    it matching end_pred. Returns (start, end, slice). Predicates are callables or exact-line strings."""
    sp, ep = _pred(start_pred), _pred(end_pred)
    starts = [i for i, ln in enumerate(lines) if sp(ln)]
    if len(starts) != 1:
        raise SystemExit(f"{label}: start line matched {len(starts)} times (expected 1): {starts[:5]}")
    s = starts[0]
    ends = [i for i in range(s, len(lines)) if ep(lines[i])]
    if not ends:
        raise SystemExit(f"{label}: no end line after line {s}")
    e = ends[0]
    return s, e, lines[s:e + 1]


def patch_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"patch {label}: {old!r} matched {n} times (expected 1)")
    return text.replace(old, new)


# ------------------------------------------------------------------------------------------------ cell 1 (ours)
# The two functions the test execs on the real train.csv. `done_uids` searches every mounted input EXCEPT the
# competition tree (a `**` glob over /kaggle/input would walk ~819k training DICOMs before finding nothing).
PREAMBLE_FUNCS = '''LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def chunk_study_ids(train_csv, shard, n_shards, limit):
    """Report-labelled studies only (the 58 gold rows are validation and never teach), sorted by UID,
    shard `shard` of `n_shards` by position, then the first `limit` (> 0)."""
    import pandas as pd
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    is_gold = tr[LABELS].notna().all(axis=1)
    ids = sorted(tr.loc[~is_gold, "StudyInstanceUID"].tolist())
    assert len(ids) == 4349, len(ids)
    assert 0 <= shard < n_shards, (shard, n_shards)
    ids = ids[shard::n_shards]
    return ids[:limit] if limit > 0 else ids

def done_uids(input_root="/kaggle/input", skip_tops=("rsna-knee-abnormality-detection", "competitions")):
    """Resume: UIDs already complete (all four views finite) in any mounted teacher npz -- a previous,
    guard-stopped run of this shard mounted through kernel_sources. The competition tree is never walked."""
    import numpy as np
    from pathlib import Path
    root, done, paths = Path(input_root), set(), []
    for top in (sorted(root.iterdir()) if root.is_dir() else []):
        if top.name in skip_tops or not top.is_dir():
            continue
        paths += sorted(top.glob("**/raptor_teacher_shard*.npz"))[:50] + sorted(top.glob("**/raptor_teacher_partial.npz"))[:50]
    for p in paths:
        try:
            z = np.load(p, allow_pickle=False)
            fin = np.isfinite(z["raw_probabilities"]).all(axis=(0, 2))
            done |= set(map(str, z["study_uids"][fin]))
        except Exception as e:
            print("  ! unreadable", p, e)
    return done
'''

PREAMBLE_TEMPLATE = '''# %%
SHARD = __SHARD__                 # sed'd at build: shard index
N_SHARDS = __N_SHARDS__              # sed'd at build: number of shards over the 4,349 report-labelled studies
LIMIT = __LIMIT__                 # sed'd at build: > 0 = first N studies of the shard (6 = smoke, 100 = the timing spike)
# Teacher pass (src/build_teacher_pass.py): the 0.942 notebook's Raptor branch, verbatim, over a chunk of TRAINING
# studies presented to it as a competition root through RSNA_COMP_ROOT.
import os, json, time, shutil
import numpy as np, pandas as pd
import torch                      # rsna_phase (cell-16 slice) reads the global `torch`
from pathlib import Path
T0 = time.time()
TIME_BUDGET = 8.0 * 3600          # rsna_deadline raises past this (the 0.942 run's own budget)
_COMP_CANDIDATES = ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]
_COMP_FOUND = [p for p in _COMP_CANDIDATES if Path(p, "train.csv").is_file() and Path(p, "train_images").is_dir()]
if not _COMP_FOUND:
    raise FileNotFoundError(f"competition root with train.csv + train_images not found in {_COMP_CANDIDATES}")
COMP_IN = _COMP_FOUND[0]
# The chunk lives OUTSIDE /kaggle/working: its test_images symlink into train_images must never sit in the output
# directory, even when a guard-stopped run never reaches the final cell's cleanup.
CHUNK = Path("/tmp/rsna_teacher_chunk"); CHUNK.mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/diagnostics").mkdir(parents=True, exist_ok=True)   # their per-study input audit appends here

__PREAMBLE_FUNCS__
ids = chunk_study_ids(f"{COMP_IN}/train.csv", SHARD, N_SHARDS, LIMIT)
skip = done_uids() & set(ids)
ids = [u for u in ids if u not in skip]
print(f"chunk: shard {SHARD}/{N_SHARDS}, limit {LIMIT} -> {len(ids)} studies ({len(skip)} already done); root {COMP_IN}", flush=True)
if not ids:
    raise RuntimeError("nothing left to run in this shard (every study is already in a mounted teacher npz)")
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(CHUNK / "test.csv", index=False)
ser = pd.read_csv(f"{COMP_IN}/train_series.csv", dtype=str)
ser[ser.StudyInstanceUID.isin(ids)].to_csv(CHUNK / "test_series.csv", index=False)
sub = pd.DataFrame({"StudyInstanceUID": ids, **{l: 0.5 for l in LABELS}}); sub.to_csv(CHUNK / "sample_submission.csv", index=False)
if not (CHUNK / "test_images").exists():
    os.symlink(f"{COMP_IN}/train_images", CHUNK / "test_images")
os.environ["RSNA_COMP_ROOT"] = str(CHUNK)
RUN = {"raptor_view_w": {"maxspan-v5": 0.60, "native384dense-v10": 0.10, "maxspan-v5-reverse": 0.10, "native384-v8": 0.20},
       "raptor_k_eval": 94}
coat_w = None                     # named in their _KE_NS, never called by the Raptor branch
_RSNA_TEST_IDS = list(ids)
'''

# ------------------------------------------------------------------------------------------------ cell 5 (ours)
FLUSHER_CELL = '''# %%
_RSNA_TEST_IDS = list(ids)        # the cell-16 slice reset it to None
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
'''

# ------------------------------------------------------------------------------------------------ cell 7 (ours)
FINAL_CELL = '''# %%
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
for p in (str(CHUNK),):
    shutil.rmtree(p, ignore_errors=True)         # the chunk (and its symlink into train_images) is never an output
'''


# ------------------------------------------------------------------------------------------------ extraction
def extract(nb):
    """The four verbatim blocks, each as text, plus the slice table (cell, start, end, first, last) for provenance."""
    if len(nb["cells"]) != N_CELLS:
        raise SystemExit(f"{NOTEBOOK} has {len(nb['cells'])} cells, expected {N_CELLS}")
    table = []

    def take(cell, start, end, label):
        lines = cell_lines(nb, cell)
        s, e, out = slice_lines(lines, start, end, label)
        table.append((label, cell, s, e, out[0], out[-1]))
        return out

    # cell 16: the runtime-integrity helpers through rsna_deadline, then rsna_phase (cell 45 calls it twice).
    c16a = take(16, lambda l: l.startswith("# Runtime integrity: no partial ensemble"),
                lambda l: l.startswith("        raise TimeoutError(f'{tag}: full-run budget exceeded"), "cell16 helpers")
    c16b = take(16, lambda l: l.startswith("def rsna_phase("), "    return current", "cell16 rsna_phase")
    for ln in c16a + c16b:
        if ln.startswith(("TIME_BUDGET = ", "T0 = ")):
            raise SystemExit("cell-16 slice now carries TIME_BUDGET / T0; the preamble owns them")
    c16 = "\n".join(c16a + [""] + c16b)
    for need in ("def rsna_strict_load(", "def rsna_deadline(", "def rsna_finite(", "def rsna_json("):
        if need not in c16:
            raise SystemExit(f"cell-16 slice lacks {need}")

    # cell 12: the asset walker/finder, then the memoising redefinitions that were in force when cell 45 ran.
    c12a = take(12, lambda l: l.startswith("ASSET_ROOTS = ["), "    return hits[0]", "cell12 asset finder")
    c12b = take(12, "_speed_original_asset_walk = _asset_walk", "        return _speed_asset_hits[key]",
                "cell12 memoised finder")
    c12 = "\n".join(c12a + [""] + c12b)

    # cell 14: the whole capacity-aware dense sampler.
    lines14 = cell_lines(nb, 14)
    table.append(("cell14 dense sampler", 14, 0, len(lines14) - 1, lines14[0], lines14[-1]))
    c14 = "\n".join(lines14).rstrip("\n")
    if "def _dense_allocate(" not in c14 or "def _dense_unique_linspace(" not in c14:
        raise SystemExit("cell 14 is not the dense sampler")

    # cell 45: imports, then _KE_SRC .. the post-run cache clear; the transformer-stack read before it
    # (_ke_primary / _ke_ours / _KE_LAB) and the raptor_input_before_coat / summary tail after it are dropped.
    c45a = take(45, "import gc as _ke_gc", "from pathlib import Path as _KePath", "cell45 imports")
    c45b = take(45, "_KE_SRC = r'''", "_ke_gc.collect()", "cell45 raptor")
    lines45 = cell_lines(nb, 45)
    after = table[-1][3] + 1
    if not lines45[after].startswith("_KePath('/kaggle/working/raptor_input_before_coat.csv')"):
        raise SystemExit(f"cell 45 line {after} is not the raptor_input_before_coat copy: {lines45[after][:80]!r}")
    c45 = "\n".join(c45a + [""] + c45b)
    c45 = patch_once(c45, "\n    outputs = [\n", '\n    outputs = globals().setdefault("_KE_TEACHER_OUTPUTS", [\n',
                     "(a) outputs open")
    c45 = patch_once(c45, "\n        for _ in arms\n    ]\n", "\n        for _ in arms\n    ])\n", "(a) outputs close")
    c45 = patch_once(c45, "\n    test_ids = test['StudyInstanceUID'].tolist()\n",
                     "\n    test_ids = test['StudyInstanceUID'].tolist()\n    globals()[\"_KE_TEACHER_IDS\"] = list(test_ids)\n",
                     "(b) test_ids")
    for bad in ("_ke_primary", "_ke_ours", "_KE_LAB", "_pipeline_stage.csv", "raptor_input_before_coat",
                "_DINOV2_MATCHED_MEMBERS"):
        if bad in c45:
            raise SystemExit(f"cell-45 slice still references {bad}")
    return {"c16": c16, "c12": c12, "c14": c14, "c45": c45}, table


def render_teacher_py(nb, shard, n_shards, limit):
    if not (0 <= int(shard) < int(n_shards)) or int(limit) < 0:
        raise SystemExit(f"bad shard/n_shards/limit: {shard}/{n_shards}/{limit}")
    blocks, _ = extract(nb)
    pre = (PREAMBLE_TEMPLATE.replace("__SHARD__", str(int(shard))).replace("__N_SHARDS__", str(int(n_shards)))
           .replace("__LIMIT__", str(int(limit))).replace("__PREAMBLE_FUNCS__", PREAMBLE_FUNCS.rstrip("\n")))
    cells = [
        pre.rstrip("\n"),
        "# %%\n# --- notebook_score_0.942.ipynb cell 16 (verbatim slices): runtime-integrity helpers + rsna_phase\n" + blocks["c16"],
        "# %%\n# --- cell 12 (verbatim slices): asset walker / finder, then its memoised redefinition\n" + blocks["c12"],
        "# %%\n# --- cell 14 (verbatim, whole cell): capacity-aware dense sampler\n" + blocks["c14"],
        FLUSHER_CELL.rstrip("\n"),
        "# %%\n# --- cell 45 (verbatim slices, 2 token patches): the Raptor branch\n" + blocks["c45"],
        FINAL_CELL.rstrip("\n"),
    ]
    src = "\n\n".join(cells) + "\n"
    for i, c in enumerate(src.split("\n# %%")):
        compile(c, f"teacher cell {i}", "exec")
    return src


def build_metadata():
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": list(DATASETS),
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }


def _render_files(shard, n_shards, limit, notebook=NOTEBOOK):
    """{filename: text} for the kernel directory, built in a temp dir through nbgen (so --check is exact)."""
    src = render_teacher_py(load_notebook(notebook), shard, n_shards, limit)
    with tempfile.TemporaryDirectory() as d:
        py, ipynb = os.path.join(d, PY_FILE), os.path.join(d, CODE_FILE)
        with open(py, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        nbgen.build(py, ipynb)
        with open(ipynb, encoding="utf-8") as f:
            nb_text = f.read()
    return {PY_FILE: src, CODE_FILE: nb_text, "kernel-metadata.json": json.dumps(build_metadata(), indent=2) + "\n"}


def write_kernel(out_dir, shard, n_shards, limit):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in _render_files(shard, n_shards, limit).items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="> 0: the first N studies of the shard (6 smoke, 100 spike)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="render in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    files = _render_files(a.shard, a.n_shards, a.limit)
    _, table = extract(load_notebook(NOTEBOOK))
    summary = (f"{a.out}: shard {a.shard}/{a.n_shards} limit {a.limit}; "
               + "; ".join(f"{lab} c{c} {s}..{e}" for lab, c, s, e, _, _ in table))
    if a.check:
        ok = True
        for name, text in files.items():
            path = os.path.join(a.out, name)
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                ok = False
                continue
            with open(path, encoding="utf-8") as f:
                disk = f.read()
            same = disk == text
            print(f"  {'same    ' if same else 'DIFFERS '}{path}")
            if not same:
                sys.stdout.writelines(list(difflib.unified_diff(disk.splitlines(True), text.splitlines(True),
                                                                path, "rendered", n=1))[:40])
            ok = ok and same
        print(("check: " + ("ok" if ok else "FAILED")) + " -- " + summary)
        return 0 if ok else 1
    write_kernel(a.out, a.shard, a.n_shards, a.limit)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
