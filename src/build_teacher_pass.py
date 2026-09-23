"""Build kaggle/rsna-knee-teacher/ -- the public 0.942 notebook's Raptor branch over chunks of TRAINING studies.

`notebook_score_0.942.ipynb` (committed, byte-identical to the public "DINOsaur V5" run) scores the hidden test set
with, among others, a "Raptor" branch (cell 45): three public CC0 CoAtNet-2 checkpoints
(`dreaddevelopment/raptor-knee-maxspan` v5 swa, `-native384` v8 swa, `-native384dense` v10), four views, two T4s,
raw per-view probabilities saved to `/kaggle/working/raptor_raw.npz` (4 x N x 12) before any fill or ranking.

This builder extracts that branch VERBATIM (plus the helper slices it names from cells 12, 14 and 16), prepends our
chunk preamble -- which writes a fake competition root (`test.csv`, `test_series.csv`, `sample_submission.csv`,
`test_series` and `test_images` -> the mounted `train_series/`) holding one shard of the 4,349 report-labelled training studies (the 58 gold rows are
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
    # the second concurrent slot / a resume: a SIBLING slug (a kernel cannot mount its own output, traps 31)
    .venv/Scripts/python.exe src/build_teacher_pass.py --slug tiankljucanin/rsna-knee-teacher-b \
        --shard 1 --n-shards 3 --kernel-source tiankljucanin/rsna-knee-teacher     # -> kaggle/rsna-knee-teacher-b/

Resume: the preamble loads the COMPLETE rows (finite in all four views) of every mounted teacher npz (shard outputs and
partials, competition tree never walked), restricted to this shard, skips those studies, and puts them FIRST in the
partial flush and in the final shard npz / csv -- so each resume's outputs hold every row so far, and the next sibling
needs to mount only the latest run.
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
KERNEL_ID = "tiankljucanin/rsna-knee-teacher"
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
# The functions the test execs (chunking on the real train.csv; resume on fake npz files). `load_prior` searches every mounted input EXCEPT the
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

_SKIP_TOPS = ("rsna-knee-abnormality-detection", "competitions")

def _teacher_npz_paths(input_root="/kaggle/input", skip_tops=_SKIP_TOPS):
    """Every mounted teacher npz (shard outputs and 5-minute partials) outside the competition tree."""
    from pathlib import Path
    root, paths = Path(input_root), []
    for top in (sorted(root.iterdir()) if root.is_dir() else []):
        if top.name in skip_tops or not top.is_dir():
            continue
        paths += sorted(top.glob("**/raptor_teacher_shard*.npz"))[:50] + sorted(top.glob("**/raptor_teacher_partial.npz"))[:50]
    return paths

def load_prior(input_root="/kaggle/input", skip_tops=_SKIP_TOPS, keep=None):
    """Resume: the COMPLETE rows (finite in all four views) of every mounted teacher npz -- earlier, guard-stopped
    runs of this shard, mounted through kernel_sources from a sibling slug (a kernel cannot mount its own output).
    Deduplicated by UID (first file in sorted order wins); `keep` (a set) restricts them to this shard's UIDs.
    Returns (uids, raw) with raw shaped (4, M, 12) float32 -- the rows every writer puts first."""
    import numpy as np
    uids, rows, seen = [], [], set()
    for p in _teacher_npz_paths(input_root, skip_tops):
        try:
            with np.load(p, allow_pickle=False) as z:
                raw = np.asarray(z["raw_probabilities"], np.float32); su = [str(u) for u in z["study_uids"]]
            if raw.ndim != 3 or raw.shape[0] != 4 or raw.shape[2] != 12 or raw.shape[1] != len(su):
                raise ValueError(f"raw_probabilities {raw.shape} for {len(su)} study_uids")
            fin = np.isfinite(raw).all(axis=(0, 2))
        except Exception as e:
            print("  ! unreadable", p, e)
            continue
        n0 = len(uids)
        for j, u in enumerate(su):
            if fin[j] and u not in seen and (keep is None or u in keep):
                seen.add(u); uids.append(u); rows.append(raw[:, j, :])
        print(f"  prior {p}: {len(uids) - n0} new complete rows", flush=True)
    return uids, (np.stack(rows, axis=1) if rows else np.zeros((4, 0, 12), np.float32))

def done_uids(input_root="/kaggle/input", skip_tops=_SKIP_TOPS):
    """UIDs already complete in any mounted teacher npz."""
    return set(load_prior(input_root, skip_tops)[0])

_IMAGE_TREES = ("train_series", "train_images")          # the mounted tree is train_series/; train_images accepted
_TREE_NAMES = ("train_series", "test_series", "train_images", "test_images")   # never walked into

def find_competition_root(candidates, input_root="/kaggle/input", max_depth=3):
    """(root, tree): the first candidate -- else the first dir within `max_depth` of `input_root`, breadth-first,
    image trees never entered -- that holds train.csv and a training image tree (hard constraint 4: no bare
    hard-coded /kaggle/input path)."""
    from pathlib import Path
    def tree_of(d):
        d = Path(d)
        if not (d / "train.csv").is_file():
            return None
        return next((t for t in _IMAGE_TREES if (d / t).is_dir()), None)
    for c in candidates:
        t = tree_of(c)
        if t:
            return str(c), t
    level = [Path(input_root)] if Path(input_root).is_dir() else []
    for _ in range(max_depth + 1):
        following = []
        for d in level:
            t = tree_of(d)
            if t:
                return str(d), t
            try:
                following += sorted(p for p in d.iterdir() if p.is_dir() and p.name not in _TREE_NAMES)
            except OSError:
                pass
        level = following
    raise FileNotFoundError(f"no competition root (train.csv + one of {_IMAGE_TREES}) in {list(candidates)} "
                            f"nor under {input_root} to depth {max_depth}")
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

__PREAMBLE_FUNCS__

# The competition mounts at /kaggle/input/competitions/rsna-knee-abnormality-detection/ with its DICOMs under
# train_series/ and test_series/ (no train_images/); the shallow glob fallback covers any other layout.
COMP_IN, TRAIN_TREE = find_competition_root(["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                                             "/kaggle/input/rsna-knee-abnormality-detection"])
print(f"competition root {COMP_IN}, training image tree {TRAIN_TREE}/", flush=True)
# The chunk lives OUTSIDE /kaggle/working: its test_series / test_images symlinks into the training image tree must
# never sit in the output directory, even when a guard-stopped run never reaches the final cell's cleanup.
CHUNK = Path("/tmp/rsna_teacher_chunk"); CHUNK.mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/diagnostics").mkdir(parents=True, exist_ok=True)   # their per-study input audit appends here
Path("/kaggle/working/raptor_raw.npz").unlink(missing_ok=True)   # the flush guard keys on this file (cell 5)
ids = chunk_study_ids(f"{COMP_IN}/train.csv", SHARD, N_SHARDS, LIMIT)
# Resume: complete rows of earlier runs of this shard (mounted from a sibling slug) are carried into every output.
_TEACHER_PRIOR_UIDS, _TEACHER_PRIOR_RAW = load_prior(keep=set(ids))
skip = set(_TEACHER_PRIOR_UIDS)
ids = [u for u in ids if u not in skip]
print(f"chunk: shard {SHARD}/{N_SHARDS}, limit {LIMIT} -> {len(ids)} studies ({len(skip)} already done); root {COMP_IN}", flush=True)
if not ids:
    raise RuntimeError("nothing left to run in this shard (every study is already in a mounted teacher npz)")
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(CHUNK / "test.csv", index=False)
ser = pd.read_csv(f"{COMP_IN}/train_series.csv", dtype=str)
ser[ser.StudyInstanceUID.isin(ids)].to_csv(CHUNK / "test_series.csv", index=False)
sub = pd.DataFrame({"StudyInstanceUID": ids, **{l: 0.5 for l in LABELS}}); sub.to_csv(CHUNK / "sample_submission.csv", index=False)
for _link in ("test_series", "test_images"):     # their reader takes root/test_series if it is a dir, else root/test_images
    if not os.path.lexists(CHUNK / _link):
        os.symlink(f"{COMP_IN}/{TRAIN_TREE}", CHUNK / _link)
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
# Their runner saves raptor_raw.npz and THEN neutral-fills failed rows of these same arrays in place (patch (a) makes
# them _KE_TEACHER_OUTPUTS): once that file exists a snapshot could publish filled rows as complete, so none is taken.
_TEACHER_RAW_NPZ = "/kaggle/working/raptor_raw.npz"
def _teacher_combine(run_uids, run_raw):
    """Prior complete rows first, then this run's rows: (uids, raw (4, N, 12) float32); no UID twice."""
    uids = list(_TEACHER_PRIOR_UIDS) + [str(u) for u in run_uids]
    if len(set(uids)) != len(uids):
        raise RuntimeError("teacher: a UID appears twice across the prior rows and this run")
    raw = np.concatenate([np.asarray(_TEACHER_PRIOR_RAW, np.float32), np.asarray(run_raw, np.float32)], axis=1)
    if raw.shape != (4, len(uids), 12):
        raise RuntimeError(f"teacher: raw_probabilities {raw.shape} for {len(uids)} uids")
    return uids, raw
def _teacher_snapshot(path):
    """Write prior + this run's rows (NaN where not yet predicted); -1 = skipped (raptor_raw.npz exists), else rows complete."""
    outs = globals().get("_KE_TEACHER_OUTPUTS"); ids = globals().get("_KE_TEACHER_IDS")
    if not outs or not ids:
        return 0
    raw = np.stack(outs)                                  # copy FIRST, then check: absent file => fill not begun at copy
    if os.path.exists(_TEACHER_RAW_NPZ):
        return -1
    uids, raw = _teacher_combine(ids, raw)
    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, study_uids=np.asarray(uids, dtype=str), raw_probabilities=raw)
    os.replace(tmp, path)                                 # a reader never sees a half-written partial
    return int(np.isfinite(raw).all(axis=(0, 2)).sum())
def _teacher_flusher():
    while not globals().get("_TEACHER_DONE"):
        time.sleep(300)
        if globals().get("_TEACHER_DONE"):
            break
        try:
            n = _teacher_snapshot("/kaggle/working/raptor_teacher_partial.npz")
            if n >= 0:
                print(f"[teacher] partial flush: {n} studies complete, {(time.time() - T0) / 3600:.2f} h", flush=True)
        except Exception as e:
            print("[teacher] flush failed:", e, flush=True)
threading.Thread(target=_teacher_flusher, daemon=True).start()
'''

# ------------------------------------------------------------------------------------------------ cell 7 (ours)
FINAL_CELL = '''# %%
globals()["_TEACHER_DONE"] = True
with np.load(_TEACHER_RAW_NPZ, allow_pickle=False) as raw:   # saved by their runner BEFORE its neutral fill: NaN = failed
    run_probs = raw["raw_probabilities"]; run_uids = [str(u) for u in raw["study_uids"]]
run_ok = np.isfinite(run_probs).all(axis=(0, 2))
uids, probs = _teacher_combine(run_uids, run_probs)   # prior complete rows first
names = [a["name"] for a in _KE_NS["ARMS"]]; weights = np.asarray([float(a["w"]) for a in _KE_NS["ARMS"]]); weights /= weights.sum()
ok = np.isfinite(probs).all(axis=(0, 2))
mean = np.tensordot(weights, np.clip(np.nan_to_num(probs, nan=0.5), 0, 1), axes=(0, 0))
np.savez_compressed(f"/kaggle/working/raptor_teacher_shard{SHARD}.npz", study_uids=np.asarray(uids, dtype=str),
                    raw_probabilities=probs, view_names=np.asarray(names, dtype=str), view_weights=weights,
                    checkpoint_sha256=np.asarray([e.get("sha256", "") for e in _RSNA_AUDIT["events"] if e.get("kind") == "raptor_checkpoint"], dtype=str))
csv = pd.DataFrame(mean[ok], columns=LABELS); csv.insert(0, "StudyInstanceUID", np.asarray(uids)[ok]); csv.to_csv(f"/kaggle/working/raptor_teacher_shard{SHARD}.csv", index=False)
elapsed = time.time() - T0
n_run, n_prior = int(run_ok.sum()), len(_TEACHER_PRIOR_UIDS)
json.dump({"shard": SHARD, "n_shards": N_SHARDS, "limit": LIMIT, "studies": n_run, "prior_studies": n_prior,
           "total_studies": int(ok.sum()), "failed_uids": [u for u, f in zip(run_uids, run_ok) if not f],
           "elapsed_s": elapsed, "sec_per_study": elapsed / max(1, n_run), "k_eval": int(RUN["raptor_k_eval"])},
          open("/kaggle/working/teacher_receipt.json", "w"), indent=1)
print(f"[teacher] {n_run} studies this run ({elapsed / max(1, n_run):.1f} s/study, {len(run_uids) - n_run} failed) + {n_prior} prior = {int(ok.sum())} complete")
for p in (str(CHUNK),):
    shutil.rmtree(p, ignore_errors=True)         # the chunk (and its symlinks into the image tree) is never an output
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


def slug_name(slug):
    """'owner/name' -> 'name' (the kernel dir, notebook and .py basename)."""
    owner, _, name = str(slug).partition("/")
    if not owner or not name or "/" in name or not all(c.isalnum() or c == "-" for c in name):
        raise SystemExit(f"--slug expects owner/kernel-name, got {slug!r}")
    return name


def slug_title(name):
    """The title Kaggle slugifies back to `name` ('rsna-knee-teacher-b' -> 'RSNA Knee Teacher B')."""
    return " ".join("RSNA" if p == "rsna" else p.capitalize() for p in name.split("-"))


def build_metadata(slug=KERNEL_ID, kernel_sources=()):
    """A resume runs in a SIBLING slug that lists the previous run in kernel_sources (traps 31: a kernel cannot mount
    its own output); two concurrent shards need two slugs (the cache2-a..d pattern)."""
    name = slug_name(slug)
    sources = [str(s) for s in kernel_sources]
    if slug in sources:
        raise SystemExit(f"--kernel-source {slug}: a kernel cannot mount its own output (traps 31); use a sibling --slug")
    if len(set(sources)) != len(sources):
        raise SystemExit(f"duplicate --kernel-source in {sources}")
    for s in sources:
        slug_name(s)
    return {
        "id": slug,
        "title": slug_title(name),
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": list(DATASETS),
        "kernel_sources": sources,
        "competition_sources": [COMPETITION],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }


def _render_files(shard, n_shards, limit, slug=KERNEL_ID, kernel_sources=(), notebook=NOTEBOOK):
    """{filename: text} for the kernel directory, built in a temp dir through nbgen (so --check is exact)."""
    name = slug_name(slug)
    meta = build_metadata(slug, kernel_sources)
    src = render_teacher_py(load_notebook(notebook), shard, n_shards, limit)
    with tempfile.TemporaryDirectory() as d:
        py, ipynb = os.path.join(d, f"{name}.py"), os.path.join(d, f"{name}.ipynb")
        with open(py, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        nbgen.build(py, ipynb)
        with open(ipynb, encoding="utf-8") as f:
            nb_text = f.read()
    return {f"{name}.py": src, f"{name}.ipynb": nb_text, "kernel-metadata.json": json.dumps(meta, indent=2) + "\n"}


def write_kernel(out_dir, shard, n_shards, limit, slug=KERNEL_ID, kernel_sources=()):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in _render_files(shard, n_shards, limit, slug, kernel_sources).items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="> 0: the first N studies of the shard (6 smoke, 100 spike)")
    ap.add_argument("--slug", default=KERNEL_ID, help="kernel id owner/name; names the dir, notebook and title")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="a previous run's kernel slug to mount for a resume (repeatable; never --slug itself)")
    ap.add_argument("--out", default=None, help="kernel dir (default kaggle/<slug name>)")
    ap.add_argument("--check", action="store_true", help="render in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    out = a.out or os.path.join("kaggle", slug_name(a.slug))
    files = _render_files(a.shard, a.n_shards, a.limit, a.slug, a.kernel_source)
    _, table = extract(load_notebook(NOTEBOOK))
    summary = (f"{out}: {a.slug} shard {a.shard}/{a.n_shards} limit {a.limit} kernel_sources {a.kernel_source}; "
               + "; ".join(f"{lab} c{c} {s}..{e}" for lab, c, s, e, _, _ in table))
    if a.check:
        ok = True
        for name, text in files.items():
            path = os.path.join(out, name)
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
    write_kernel(out, a.shard, a.n_shards, a.limit, a.slug, a.kernel_source)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
