"""Build kaggle/rsna-knee-teacher-d4/ -- the public 0.942 notebook's D4 CoAtNet child over chunks of TRAINING studies (P-45).

`notebook_score_0.942.ipynb` scores the hidden test set with, inside its CoAt family block (cell 45's `_coat_substitute`),
the public CC0 "D4" model (`mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`: CoAtNet-rmlp-2 @384 Global96 parent
`coatnet_d4_parent_rank8_swa2.pt` + the depth-zone adapter `coatnet_d4_depthzone_swa3_adapter.pt`, run by the Dataset's own
runtime as two GPU-owner processes, one per T4). Cell 12 holds the wrapper: `_asset_run_d4` writes a worker script that
checks the runtime (`_d4_check_runtime`: manifest/adapter hashes, timm 1.0.22, cv2 4.12.0, exactly two T4s), patches D4's
input preparation with cell 14's capacity-aware dense Global96 grid (`_asset_patch_d4`), runs `run_submission` and checks
its outputs (`_d4_check_outputs`). `_coat_substitute` installs the two pinned offline wheels (opencv-python-headless
4.12.0.88 into /kaggle/working/_coat_env, timm 1.0.22 into /kaggle/working/_d4_env) and calls `_asset_run_d4`.

This builder takes cell 12 and cell 14 WHOLE and the wheel / D4-artifact lines of `_coat_substitute` (re-indented), all
verbatim but for the three token patches in PATCHES, and wraps them in our cells. Nothing else of the 0.942 graph (DINO x20,
A5, RadImageNet, calibrator, Raptor, resgated CoAt, FineSpacing) is included. Six cells:
    1 preamble (ours)    2 cell 12 (3 token patches)    3 cell 14 (whole)    4 cell-45 slices: wheels + D4 artifact root
    5 sub-chunk driver (ours)    6 final writer (ours)

The driver presents each sub-chunk of <= SUB_CHUNK studies to `_asset_run_d4` as a competition root (test.csv /
test_series.csv / sample_submission.csv + test_series and test_images symlinked to the mounted train_series/), reads D4's
RAW sigmoid probabilities (1, n, 12) from the runtime's own `coatnet_d4_depthzone_swa3_predictions.npz` (written before
any ranking), and flushes `d4_teacher_partial.npz` after every sub-chunk. A failing child is retried without the study
its log names (their "preparation failed for <uid>" / "packed inference failed for studies [...]"), else bisected; a time
guard stops the loop before a sub-chunk that would not finish, and the final cell still writes every output.

Input grid (`--grid`): `notebook` (default) = the 0.942 graph's dense grid, exactly as it ran there; `original` = D4's own
Global96 grid (the injected `_dense_stack` hands the reference stack back unchanged) -- the grid D4's gold-58 reference
(`d4_gold58_reference.npz`, macro AUC 0.9302) was computed on. `--gold` runs the 58 gold studies under BOTH grids (primary
grid -> d4_teacher_shard0.*, the other -> d4_teacher_gold_<grid>.*) and scores each against that reference.

Kernel outputs: d4_teacher_shard{SHARD}.npz (study_uids, raw_probabilities 1xNx12 -- prior rows first, NaN = failed,
view_names ["d4-swa3-<grid>"], view_weights [1.0], checkpoint_sha256, pass_mode), d4_teacher_shard{SHARD}.csv (UID + 12),
d4_teacher_partial.npz (after every sub-chunk), d4_teacher_receipt.json, d4_sub/<run>/ (each child's own receipts + log).

    export PYTHONUTF8=1 PYTHONPATH=src
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --gold                     # the 58-study spike (both grids)
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --limit 6                  # smoke: 6 studies of shard 0/1
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --shard 0 --n-shards 2     # one production chunk
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --slug tiankljucanin/rsna-knee-teacher-d4-b --shard 1 --n-shards 2
    # a resume: a SIBLING slug mounting the guard-stopped run (a kernel cannot mount its own output, traps 31)
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --slug tiankljucanin/rsna-knee-teacher-d4-c --shard 0 --n-shards 2 \\
        --kernel-source tiankljucanin/rsna-knee-teacher-d4
    .venv/Scripts/python.exe src/build_d4_teacher_pass.py --gold --check            # rebuild in memory, diff vs disk

The Raptor builder (src/build_teacher_pass.py) is imported for its notebook / slicing / slug helpers and its chunker and
root finder, and is never modified: its renders stay byte-identical.
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
import build_teacher_pass as tp  # noqa: E402

NOTEBOOK = tp.NOTEBOOK
KERNEL_ID = "tiankljucanin/rsna-knee-teacher-d4"
COMPETITION = tp.COMPETITION
# The D4 Dataset (runtime .py files, manifest, parent + adapter checkpoints, the timm wheel, the gold-58 reference) and the
# pinned opencv wheel the 0.942 graph installs for every CoAt child (`_d4_check_runtime` demands cv2 4.12.0).
DATASETS = ["mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2", "mattiaangeli/opencv-python-headless-4120088-x86"]
LABELS = tp.LABELS
GRIDS = ("notebook", "original")
SUB_CHUNK = 400

# (label, old, new, why) -- applied to cell 12 with tp.patch_once (each anchor must match exactly once).
PATCHES = [
    ("(1) prep warnings not fatal",
     "        need(int(s['fallback_studies']) == 0 and int(s['preparation_warnings']) == 0, 'failed study/preparation')",
     "        need(int(s['fallback_studies']) == 0, 'failed study')  # teacher pass: preparation warnings are recorded, not fatal",
     "their release gate kills the whole child when any study logs a DICOM preparation warning (an unreadable file, a "
     "missing primary series); over 4,349 training studies that would discard whole sub-chunks. The count stays in the "
     "receipt. (`fallback_studies` is hard-coded 0 by their runtime -- a real failure raises -- so that check is kept.)"),
    ("(2) grid switch: signature",
     "def _asset_run_d4(artifact, environment_dir, competition, output):",
     "def _asset_run_d4(artifact, environment_dir, competition, output, grid_suffix=''):",
     "--grid original: a suffix appended to the grid source the worker execs (default '' = their notebook grid)"),
    ("(3) grid switch: source",   # anchored on the line end, so the new text does not contain the anchor
     "(inspect.getsource(f) for f in (_dense_quotas, _dense_stack)))\n",
     "(inspect.getsource(f) for f in (_dense_quotas, _dense_stack))) + grid_suffix\n",
     "the suffix redefines _dense_stack as the identity, so D4 sees its own Global96 grid; '' leaves their wrapper "
     "byte-identical"),
]

# ------------------------------------------------------------------------------------------------ cell 1 (ours)
_PF = tp.PREAMBLE_FUNCS


def _between(text, start, stop, label):
    """text[start-marker : stop-marker], both markers unique."""
    for m in (start, stop):
        if text.count(m) != 1:
            raise SystemExit(f"{label}: marker {m!r} found {text.count(m)} times in the Raptor preamble functions")
    return text[text.index(start):text.index(stop)]


# The Raptor preamble's chunker (chunk_study_ids, gold_study_ids, _SKIP_TOPS) and root finder, reused verbatim -- so a D4
# shard k of n is the same UID set as the Raptor shard k of n. Its resume functions are 4-view / raptor-named: replaced.
SHARED_HEAD = _between(_PF, "LABELS = [", "def _teacher_npz_paths(", "shared head")
SHARED_TAIL = _PF[_PF.index("_IMAGE_TREES = "):]
if _PF.count("_IMAGE_TREES = ") != 1 or "def find_competition_root(" not in SHARED_TAIL:
    raise SystemExit("the Raptor preamble functions changed: root finder not found")

D4_RESUME_FUNCS = '''def _teacher_npz_paths(input_root="/kaggle/input", skip_tops=_SKIP_TOPS):
    """Every mounted D4 teacher npz (shard outputs and per-sub-chunk partials) outside the competition tree."""
    from pathlib import Path
    root, paths = Path(input_root), []
    for top in (sorted(root.iterdir()) if root.is_dir() else []):
        if top.name in skip_tops or not top.is_dir():
            continue
        paths += sorted(top.glob("**/d4_teacher_shard*.npz"))[:50] + sorted(top.glob("**/d4_teacher_partial.npz"))[:50]
    return paths

def load_prior(input_root="/kaggle/input", skip_tops=_SKIP_TOPS, keep=None, view=None):
    """Resume: the COMPLETE rows (finite) of every mounted D4 teacher npz of the SAME input grid (`view`, e.g.
    "d4-swa3-notebook": a file of another grid, a gold pass or a 4-view Raptor file is skipped, never mixed in) -- earlier,
    guard-stopped runs of this shard, mounted through kernel_sources from a sibling slug (a kernel cannot mount its own
    output). Deduplicated by UID (first file in sorted order wins); `keep` (a set) restricts them to this shard's UIDs.
    Returns (uids, raw) with raw shaped (1, M, 12) float32 -- the rows every writer puts first."""
    import numpy as np
    uids, rows, seen = [], [], set()
    for p in _teacher_npz_paths(input_root, skip_tops):
        try:
            with np.load(p, allow_pickle=False) as z:
                raw = np.asarray(z["raw_probabilities"], np.float32); su = [str(u) for u in z["study_uids"]]
                names = [str(v) for v in z["view_names"]] if "view_names" in z.files else None
                mode = str(z["pass_mode"]) if "pass_mode" in z.files else None
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 12 or raw.shape[1] != len(su):
                raise ValueError(f"raw_probabilities {raw.shape} for {len(su)} study_uids")
            if view is not None and names != [view]:
                raise ValueError(f"view_names {names}, this run is {view!r}")
            if mode != "train":
                raise ValueError(f"pass_mode {mode!r} (only a training pass resumes)")
            fin = np.isfinite(raw).all(axis=(0, 2))
        except Exception as e:
            print("  ! skipped", p, e)
            continue
        n0 = len(uids)
        for j, u in enumerate(su):
            if fin[j] and u not in seen and (keep is None or u in keep):
                seen.add(u); uids.append(u); rows.append(raw[:, j, :])
        print(f"  prior {p}: {len(uids) - n0} new complete rows", flush=True)
    return uids, (np.stack(rows, axis=1) if rows else np.zeros((1, 0, 12), np.float32))

def done_uids(input_root="/kaggle/input", skip_tops=_SKIP_TOPS, view=None):
    """UIDs already complete in any mounted D4 teacher npz of this grid."""
    return set(load_prior(input_root, skip_tops, view=view)[0])

'''

# Our helpers for the driver / writer cells: pure functions, exec'd by the tests.
D4_FUNCS = r'''def write_chunk_root(root, uids, series, image_tree):
    """A fake competition root holding `uids` as the test set: test.csv, test_series.csv (their rows of train_series.csv,
    strings verbatim), sample_submission.csv, and test_series / test_images -> the training image tree (None: no links)."""
    import os
    from pathlib import Path
    import pandas as pd
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    uids = [str(u) for u in uids]
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(root / "test.csv", index=False)
    series[series.StudyInstanceUID.isin(set(uids))].to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": uids, **{l: 0.5 for l in LABELS}}).to_csv(root / "sample_submission.csv", index=False)
    for link in ("test_series", "test_images"):     # their reader takes root/test_series
        if image_tree and not os.path.lexists(root / link):
            os.symlink(str(image_tree), root / link)
    return root

def d4_subchunks(uids, size):
    """Consecutive near-equal sub-chunks of at most `size` (>= 4) studies -- so none holds a single study."""
    import numpy as np
    uids, size = list(uids), int(size)
    if size < 4:
        raise ValueError(f"sub-chunk size {size} < 4")
    if not uids:
        return []
    k = -(-len(uids) // size)
    b = np.linspace(0, len(uids), k + 1).round().astype(int)
    return [uids[i:j] for i, j in zip(b[:-1], b[1:])]

def d4_failed_uids(log_text, uids):
    """{uid: reason} for the studies a failed D4 child names in its log: their "preparation failed for <uid>; fallback is
    forbidden" and "[start:stop] packed inference failed for studies [i, ...]" (indices local to the worker range)."""
    import re
    uids = [str(u) for u in uids]; known = set(uids); bad = {}
    for u in re.findall(r"preparation failed for ([^\s;]+); fallback is forbidden", log_text):
        if u in known:
            bad[u] = "D4 preparation failed (see the child log)"
    for a, b, idx in re.findall(r"\[(\d+):(\d+)\] packed inference failed for studies \[([0-9, ]*)\]", log_text):
        for i in [int(x) for x in idx.replace(" ", "").split(",") if x]:
            j = int(a) + i
            if j < min(int(b), len(uids)):
                bad[uids[j]] = "D4 packed inference failed (see the child log)"
    return bad

def d4_progress_seconds(log_text):
    """Per-study seconds of one GPU worker, from their "[start:stop] studies k/n | elapsed Xs" lines (printed every 10
    studies per worker); the first interval (model load + warm-up) is left out."""
    import re
    import numpy as np
    per = {}
    for a, b, k, n, s in re.findall(r"\[(\d+):(\d+)\] studies (\d+)/(\d+) \| elapsed ([0-9.]+)s", log_text):
        per.setdefault((a, b), set()).add((int(k), float(s)))
    out = []
    for pts in per.values():
        pts = sorted(pts)
        for (k0, s0), (k1, s1) in zip(pts[:-1], pts[1:]):
            if k1 > k0:
                out += [(s1 - s0) / (k1 - k0)] * (k1 - k0)
    return np.asarray(out, float)

def d4_rows(attempted, done):
    """(1, len(attempted), 12) float32 in `attempted` order: the done rows, NaN for the rest (failed)."""
    import numpy as np
    raw = np.full((1, len(attempted), 12), np.nan, np.float32)
    for i, u in enumerate(attempted):
        if u in done:
            raw[0, i] = done[u]
    return raw

def d4_combine(prior_uids, prior_raw, run_uids, run_raw):
    """Prior complete rows first, then this run's rows: (uids, raw (1, N, 12) float32); no UID twice."""
    import numpy as np
    uids = [str(u) for u in prior_uids] + [str(u) for u in run_uids]
    if len(set(uids)) != len(uids):
        raise RuntimeError("teacher: a UID appears twice across the prior rows and this run")
    raw = np.concatenate([np.asarray(prior_raw, np.float32).reshape(1, -1, 12),
                          np.asarray(run_raw, np.float32).reshape(1, -1, 12)], axis=1)
    if raw.shape != (1, len(uids), 12):
        raise RuntimeError(f"teacher: raw_probabilities {raw.shape} for {len(uids)} uids")
    return uids, raw

def d4_save(path, uids, raw, view, sha256, mode):
    """One D4 teacher npz (merge_teacher's input), written atomically (a reader never sees a half-written file)."""
    import os
    import numpy as np
    path = str(path); tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, study_uids=np.asarray([str(u) for u in uids], dtype=str),
                        raw_probabilities=np.asarray(raw, np.float32), view_names=np.asarray([view], dtype=str),
                        view_weights=np.asarray([1.0]), checkpoint_sha256=np.asarray(list(sha256), dtype=str),
                        pass_mode=np.asarray(mode))
    os.replace(tmp, path)
    return path

def d4_csv(path, uids, raw):
    """UID + the 12 probabilities of the complete rows."""
    import numpy as np
    import pandas as pd
    ok = np.isfinite(raw).all(axis=(0, 2))
    out = pd.DataFrame(np.clip(raw[0, ok], 0, 1), columns=LABELS)
    out.insert(0, "StudyInstanceUID", np.asarray([str(u) for u in uids], dtype=object)[ok])
    out.to_csv(path, index=False)
    return int(ok.sum())

def find_input_file(name, sha256=None, input_root="/kaggle/input", skip_tops=_SKIP_TOPS):
    """The ONE mounted file called `name` outside the competition tree (a glob, no hard-coded dataset path -- hard
    constraint 4), checked against `sha256` when given."""
    import hashlib
    from pathlib import Path
    root, hits = Path(input_root), []
    for top in (sorted(root.iterdir()) if root.is_dir() else []):
        if top.name not in skip_tops and top.is_dir():
            hits += sorted(top.glob(f"**/{name}"))
    hits = list({p.resolve(): p for p in hits}.values())
    if sha256:
        hits = [p for p in hits if hashlib.sha256(p.read_bytes()).hexdigest() == sha256]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one mounted {name} (sha256 {sha256}), found {hits}")
    return hits[0]

def label_aucs(truth, pred):
    """Per-label ROC AUC by average ranks (NaN where a label has one class)."""
    import numpy as np
    import pandas as pd
    out = []
    for j in range(truth.shape[1]):
        y = np.asarray(truth[:, j]) > 0
        r = pd.Series(np.asarray(pred[:, j], float)).rank(method="average").to_numpy()
        n1, n0 = int(y.sum()), int((~y).sum())
        out.append((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan"))
    return np.asarray(out, float)

# The authors' own T4-vs-H100 tolerances (repairv1/coatnet_repairv1_full_gold58_t4_cv412_audit.json in the D4 Dataset):
# Kaggle x86 + cv2 4.12 rendering is not byte-exact to their ARM training cache, so their own sibling model moved by up
# to 0.037 per probability on T4. The strict 1e-3 line is the spike's acceptance test; this one says whether a miss is
# portability noise or a wiring error.
PORTABILITY = {"max_abs": 0.03, "mean_abs": 0.003, "auc_drop": 0.003}

def d4_gold_report(uids, raw, ref_path, grid, tol=1e-3):
    """Our D4 probabilities on the gold studies vs D4's own `d4_gold58_reference.npz` (probability_mean, truth)."""
    import numpy as np
    with np.load(ref_path, allow_pickle=False) as z:
        ref_u = [str(u) for u in z["study_uids"]]
        ref = np.asarray(z["probability_mean"], np.float64); truth = np.asarray(z["truth"])
    pos = {str(u): i for i, u in enumerate(uids)}
    keep = [j for j, u in enumerate(ref_u) if u in pos and np.isfinite(raw[0, pos[u]]).all()]
    ours = np.asarray([raw[0, pos[ref_u[j]]] for j in keep], np.float64).reshape(-1, 12)
    a_ref = label_aucs(truth, ref)
    rep = {"grid": grid, "compared": len(keep), "reference_studies": len(ref_u), "ref_macro_auc": float(np.nanmean(a_ref))}
    if not keep:
        rep.update(strict_pass=False, portability_pass=False)
        print(f"[d4-gold:{grid}] no gold study predicted -- FAIL", flush=True)
        return rep
    diff = np.abs(ours - ref[keep])
    a = label_aucs(truth[keep], ours)
    rep.update(max_abs=float(diff.max()), mean_abs=float(diff.mean()), macro_auc=float(np.nanmean(a)),
               per_label_auc={l: round(float(v), 4) for l, v in zip(LABELS, a)},
               per_label_max_abs={l: round(float(v), 5) for l, v in zip(LABELS, diff.max(axis=0))})
    rep["strict_pass"] = bool(len(keep) == len(ref_u) and rep["max_abs"] <= tol)
    rep["portability_pass"] = bool(len(keep) == len(ref_u) and rep["max_abs"] <= PORTABILITY["max_abs"]
                                   and rep["mean_abs"] <= PORTABILITY["mean_abs"]
                                   and rep["ref_macro_auc"] - rep["macro_auc"] <= PORTABILITY["auc_drop"])
    print(f"[d4-gold:{grid}] {len(keep)}/{len(ref_u)} gold studies; max |ours - reference| {rep['max_abs']:.2e}, "
          f"mean {rep['mean_abs']:.2e}", flush=True)
    print(f"[d4-gold:{grid}] macro AUC vs truth: ours {rep['macro_auc']:.4f}, reference {rep['ref_macro_auc']:.4f} "
          f"(delta {rep['macro_auc'] - rep['ref_macro_auc']:+.4f})", flush=True)
    print(f"[d4-gold:{grid}] per label: " + ", ".join(f"{l} {v:.3f}" for l, v in zip(LABELS, a)), flush=True)
    print(f"[d4-gold:{grid}] " + ("PASS" if rep["strict_pass"] else "STRICT FAIL")
          + f": max abs diff {rep['max_abs']:.2e} vs tolerance {tol:.0e} (all {len(ref_u)} studies required)", flush=True)
    print(f"[d4-gold:{grid}] portability (authors' T4 tolerances max {PORTABILITY['max_abs']}, mean "
          f"{PORTABILITY['mean_abs']}, AUC drop {PORTABILITY['auc_drop']}): "
          + ("PASS" if rep["portability_pass"] else "FAIL"), flush=True)
    return rep

def d4_timing_summary(runs, setup_s, sub_chunk, n_full=4349):
    """Timing of the child runs, and the projection for a full pass in one T4x2 kernel."""
    import numpy as np
    if not runs:
        return {}
    n = sum(r["n"] for r in runs); wall = sum(r["wall_s"] for r in runs); gpu = sum(r["gpu_s"] for r in runs)
    per = np.asarray([x for r in runs for x in r.get("per_study_s", [])], float)
    over = float(np.mean([max(0.0, r["wall_s"] - r["gpu_s"]) for r in runs]))
    proj = (setup_s + n_full * gpu / n + -(-n_full // int(sub_chunk)) * over) / 3600
    out = {"runs": len(runs), "studies": n, "wall_s": wall, "wall_s_per_study": wall / n, "gpu_wall_s_per_study": gpu / n,
           "overhead_s_per_run": over, "setup_s": setup_s, "projected_full_pass_h": proj,
           "worker_s_per_study": ({"median": float(np.median(per)), "p90": float(np.percentile(per, 90)),
                                   "max": float(per.max()), "n": int(per.size)} if per.size else {}),
           "model_load_s": max(r.get("model_load_s", 0.0) for r in runs),
           "peak_reserved_gb": max(r.get("peak_reserved_gb", 0.0) for r in runs),
           "preparation_warnings": sum(r.get("warnings", 0) for r in runs)}
    print(f"[d4-timing] setup (wheels, asset hashes) {setup_s:.0f} s; {len(runs)} child run(s), {n} studies, "
          f"{wall:.0f} s wall = {wall / n:.2f} s/study (T4x2, incl. {over:.0f} s per-run overhead)", flush=True)
    if per.size:
        print(f"[d4-timing] per GPU worker: {np.median(per):.2f} s/study median, p90 {np.percentile(per, 90):.2f}, "
              f"max {per.max():.2f} (n {per.size}); model load {out['model_load_s']:.1f} s; peak reserved "
              f"{out['peak_reserved_gb']:.2f} GiB per T4", flush=True)
    print(f"[d4-timing] projection: {n_full} studies at {gpu / n:.2f} s/study + {-(-n_full // int(sub_chunk))} runs x "
          f"{over:.0f} s ~= {proj:.2f} h in one T4x2 kernel (half that per shard on two slugs; queue not included)",
          flush=True)
    return out
'''

PREAMBLE_FUNCS = (SHARED_HEAD + D4_RESUME_FUNCS + SHARED_TAIL.rstrip("\n") + "\n\n" + D4_FUNCS.rstrip("\n") + "\n")

CHUNK_CALL = 'ALL_IDS = chunk_study_ids(f"{COMP_IN}/train.csv", SHARD, N_SHARDS, LIMIT)'
GOLD_CALL = 'ALL_IDS = gold_study_ids(f"{COMP_IN}/train.csv")   # --gold: the 58 gold studies, not a shard'

PREAMBLE_TEMPLATE = '''# %%
SHARD = __SHARD__                 # sed'd at build: shard index
N_SHARDS = __N_SHARDS__              # sed'd at build: number of shards over the 4,349 report-labelled studies
LIMIT = __LIMIT__                 # sed'd at build: > 0 = first N studies of the shard (6 = smoke)
GOLD = __GOLD__                # sed'd at build: True = the 58 gold studies (the spike), under BOTH input grids
D4_GRID = "__GRID__"         # sed'd at build: "notebook" = the 0.942 graph's dense grid; "original" = D4's own Global96 grid
SUB_CHUNK = __SUB_CHUNK__               # sed'd at build: studies per D4 child run; a partial npz is flushed after each
# D4 teacher pass (src/build_d4_teacher_pass.py): the 0.942 notebook's D4 CoAtNet child, verbatim, run over sub-chunks of
# TRAINING studies, each presented to it as a competition root.
import os, re, json, time, shutil, subprocess
import numpy as np, pandas as pd
from pathlib import Path
T0 = time.time()
TIME_BUDGET = 8.0 * 3600          # _run_required_child (cell 12) kills a child past this; the driver stops before that
INPUT_ROOT = "/kaggle/input"
assert D4_GRID in ("notebook", "original"), D4_GRID
D4_VIEW = f"d4-swa3-{D4_GRID}"    # the view name records the grid: a resume and merge_teacher never mix grids

__PREAMBLE_FUNCS__

# The competition mounts at /kaggle/input/competitions/rsna-knee-abnormality-detection/ with its DICOMs under
# train_series/ and test_series/ (no train_images/); the shallow glob fallback covers any other layout.
COMP_IN, TRAIN_TREE = find_competition_root(["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                                             "/kaggle/input/rsna-knee-abnormality-detection"])
IMAGE_TREE = f"{COMP_IN}/{TRAIN_TREE}"
print(f"competition root {COMP_IN}, training image tree {TRAIN_TREE}/", flush=True)
# Chunk roots live OUTSIDE /kaggle/working: their test_series / test_images symlinks into the training image tree must
# never sit in the output directory, even when a guard-stopped run never reaches the final cell's cleanup.
CHUNK_ROOT = Path("/tmp/rsna_d4_chunk"); CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working/d4_sub"); WORK.mkdir(parents=True, exist_ok=True)
__IDS_CALL__
# Resume: complete rows of earlier runs of this shard and grid (mounted from a sibling slug) go first in every output.
_TEACHER_PRIOR_UIDS, _TEACHER_PRIOR_RAW = load_prior(INPUT_ROOT, keep=set(ALL_IDS), view=D4_VIEW)
skip = set(_TEACHER_PRIOR_UIDS)
ids = [u for u in ALL_IDS if u not in skip]
print(f"chunk: {'GOLD-58' if GOLD else f'shard {SHARD}/{N_SHARDS}, limit {LIMIT}'}, grid {D4_GRID}, sub-chunk {SUB_CHUNK} "
      f"-> {len(ids)} studies ({len(skip)} already done)", flush=True)
if not ids:
    raise RuntimeError("nothing left to run in this shard (every study is already in a mounted D4 teacher npz)")
SERIES = pd.read_csv(f"{COMP_IN}/train_series.csv", dtype=str)
'''

# ------------------------------------------------------------------------------------------------ cell 5 (ours)
DRIVER_CELL = r'''# %%
# --- ours: the sub-chunk driver. One _asset_run_d4 call (cell 12) per sub-chunk; their wrapper checks the runtime, runs D4
# on both T4s and checks the outputs. The RAW (1, n, 12) sigmoid probabilities come from the D4 runtime's own predictions
# npz (written before any ranking); a partial npz (prior rows + every finished sub-chunk) is flushed after each.
D4_SETUP_S = time.time() - T0
_D4_MANIFEST = json.loads(Path(d4_manifest).read_text())
D4_PREDICTIONS = str(_D4_MANIFEST["predictions_name"])
D4_SHA256 = [str(_D4_MANIFEST["checkpoint"]["sha256"]), str(_D4_MANIFEST["slice_depth"]["adapter_sha256"])]
# --grid original: the worker's injected _dense_stack hands D4's own Global96 stack back unchanged (input audit still runs)
D4_ORIGINAL_GRID = ("\ndef _dense_stack(reference, offsets, valid, source_depth):\n"
                    "    return (reference, [{'grid': 'original'}])\n")
D4_RETRY_BUDGET = 6               # failed child runs one sub-chunk may spend on dropping / bisecting bad studies
D4_RUNS = []                      # one record per successful child run: timing, warnings, memory
PARTIAL = "/kaggle/working/d4_teacher_partial.npz"
print(f"[d4] setup {D4_SETUP_S:.0f} s; artifact root {d4_manifest.parent}; predictions file {D4_PREDICTIONS}", flush=True)

class D4GuardStop(Exception):
    pass

def d4_estimate_s(n):
    """Wall seconds for one child over n studies: measured so far (x1.25), else a conservative 90 s + 3 s/study."""
    if not D4_RUNS:
        return 90.0 + 3.0 * n
    per = sum(r["gpu_s"] for r in D4_RUNS) / max(1, sum(r["n"] for r in D4_RUNS))
    over = max(max(0.0, r["wall_s"] - r["gpu_s"]) for r in D4_RUNS)
    return 1.25 * (over + per * n)

def d4_run_chunk(uids, tag, grid):
    """One D4 child over `uids` presented as a competition root -> raw (1, len(uids), 12) float32."""
    run_ids = [str(u) for u in uids]
    if len(run_ids) == 1:              # their dual-T4 runtime refuses < 2 studies: pad with a companion, drop its row
        run_ids.append(next(u for u in ALL_IDS if u != run_ids[0]))
    need_s = d4_estimate_s(len(run_ids))
    if time.time() - T0 + need_s > TIME_BUDGET:
        raise D4GuardStop(f"{tag}: {len(run_ids)} studies need ~{need_s:.0f} s, "
                          f"{TIME_BUDGET - (time.time() - T0):.0f} s of the budget left")
    root = write_chunk_root(CHUNK_ROOT / tag, run_ids, SERIES, IMAGE_TREE)
    out = WORK / tag / "d4.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        receipt = _asset_run_d4(d4_manifest.parent, d4_env, root, out, "" if grid == "notebook" else D4_ORIGINAL_GRID)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    wall = time.time() - t
    with np.load(out.parent / D4_PREDICTIONS, allow_pickle=False) as z:
        got = [str(u) for u in z["study_uids"]]
        raw = np.asarray(z["raw_probabilities"], np.float32)
    if got != run_ids or raw.shape != (1, len(run_ids), 12) or not np.isfinite(raw).all():
        raise RuntimeError(f"{tag}: D4 predictions {raw.shape} do not match the {len(run_ids)} chunk studies")
    shards = list(receipt.get("shards", []))
    gpu_s = max([float(s["elapsed_seconds"]) - float(s.get("model_load_seconds", 0)) for s in shards] or [wall])
    log = out.with_suffix(".log")
    D4_RUNS.append({"tag": tag, "grid": grid, "n": len(run_ids), "wall_s": wall, "gpu_s": gpu_s,
                    "warnings": int(sum(int(s.get("preparation_warnings", 0)) for s in shards)),
                    "peak_reserved_gb": max([int(s.get("peak_reserved_bytes", 0)) for s in shards] or [0]) / 2 ** 30,
                    "model_load_s": max([float(s.get("model_load_seconds", 0)) for s in shards] or [0.0]),
                    "per_study_s": d4_progress_seconds(log.read_text(errors="replace") if log.is_file() else "").tolist()})
    return raw[:, :len(uids)]

def d4_run_robust(uids, tag, grid, budget):
    """d4_run_chunk with failure handling -> (done {uid: (12,)}, failed {uid: reason}). A study the child's log names is
    dropped and the rest re-run; an unattributed failure is bisected; `budget` ([int]) caps the failed runs. Raises only
    D4GuardStop (the time budget)."""
    try:
        raw = d4_run_chunk(uids, tag, grid)
        return {u: raw[0, i] for i, u in enumerate(uids)}, {}
    except D4GuardStop:
        raise
    except (TimeoutError, subprocess.TimeoutExpired) as e:
        raise D4GuardStop(f"{tag}: {type(e).__name__}: {e}")
    except Exception as e:
        log = WORK / tag / "d4.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        reason = f"{type(e).__name__}: {(str(e).strip().splitlines() or [''])[0][:300]}"
        bad = d4_failed_uids(text, uids)
        print(f"[d4] {tag}: child failed ({reason}); studies named in its log: {sorted(bad) or 'none'}", flush=True)
        budget[0] -= 1
        if budget[0] < 0:
            return {}, {u: "retry budget exhausted; " + reason for u in uids}
        if bad:
            rest = [u for u in uids if u not in bad]
            done, failed = d4_run_robust(rest, tag + "r", grid, budget) if rest else ({}, {})
            return done, {**bad, **failed}
        if len(uids) == 1:
            return {}, {uids[0]: reason}
        h = len(uids) // 2
        d1, f1 = d4_run_robust(uids[:h], tag + "a", grid, budget)
        d2, f2 = d4_run_robust(uids[h:], tag + "b", grid, budget)
        return {**d1, **d2}, {**f1, **f2}

def d4_pass(run_ids, grid, label, flush=True):
    """Every sub-chunk of run_ids under one grid -> (attempted uids, raw (1, n, 12) NaN where failed, failed, stop reason)."""
    subs = d4_subchunks(run_ids, SUB_CHUNK)
    done, failed, attempted, stopped = {}, {}, [], None
    for k, sub in enumerate(subs):
        t = time.time()
        try:
            d, f = d4_run_robust(sub, f"{label}_sub{k:03d}", grid, [D4_RETRY_BUDGET])
        except D4GuardStop as e:
            stopped = f"time guard before sub-chunk {k + 1}/{len(subs)}: {e}"
            print(f"[d4] {label}: stopping -- {stopped}", flush=True)
            break
        done.update(d); failed.update(f); attempted += list(sub)
        if flush:
            d4_save(PARTIAL, *d4_combine(_TEACHER_PRIOR_UIDS, _TEACHER_PRIOR_RAW, attempted, d4_rows(attempted, done)),
                    view=D4_VIEW, sha256=D4_SHA256, mode="gold" if GOLD else "train")
        print(f"[d4] {label} sub-chunk {k + 1}/{len(subs)}: {len(d)}/{len(sub)} studies in {time.time() - t:.0f} s; "
              f"{len(done)} done, {len(failed)} failed this run; {(time.time() - T0) / 3600:.2f} h"
              + ("; partial flushed" if flush else ""), flush=True)
        if not d:
            stopped = f"sub-chunk {k + 1}/{len(subs)} produced no study: {next(iter(f.values()), '?')}"
            print(f"[d4] {label}: stopping -- {stopped}", flush=True)
            break
    return attempted, d4_rows(attempted, done), failed, stopped

RUN_UIDS, RUN_RAW, FAILED, STOPPED = d4_pass(ids, D4_GRID, f"gold_{D4_GRID}" if GOLD else "train")
GOLD_EXTRA = {}
if GOLD:                          # the spike also runs the other grid: the original one is what the reference measured
    for _g in [g for g in ("notebook", "original") if g != D4_GRID]:
        GOLD_EXTRA[_g] = d4_pass(list(ALL_IDS), _g, f"gold_{_g}", flush=False)
'''

# ------------------------------------------------------------------------------------------------ cell 6 (ours)
FINAL_CELL = r'''# %%
uids, probs = d4_combine(_TEACHER_PRIOR_UIDS, _TEACHER_PRIOR_RAW, RUN_UIDS, RUN_RAW)   # prior complete rows first
_mode = "gold" if GOLD else "train"
d4_save(f"/kaggle/working/d4_teacher_shard{SHARD}.npz", uids, probs, view=D4_VIEW, sha256=D4_SHA256, mode=_mode)
n_total = d4_csv(f"/kaggle/working/d4_teacher_shard{SHARD}.csv", uids, probs)
run_ok = np.isfinite(RUN_RAW).all(axis=(0, 2))
gold = {}
if GOLD:
    _ref = find_input_file("d4_gold58_reference.npz", _D4_MANIFEST["files"]["d4_gold58_reference.npz"], INPUT_ROOT)
    gold[D4_GRID] = d4_gold_report(RUN_UIDS, RUN_RAW, _ref, D4_GRID)
    for _g, (_u, _r, _f, _s) in GOLD_EXTRA.items():
        d4_save(f"/kaggle/working/d4_teacher_gold_{_g}.npz", _u, _r, view=f"d4-swa3-{_g}", sha256=D4_SHA256, mode="gold")
        d4_csv(f"/kaggle/working/d4_teacher_gold_{_g}.csv", _u, _r)
        gold[_g] = {**d4_gold_report(_u, _r, _ref, _g), "failed": _f, "stopped": _s}
timing = d4_timing_summary(D4_RUNS, D4_SETUP_S, SUB_CHUNK)
elapsed = time.time() - T0
n_run, n_prior = int(run_ok.sum()), len(_TEACHER_PRIOR_UIDS)
not_attempted = len(ids) - len(RUN_UIDS)
json.dump({"teacher": "d4", "grid": D4_GRID, "view": D4_VIEW, "gold": GOLD, "shard": SHARD, "n_shards": N_SHARDS,
           "limit": LIMIT, "sub_chunk": SUB_CHUNK, "studies": n_run, "prior_studies": n_prior, "total_studies": n_total,
           "failed_uids": FAILED, "not_attempted": not_attempted, "stopped": STOPPED, "elapsed_s": elapsed,
           "sec_per_study": elapsed / max(1, n_run), "checkpoint_sha256": D4_SHA256, "timing": timing, "gold_report": gold,
           "runs": [{k: v for k, v in r.items() if k != "per_study_s"} for r in D4_RUNS]},
          open("/kaggle/working/d4_teacher_receipt.json", "w"), indent=1, default=str)
_complete = not FAILED and not not_attempted
print(f"[teacher] {n_run} studies this run ({elapsed / max(1, n_run):.1f} s/study incl. setup), {len(FAILED)} failed, "
      f"{not_attempted} not attempted + {n_prior} prior = {n_total} complete -- "
      + ("COMPLETE" if _complete else "INCOMPLETE: resume in a sibling slug (--kernel-source this one)"), flush=True)
for p in (str(CHUNK_ROOT), "/kaggle/working/_coat_env", "/kaggle/working/_d4_env"):
    shutil.rmtree(p, ignore_errors=True)         # chunk symlinks and the installed wheels are never outputs
'''


# ------------------------------------------------------------------------------------------------ extraction
def _dedent(lines, n, label):
    pad = " " * n
    out = []
    for ln in lines:
        if ln.strip() and not ln.startswith(pad):
            raise SystemExit(f"{label}: line not indented by {n}: {ln[:80]!r}")
        out.append(ln[n:] if ln.strip() else "")
    return out


def extract(nb):
    """The three verbatim blocks as text, plus the slice table (label, cell, start, end, first, last) for provenance."""
    if len(nb["cells"]) != tp.N_CELLS:
        raise SystemExit(f"{NOTEBOOK} has {len(nb['cells'])} cells, expected {tp.N_CELLS}")
    table = []

    def take(cell, start, end, label):
        s, e, out = tp.slice_lines(tp.cell_lines(nb, cell), start, end, label)
        table.append((label, cell, s, e, out[0], out[-1]))
        return out

    # cell 12, whole: thread env defaults, the D4 runtime / output checks, the child runner, the asset finder (+ its
    # memoised redefinition), _asset_patch_d4 and _asset_run_d4. Three token patches.
    lines12 = tp.cell_lines(nb, 12)
    table.append(("cell12 whole", 12, 0, len(lines12) - 1, lines12[0], lines12[-1]))
    c12 = "\n".join(lines12).rstrip("\n")
    for need, times in (("def _d4_check_runtime(", 1), ("def _d4_check_outputs(", 1), ("def _run_required_child(", 1),
                        ("def _asset_find_asset(", 2), ("def _asset_patch_d4(", 1), ("def _asset_run_d4(", 1),
                        ("_speed_original_find_asset = _asset_find_asset", 1)):   # the finder: original + memoised
        if c12.count(need) != times:
            raise SystemExit(f"cell 12 holds {need!r} {c12.count(need)} times, expected {times}")
    for label, old, new, _ in PATCHES:
        c12 = tp.patch_once(c12, old, new, label)

    # cell 14, whole: the dense sampler (_dense_quotas / _dense_stack are shipped into the worker by inspect.getsource).
    lines14 = tp.cell_lines(nb, 14)
    table.append(("cell14 whole", 14, 0, len(lines14) - 1, lines14[0], lines14[-1]))
    c14 = "\n".join(lines14).rstrip("\n")
    if "def _dense_quotas(" not in c14 or "def _dense_stack(" not in c14:
        raise SystemExit("cell 14 is not the dense sampler")

    # cell 45, `_coat_substitute`: its imports, the opencv wheel pin + sha() / find() helpers + install, and the D4 block
    # through the timm install (the d4_input dir, the _asset_run_d4 call and the flagged except are the driver's job).
    lines45 = tp.cell_lines(nb, 45)
    fn = [i for i, ln in enumerate(lines45) if ln == "def _coat_substitute():"]
    if len(fn) != 1:
        raise SystemExit("cell 45: def _coat_substitute() not found exactly once")
    a = take(45, "    import hashlib as _h, os as _o, subprocess as _sp, sys as _sy", "    from pathlib import Path as _P",
             "cell45 imports")
    b = take(45, "    WHL_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'",
             "    WHL_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'", "cell45 cv2 wheel pin")
    c = take(45, "    def sha(p):", "        return _asset_find_asset(name, want)", "cell45 sha/find")
    d = take(45, "    whl = find('opencv_python_headless-4.12.0.88-*.whl', WHL_SHA)", "    )", "cell45 cv2 install")
    e = take(45, "        d4_manifest = _asset_find_asset('coatnet_pairfilm_manifest.json',",
             "            '--target',str(d4_env),str(timm_whl)],check=True)", "cell45 D4 root + timm install")
    for lab, _, s, *_ in table[-5:]:
        if s <= fn[0]:
            raise SystemExit(f"{lab}: slice starts before _coat_substitute")
    c45 = "\n".join(_dedent(a, 4, "imports") + _dedent(b, 4, "pin") + [""] + _dedent(c, 4, "sha/find") + [""]
                    + _dedent(d, 4, "cv2") + [""] + _dedent(e, 8, "d4"))
    for need in ("envd = _P('/kaggle/working/_coat_env')", "d4_env = _P('/kaggle/working/_d4_env')",
                 "raise RuntimeError('D4 timm wheel changed')"):
        if c45.count(need) != 1:
            raise SystemExit(f"cell-45 slices hold {need!r} {c45.count(need)} times")
    for bad in ("_ke_ours", "_KE_LAB", "rsna_phase", "rsna_event", "_KE_NS", "_asset_run_d4(", "MAN_SHA"):
        if bad in c45:
            raise SystemExit(f"cell-45 slices still reference {bad}")
    return {"c12": c12, "c14": c14, "c45": c45}, table


def render_teacher_py(nb, shard=0, n_shards=1, limit=0, gold=False, grid="notebook", sub_chunk=SUB_CHUNK):
    if not (0 <= int(shard) < int(n_shards)) or int(limit) < 0:
        raise SystemExit(f"bad shard/n_shards/limit: {shard}/{n_shards}/{limit}")
    if gold and (int(shard), int(n_shards), int(limit)) != (0, 1, 0):
        raise SystemExit("--gold renders the 58 gold studies as one pass: leave --shard 0 --n-shards 1 --limit 0")
    if grid not in GRIDS:
        raise SystemExit(f"--grid must be one of {GRIDS}, got {grid!r}")
    if int(sub_chunk) < 4:
        raise SystemExit(f"--sub-chunk must be >= 4, got {sub_chunk}")
    blocks, _ = extract(nb)
    pre = (PREAMBLE_TEMPLATE.replace("__SHARD__", str(int(shard))).replace("__N_SHARDS__", str(int(n_shards)))
           .replace("__LIMIT__", str(int(limit))).replace("__GOLD__", str(bool(gold))).replace("__GRID__", grid)
           .replace("__SUB_CHUNK__", str(int(sub_chunk))).replace("__IDS_CALL__", GOLD_CALL if gold else CHUNK_CALL)
           .replace("__PREAMBLE_FUNCS__", PREAMBLE_FUNCS.rstrip("\n")))
    cells = [
        pre.rstrip("\n"),
        "# %%\n# --- notebook_score_0.942.ipynb cell 12 (verbatim, whole; 3 token patches): D4 runtime / output checks, the\n"
        "# child runner, the asset finder, _asset_patch_d4 and _asset_run_d4\n" + blocks["c12"],
        "# %%\n# --- cell 14 (verbatim, whole): the capacity-aware dense sampler (_dense_quotas / _dense_stack go to the D4 worker)\n"
        + blocks["c14"],
        "# %%\n# --- cell 45 (verbatim slices of _coat_substitute, re-indented): the pinned opencv wheel, then the D4 artifact\n"
        "# root and its timm wheel\n" + blocks["c45"],
        DRIVER_CELL.rstrip("\n"),
        FINAL_CELL.rstrip("\n"),
    ]
    src = "\n\n".join(cells) + "\n"
    for i, c in enumerate(src.split("\n# %%")):
        compile(c, f"d4 teacher cell {i}", "exec")
    return src


def build_metadata(slug=KERNEL_ID, kernel_sources=()):
    """As the Raptor builder's (traps 31: a resume runs in a SIBLING slug), with the D4 Dataset + the opencv wheel."""
    meta = tp.build_metadata(slug, kernel_sources)
    meta["dataset_sources"] = list(DATASETS)
    if meta["machine_shape"] != "NvidiaTeslaT4" or not meta["enable_gpu"] or meta["enable_internet"]:
        raise SystemExit("metadata must be T4 x2, GPU on, internet off")
    return meta


def _render_files(shard, n_shards, limit, slug=KERNEL_ID, kernel_sources=(), notebook=NOTEBOOK, gold=False,
                  grid="notebook", sub_chunk=SUB_CHUNK):
    """{filename: text} for the kernel directory, built in a temp dir through nbgen (so --check is exact)."""
    name = tp.slug_name(slug)
    meta = build_metadata(slug, kernel_sources)
    src = render_teacher_py(tp.load_notebook(notebook), shard, n_shards, limit, gold=gold, grid=grid, sub_chunk=sub_chunk)
    with tempfile.TemporaryDirectory() as d:
        py, ipynb = os.path.join(d, f"{name}.py"), os.path.join(d, f"{name}.ipynb")
        with open(py, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        nbgen.build(py, ipynb)
        with open(ipynb, encoding="utf-8") as f:
            nb_text = f.read()
    return {f"{name}.py": src, f"{name}.ipynb": nb_text, "kernel-metadata.json": json.dumps(meta, indent=2) + "\n"}


def write_kernel(out_dir, shard, n_shards, limit, slug=KERNEL_ID, kernel_sources=(), gold=False, grid="notebook",
                 sub_chunk=SUB_CHUNK):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in _render_files(shard, n_shards, limit, slug, kernel_sources, gold=gold, grid=grid,
                                    sub_chunk=sub_chunk).items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="> 0: the first N studies of the shard (6 = smoke)")
    ap.add_argument("--gold", action="store_true", help="the 58 gold studies under both grids, scored vs D4's reference")
    ap.add_argument("--grid", choices=GRIDS, default="notebook",
                    help="D4 input grid: notebook = the 0.942 graph's dense grid (default); original = D4's own")
    ap.add_argument("--sub-chunk", type=int, default=SUB_CHUNK, help="studies per D4 child run / partial flush")
    ap.add_argument("--slug", default=KERNEL_ID, help="kernel id owner/name; names the dir, notebook and title")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="a previous run's kernel slug to mount for a resume (repeatable; never --slug itself)")
    ap.add_argument("--out", default=None, help="kernel dir (default kaggle/<slug name>)")
    ap.add_argument("--check", action="store_true", help="render in memory and compare with the files on disk")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    out = a.out or os.path.join("kaggle", tp.slug_name(a.slug))
    files = _render_files(a.shard, a.n_shards, a.limit, a.slug, a.kernel_source, gold=a.gold, grid=a.grid,
                          sub_chunk=a.sub_chunk)
    _, table = extract(tp.load_notebook(NOTEBOOK))
    summary = (f"{out}: {a.slug} " + ("GOLD-58 (both grids)" if a.gold else f"shard {a.shard}/{a.n_shards} limit {a.limit}")
               + f", grid {a.grid}, sub-chunk {a.sub_chunk}, kernel_sources {a.kernel_source}; "
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
    write_kernel(out, a.shard, a.n_shards, a.limit, a.slug, a.kernel_source, gold=a.gold, grid=a.grid,
                 sub_chunk=a.sub_chunk)
    print("wrote " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
