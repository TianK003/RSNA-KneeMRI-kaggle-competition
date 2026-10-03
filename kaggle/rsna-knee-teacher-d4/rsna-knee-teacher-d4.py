# %%
SHARD = 0                 # sed'd at build: shard index
N_SHARDS = 1              # sed'd at build: number of shards over the 4,349 report-labelled studies
LIMIT = 0                 # sed'd at build: > 0 = first N studies of the shard (6 = smoke)
GOLD = False                # sed'd at build: True = the 58 gold studies (the spike), under BOTH input grids
D4_GRID = "original"         # sed'd at build: "notebook" = the 0.942 graph's dense grid; "original" = D4's own Global96 grid
SUB_CHUNK = 400               # sed'd at build: studies per D4 child run; a partial npz is flushed after each
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

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
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

def gold_study_ids(train_csv):
    """P-49 (--gold): the 58 radiologist-labelled studies, sorted by UID -- the Raptor teacher's quality on held-out
    truth. Never a training table: the pipeline masks every teacher table on gold rows."""
    import pandas as pd
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    ids = sorted(tr.loc[tr[LABELS].notna().all(axis=1), "StudyInstanceUID"].tolist())
    assert len(ids) == 58, len(ids)
    return ids

_SKIP_TOPS = ("rsna-knee-abnormality-detection", "competitions")

def _teacher_npz_paths(input_root="/kaggle/input", skip_tops=_SKIP_TOPS):
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

def write_chunk_root(root, uids, series, image_tree):
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
ALL_IDS = chunk_study_ids(f"{COMP_IN}/train.csv", SHARD, N_SHARDS, LIMIT)
# Resume: complete rows of earlier runs of this shard and grid (mounted from a sibling slug) go first in every output.
_TEACHER_PRIOR_UIDS, _TEACHER_PRIOR_RAW = load_prior(INPUT_ROOT, keep=set(ALL_IDS), view=D4_VIEW)
skip = set(_TEACHER_PRIOR_UIDS)
ids = [u for u in ALL_IDS if u not in skip]
print(f"chunk: {'GOLD-58' if GOLD else f'shard {SHARD}/{N_SHARDS}, limit {LIMIT}'}, grid {D4_GRID}, sub-chunk {SUB_CHUNK} "
      f"-> {len(ids)} studies ({len(skip)} already done)", flush=True)
if not ids:
    raise RuntimeError("nothing left to run in this shard (every study is already in a mounted D4 teacher npz)")
SERIES = pd.read_csv(f"{COMP_IN}/train_series.csv", dtype=str)

# %%
# --- notebook_score_0.942.ipynb cell 12 (verbatim, whole; 3 token patches): D4 runtime / output checks, the
# child runner, the asset finder, _asset_patch_d4 and _asset_run_d4
import os
for _thread_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_env, '4')
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG', '1024')

def _d4_check_runtime(rt, artifact_root):
    import hashlib
    import json
    from pathlib import Path
    import torch
    root = Path(artifact_root)
    manifest_path = root / 'coatnet_pairfilm_manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 reference manifest identity changed')
    expected = json.loads(manifest_path.read_text())
    manifest, paths = rt.validate_artifact_root(root, verify_hashes=True)
    if manifest != expected or len(paths) != 1 or rt.N_MODELS != 1:
        raise RuntimeError('D4 must load exactly the reference SWA model')
    if Path(paths[0]).name != 'coatnet_d4_parent_rank8_swa2.pt':
        raise RuntimeError('D4 must use its reference rank-8 parent checkpoint')
    if manifest['slice_depth']['adapter_sha256'] != 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a':
        raise RuntimeError('D4 reference adapter identity changed')
    if rt.GPU_BATCH_STUDIES != 2 or rt.BACKBONE_MICRO_IMAGES != 8:
        raise RuntimeError('D4 reference batching changed')
    if rt.CUDNN_BENCHMARK is not False:
        raise RuntimeError('D4 reference convolution policy changed')
    if rt.base.cv2.__version__ != '4.12.0' or rt.parent.timm.__version__ != '1.0.22':
        raise RuntimeError('D4 reference dependency versions changed')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('D4 requires exactly two T4 GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all(('T4' in name for name in names)):
        raise RuntimeError(f'D4 reference device mismatch: {names}')
    return (manifest, paths)

def _d4_check_outputs(rt, manifest, receipt, output, competition):
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    def need(condition, message):
        if not condition:
            raise RuntimeError('D4 release check: ' + message)
    need(receipt['status'] == rt.SUBMISSION_STATUS, 'unsuccessful status')
    expected_fields = {'models': 1, 'fallback_studies': 0, 'dicom_preparations_per_study': 1, 'model_passes_per_study': 1, 'rank_after_probability_average': True, 'canonical_sagittal': True, 'common_physical_triplet_warp': False, 'resolution': 384, 'slice_depth_arm': 'd4_zones', 'slice_depth_state_elements': 3255, 'slice_depth_parent_checkpoint_sha256': '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'fsx_attention_heads': 2, 'fsx_attention_width': 128, 'fsx_serving_delta_cap': 0.3, 'fsx_training_delta_cap': 0.5, 'head': 'rank8_swa2_plus_d4_three_zone_expanded_fsx', 'slice_depth_swa_member_epochs': [11, 9, 5], 'slice_depth_checkpoint_sha256': manifest['slice_depth']['adapter_sha256']}
    for key, value in expected_fields.items():
        need(receipt[key] == value, key)
    need(1 <= int(receipt['maximum_eval_windows']) <= 94, 'window limit')
    checkpoints = receipt['checkpoints']
    need(len(checkpoints) == 1, 'parent checkpoint count')
    need(checkpoints[0]['sha256'] == '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'parent checkpoint hash')
    need(checkpoints[0]['member_epochs'] == [15, 16], 'parent SWA member epochs')
    need(receipt['slice_depth_checkpoint_sha256'] == 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a', 'adapter checkpoint hash')
    processes = receipt['processes']
    need(len(processes) == 2 and all((int(p['returncode']) == 0 for p in processes)), 'worker process failure')
    shards = receipt['shards']
    need(len(shards) == 2, 'shard count')
    for s in shards:
        need(s['device'] == 'cuda', 'non-CUDA shard')
        need(int(s['fallback_studies']) == 0, 'failed study')  # teacher pass: preparation warnings are recorded, not fatal
        need(int(s['models_resident']) == 1, 'resident model count')
        need(int(s['peak_reserved_bytes']) < int(15.5 * 1024 ** 3), 'reference memory gate')
        need(s['specialized_worker_loader_contract'] == manifest['worker_loader_contract'], 'specialized loader')
    output = Path(output)
    prediction_path = output.parent / rt.PREDICTIONS_NAME
    with np.load(prediction_path, allow_pickle=False) as payload:
        uids = payload['study_uids'].astype(str).tolist()
        raw = np.asarray(payload['raw_probabilities'], dtype=np.float32)
        mean = np.asarray(payload['probability_mean'], dtype=np.float32)
        ranked = np.asarray(payload['submission_percentile_rank'], dtype=np.float64)
    ids = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    need(len(ids) > 0 and len(ids) == len(set(ids)), 'test identity')
    need(uids == ids and int(receipt['studies']) == len(ids), 'prediction identity/count')
    need(raw.shape == (1, len(ids), 12), 'raw probability shape')
    need(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'raw probabilities')
    need(np.array_equal(raw.astype(np.float64).mean(0).astype(np.float32), mean), 'probability mean')
    need(np.array_equal(rt.percentile_rank64(mean), ranked), 'global percentile ranks')
    frame = pd.read_csv(output, dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    need(frame.columns.tolist() == ['StudyInstanceUID', *rt.base.LABELS], 'CSV labels')
    need(frame.StudyInstanceUID.tolist() == ids, 'CSV identity')
    need(np.array_equal(frame.iloc[:, 1:].to_numpy(np.float64), ranked), 'CSV numeric round trip')
    actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    need(actual_hash == receipt['output_sha256'], 'CSV hash')
    receipt['reference_wrapper_checks'] = {'export_sha256': '7f8410fdf1b60cd3831d50f517bb25798a4e4a3404c59dc269193d5fe7cce729', 'validated_manifest_sha256': '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d', 'probability_mean_exact': True, 'ranks_exact': True, 'csv_roundtrip_exact': True, 'source_and_model_hashes_validated': True, 'input_identity_equals_original_grid': False}
    return receipt

def _run_required_child(command, environment, logfile):
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path
    remaining = TIME_BUDGET - (time.time() - T0)
    if remaining <= 0:
        raise TimeoutError('No time remains for a required model branch')
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log_handle:
        process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode != 0:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 5000))
            tail = handle.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Required model branch failed ({returncode}); {path}\n{tail}')
    return returncode
ASSET_ROOTS = ['/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan', '/kaggle/input/raptor-knee-maxspan', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384', '/kaggle/input/raptor-knee-native384', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense', '/kaggle/input/raptor-knee-native384dense', '/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights', '/kaggle/input/knee-mri-fold-weights', '/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86', '/kaggle/input/opencv-python-headless-4120088-x86', '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan', '/kaggle/input/resnet-50-radimagenet-marwan', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels', '/kaggle/input/rsna-knee-llm-labels', '/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/datasets/pilkwang/rsna-knee-weights', '/kaggle/input/rsna-knee-weights', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train', '/kaggle/input/rsna-knee-e11-train', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train', '/kaggle/input/rsna-knee-e13-train', '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1', '/kaggle/input/dinov2/pytorch/small/1']

def _asset_walk():
    from pathlib import Path
    seen = set()
    for root in ASSET_ROOTS:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        level = [root]
        for _depth in range(6):
            following = []
            for directory in sorted(level):
                if str(directory) in seen:
                    continue
                seen.add(str(directory))
                children = sorted(directory.iterdir())
                dirs = [p for p in children if p.is_dir() and p.name not in ('train_series', 'test_series', 'train_images', 'test_images')]
                files = [p.name for p in children if p.is_file()]
                yield (str(directory), [p.name for p in dirs], files)
                following.extend(dirs)
            level = following

def _asset_find_asset(name, digest=None):
    import fnmatch, hashlib
    from pathlib import Path
    hits = []
    for root, _, files in _asset_walk():
        for file in files:
            if fnmatch.fnmatchcase(file, name):
                p = Path(root) / file
                if digest:
                    h = hashlib.sha256()
                    with p.open('rb') as handle:
                        for block in iter(lambda: handle.read(8 << 20), b''):
                            h.update(block)
                    if h.hexdigest() != digest:
                        continue
                if p.resolve() not in [h.resolve() for h in hits]:
                    hits.append(p)
    if len(hits) != 1:
        raise RuntimeError(f'expected one pinned asset {name}, got {hits}')
    return hits[0]

def _asset_half_rank_mix(top3, swa3):
    if top3.shape != swa3.shape or not top3.index.equals(swa3.index):
        raise ValueError('CoAt family alignment changed')
    if top3.isna().any().any() or swa3.isna().any().any():
        raise ValueError('missing CoAt family prediction')
    units = top3.rank(method='average') + swa3.rank(method='average')
    return units.rank(method='average', pct=True)

def _asset_write_coat_runtime(artifact, output, helper_source):
    from pathlib import Path
    import hashlib
    source = Path(artifact) / 'coatnet_resgated_ep10_top3_inference.py'
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9':
        raise RuntimeError('unexpected parent CoAt runtime')
    edits = {'if not 1 <= windows <= MAX_EVAL_WINDOWS:': 'if not 1 <= windows <= 94:', 'or int(counts.max()) > MAX_EVAL_WINDOWS:': 'or int(counts.max()) > 94:', '"runtime_contract": RUNTIME_CONTRACT,': '"runtime_contract": RUNTIME_CONTRACT, "input_max_windows": 94,', '"eval_grid": "saved_training_gold_unique_center_slot_budgets",': '"eval_grid": "input_unique_native_centers_six_slot_v1",'}
    for old, new in edits.items():
        if old not in text:
            raise RuntimeError('CoAt patch target missing: ' + old)
        text = text.replace(old, new)
    text = text.replace('"gold58_rank_ensemble_auc": manifest["selection"]', '"historical_parent_gold58_rank_ensemble_auc": manifest["selection"]')
    text = text.replace('"t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]', '"historical_parent_t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]')
    marker = 'if __name__ == "__main__":'
    if text.count(marker) != 1:
        raise RuntimeError('CoAt entrypoint changed')
    patch = helper_source + ('\n_original_train_faithful_eval_specs = train_faithful_eval_specs\n'
                             'def _tolerant_coat_specs(study):\n    try:\n        return _dense_coat_specs(study)\n'
                             '    except Exception as exc:\n        print("[dense-fallback] " + str(getattr(study, "study_uid", "?")) + ": " + type(exc).__name__ + ": " + str(exc), flush=True)\n'
                             '        return _original_train_faithful_eval_specs(study)\n'
                             'train_faithful_eval_specs = _tolerant_coat_specs\nRUNTIME_CONTRACT = "resgated_ep10_input_nativecenters_v1"\n')
    text = text.replace(marker, patch + '\n' + marker)
    compile(text, str(output), 'exec')
    Path(output).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def _asset_patch_d4(rt, grid_source):
    import ast
    import hashlib
    import inspect
    import json
    import os
    import textwrap
    from pathlib import Path
    audited = rt.parent.audited
    original = audited.prepare_global96_bag
    original_shard = rt._PARENT_INFER_SHARD
    if getattr(original, '_input_patch_installed', False):
        raise RuntimeError('D4 input preparation was already patched')
    src = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or fn.name != 'prepare_global96_bag':
        raise RuntimeError('D4 preparation function identity changed')
    if 'study_uid' not in inspect.signature(original).parameters:
        raise RuntimeError('D4 preparation study-UID contract changed')
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'specs' for t in n.targets)) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and (n.value.func.value.id == 'global_stack') and (n.value.func.attr == 'sample_global_windows')]
    if len(candidates) != 1 or candidates[0] not in fn.body:
        raise RuntimeError('D4 expected one top-level Global96 specs constructor')
    call = candidates[0]
    keywords = {k.arg: k.value for k in call.value.keywords}
    if not (isinstance(keywords.get('count'), ast.Constant) and keywords['count'].value is None and isinstance(keywords.get('train'), ast.Constant) and (keywords['train'].value is False)):
        raise RuntimeError('D4 original evaluation must use count=None, train=False')
    exec(compile(grid_source, '<input-grid>', 'exec'), audited.__dict__)
    audit_dir = Path(os.environ['RSNA_D4_INPUT_AUDIT_DIR'])
    audit_dir.mkdir(parents=True, exist_ok=True)

    def record_input(study_uid, stack, audit):
        import numpy as np
        arrays = ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')
        digest = hashlib.sha256()
        for name in arrays:
            value = np.ascontiguousarray(getattr(stack, name))
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        record = {'study_uid': str(study_uid), 'pid': os.getpid(), 'positions': int(len(stack.global_indices)), 'windows_expected': int(len(stack.global_indices) - 2), 'slot_widths': list(map(int, stack.slot_widths)), 'stack_sidecars_sha256': digest.hexdigest(), 'slots': audit}
        with (audit_dir / f'input_{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    audited.__dict__['_record_input'] = record_input
    injection = ast.parse('try:\n    stack, input_audit = _dense_stack(stack, study.offsets, study.valid, study.source_depth)\n'
                          'except Exception as _dense_exc:\n    print("[dense-fallback] " + str(study_uid) + ": " + type(_dense_exc).__name__ + ": " + str(_dense_exc), flush=True)\n'
                          '    input_audit = [{"fallback": type(_dense_exc).__name__ + ": " + str(_dense_exc)}]\n'
                          '_record_input(study_uid, stack, input_audit)\n').body
    where = fn.body.index(call)
    fn.body[where:where] = injection
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<D4-input-preparation-only>', 'exec'), audited.__dict__)
    patched = audited.prepare_global96_bag
    patched._input_patch_installed = True
    for module in (rt, rt.parent):
        if getattr(module, 'prepare_global96_bag', None) is original:
            module.prepare_global96_bag = patched
    if rt._PARENT_INFER_SHARD is not original_shard:
        raise RuntimeError('D4 parent shard must remain the official callable')
    files = {}
    for name, module in [('d4_runtime', rt), ('parent_runtime', rt.parent), ('audited_runtime', audited)]:
        path = Path(module.__file__)
        if path.is_file():
            files[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'contract': 'd4-original-runtime-input-preparation-only-v2', 'reference_script_version_id': 348764100, 'reference_export_available': True, 'original_prepare_source_sha256': hashlib.sha256(src.encode()).hexdigest(), 'patched_prepare_source_sha256': hashlib.sha256(ast.unparse(tree).encode()).hexdigest(), 'model_loader_modified': False, 'zone_head_modified': False, 'parent_infer_shard_replaced': False, 'grid_source_sha256': hashlib.sha256(grid_source.encode()).hexdigest(), 'artifact_source_files': files}
    (audit_dir / f'patch_{os.getpid()}.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def _asset_run_d4(artifact, environment_dir, competition, output, grid_suffix=''):
    import hashlib, inspect, json, os, sys
    from pathlib import Path
    import pandas as pd
    artifact, output = (Path(artifact), Path(output))
    manifest = artifact / 'coatnet_pairfilm_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 SWA3 manifest changed')
    grid = 'import numpy as np\nfrom raptor_global_stack_smart336 import FixedGlobalStack, INFERRED96_WIDTHS\n' + '\n'.join((inspect.getsource(f) for f in (_dense_quotas, _dense_stack))) + grid_suffix
    audit_dir = output.parent / 'study_input_receipts'
    audit_dir.mkdir(parents=True, exist_ok=True)
    for stale in audit_dir.glob('*.json*'):
        stale.unlink()
    output.unlink(missing_ok=True)
    output.with_suffix('.receipt.json').unlink(missing_ok=True)
    wrapper = output.parent / 'd4_worker_entry.py'
    code = f'import sys\nsys.path.insert(0,{str(environment_dir)!r})\nsys.path.insert(0,{str(artifact)!r})\nfrom pathlib import Path\nimport json\nimport coatnet_d4_depthzone_swa_inference as rt\n' + inspect.getsource(_asset_patch_d4) + '\n' + inspect.getsource(_d4_check_runtime) + '\n' + inspect.getsource(_d4_check_outputs) + '\n' + f"if __name__ == '__main__' and '--worker-output' not in sys.argv:\n    _checked_manifest, _checked_paths = _d4_check_runtime(rt, Path({str(artifact)!r}))\n" + f'_asset_patch_d4(rt,{grid!r})\n' + 'rt.parent.audited.__file__ = __file__\n' + "if __name__ == '__main__':\n" + "    if '--worker-output' in sys.argv:\n        raise SystemExit(rt.main())\n" + '    manifest, paths = _checked_manifest, _checked_paths\n' + '    r = rt.run_submission(\n' + f'        competition_root=Path({str(competition)!r}), artifact_root=Path({str(artifact)!r}),\n' + f'        output_path=Path({str(output)!r}), gpu_batch_studies=2, backbone_micro_images=8)\n' + f'    r = _d4_check_outputs(rt, manifest, r, Path({str(output)!r}), Path({str(competition)!r}))\n' + f"    Path({str(output.with_suffix('.receipt.json'))!r}).write_text(json.dumps(r,indent=2,allow_nan=False))\n"
    compile(code, str(wrapper), 'exec')
    wrapper.write_text(code)
    env = dict(os.environ)
    env['RSNA_D4_INPUT_AUDIT_DIR'] = str(audit_dir)
    env['PYTHONPATH'] = f'{environment_dir}:/kaggle/working/_coat_env:{artifact}:' + env.get('PYTHONPATH', '')
    env.pop('CUDNN_CONV_WSCAP_DBG', None)
    _run_required_child([sys.executable, str(wrapper)], env, output.with_suffix('.log'))
    receipt = json.loads(output.with_suffix('.receipt.json').read_text())
    expected = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    records = []
    for path in sorted(audit_dir.glob('input_*.jsonl')):
        records.extend((json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    ids = [r['study_uid'] for r in records]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError('D4 input hook must execute exactly once for every test study')
    if not all((1 <= r['windows_expected'] <= 94 and r['positions'] == r['windows_expected'] + 2 for r in records)):
        raise RuntimeError('D4 input record cardinality drift')
    receipt['input_coverage'] = {'studies': len(records), 'folder': str(audit_dir), 'min_windows': min((r['windows_expected'] for r in records)), 'max_windows': max((r['windows_expected'] for r in records)), 'original_model_loader_retained': True, 'original_shard_scheduler_retained': True}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt
_speed_original_asset_walk = _asset_walk
_speed_original_find_asset = _asset_find_asset
'Catalogue named immutable attachments once; no competition-tree search.'
import threading as _speed_threading
_speed_asset_lock = _speed_threading.RLock()
_speed_asset_catalogue = None
_speed_asset_hits = {}

def _asset_walk():
    global _speed_asset_catalogue
    with _speed_asset_lock:
        if _speed_asset_catalogue is None:
            _speed_asset_catalogue = tuple(((root, tuple(dirs), tuple(files)) for root, dirs, files in _speed_original_asset_walk()))
        catalogue = _speed_asset_catalogue
    for root, dirs, files in catalogue:
        yield (root, list(dirs), list(files))

def _asset_find_asset(name, digest=None):
    key = (str(name), digest)
    with _speed_asset_lock:
        if key not in _speed_asset_hits:
            _speed_asset_hits[key] = _speed_original_find_asset(name, digest)
        return _speed_asset_hits[key]

# %%
# --- cell 14 (verbatim, whole): the capacity-aware dense sampler (_dense_quotas / _dense_stack go to the D4 worker)
import numpy as np

def _dense_allocate(capacities, proportions, target):
    capacities = np.asarray(capacities, dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if capacities.ndim != 1 or capacities.shape != proportions.shape or np.any(capacities < 0) or (not np.isfinite(proportions).all()) or np.any(proportions <= 0) or (int(target) < 1):
        raise ValueError('invalid capacity-aware sampling allocation')
    ideal = int(target) * proportions / proportions.sum()
    initial = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - initial), kind='stable')[:int(target) - int(initial.sum())]:
        initial[i] += 1
    quotas = np.minimum(initial, capacities)
    wanted = min(int(target), int(capacities.sum()))
    deficit = np.zeros(len(quotas), dtype=np.float64)
    while int(quotas.sum()) < wanted:
        donors = quotas < capacities
        weights = np.where(donors, proportions, 0.0)
        deficit[donors] += weights[donors] / weights.sum()
        chosen = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[chosen] += 1
        deficit[chosen] -= 1
    assert int(quotas.sum()) == wanted and np.all(quotas <= capacities)
    return quotas

def _dense_unique_linspace(capacity, count):
    if not 0 <= count <= capacity:
        raise ValueError('cannot create unique picks beyond physical capacity')
    result = np.linspace(0, capacity - 1, count).round().astype(np.int64)
    if count and (result.min() < 0 or result.max() >= capacity):
        raise AssertionError('out-of-range source')
    assert len(np.unique(result)) == count
    return result

def _dense_coat_specs(study):
    import raptor_light224 as sampler
    capacities = []
    for row in study.series_rows:
        if int(row) < 0:
            capacities.append(0)
        else:
            centers, _, _, _ = sampler._candidate_centers(int(row), study.offsets, study.valid, study.source_depth)
            capacities.append(len(centers))
    if sum(capacities) < 1:
        raise ValueError('no usable CoAt triplets')
    quotas = _dense_allocate(capacities, (20, 12, 12, 8, 16, 8), 94)
    specs = sampler.sample_eval_windows(study.series_rows, study.offsets, study.valid, study.source_depth, count=None, slot_budgets=np.maximum(quotas, 1).tolist())
    if len(specs) != min(94, sum(capacities)):
        raise RuntimeError('capacity-aware sampling window-count drift')
    keys = [(s.series_row, s.center_local) for s in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError('capacity-aware sampling repeated a native center')
    for s in specs:
        start = int(study.offsets[s.series_row])
        if not np.all(study.valid[start + np.asarray(s.local_indices)]):
            raise RuntimeError('capacity-aware sampling invalid source channel')
    return specs
INFERRED96_WIDTHS = (27, 21, 18, 12, 18)

def _dense_quotas(capacities) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    base = np.asarray(INFERRED96_WIDTHS, dtype=np.int64)
    if capacities.shape != (5,) or bool((capacities < 0).any()):
        raise ValueError('expected five nonnegative source capacities')
    quotas = np.minimum(base, capacities)
    target = min(96, int(capacities.sum()))
    deficit = np.zeros(5, dtype=np.float64)
    while int(quotas.sum()) < target:
        donors = quotas < capacities
        weights = np.where(donors, base, 0).astype(np.float64)
        deficit[donors] += weights[donors] / weights.sum()
        selected = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[selected] += 1
        deficit[selected] -= 1
    if int(quotas.sum()) != target or bool((quotas > capacities).any()):
        raise RuntimeError('additional quota allocation changed')
    return quotas

def _dense_stack(reference, offsets, valid, source_depth):
    from types import SimpleNamespace
    from raptor_global_stack_smart336 import FixedGlobalStack
    if tuple(reference.slot_widths) != tuple(INFERRED96_WIDTHS):
        raise ValueError('capacity-aware sampling is pinned to the Global96 reference layout')
    pools = []
    audit = []
    begin = 0
    for slot, width in enumerate(reference.slot_widths, 1):
        end = begin + width
        old = reference.global_indices[begin:end]
        row = int(reference.source_series_rows[begin])
        info = {'slot': slot, 'original_width': width, 'source_row': row}
        audit.append(info)
        if row < 0 or bool((old < 0).all()):
            pools.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            begin = end
            continue
        start, stop = map(int, offsets[row:row + 2])
        if not 0 <= start < stop <= len(valid) == len(source_depth):
            raise ValueError('source offset drift')
        depth = np.asarray(source_depth[start:stop], dtype=np.float32)
        keep = np.asarray(valid[start:stop], dtype=bool) & np.isfinite(depth) & (depth >= 0.02) & (depth <= 0.98)
        eligible = np.flatnonzero(keep).astype(np.int64) + start
        reverse = bool(old[0] > old[-1] or (old[0] == old[-1] and np.float32(source_depth[old[0]]) != reference.canonical_depths[begin]))
        if reverse:
            eligible = eligible[::-1].copy()
        depth = np.asarray(source_depth[eligible], dtype=np.float32)
        if reverse:
            depth = (1.0 - depth).astype(np.float32)
        pools.append((eligible, depth))
        begin = end
    quotas = _dense_quotas([len(pool[0]) for pool in pools])
    if int(quotas.sum()) < 3:
        raise ValueError('study has fewer than three usable unique source slices')
    parts = {key: [] for key in ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')}
    for slot, ((eligible, depth), quota, info) in enumerate(zip(pools, quotas, audit), 1):
        quota = int(quota)
        info.update(eligible=len(eligible), allocated_width=quota, additional=max(0, quota - info['original_width']))
        info['reason'] = 'missing_no_padding' if not quota else 'short_slot_all_unique' if len(eligible) < info['original_width'] else 'uniform_native_centers' if info['additional'] else 'uniform_native_centers'
        positions = np.linspace(0, len(eligible) - 1, quota).round().astype(np.int64)
        if len(positions) != quota or not bool((np.diff(positions) > 0).all()):
            raise RuntimeError('capacity-aware sampling slot selection repeated a source')
        mask = reference.nominal_slots == slot
        source_slot = int(reference.source_slot_ids[mask][0])
        parts['global_indices'].append(eligible[positions])
        parts['canonical_depths'].append(depth[positions])
        parts['nominal_slots'].append(np.full(quota, slot, dtype=np.int8))
        parts['nominal_steps'].append(np.full(quota, 0.96 / max(quota - 1, 1), dtype=np.float32))
        parts['source_series_rows'].append(np.full(quota, info['source_row'], dtype=np.int32))
        parts['source_slot_ids'].append(np.full(quota, source_slot, dtype=np.int8))
    result = FixedGlobalStack(**{key: np.concatenate(value) for key, value in parts.items()}, slot_widths=tuple((int(value) for value in quotas)))
    if len(np.unique(result.global_indices)) != len(result.global_indices) or bool((result.global_indices < 0).any()):
        raise RuntimeError('capacity-aware sampling output contains a repeated/missing source')
    return (result, audit)

# %%
# --- cell 45 (verbatim slices of _coat_substitute, re-indented): the pinned opencv wheel, then the D4 artifact
# root and its timm wheel
import hashlib as _h, os as _o, subprocess as _sp, sys as _sy
from pathlib import Path as _P
WHL_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'

def sha(p):
    d = _h.sha256()
    with _P(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            d.update(b)
    return d.hexdigest()

def find(name, want):
    return _asset_find_asset(name, want)

whl = find('opencv_python_headless-4.12.0.88-*.whl', WHL_SHA)
envd = _P('/kaggle/working/_coat_env')
_sp.run(
    [
        _sy.executable,
        '-m',
        'pip',
        'install',
        '--no-deps',
        '--quiet',
        '--target',
        str(envd),
        str(whl),
    ],
    check=True,
)

d4_manifest = _asset_find_asset('coatnet_pairfilm_manifest.json',
    '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d')
timm_whl = d4_manifest.parent/'timm-1.0.22-py3-none-any.whl'
if sha(timm_whl) != '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
    raise RuntimeError('D4 timm wheel changed')
d4_env = _P('/kaggle/working/_d4_env')
_sp.run([_sy.executable,'-m','pip','install','--no-deps','--quiet',
    '--target',str(d4_env),str(timm_whl)],check=True)

# %%
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

# %%
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
