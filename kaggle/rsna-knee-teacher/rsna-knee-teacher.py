# %%
SHARD = 0                 # sed'd at build: shard index
N_SHARDS = 1              # sed'd at build: number of shards over the 4,349 report-labelled studies
LIMIT = 6                 # sed'd at build: > 0 = first N studies of the shard (6 = smoke, 100 = the timing spike)
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

# %%
# --- notebook_score_0.942.ipynb cell 16 (verbatim slices): runtime-integrity helpers + rsna_phase
# Runtime integrity: no partial ensemble is ever published as submission.csv.
import os, json, hashlib, time, threading, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

_RSNA_AUDIT_LOCK = threading.RLock()
_RSNA_AUDIT = {'contract': 'btkd_speedy_v558_input_uniform', 'events': [], 'phases': {}}
_RSNA_ORDER_MEMO = {}
_RSNA_HEADERS_MEMO = {}
_RSNA_CACHE_FILES = []
_RSNA_TEST_IDS = None
_RSNA_LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def rsna_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rsna_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def rsna_event(kind, **details):
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['events'].append({'kind':kind, **details})

def rsna_finite(values, tag, probability=False):
    x = np.asarray(values)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise RuntimeError(f'{tag}: nonfinite/non-numeric values BEFORE ranking')
    if probability and x.size and (x.min() < 0 or x.max() > 1):
        raise RuntimeError(f'{tag}: probability outside [0,1]')
    return x

def rsna_rank01(values):
    """Parent endpoint scale; tied values share their average rank."""
    x = np.asarray(rsna_finite(values, 'rank01'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rank01 requires a nonempty [study,finding] matrix')
    if len(x) == 1:
        return np.full_like(x, .5)
    return (pd.DataFrame(x).rank(method='average').to_numpy() - 1) / (len(x)-1)

def rsna_rankpct(values):
    x = np.asarray(rsna_finite(values, 'rankpct'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rankpct requires a nonempty matrix')
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy(np.float64)

def rsna_frame(frame, ids, labels, tag):
    ids = [str(u) for u in ids]; labels = list(labels)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'{tag}: duplicate expected UID')
    if frame.columns.tolist() != ['StudyInstanceUID', *labels]:
        raise RuntimeError(f'{tag}: label order/schema mismatch')
    frame = frame.copy(); frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError(f'{tag}: missing/extra/duplicate UID')
    frame = frame.set_index('StudyInstanceUID').loc[ids].reset_index()
    rsna_finite(frame[labels].to_numpy(), tag, probability=True)
    return frame

def rsna_save_predictions(tag, ids, values, labels=None):
    x = rsna_finite(values, tag)
    target = Path('/kaggle/working/diagnostics'); target.mkdir(parents=True, exist_ok=True)
    path = target/(tag+'.npz')
    np.savez_compressed(path, study_uids=np.asarray(ids,dtype=str), labels=np.asarray(labels or _RSNA_LABELS,dtype=str), values=x)
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['phases'][tag]={'path':str(path),'sha256':rsna_sha(path),'shape':list(x.shape)}

def rsna_strict_load(model, state, tag):
    """A random frozen encoder is never an acceptable missing-state fallback."""
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state)); extra = sorted(set(state)-set(expected))
    bad_shape = [k for k in set(expected)&set(state) if tuple(expected[k].shape)!=tuple(state[k].shape)]
    if missing or extra or bad_shape:
        raise RuntimeError(f'{tag}: incomplete/mismatched checkpoint; missing={missing[:20]}, extra={extra[:20]}, shapes={bad_shape[:20]}. A partial encoder needs its exact pinned source weights; random initialization is forbidden.')
    for k,t in state.items():
        if t.is_floating_point() and not bool(t.isfinite().all()):
            raise RuntimeError(f'{tag}: nonfinite checkpoint tensor {k}')
    return model.load_state_dict(state, strict=True)

def rsna_deadline(tag):
    if time.time() - T0 > TIME_BUDGET:
        raise TimeoutError(f'{tag}: full-run budget exceeded; no incomplete output may be submitted')

def rsna_phase(stage, status='START', **extra):
    """Local phase evidence; this does not expose Kaggle's hidden rerun logs."""
    import json, os, resource, time
    from pathlib import Path
    work = Path('/kaggle/working/diagnostics')
    work.mkdir(parents=True, exist_ok=True)
    current = {'stage': stage, 'status': status,
               'elapsed_seconds': time.time()-T0,
               'max_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), **extra}
    for path in ['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max']:
        try: current[path.rsplit('/', 1)[-1]] = Path(path).read_text().strip()
        except OSError: pass
    if torch.cuda.is_initialized():
        current['gpus'] = [{'id': i, 'allocated_bytes': torch.cuda.memory_allocated(i),
                            'reserved_bytes': torch.cuda.memory_reserved(i)}
                           for i in range(torch.cuda.device_count())]
    print('[phase] '+json.dumps(current, sort_keys=True), flush=True)
    rsna_json(work/'current_phase.json', current)
    with (work/'phase_events.jsonl').open('a') as f:
        f.write(json.dumps(current, sort_keys=True)+'\n')
    return current

# %%
# --- cell 12 (verbatim slices): asset walker / finder, then its memoised redefinition
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
# --- cell 14 (verbatim, whole cell): capacity-aware dense sampler
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

# %%
# --- cell 45 (verbatim slices, 2 token patches): the Raptor branch
import gc as _ke_gc
import os as _ke_os
import time as _ke_time
from concurrent.futures import ThreadPoolExecutor as _KeThreadPool
import numpy as _ke_np
import pandas as _ke_pd
from pathlib import Path as _KePath

_KE_SRC = r'''
import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = False  # ragged 94-window chunks: no repeated autotuning
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(slot[2] for slot in SLOTS)
K_EVAL = 62
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]
_VIEW_W = RUN["raptor_view_w"]
for _arm in ARMS:
    _arm["w"] = float(_VIEW_W[_arm["name"]])

def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(('maxvit', 'maxxvit', 'coatnet', 'coat_', 'convnext'))
    is_vit = not hybrid and any((k in arch for k in ('vit', 'deit', 'dinov2', 'eva', 'beit')))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool='token', dynamic_img_size=True)
    else:
        kw.update(global_pool='avg')
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):

    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', arch_default)
    if arch != arch_default:
        raise RuntimeError(f'Raptor architecture drift: {arch} != {arch_default}')
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    rsna_strict_load(model,ck['model'],str(pt_path))
    rsna_event('raptor_checkpoint',file=str(pt_path),sha256=rsna_sha(pt_path),arch=arch,res=ck_res)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)



def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM):
    """Resize each selected source plane once, then gather the unchanged RGB triplets."""
    volume = np.asarray(vol)
    depth = int(volume.shape[0])
    centers = np.asarray(_eval_centers(mask, depth, k), dtype=np.int64)
    centers = np.clip(centers, 1, depth - 2)
    if tuple(volume.shape[-2:]) == (res, res):
        # Native384 needs no interpolation; the existing small per-triplet copy
        # avoids a slower large advanced-index gather and preserves its exact math.
        wins = np.empty((len(centers), 3, res, res), np.float32)
        for j, center in enumerate(centers):
            wins[j] = np.stack([volume[center - 1], volume[center],
                                volume[center + 1]], axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(wins)
        if norm == 'imagenet':
            x = (x - _MEAN) / _STD
        return x
    triplets = centers[:, None] + np.asarray([-1, 0, 1], dtype=np.int64)
    # Use only planes referenced by the original eval-center recipe.
    unique, inverse = np.unique(triplets.reshape(-1), return_inverse=True)
    source = volume[unique].astype(np.float32) / 255.0
    planes = torch.from_numpy(source)
    if tuple(planes.shape[-2:]) != (res, res):
        resized = torch.empty((len(planes), res, res), dtype=torch.float32)
        for start in range(0, len(planes), 16):
            resized[start:start+16] = F.interpolate(
                planes[start:start+16, None], size=(res, res),
                mode='bilinear', align_corners=False)[:, 0]
        planes = resized
    x = planes[torch.from_numpy(inverse.reshape(-1, 3))]
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x





def rankpct(x):
    return rsna_rank01(x)

def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + '/*.dcm')
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    instance = getattr(h, 'InstanceNumber', None)
                    if instance is None:
                        rsna_event('raptor_order_index_fallback', file=str(f)); instance = len(recs)
                    pos = float(instance)
                if not np.isfinite(pos):
                    rsna_event('raptor_order_nonfinite_fallback', file=str(f)); pos = float(len(recs))
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception as exc:
                rsna_event('raptor_order_unreadable_skipped', file=str(f), error=f'{type(exc).__name__}: {exc}'); continue
        recs.sort(key=lambda x: (x[0], x[1]))
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (order_and_meta, read_px, mm_crop_resize)

def _pick_series_for_slot(rows, plane, fluid, used):
    """An unknown protocol flag is not false and must not reach int(NaN)."""
    def flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ('', 'nan', 'none', '<na>', 'unknown'):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ('true', 'yes', 'y', 't'):
                return 1
            if text in ('false', 'no', 'n', 'f'):
                return 0
            return None  # unknown code: neither false nor fatal; the plane-level fallback applies
        if not np.isfinite(number):
            return None
        if number not in (0.0, 1.0):
            return None
        return int(number)
    candidates = [r for r in rows if r['Anatomical_Plane'] == plane
                  and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if flag(r.get('Fluid_Sensitive')) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None



def find_test_root():
    from pathlib import Path
    for value in [os.environ.get('RSNA_COMP_ROOT', ''),
                  '/kaggle/input/competitions/rsna-knee-abnormality-detection',
                  '/kaggle/input/rsna-knee-abnormality-detection']:
        if value and (Path(value)/'test.csv').is_file():
            return value
    raise FileNotFoundError('explicit competition root absent')

def find_weight_file(fname):
    return str(_asset_find_asset(fname))
'''
_KE_NS = {'__name__': '_ke_raptor', 'RUN': RUN, 'coat_w': coat_w, '_asset_find_asset': _asset_find_asset, 'rsna_rank01':rsna_rank01, 'rsna_strict_load':rsna_strict_load, 'rsna_event':rsna_event, 'rsna_sha':rsna_sha}
exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)
from concurrent.futures import Future as _KeFuture
import threading as _ke_threading

_ke_base_make_reader = _KE_NS['_make_reader']
_ke_order_cache = {}
from collections import OrderedDict as _KeOrderedDict
_ke_pixel_cache = _KeOrderedDict()
_ke_order_cache_lock = _ke_threading.Lock()


def _ke_make_cached_reader():
    """Share immutable DICOM ordering metadata across Raptor views."""
    base_order, read_pixel, crop_resize = _ke_base_make_reader()

    def cached_order(series_dir):
        key = str(series_dir)
        owner = False
        with _ke_order_cache_lock:
            future = _ke_order_cache.get(key)
            if future is None:
                future = _KeFuture()
                _ke_order_cache[key] = future
                owner = True
        if owner:
            try:
                future.set_result(base_order(series_dir))
            except BaseException as error:
                future.set_exception(error)
                with _ke_order_cache_lock:
                    _ke_order_cache.pop(key, None)
                raise
        return future.result()

    def cached_pixel(path):
        global _ke_pixel_cache_bytes
        from collections import OrderedDict
        key=str(path)
        with _ke_order_cache_lock:
            got=_ke_pixel_cache.get(key)
            if got is not None:
                _ke_pixel_cache.move_to_end(key);return got
        value=read_pixel(path)
        if value.ndim!=2 or not _ke_np.isfinite(value).all():
            raise RuntimeError(f'Raptor invalid pixels {path}')
        value.setflags(write=False)
        with _ke_order_cache_lock:
            previous = _ke_pixel_cache.pop(key, None)
            if previous is not None:
                _ke_pixel_cache_bytes -= previous.nbytes
            _ke_pixel_cache[key] = value
            _ke_pixel_cache_bytes += value.nbytes
            while _ke_pixel_cache_bytes > (192 << 20):
                _, removed = _ke_pixel_cache.popitem(last=False)
                _ke_pixel_cache_bytes -= removed.nbytes
        return value
    return cached_order, cached_pixel, crop_resize


_KE_NS['_make_reader'] = _ke_make_cached_reader
if _ke_os.environ.get('RSNA_COMP_ROOT'):
    _KE_NS['find_test_root'] = lambda: _ke_os.environ['RSNA_COMP_ROOT']


def _ke_prepare_windows(arm, study_uid, series, series_root, reader):







    order_and_meta, read_px, _ = reader
    image_size = int(arm['img'])
    slots = list(arm['slots'])
    span_lo, span_hi = map(float, arm['span'])
    used, pools = set(), []
    rows = series.get(study_uid, [])
    for plane, fluid, count in slots:
        record = _KE_NS['_pick_series_for_slot'](rows, plane, fluid, used)
        if record is None:
            pools.append(None)
            continue
        used.add(record['SeriesInstanceUID'])
        files, median_spacing = order_and_meta(
            f"{series_root}/{study_uid}/{record['SeriesInstanceUID']}")
        if not files:
            rsna_event('raptor_acquisition_empty', study=str(study_uid), series=str(record['SeriesInstanceUID'])); pools.append(None); continue
        lo = int(len(files) * span_lo)
        hi = max(int(len(files) * span_hi) - 1, lo)
        original = _ke_np.linspace(lo, hi, count).round().astype(int)
        pools.append((files, median_spacing, lo, hi, original))
    capacities = [0 if p is None else p[3] - p[2] + 1 for p in pools]
    quotas = _dense_allocate(capacities, [s[2] for s in slots], 96)
    volume, sources = [], []
    import cv2
    for pool, quota in zip(pools, quotas):
        if pool is None or not quota:
            continue
        files, median_spacing, lo, hi, original = pool
        picks = lo + _dense_unique_linspace(hi - lo + 1, int(quota))
        arrays = {}
        # Decode union once. Failure is explicit, not an unreported 0.5 score.
        for i in sorted(set(original.tolist()) | set(picks.tolist())):
            arrays[i] = read_px(files[i][0])
        all_pixels = _ke_np.concatenate([arrays[int(i)].ravel() for i in original])
        low, high = _ke_np.percentile(all_pixels, [2.0, 98.0])
        for i in picks:
            path, spacing = files[int(i)]
            a = arrays[int(i)]
            a = _ke_np.clip((a-low)/(high-low+1e-6), 0, 1)
            spacing = spacing if spacing > 0 else median_spacing
            h, w = a.shape
            c = min(int(round(140.0/max(spacing, .001))), min(h, w))
            y, x = (h-c)//2, (w-c)//2
            a = cv2.resize(a[y:y+c, x:x+c], (image_size,image_size),
                           interpolation=cv2.INTER_AREA)
            volume.append((a*255).astype(_ke_np.uint8))
            sources.append(str(path))
    if len(sources) != len(set(sources)):
        raise RuntimeError('capacity-aware sampling selected duplicate source files')
    if len(volume) < 3:
        raise RuntimeError(f'{study_uid}: fewer than three unique source slices')
    volume = _ke_np.stack(volume)
    # Presence means an acquired source, not an intensity test. A genuinely
    # black acquired slice is not interchangeable with padding.
    mask = _ke_np.ones(len(volume), dtype=_ke_np.uint8)
    windows = _KE_NS['eval_windows'](
        volume, mask, k=len(volume)-2, res=int(arm['res']), norm=_KE_NS['NORM'])
    if len(windows) != min(96, sum(capacities))-2:
        raise RuntimeError('capacity-aware sampling Raptor cardinality drift')
    import hashlib, json
    signature = hashlib.sha256(json.dumps(sources, ensure_ascii=False,
                                separators=(',', ':')).encode()).hexdigest()
    audit = {'arm':arm['name'], 'uid':str(study_uid),
             'capacities':capacities, 'quotas':quotas.tolist(),
             'source_count':len(sources), 'source_list_sha256':signature,
             'windows':len(windows), 'duplicate_sources':0}
    identity = (arm['name'], str(study_uid))
    with _ke_input_lock:
        if identity in _ke_input_ids:
            raise RuntimeError(f'Duplicate Raptor preparation: {identity}')
        _ke_input_ids.add(identity)
        with _ke_input_audit_path.open('a') as handle:
            handle.write(json.dumps(audit, sort_keys=True, allow_nan=False) + '\n')
    return windows


def _ke_infer_input(model, xwins, device):
    """Bound backbone workspace; keep all window features for one head call."""
    torch = _KE_NS['torch']
    def _forward(_amp):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=_amp):
            features = [model.backbone(xwins[i:i+8].to(device))
                        for i in range(0,len(xwins),8)]
            feats = torch.cat(features,dim=0).unsqueeze(0)
            return torch.sigmoid(model.head(feats).float())[0].cpu().numpy()
    probabilities = _forward(True)
    if not _ke_np.isfinite(probabilities).all():
        rsna_event('raptor_fp16_nonfinite_retry_fp32')
        probabilities = _forward(False)
    if not _ke_np.isfinite(probabilities).all():
        raise RuntimeError('nonfinite Raptor prediction')
    return probabilities


def _ke_prefetched_windows(arm, test_ids, series, series_root, reader):
    """Bound host memory while decoding one study ahead of its GPU forward."""
    depth = int(_ke_os.environ.get('RSNA_RAPTOR_PREFETCH', '2'))
    if not 1 <= depth <= 4: raise ValueError('Raptor prefetch must be 1..4')
    with _KeThreadPool(max_workers=1) as executor:
        pending = {}
        submit_at = 0
        while submit_at < min(depth, len(test_ids)):
            pending[submit_at] = executor.submit(
                _ke_prepare_windows,
                arm,
                test_ids[submit_at],
                series,
                series_root,
                reader,
            )
            submit_at += 1
        for study_index, study_uid in enumerate(test_ids):
            future = pending.pop(study_index)
            if submit_at < len(test_ids):
                pending[submit_at] = executor.submit(
                    _ke_prepare_windows,
                    arm,
                    test_ids[submit_at],
                    series,
                    series_root,
                    reader,
                )
                submit_at += 1
            yield study_index, study_uid, future


def _ke_run_raptor_arms():
    """Run the four unchanged Raptor arms across exactly two GPUs."""
    torch = _KE_NS['torch']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"optimized Raptor requires exactly two GPUs, got {torch.cuda.device_count()}"
        )
    started = _ke_time.time()
    root = _KE_NS['find_test_root']()
    series_root = root + '/test_series'
    if not _ke_os.path.isdir(series_root):
        series_root = root + '/test_images'
    test = _ke_pd.read_csv(root + '/test.csv')
    test['StudyInstanceUID'] = test['StudyInstanceUID'].astype(str)
    test_ids = test['StudyInstanceUID'].tolist()
    globals()["_KE_TEACHER_IDS"] = list(test_ids)
    test_series = _ke_pd.read_csv(root + '/test_series.csv')
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    series = {
        key: frame.to_dict('records')
        for key, frame in test_series.groupby('StudyInstanceUID')
    }
    sample = root + '/sample_submission.csv'
    columns = ['StudyInstanceUID', *_KE_NS['LAB']]
    if _ke_os.path.exists(sample):
        columns = list(_ke_pd.read_csv(sample, nrows=1).columns)
    arms = list(_KE_NS['ARMS'])
    outputs = globals().setdefault("_KE_TEACHER_OUTPUTS", [
        _ke_np.full((len(test_ids), len(_KE_NS['LAB'])), np.nan, _ke_np.float32)
        for _ in arms
    ])
    print(
        f"[raptor-fast] {len(test_ids)} studies; balanced arm groups "
        f"cuda:0=[0,2], cuda:1=[1,3]",
        flush=True,
    )

    def run_single(arm_index, device):
        arm = arms[arm_index]
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](arm['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, arm['arch'], arm['res'], device
        )
        if int(resolution) != int(arm['res']):
            raise RuntimeError(
                f"{arm['name']} checkpoint resolution {resolution} != {arm['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            arm, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[arm_index][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(arm['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((arm['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def run_shared_maxspan(device):
        first, reverse = arms[0], arms[2]
        comparable = ('file', 'arch', 'res', 'img', 'slots', 'span', 'k_eval')
        if any(first[key] != reverse[key] for key in comparable):
            raise RuntimeError('MaxSpan forward/reverse arms no longer share preprocessing')
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](first['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, first['arch'], first['res'], device
        )
        if int(resolution) != int(first['res']):
            raise RuntimeError(
                f"MaxSpan checkpoint resolution {resolution} != {first['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            first, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[0][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                outputs[2][study_index] = _KE_NS['infer_probs'](
                    model, windows.flip(1).contiguous(), device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(first['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((first['name'], str(study_uid))); _ke_input_ids.add((reverse['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def gpu_zero():
        with torch.cuda.device(0):
            run_shared_maxspan(torch.device('cuda:0'))

    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))

    with _KeThreadPool(max_workers=2) as executor:
        workers = [executor.submit(gpu_zero), executor.submit(gpu_one)]
        for worker in workers:
            worker.result()

    _ke_np.savez_compressed('/kaggle/working/raptor_raw.npz',
        study_uids=_ke_np.asarray(test_ids), raw_probabilities=_ke_np.stack(outputs))
    for _ai, _arr in enumerate(outputs):
        _bad = ~_ke_np.isfinite(_arr).all(axis=1)
        if _bad.any():
            _fill = _ke_np.nanmean(_ke_np.where(_ke_np.isfinite(_arr), _arr, _ke_np.nan), axis=0) if (~_bad).any() else _ke_np.full(_arr.shape[1], .5, _arr.dtype)
            _arr[_bad] = _ke_np.where(_ke_np.isfinite(_fill), _fill, .5)
            rsna_event('raptor_neutral_fill', arm=str(arms[_ai]['name']), studies=int(_bad.sum()))
    rsna_finite(_ke_np.stack(outputs),'all four Raptor views',probability=True)
    weights = _ke_np.asarray([float(arm['w']) for arm in arms], _ke_np.float64)
    weights /= weights.sum()
    probability_blend = _ke_np.tensordot(
        weights,
        _ke_np.stack([_ke_np.clip(value, 0, 1) for value in outputs]),
        axes=(0, 0),
    )
    ranks = _KE_NS['rankpct'](probability_blend)
    rsna_finite(ranks,'Raptor ranked ensemble',probability=True)
    submission = _ke_pd.DataFrame(ranks.astype(_ke_np.float32), columns=_KE_NS['LAB'])
    submission.insert(0, 'StudyInstanceUID', test_ids)
    submission = submission[columns]
    if submission['StudyInstanceUID'].tolist() != test_ids:
        raise RuntimeError('Raptor study order drift')
    if not _ke_np.isfinite(submission[_KE_NS['LAB']].to_numpy()).all():
        raise RuntimeError('Raptor produced non-finite predictions')
    output = _KePath('/kaggle/working/_raptor.csv')
    submission.to_csv(output, index=False)
    print(
        f"[raptor-fast] wrote {output}; elapsed {_ke_time.time() - started:.1f}s",
        flush=True,
    )


_ke_input_lock = _ke_threading.Lock()
_ke_input_ids = set()
_ke_pixel_cache_bytes = 0
_ke_input_audit_path = _KePath('/kaggle/working/diagnostics/raptor_inputs.jsonl')
_ke_input_audit_path.unlink(missing_ok=True)
_KE_NS['infer_probs'] = _ke_infer_input
for _arm in _KE_NS['ARMS']:
    _arm['parent_k_eval'] = _arm['k_eval']
    _arm['k_eval'] = int(RUN['raptor_k_eval'] or _arm['parent_k_eval'])
rsna_phase('public_raptor', 'START')
_ke_run_raptor_arms()
rsna_phase('public_raptor', 'COMPLETE')
_ke_pixel_cache.clear()
_ke_order_cache.clear()
_ke_gc.collect()

# %%
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
