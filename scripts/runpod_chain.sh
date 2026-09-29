#!/usr/bin/env bash
# One unattended pod job: inputs -> four PARALLEL c02 cache pulls -> blob verification -> train <arm> -> ship <arm>.
# Written 2026-09-26 from the chained job the v09t run (2026-09-23, 41 min on a 4090) typed inline over SSH; the steps and
# their order are that job's. runpod_bootstrap.sh `setup` does the same inputs but pulls the four shards one after another
# (~4x slower), and a truncated blob is skipped rather than re-fetched by a re-pull (traps 38) -- hence the verification.
#
# Usage (on the pod, as root, from a clone of the repo, after copying a FRESH ~/.kaggle/credentials.json -- the OAuth token
# does not refresh itself and lasts ~3 h):
#   CACHE_ROOT=/workspace/cache RSNA_TEACHER_TABLES='("raptor_teacher",)' \
#     nohup bash scripts/runpod_chain.sh v09r > /workspace/job_v09r.log 2>&1 &
# CACHE_ROOT: where the 36 GB of blobs land (a local NVMe /workspace, or /dev/shm when it has > 40 GB free -- check
# `stat -f -c %T /workspace; df -h /dev/shm` first); /kaggle/input/<shard> is symlinked to it.
# CACHE_PREFIX (default rsna-knee-cache2): the four cache kernels <prefix>-a..d -- rsna-knee-cache3 for the c03 input
# (P-56 / P-60; ~51 GB, same manifest names). Several arms (`runpod_chain.sh v11n v11n2`) train IN PARALLEL, one per GPU
# (CUDA_VISIBLE_DEVICES = the arm's position modulo the pod's GPU count, so two arms share a 1-GPU pod), then ship
# one by one -- the cache is pulled once for both.
# EXPECT_TEACHER (default: every table named in RSNA_TEACHER_TABLES): each must be in the downloaded teacher-tables Dataset
# with >= EXPECT_ROWS (4349) rows; TEACHER_WAIT_MIN (default 0) = minutes to keep re-downloading until it is.
set -euo pipefail

ARMS=("$@")
[ "${#ARMS[@]}" -ge 1 ] || { echo "usage: $0 <arm> [<arm> ...]"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN=/kaggle/input
WORK=/kaggle/working
COMP=rsna-knee-abnormality-detection
OWNER=tiankljucanin
CACHE_ROOT="${CACHE_ROOT:-/workspace/cache}"
CACHE_PREFIX="${CACHE_PREFIX:-rsna-knee-cache2}"
CACHE2=("$CACHE_PREFIX-a" "$CACHE_PREFIX-b" "$CACHE_PREFIX-c" "$CACHE_PREFIX-d")
LABELS=(pilkwang/rsna-knee-llm-labels stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56
        tiankljucanin/rsna-knee-teacher-tables)
WEIGHTS=(timm-coatnet-rmlp-1-rw-224 timm-coatnet-rmlp-2-rw-384 convnext-tiny-224-hf)

log() { echo "[$(date +%H:%M:%S)] $*"; }
# 2026-09-29: a 429 (Too Many Requests) on train_series.csv, while the four cache pulls ran, killed the job under set -e
# (traps 45). Every small download now retries with back-off instead of aborting the chain.
retry() { local n=0; until "$@"; do n=$((n + 1)); [ "$n" -ge 6 ] && { echo "!! gave up after $n tries: $*"; return 1; }
          echo "  retry $n in $((30 * n)) s: $*"; sleep $((30 * n)); done; }

mkdir -p "$IN/competitions/$COMP" "$IN/models/metaresearch/dinov2/pytorch/small/1" "$WORK" "$CACHE_ROOT"
log "python deps (torch comes from the image)"
PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q -r "$REPO/requirements-gpu.txt"   # the runpod/pytorch image marks its Python externally managed (PEP 668)
python -c "import torch, timm, pydicom, safetensors; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
kaggle --version

log "cache shards: 4 parallel pulls -> $CACHE_ROOT"
pids=()
for k in "${CACHE2[@]}"; do
  mkdir -p "$CACHE_ROOT/$k"; ln -sfn "$CACHE_ROOT/$k" "$IN/$k"
  ( kaggle kernels output "$OWNER/$k" -p "$CACHE_ROOT/$k" > "$CACHE_ROOT/$k.pull.log" 2>&1 || true ) &
  pids+=($!)
done

log "competition CSVs"
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  [ -f "$IN/competitions/$COMP/$f" ] || retry kaggle competitions download -c "$COMP" -f "$f" -p "$IN/competitions/$COMP" > /dev/null
done
( cd "$IN/competitions/$COMP" && for z in *.zip; do [ -f "$z" ] && unzip -oq "$z" && rm -f "$z"; done; true )
ls "$IN/competitions/$COMP"

log "label + teacher tables (always re-downloaded: a stale teacher-tables copy would miss a new table)"
for d in "${LABELS[@]}"; do
  slug="${d#*/}"; rm -rf "${IN:?}/$slug"
  retry kaggle datasets download -d "$d" -p "$IN/$slug" --unzip > /dev/null
  echo "  $slug: $(find "$IN/$slug" -type f | wc -l) files"
done
# TEACHER_WAIT_MIN > 0: the pod may start before the table's Dataset version is published/processed -- re-download the
# teacher-tables Dataset once a minute until every expected table is there (the cache pulls keep running meanwhile).
EXPECT_TEACHER="${EXPECT_TEACHER:-$(echo "${RSNA_TEACHER_TABLES:-}" | tr -d '()"'"'"' ' | tr ',' ' ')}"
EXPECT_ROWS="${EXPECT_ROWS:-4349}"
waited=0
while :; do
  missing=""
  for t in $EXPECT_TEACHER; do
    f="$IN/rsna-knee-teacher-tables/$t.csv"
    if [ ! -f "$f" ] || [ "$(($(wc -l < "$f") - 1))" -lt "$EXPECT_ROWS" ]; then missing="$missing $t"; fi
  done
  [ -z "$missing" ] && break
  if [ "$waited" -ge "${TEACHER_WAIT_MIN:-0}" ]; then
    echo "!! teacher table(s)$missing missing or < $EXPECT_ROWS rows in the downloaded Dataset -- publish first"
    ls -la "$IN/rsna-knee-teacher-tables"; exit 3
  fi
  sleep 60; waited=$((waited + 1))
  rm -rf "$IN/rsna-knee-teacher-tables"
  kaggle datasets download -d tiankljucanin/rsna-knee-teacher-tables -p "$IN/rsna-knee-teacher-tables" --unzip > /dev/null 2>&1 || true
done
for t in $EXPECT_TEACHER; do
  echo "  $t.csv: $(($(wc -l < "$IN/rsna-knee-teacher-tables/$t.csv") - 1)) rows (waited ${waited} min)"
done

log "backbone weights"
kaggle models instances versions download metaresearch/dinov2/PyTorch/small/1 -p "$IN/models/metaresearch/dinov2/pytorch/small/1" --untar > /dev/null || true
for w in "${WEIGHTS[@]}"; do
  [ -d "$IN/$w" ] || retry kaggle datasets download -d "$OWNER/$w" -p "$IN/$w" --unzip > /dev/null
  echo "  $w: $(find "$IN/$w" -type f | wc -l) files"
done

log "waiting for the cache pulls"
for p in "${pids[@]}"; do wait "$p" || true; done

verify() {   # every blob named by a c02 manifest exists and holds exactly the manifest's rows for it
  python - "$CACHE_ROOT" <<'EOF'
import glob, os, sys
import numpy as np, pandas as pd
root = sys.argv[1]
n_blob = n_bad = n_study = 0
for mpath in sorted(glob.glob(os.path.join(root, "*", "manifest_shard*_c02.csv"))):
    m = pd.read_csv(mpath)
    m = m[m.get("cached", 1) == 1]
    arr_dir = os.path.join(os.path.dirname(mpath), str(m.cache_version.iloc[0]))
    for blob, g in m.groupby("blob"):
        n_blob += 1
        p = os.path.join(arr_dir, str(blob))
        try:
            n = np.load(p, mmap_mode="r").shape[0]
            ok = n == len(g) and int(g.row.max()) < n
        except Exception as e:                      # missing or truncated: np.load raises
            ok, n = False, repr(e)[:80]
        if not ok:
            n_bad += 1
            print(f"  BAD {p}: {n} vs {len(g)} manifest rows")
            if os.path.exists(p):
                os.remove(p)                        # a re-pull skips an existing file (traps 38)
    n_study += len(m)
print(f"blobs {n_blob}, bad {n_bad}, studies {n_study}")
sys.exit(0 if n_bad == 0 and n_blob > 0 and n_study == 4407 else 1)
EOF
}
for attempt in 1 2 3; do
  log "verify blobs (attempt $attempt)"
  if verify; then break; fi
  [ "$attempt" -lt 3 ] || { echo "!! cache still incomplete after 3 pulls"; exit 2; }
  for k in "${CACHE2[@]}"; do kaggle kernels output "$OWNER/$k" -p "$CACHE_ROOT/$k" > /dev/null 2>&1 || true; done
done
du -sh "${CACHE2[@]/#/$CACHE_ROOT/}"

tpids=()
n_gpu=$(nvidia-smi -L | wc -l)
[ "$n_gpu" -ge 1 ] || n_gpu=1
for i in "${!ARMS[@]}"; do
  ARM="${ARMS[$i]}"
  gpu=$(( i % n_gpu ))                               # more arms than GPUs share one (a 1 x 5090 pod runs both)
  log "TRAIN $ARM on GPU $gpu (TEACHER_TABLES ${RSNA_TEACHER_TABLES:-()}) -> $WORK/job_train_$ARM.log"
  ( CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/runpod_bootstrap.sh" train "$ARM" > "$WORK/job_train_$ARM.log" 2>&1 ) &
  tpids+=($!)
done
set +e
for p in "${tpids[@]}"; do wait "$p"; done
set -e
shipped=0
for ARM in "${ARMS[@]}"; do
  ls -la "$WORK" | grep "$ARM" || true
  if [ -f "$WORK/${ARM}_fold0_best.pt" ]; then
    log "SHIP $ARM"
    bash "$REPO/scripts/runpod_bootstrap.sh" ship "$ARM" && shipped=$((shipped + 1))
  else
    echo "!! no ${ARM}_fold0_best.pt -- not shipping $ARM (tail of its log:)"; tail -20 "$WORK/job_train_$ARM.log" || true
  fi
done
log "job done: shipped $shipped / ${#ARMS[@]}"
[ "$shipped" -eq "${#ARMS[@]}" ] || exit 4
