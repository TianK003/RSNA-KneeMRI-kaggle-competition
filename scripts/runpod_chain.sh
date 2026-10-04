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
# 2026-10-04 (P-66):
# SEQ_ARMS=1: train the arms ONE AFTER ANOTHER on GPU 0, in the order given (an arm that fails does not stop the next). Use it
# when the arms do not fit one GPU together, or to bound the spend arm by arm.
# AUTO_STOP=1: `runpodctl stop pod $RUNPOD_POD_ID` when the job ends, success or failure -- GPU billing stops even if nobody is
# watching (the volume is kept; delete the pod by hand after the ship is confirmed).
# Before the chain: `mkdir -p /workspace/kaggle && ln -sfn /workspace/kaggle /kaggle` so /kaggle/working (every _last.pt)
# lives on the persistent volume and survives a stop (traps 46); put CACHE_ROOT on fast local storage, never on MooseFS.
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
WEIGHTS=(timm-coatnet-rmlp-1-rw-224 timm-coatnet-rmlp-2-rw-384 convnext-tiny-224-hf
         timm-resnet50-a1 timm-efficientnet-b0-ra timm-efficientnet-b3-ra2)   # the CNN line (P-64 / P-66)
# The cache scheme the manifests carry in their names (manifest_shard<k>_<scheme>.csv): c02 unless the c03 kernels are pulled.
CACHE_SCHEME="${CACHE_SCHEME:-$([ "$CACHE_PREFIX" = rsna-knee-cache3 ] && echo c03 || echo c02)}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
# The pod's own RUNPOD_API_KEY / RUNPOD_POD_ID live in PID 1's environment, not in an ssh session's: export them first
# (`tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | sed "s/^/export /"`). That pod-scoped key is
# accepted by the GraphQL API for its own pod; runpodctl 1.14 answers "Unauthorized" to it (2026-10-04), so it is a fallback.
self_stop() {
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-}" -H "Content-Type: application/json" \
       -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}" ||
  runpodctl stop pod "${RUNPOD_POD_ID:-}" || true
}
if [ "${AUTO_STOP:-0}" = 1 ]; then
  [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || { echo "!! AUTO_STOP=1 needs RUNPOD_API_KEY and RUNPOD_POD_ID"; exit 1; }
  trap 'rc=$?; log "job exit rc=$rc -- AUTO_STOP: stopping pod ${RUNPOD_POD_ID}"; self_stop' EXIT
fi
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

verify() {   # every blob named by a c02 / c03 manifest exists and holds exactly the manifest's rows for it
  python - "$CACHE_ROOT" "$CACHE_SCHEME" <<'EOF'
import glob, os, sys
import numpy as np, pandas as pd
root, scheme = sys.argv[1], sys.argv[2]
n_blob = n_bad = n_study = 0
for mpath in sorted(glob.glob(os.path.join(root, "*", f"manifest_shard*_{scheme}.csv"))):
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

log "kaggle auth check (the ship at the end needs it too)"
retry kaggle datasets files "$OWNER/rsna-knee-teacher-tables" > /dev/null

tpids=()
n_gpu=$(nvidia-smi -L | wc -l)
[ "$n_gpu" -ge 1 ] || n_gpu=1
set +e
for i in "${!ARMS[@]}"; do
  ARM="${ARMS[$i]}"
  gpu=$(( i % n_gpu ))                               # more arms than GPUs share one (a 1 x 5090 pod runs both)
  [ "${SEQ_ARMS:-0}" = 1 ] && gpu=0
  log "TRAIN $ARM on GPU $gpu (TEACHER_TABLES ${RSNA_TEACHER_TABLES:-()}, MIX ${RSNA_TEACHER_MIX:-0.5}) -> $WORK/job_train_$ARM.log"
  if [ "${SEQ_ARMS:-0}" = 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/runpod_bootstrap.sh" train "$ARM" > "$WORK/job_train_$ARM.log" 2>&1
    log "TRAIN $ARM ended rc=$?"
  else
    ( CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/runpod_bootstrap.sh" train "$ARM" > "$WORK/job_train_$ARM.log" 2>&1 ) &
    tpids+=($!)
  fi
done
for p in "${tpids[@]}"; do wait "$p"; done
set -e
# Ship only a finished member: a guard-stopped run also leaves a _best.pt (a mid-schedule EMA under ckpt_policy="last",
# traps 47). A ship can land inside the Kaggle token's 30-min post-expiry window (traps 20): retry for ~45 min.
shipped=0
for ARM in "${ARMS[@]}"; do
  ls -la "$WORK" | grep "$ARM" || true
  L="$WORK/train_$ARM.log"
  if [ -f "$WORK/${ARM}_fold0_best.pt" ] && grep -q "SWA of last" "$L" && ! grep -q "stopping: runtime guard" "$L"; then
    for t in 1 2 3 4 5 6 7 8 9 10; do
      log "SHIP $ARM (try $t)"
      if bash "$REPO/scripts/runpod_bootstrap.sh" ship "$ARM"; then shipped=$((shipped + 1)); break; fi
      sleep 300
    done
  else
    echo "!! $ARM not finished (no _best.pt, no 'SWA of last', or a runtime-guard stop) -- not shipping (tail of its log:)"
    tail -20 "$WORK/job_train_$ARM.log" || true
  fi
done
log "job done: shipped $shipped / ${#ARMS[@]}"
[ "$shipped" -eq "${#ARMS[@]}" ] || exit 4
