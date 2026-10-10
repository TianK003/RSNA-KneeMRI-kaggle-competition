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
# 2026-10-06 (P-69, critic): with SEQ_ARMS=1 each arm SHIPS AS SOON AS IT FINISHES, so a stop during the next arm cannot lose
# it (it used to ship every arm at the end). SHIP_TRIES / SHIP_WAIT_S (default 10 x 300 s) bound the ship retries.
# MAX_POD_H > 0 (with SEQ_ARMS=1): start the next arm only if (now - POD_T0) + (the last arm's training time x 1.05) + 0.1 h
# <= MAX_POD_H; otherwise the remaining arms are skipped (logged, not a failure). POD_T0 = the pod's creation time in epoch
# seconds (default: when this job started). Pair it with scripts/runpod_stopper.sh, the hard deadline on the pod itself.
# Before the chain: `mkdir -p /workspace/kaggle && ln -sfn /workspace/kaggle /kaggle` so /kaggle/working (every _last.pt)
# lives on the persistent volume and survives a stop (traps 46); put CACHE_ROOT on fast local storage, never on MooseFS.
# 2026-10-08 (P-81): OAI=1 also builds the OAI cache shard ON THE POD -- the OAI data never goes to Kaggle or GitHub. It needs
# three files copied to the pod by hand (scp; all gitignored): $REPO/.env (NDA_USERNAME / NDA_PASSWORD for nda-tools),
# $REPO/data/oai/nda_pkg_meta/image03.txt and $REPO/artifacts/oai/oai_targets.csv. The ~58 GB of chosen series (7,196 tarballs,
# NDA package OAI_PACKAGE, default 1249779) download in parallel with the cache pulls into OAI_RAW (default /workspace/oai_raw);
# src/build_oai_cache.py then writes shard 90 (~36 GB) to $IN/rsna-knee-oai-cache, deleting each knee's tarballs once it is in a
# blob, and RSNA_OAI_TARGETS is exported for the arms (an arm without oai=True drops every OAI study: apply_oai). Disk: put
# /workspace on >= 200 GB (c03 51 GB + OAI cache 36 GB + the not-yet-built tarballs).
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
         timm-resnet50-a1 timm-efficientnet-b0-ra timm-efficientnet-b3-ra2   # the CNN line (P-64 / P-66)
         timm-convnext-tiny-in12k)                                           # P-69 v15c (2026-10-06)
# The scheme suffix in the manifest names (manifest_shard<k>_<scheme>.csv). It is c02 for the c03 cache too: c03 is the c02
# scheme with denser slot budgets (cache version c02_p336_b24-24-24-14-8-8_..., written by the same code path). Verified on
# the pod 2026-10-04 -- a "_c03" glob found 0 blobs.
CACHE_SCHEME="${CACHE_SCHEME:-c02}"

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

if [ "${OAI:-0}" = 1 ]; then
  for f in .env data/oai/nda_pkg_meta/image03.txt artifacts/oai/oai_targets.csv; do
    [ -f "$REPO/$f" ] || { echo "!! OAI=1 needs $REPO/$f (copy it to the pod by hand; it is gitignored)"; exit 5; }
  done
  PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q nda-tools
  OAI_RAW="${OAI_RAW:-/workspace/oai_raw}"
  mkdir -p "$OAI_RAW"
  log "OAI: the chosen series -> $OAI_RAW (nda-tools, in the background; log /workspace/oai_download.log)"
  # The NDA password must not outlive the download on the pod's disk (critic, 10-08): .env is shredded as soon as nda-tools
  # returns, success or failure.
  ( cd "$REPO" && python src/build_oai_cache.py --links /workspace/oai_links.txt \
      && python scripts/nda_run.py .env -dp "${OAI_PACKAGE:-1249779}" -t /workspace/oai_links.txt -d "$OAI_RAW" -wt 16
    rc=$?; shred -u "$REPO/.env" 2>/dev/null || rm -f "$REPO/.env"; exit $rc
  ) > /workspace/oai_download.log 2>&1 &
  oai_pid=$!
fi

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

if [ "${OAI:-0}" = 1 ]; then
  log "OAI: waiting for the download"
  wait "$oai_pid" || { echo "!! OAI download failed:"; tail -20 /workspace/oai_download.log; exit 5; }
  echo "  $(find "$OAI_RAW" -name '*.tar.gz' | wc -l) tarballs, $(du -sh "$OAI_RAW" | cut -f1)"
  log "OAI: cache shard 90 -> $IN/rsna-knee-oai-cache"
  # nproc - 8 workers (at least 4): the training loader uses 8 (critic, 10-08).
  ( cd "$REPO" && python src/build_oai_cache.py --build "$OAI_RAW" --out "$IN/rsna-knee-oai-cache" \
      --workers "$(( $(nproc) > 12 ? $(nproc) - 8 : 4 ))" --delete-tars )
  n_oai=$(python -c "import pandas as pd; print(int(pd.read_csv('$IN/rsna-knee-oai-cache/manifest_shard90_oai.csv').cached.sum()))")
  [ "$n_oai" -ge 2300 ] || { echo "!! the OAI shard holds only $n_oai knees (expected ~2,398)"; exit 5; }
  export RSNA_OAI_TARGETS="$REPO/artifacts/oai/oai_targets.csv"
  echo "  OAI shard: $n_oai knees; RSNA_OAI_TARGETS=$RSNA_OAI_TARGETS"
  # OAI_MAX_PREP_H (default 1.2): the preparation must end this long after POD_T0 (pass POD_T0 = the pod's creation time),
  # or the job stops before training -- a slow download would push the training into the stopper's deadline and lose it
  # mid-SWA (critic, 10-08).
  prep_s=$(( $(date +%s) - ${POD_T0:-$(date +%s)} ))
  max_prep_s=$(awk "BEGIN { print int(${OAI_MAX_PREP_H:-1.2} * 3600) }")
  [ "$prep_s" -le "$max_prep_s" ] || { echo "!! OAI preparation took $((prep_s / 60)) min > OAI_MAX_PREP_H -- not training"; exit 6; }
  echo "  preparation done $((prep_s / 60)) min after POD_T0 (limit $((max_prep_s / 60)) min)"
fi

log "kaggle auth check (the ship at the end needs it too)"
# 2026-10-10: a warning, not an exit -- the check can land in the token's 30-min post-expiry window (traps 20), and an exit here
# would throw away the whole preparation; the ship retries SHIP_TRIES x SHIP_WAIT_S on its own.
retry kaggle datasets files "$OWNER/rsna-knee-teacher-tables" > /dev/null ||
  log "!! kaggle auth check failed (token window?) -- training anyway; the ship retries"

# Ship only a finished member: a guard-stopped run also leaves a _best.pt (a mid-schedule EMA under ckpt_policy="last",
# traps 47). A ship can land inside the Kaggle token's 30-min post-expiry window (traps 20): retry SHIP_TRIES x SHIP_WAIT_S.
SHIP_TRIES="${SHIP_TRIES:-10}"
SHIP_WAIT_S="${SHIP_WAIT_S:-300}"
shipped=0
skipped=0
ship_arm() {
  local ARM="$1" L="$WORK/train_$1.log" t
  ls -la "$WORK" | grep "$ARM" || true
  if [ -f "$WORK/${ARM}_fold0_best.pt" ] && grep -q "SWA of last" "$L" && ! grep -q "stopping: runtime guard" "$L"; then
    for t in $(seq 1 "$SHIP_TRIES"); do
      log "SHIP $ARM (try $t of $SHIP_TRIES)"
      if bash "$REPO/scripts/runpod_bootstrap.sh" ship "$ARM"; then shipped=$((shipped + 1)); log "SHIPPED $ARM"; return 0; fi
      [ "$t" -lt "$SHIP_TRIES" ] && sleep "$SHIP_WAIT_S"
    done
    echo "!! $ARM: the ship failed $SHIP_TRIES times"
  else
    echo "!! $ARM not finished (no _best.pt, no 'SWA of last', or a runtime-guard stop) -- not shipping (tail of its log:)"
    tail -20 "$WORK/job_train_$ARM.log" || true
  fi
  return 1
}

POD_T0="${POD_T0:-$(date +%s)}"
MAX_POD_S=$(awk "BEGIN { print int(${MAX_POD_H:-0} * 3600) }")
tpids=()
n_gpu=$(nvidia-smi -L | wc -l)
[ "$n_gpu" -ge 1 ] || n_gpu=1
last_train_s=0
set +e
for i in "${!ARMS[@]}"; do
  ARM="${ARMS[$i]}"
  gpu=$(( i % n_gpu ))                               # more arms than GPUs share one (a 1 x 5090 pod runs both)
  [ "${SEQ_ARMS:-0}" = 1 ] && gpu=0
  if [ "${SEQ_ARMS:-0}" = 1 ] && [ "$MAX_POD_S" -gt 0 ] && [ "$last_train_s" -gt 0 ]; then
    proj=$(( $(date +%s) - POD_T0 + last_train_s * 105 / 100 + 360 ))
    if [ "$proj" -gt "$MAX_POD_S" ]; then
      log "SKIP ${ARMS[*]:$i}: projected pod time $((proj / 60)) min > MAX_POD_H ${MAX_POD_H} h (last arm trained $((last_train_s / 60)) min)"
      skipped=$(( ${#ARMS[@]} - i ))
      break
    fi
    log "next arm fits: projected pod time $((proj / 60)) min <= MAX_POD_H ${MAX_POD_H} h"
  fi
  log "TRAIN $ARM on GPU $gpu (TEACHER_TABLES ${RSNA_TEACHER_TABLES:-()}, MIX ${RSNA_TEACHER_MIX:-0.5}) -> $WORK/job_train_$ARM.log"
  if [ "${SEQ_ARMS:-0}" = 1 ]; then
    t_arm=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/runpod_bootstrap.sh" train "$ARM" > "$WORK/job_train_$ARM.log" 2>&1
    log "TRAIN $ARM ended rc=$?"
    last_train_s=$(( $(date +%s) - t_arm ))
    ship_arm "$ARM"                                  # at once: a stop during the next arm cannot lose this one
  else
    ( CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/runpod_bootstrap.sh" train "$ARM" > "$WORK/job_train_$ARM.log" 2>&1 ) &
    tpids+=($!)
  fi
done
for p in "${tpids[@]}"; do wait "$p"; done
if [ "${SEQ_ARMS:-0}" != 1 ]; then
  for ARM in "${ARMS[@]}"; do ship_arm "$ARM"; done
fi
set -e
log "job done: shipped $shipped / ${#ARMS[@]} (skipped by MAX_POD_H: $skipped)"
[ $(( shipped + skipped )) -eq "${#ARMS[@]}" ] || exit 4
