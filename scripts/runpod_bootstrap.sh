#!/usr/bin/env bash
# RunPod (or any Ubuntu + CUDA box) runner for the RSNA knee training pipeline (P-24b).
#
# Training needs only: the c02 cache blobs (~36 GB, 4 Kaggle kernel outputs), the competition
# CSVs, the three LLM label tables, and the backbone weights -- never the DICOMs. This script
# lays them out under /kaggle/input EXACTLY as Kaggle mounts them, so src/kaggle_pipeline.py
# runs unchanged (ON_KAGGLE flips to true; print_input_layout() shows what it found), trains
# with RSNA_TRAIN_ONLY=1 (stop before the test-set gate: there is no test tree here), and ships
# the checkpoints + OOF csvs back as a private Kaggle Dataset that rsna-knee-infer mounts.
#
# Usage (inside the pod, as root, from a clone of the repo):
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...        # or copy ~/.kaggle/kaggle.json
#   bash scripts/runpod_bootstrap.sh setup           # pip, CSVs, labels, weights, caches (~40 min)
#   bash scripts/runpod_bootstrap.sh train v09h      # one arm from ARMS / ARM_V10C, fold 0
#   bash scripts/runpod_bootstrap.sh ship v09h       # checkpoints -> Dataset rsna-knee-ckpt-<arm>
#
# Pod sizing: coatnet_rmlp_2_rw_384 with train_windows=24 needs grad_checkpoint=True on a 24 GB
# card (4090 / A5000); an A100 40/80 GB runs it without. Disk: >= 60 GB local NVMe.
# Cache reads are the throughput floor (0.09 s/study on Kaggle's FUSE mount): keep the blobs on
# the pod's local disk, never on a network volume, and use RSNA_WORKERS=8.
set -euo pipefail

STEP="${1:-setup}"
ARM="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN=/kaggle/input
WORK=/kaggle/working
COMP=rsna-knee-abnormality-detection
OWNER=tiankljucanin
CACHE2=(rsna-knee-cache2-a rsna-knee-cache2-b rsna-knee-cache2-c rsna-knee-cache2-d)
LABELS=(pilkwang/rsna-knee-llm-labels stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56)
WEIGHTS=(timm-coatnet-rmlp-1-rw-224 timm-coatnet-rmlp-2-rw-384 convnext-tiny-224-hf)

log() { echo "[$(date +%H:%M:%S)] $*"; }

verify_count() {   # verify_count <dir> <min files> <min GB>  -- 429s return exit 0 (traps 14)
  local n gb
  n=$(find "$1" -type f | wc -l)
  gb=$(du -sb "$1" | awk '{printf "%.1f", $1/1e9}')
  log "  $1: $n files, ${gb} GB"
  if [ "$n" -lt "$2" ] || [ "$(echo "$gb < $3" | bc)" -eq 1 ]; then
    echo "!! $1 looks incomplete (need >= $2 files, >= $3 GB) -- Kaggle rate-limited the pull; re-run setup" >&2
    exit 2
  fi
}

case "$STEP" in
setup)
  log "python deps"
  pip install -q -r "$REPO/requirements-gpu.txt"
  python -c "import torch, timm, transformers, pydicom, safetensors; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
  mkdir -p "$IN/competitions/$COMP" "$IN/models/metaresearch/dinov2/pytorch/small/1" "$WORK"

  log "competition CSVs (no images)"
  for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
    [ -f "$IN/competitions/$COMP/$f" ] || kaggle competitions download -c "$COMP" -f "$f" -p "$IN/competitions/$COMP"
  done
  ( cd "$IN/competitions/$COMP" && for z in *.zip; do [ -f "$z" ] && unzip -oq "$z" && rm -f "$z"; done; true )

  log "LLM label tables"
  for d in "${LABELS[@]}"; do
    slug="${d#*/}"; [ -d "$IN/$slug" ] && continue
    kaggle datasets download -d "$d" -p "$IN/$slug" --unzip
  done

  log "backbone weights"
  kaggle models instances versions download metaresearch/dinov2/PyTorch/small/1 -p "$IN/models/metaresearch/dinov2/pytorch/small/1" --untar || true
  for w in "${WEIGHTS[@]}"; do
    [ -d "$IN/$w" ] && continue
    kaggle datasets download -d "$OWNER/$w" -p "$IN/$w" --unzip
  done

  log "c02 cache blobs (4 kernel outputs, ~36 GB; ~18 blobs + csvs each)"
  for k in "${CACHE2[@]}"; do
    mkdir -p "$IN/$k"
    kaggle kernels output "$OWNER/$k" -p "$IN/$k" || true
    verify_count "$IN/$k" 4 6.0          # ~18 blobs; the threshold catches a half-pulled shard
  done
  log "layout under $IN:"; find "$IN" -maxdepth 2 | head -40
  ;;

train)
  [ -n "$ARM" ] || { echo "usage: $0 train <arm>  (v08w | v09h | v10c)"; exit 1; }
  cd "$REPO"
  # Real run of ONE arm: the same sed the Kaggle real-run push uses (FORCE_SMOKE False, MODE
  # train); the arm is selected by RSNA_ARM inside the pipeline (ARMS or ARM_V10C), the session
  # guard is lifted to a long fold, and the loader gets RSNA_WORKERS workers. A crash-and-resume
  # works as on Kaggle: {arm}_fold0_last.pt in $WORK is picked up by the next run.
  mkdir -p artifacts
  sed -e 's/^FORCE_SMOKE = True/FORCE_SMOKE = False/' -e 's/^MODE = "auto"/MODE = "train"/' \
      src/kaggle_pipeline.py > artifacts/runpod_train.py
  export RSNA_ARM="$ARM" RSNA_TRAIN_ONLY=1 RSNA_WORKERS="${RSNA_WORKERS:-8}" \
         RSNA_RUNTIME_H="${RSNA_RUNTIME_H:-40}" PYTHONUTF8=1 PYTHONPATH=src
  log "training $ARM (log -> $WORK/train_$ARM.log); resume = re-run this command"
  python artifacts/runpod_train.py 2>&1 | tee "$WORK/train_$ARM.log"
  ;;

ship)
  [ -n "$ARM" ] || { echo "usage: $0 ship <arm>"; exit 1; }
  # Fresh ship dir every time (a stale one from an earlier fold-0 ship would carry old files into
  # the new Dataset version). Only the checkpointed-epoch files go: `fold[0-9]_` excludes the
  # per-epoch `_ep*_oof.csv` and the `_tta_oof.csv` that the old `fold*_oof.csv` glob swept in
  # (the fold-0 v09h ship uploaded 8 per-epoch csvs that way). Zero checkpoints = a loud failure,
  # not an empty Dataset version.
  OUT="$WORK/ship_$ARM"; rm -rf "$OUT"; mkdir -p "$OUT"
  cp "$WORK/${ARM}"_fold[0-9]_best.pt "$WORK/${ARM}"_fold[0-9]_oof.csv "$OUT/" 2>/dev/null || true
  n_ckpt=$(ls "$OUT"/*_best.pt 2>/dev/null | wc -l)
  [ "$n_ckpt" -ge 1 ] || { echo "!! ship $ARM: no ${ARM}_fold*_best.pt in $WORK -- nothing to publish"; exit 2; }
  echo "shipping $n_ckpt checkpoint(s) + $(ls "$OUT"/*_oof.csv 2>/dev/null | wc -l) oof csv(s)"
  ls -la "$OUT"
  cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "RSNA knee ckpt $ARM", "id": "$OWNER/rsna-knee-ckpt-$ARM", "licenses": [{"name": "other"}]}
EOF
  ( cd "$OUT" && kaggle datasets create -p . || kaggle datasets version -p . -m "update $ARM" )
  log "add $OWNER/rsna-knee-ckpt-$ARM to kaggle/rsna-knee-infer/kernel-metadata.json dataset_sources"
  ;;
*)
  echo "unknown step $STEP (setup | train <arm> | ship <arm>)"; exit 1 ;;
esac
