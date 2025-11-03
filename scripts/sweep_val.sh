#!/usr/bin/env bash
set -euo pipefail

# Minimal DL val-score sweep runner (sequential).
# Uses 100 epochs per run. Writes models/ + sidecars under a timestamped dir.

ROOT_DIR=${1:-data}
DEVICE=${DEVICE:-cuda}
W=${WORKERS:-4}
PF=${PREFETCH:-2}

STAMP=$(date +%Y%m%d_%H%M%S)
export VKB_MODELS_DIR="runs/models_${STAMP}"
mkdir -p "$VKB_MODELS_DIR"
echo "Models dir: $VKB_MODELS_DIR"

run() {
  BS=$1; WD=$2; DP=$3; DO=$4; AUG=$5; BR=$6; WR=$7; CW=$8
  echo "=== RUN bs=$BS wd=$WD dp=$DP do=$DO aug=$AUG br=$BR warp=$WR cw=$CW ==="
  .venv/bin/python train_frames.py \
    --data "$ROOT_DIR" --clf dl --device "$DEVICE" \
    --embed-model mobilenetv3_small_100 \
    --epochs 100 --batch-size "$BS" \
    --lr 1e-4 --wd "$WD" --drop-path "$DP" --dropout "$DO" \
    --aug "$AUG" --brightness "$BR" --warp "$WR" \
    --class-weights "$CW" \
    --workers "$W" --prefetch "$PF" --visdom-aug 0

  LATEST=$(readlink -f "$VKB_MODELS_DIR/LATEST" 2>/dev/null || true)
  META="${LATEST}.meta.json"
  if [[ -f "$META" ]]; then
    export META
    .venv/bin/python - <<'PY'
import json, os
meta_path = os.environ.get('META')
with open(meta_path) as f:
    m = json.load(f)
print('RESULT', 'val_acc=', m.get('val_acc'), 'best_epoch=', m.get('best_epoch'), 'model=', os.path.basename(meta_path[:-10]))
PY
  else
    echo "WARN: sidecar not found for $LATEST"
  fi
}

# Sweep (keep small; adjust if you want more)
run 96  0.0003 0.10 0.10 rot360 0.10 0.20 auto
run 96  0.0005 0.05 0.00 rot360 0.10 0.25 auto
run 128 0.0003 0.10 0.00 rot360 0.15 0.20 auto
run 96  0.0001 0.30 0.00 rot360 0.10 0.10 auto
run 96  0.0003 0.05 0.10 rot360 0.05 0.20 auto
run 96  0.0003 0.10 0.10 rot360 0.10 0.20 none

echo "Sweep complete. Models in $VKB_MODELS_DIR"
