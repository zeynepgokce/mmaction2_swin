#!/bin/bash
set -e

PYTHON=/home/zeynep/anaconda3/envs/open-mmlab/bin/python
BASE_DIR=/home/zeynep/Thesis/code/mmaction2
WDIR=$BASE_DIR/workdir

run_train() {
  local CONFIG="$1"
  local WORK_DIR="$2"
  mkdir -p "$WORK_DIR"
  echo "=========================================="
  echo "CONFIG  : $CONFIG"
  echo "WORK_DIR: $WORK_DIR"
  echo "=========================================="
  $PYTHON "$BASE_DIR/tools/train.py" "$CONFIG" \
    --work-dir "$WORK_DIR" \
    2>&1 | tee -a "$WORK_DIR/train.log"
}

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_small/swin_small_wlasl100_train64_lr1e-4.py" \
  "$WDIR/swin_small/wlasl100/train_64_lr1e-4"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_small/swin_small_wlasl100_train256.py" \
  "$WDIR/swin_small/wlasl100/train_256"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_small/swin_small_wlasl100_train256_lr1e-4.py" \
  "$WDIR/swin_small/wlasl100/train_256_lr1e-4"

echo "All swin_small training runs completed."
