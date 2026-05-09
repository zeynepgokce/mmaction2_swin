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
  "/home/zeynep/Thesis/code/mmaction2/configs/SLR/bench_uniformer_v2_base/uniformer_v2_base_wlasl100_train256.py" \
  "/home/zeynep/Thesis/code/mmaction2/workdir/uniformer_v2_base/wlasl100/train_256"

echo "All uniformer_v2_base training runs completed."
