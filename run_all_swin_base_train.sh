#!/bin/bash
set -e

BASE_DIR=/home/zeynep/Thesis/code/mmaction2
WDIR=$BASE_DIR/workdir
PYTHON=/home/zeynep/anaconda3/envs/open-mmlab/bin/python

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

# ── WLASL100 ────────────────────────────────────────────────────────────────
run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_wlasl100_train64.py" \
  "$WDIR/swin_base/wlasl100/train_64"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_wlasl100_train64_lr1e-4.py" \
  "$WDIR/swin_base/wlasl100/train_64_lr1e-4"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_wlasl100_train256.py" \
  "$WDIR/swin_base/wlasl100/train_256"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_wlasl100_train256_lr1e-4.py" \
  "$WDIR/swin_base/wlasl100/train_256_lr1e-4"

# ── ASLCitizen100 ────────────────────────────────────────────────────────────
run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_aslcitizen100_train64.py" \
  "$WDIR/swin_base/aslcitizen100/train_64"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_aslcitizen100_train64_lr1e-4.py" \
  "$WDIR/swin_base/aslcitizen100/train_64_lr1e-4"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_aslcitizen100_train256.py" \
  "$WDIR/swin_base/aslcitizen100/train_256"

run_train \
  "$BASE_DIR/configs/SLR/bench_swin_base/swin_base_aslcitizen100_train256_lr1e-4.py" \
  "$WDIR/swin_base/aslcitizen100/train_256_lr1e-4"

echo "All swin_base training runs completed."
