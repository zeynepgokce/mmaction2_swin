#!/bin/bash
set -e

PYTHON=/home/zeynep/anaconda3/envs/open-mmlab/bin/python
BASE_DIR=/home/zeynep/Thesis/code/mmaction2
WDIR=$BASE_DIR/workdir

run_test() {
  local CONFIG="$1"
  local TRAIN_WORKDIR="$2"
  local EVAL_DIR="$3"

  CKPT_PATH="$(find "$TRAIN_WORKDIR" -type f -name 'best_acc_top1_epoch*.pth' \
    -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: No best checkpoint found in $TRAIN_WORKDIR — skipping." >&2
    return 1
  fi

  mkdir -p "$EVAL_DIR"
  echo "=========================================="
  echo "CONFIG    : $CONFIG"
  echo "CKPT      : $CKPT_PATH"
  echo "EVAL_DIR  : $EVAL_DIR"
  echo "=========================================="
  $PYTHON "$BASE_DIR/tools/test.py" "$CONFIG" "$CKPT_PATH" \
    --work-dir "$EVAL_DIR" \
    2>&1 | tee -a "$EVAL_DIR/eval.log"
}


# --- aslcitizen100 / train_64 → test64 ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_aslcitizen100_from_train64_test64.py" \
  "$WDIR/uniformer_v2_base/aslcitizen100/train_64" \
  "$WDIR/uniformer_v2_base/aslcitizen100/eval_from_train64_test64"

# --- aslcitizen100 / train_256 → test256 ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_aslcitizen100_from_train256_test256.py" \
  "$WDIR/uniformer_v2_base/aslcitizen100/train_256" \
  "$WDIR/uniformer_v2_base/aslcitizen100/eval_from_train256_test256"

# --- aslcitizen100 / train_256 → test256bilinear ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_aslcitizen100_from_train256_test256bilinear.py" \
  "$WDIR/uniformer_v2_base/aslcitizen100/train_256" \
  "$WDIR/uniformer_v2_base/aslcitizen100/eval_from_train256_test256bilinear"

# --- aslcitizen100 / train_256 → testSR ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_aslcitizen100_from_train256_testSR.py" \
  "$WDIR/uniformer_v2_base/aslcitizen100/train_256" \
  "$WDIR/uniformer_v2_base/aslcitizen100/eval_from_train256_testSR"

# --- wlasl100 / train_64 → test64 ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_wlasl100_from_train64_test64.py" \
  "$WDIR/uniformer_v2_base/wlasl100/train_64" \
  "$WDIR/uniformer_v2_base/wlasl100/eval_from_train64_test64"

# --- wlasl100 / train_256 → test256 ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_wlasl100_from_train256_test256.py" \
  "$WDIR/uniformer_v2_base/wlasl100/train_256" \
  "$WDIR/uniformer_v2_base/wlasl100/eval_from_train256_test256"

# --- wlasl100 / train_256 → test256bilinear ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_wlasl100_from_train256_test256bilinear.py" \
  "$WDIR/uniformer_v2_base/wlasl100/train_256" \
  "$WDIR/uniformer_v2_base/wlasl100/eval_from_train256_test256bilinear"

# --- wlasl100 / train_256 → testSR ---
run_test \
  "$BASE_DIR/configs/SLR/bench_uniformer_v2_base/eval/uniformer_v2_base_wlasl100_from_train256_testSR.py" \
  "$WDIR/uniformer_v2_base/wlasl100/train_256" \
  "$WDIR/uniformer_v2_base/wlasl100/eval_from_train256_testSR"

echo "All uniformer_v2_base test runs completed."
