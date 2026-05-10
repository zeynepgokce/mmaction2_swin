#!/bin/bash

PYTHON=/home/zeynep/anaconda3/envs/open-mmlab/bin/python
BASE_DIR=/home/zeynep/Thesis/code/mmaction2
CFG=$BASE_DIR/configs/SLR/bench_uniformer_base/eval
WDIR=$BASE_DIR/workdir/uniformer_base

run_test() {
  local CONFIG="$1"
  local TRAIN_WORKDIR="$2"
  local EVAL_DIR="$3"

  CKPT_PATH="$(find "$TRAIN_WORKDIR" -maxdepth 1 -type f -name 'best_acc_top1_epoch*.pth' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [ -z "$CKPT_PATH" ]; then
    echo "SKIP: checkpoint bulunamadı — $TRAIN_WORKDIR" >&2
    return 0
  fi

  mkdir -p "$EVAL_DIR"
  echo "=========================================="
  echo "CONFIG   : $CONFIG"
  echo "CKPT     : $CKPT_PATH"
  echo "EVAL_DIR : $EVAL_DIR"
  echo "=========================================="
  $PYTHON "$BASE_DIR/tools/test.py" "$CONFIG" "$CKPT_PATH" \
    --work-dir "$EVAL_DIR" \
    2>&1 | tee -a "$EVAL_DIR/eval.log"
}

# ═══════════════════════════════════════════════════════
#  aslcitizen100
# ═══════════════════════════════════════════════════════

run_test "$CFG/uniformer_base_aslcitizen100_from_train64_test64.py" \
         "$WDIR/aslcitizen100/train_64" \
         "$WDIR/aslcitizen100/eval_from_train64_test64"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_test256.py" \
         "$WDIR/aslcitizen100/train_256" \
         "$WDIR/aslcitizen100/eval_from_train256_test256"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_test256bilinear.py" \
         "$WDIR/aslcitizen100/train_256" \
         "$WDIR/aslcitizen100/eval_from_train256_test256bilinear"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_testSR.py" \
         "$WDIR/aslcitizen100/train_256" \
         "$WDIR/aslcitizen100/eval_from_train256_testSR"

run_test "$CFG/uniformer_base_aslcitizen100_from_train64_lr1e-5_test64.py" \
         "$WDIR/aslcitizen100/train_64_lr1e-5" \
         "$WDIR/aslcitizen100/eval_from_train64_lr1e-5_test64"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-5_test256.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-5" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-5_test256"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-5_test256bilinear.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-5" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-5_test256bilinear"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-5_testSR.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-5" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-5_testSR"

run_test "$CFG/uniformer_base_aslcitizen100_from_train64_lr1e-4_test64.py" \
         "$WDIR/aslcitizen100/train_64_lr1e-4" \
         "$WDIR/aslcitizen100/eval_from_train64_lr1e-4_test64"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-4_test256.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-4" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-4_test256"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-4_test256bilinear.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-4" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-4_test256bilinear"

run_test "$CFG/uniformer_base_aslcitizen100_from_train256_lr1e-4_testSR.py" \
         "$WDIR/aslcitizen100/train_256_lr1e-4" \
         "$WDIR/aslcitizen100/eval_from_train256_lr1e-4_testSR"

# ═══════════════════════════════════════════════════════
#  wlasl100
# ═══════════════════════════════════════════════════════

run_test "$CFG/uniformer_base_wlasl100_from_train64_test64.py" \
         "$WDIR/wlasl100/train_64" \
         "$WDIR/wlasl100/eval_from_train64_test64"

run_test "$CFG/uniformer_base_wlasl100_from_train256_test256.py" \
         "$WDIR/wlasl100/train_256" \
         "$WDIR/wlasl100/eval_from_train256_test256"

run_test "$CFG/uniformer_base_wlasl100_from_train256_test256bilinear.py" \
         "$WDIR/wlasl100/train_256" \
         "$WDIR/wlasl100/eval_from_train256_test256bilinear"

run_test "$CFG/uniformer_base_wlasl100_from_train256_testSR.py" \
         "$WDIR/wlasl100/train_256" \
         "$WDIR/wlasl100/eval_from_train256_testSR"

run_test "$CFG/uniformer_base_wlasl100_from_train64_lr1e-5_test64.py" \
         "$WDIR/wlasl100/train_64_lr1e-5" \
         "$WDIR/wlasl100/eval_from_train64_lr1e-5_test64"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-5_test256.py" \
         "$WDIR/wlasl100/train_256_lr1e-5" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-5_test256"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-5_test256bilinear.py" \
         "$WDIR/wlasl100/train_256_lr1e-5" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-5_test256bilinear"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-5_testSR.py" \
         "$WDIR/wlasl100/train_256_lr1e-5" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-5_testSR"

run_test "$CFG/uniformer_base_wlasl100_from_train64_lr1e-4_test64.py" \
         "$WDIR/wlasl100/train_64_lr1e-4" \
         "$WDIR/wlasl100/eval_from_train64_lr1e-4_test64"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-4_test256.py" \
         "$WDIR/wlasl100/train_256_lr1e-4" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-4_test256"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-4_test256bilinear.py" \
         "$WDIR/wlasl100/train_256_lr1e-4" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-4_test256bilinear"

run_test "$CFG/uniformer_base_wlasl100_from_train256_lr1e-4_testSR.py" \
         "$WDIR/wlasl100/train_256_lr1e-4" \
         "$WDIR/wlasl100/eval_from_train256_lr1e-4_testSR"

echo "All uniformer_base test runs completed."
