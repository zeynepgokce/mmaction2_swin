#!/bin/bash
set -e

BASE_DIR=/home/zeynep/Thesis/code/mmaction2
WDIR=$BASE_DIR/workdir
PYTHON=/home/zeynep/anaconda3/envs/open-mmlab/bin/python

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

EVAL="$BASE_DIR/configs/SLR/bench_swin_large/eval"

# ── WLASL100 / train_256 ────────────────────────────────────────────────────
run_test "$EVAL/swin_large_wlasl100_from_train256_test256.py"         "$WDIR/swin_large/wlasl100/train_256"       "$WDIR/swin_large/wlasl100/eval_from_train256_test256"
run_test "$EVAL/swin_large_wlasl100_from_train256_testSR.py"          "$WDIR/swin_large/wlasl100/train_256"       "$WDIR/swin_large/wlasl100/eval_from_train256_testSR"
run_test "$EVAL/swin_large_wlasl100_from_train256_test256bilinear.py" "$WDIR/swin_large/wlasl100/train_256"       "$WDIR/swin_large/wlasl100/eval_from_train256_test256bilinear"

# ── WLASL100 / train_256_lr1e-4 ─────────────────────────────────────────────
run_test "$EVAL/swin_large_wlasl100_from_train256_lr1e-4_test256.py"         "$WDIR/swin_large/wlasl100/train_256_lr1e-4" "$WDIR/swin_large/wlasl100/eval_from_train256_lr1e-4_test256"
run_test "$EVAL/swin_large_wlasl100_from_train256_lr1e-4_testSR.py"          "$WDIR/swin_large/wlasl100/train_256_lr1e-4" "$WDIR/swin_large/wlasl100/eval_from_train256_lr1e-4_testSR"
run_test "$EVAL/swin_large_wlasl100_from_train256_lr1e-4_test256bilinear.py" "$WDIR/swin_large/wlasl100/train_256_lr1e-4" "$WDIR/swin_large/wlasl100/eval_from_train256_lr1e-4_test256bilinear"

# ── WLASL100 / train_64 ─────────────────────────────────────────────────────
run_test "$EVAL/swin_large_wlasl100_from_train64_test64.py"       "$WDIR/swin_large/wlasl100/train_64"       "$WDIR/swin_large/wlasl100/eval_from_train64_test64"

# ── WLASL100 / train_64_lr1e-4 ──────────────────────────────────────────────
run_test "$EVAL/swin_large_wlasl100_from_train64_lr1e-4_test64.py" "$WDIR/swin_large/wlasl100/train_64_lr1e-4" "$WDIR/swin_large/wlasl100/eval_from_train64_lr1e-4_test64"

# ── ASLCitizen100 / train_256 ───────────────────────────────────────────────
run_test "$EVAL/swin_large_aslcitizen100_from_train256_test256.py"         "$WDIR/swin_large/aslcitizen100/train_256"       "$WDIR/swin_large/aslcitizen100/eval_from_train256_test256"
run_test "$EVAL/swin_large_aslcitizen100_from_train256_testSR.py"          "$WDIR/swin_large/aslcitizen100/train_256"       "$WDIR/swin_large/aslcitizen100/eval_from_train256_testSR"
run_test "$EVAL/swin_large_aslcitizen100_from_train256_test256bilinear.py" "$WDIR/swin_large/aslcitizen100/train_256"       "$WDIR/swin_large/aslcitizen100/eval_from_train256_test256bilinear"

# ── ASLCitizen100 / train_256_lr1e-4 ────────────────────────────────────────
run_test "$EVAL/swin_large_aslcitizen100_from_train256_lr1e-4_test256.py"         "$WDIR/swin_large/aslcitizen100/train_256_lr1e-4" "$WDIR/swin_large/aslcitizen100/eval_from_train256_lr1e-4_test256"
run_test "$EVAL/swin_large_aslcitizen100_from_train256_lr1e-4_testSR.py"          "$WDIR/swin_large/aslcitizen100/train_256_lr1e-4" "$WDIR/swin_large/aslcitizen100/eval_from_train256_lr1e-4_testSR"
run_test "$EVAL/swin_large_aslcitizen100_from_train256_lr1e-4_test256bilinear.py" "$WDIR/swin_large/aslcitizen100/train_256_lr1e-4" "$WDIR/swin_large/aslcitizen100/eval_from_train256_lr1e-4_test256bilinear"

# ── ASLCitizen100 / train_64 ─────────────────────────────────────────────────
run_test "$EVAL/swin_large_aslcitizen100_from_train64_test64.py"       "$WDIR/swin_large/aslcitizen100/train_64"       "$WDIR/swin_large/aslcitizen100/eval_from_train64_test64"

# ── ASLCitizen100 / train_64_lr1e-4 ──────────────────────────────────────────
run_test "$EVAL/swin_large_aslcitizen100_from_train64_lr1e-4_test64.py" "$WDIR/swin_large/aslcitizen100/train_64_lr1e-4" "$WDIR/swin_large/aslcitizen100/eval_from_train64_lr1e-4_test64"

echo "All swin_large test runs completed."
