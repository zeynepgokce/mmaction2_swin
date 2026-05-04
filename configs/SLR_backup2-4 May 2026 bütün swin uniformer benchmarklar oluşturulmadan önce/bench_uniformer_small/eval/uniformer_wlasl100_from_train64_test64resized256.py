# Eval config: UniFormer-Small / WLASL100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      64×64 actual source → bilinear resize to 256×256 (matches training)
_base_ = ['../uniformer_wlasl100_train64_resize256.py']
# No overrides needed.
