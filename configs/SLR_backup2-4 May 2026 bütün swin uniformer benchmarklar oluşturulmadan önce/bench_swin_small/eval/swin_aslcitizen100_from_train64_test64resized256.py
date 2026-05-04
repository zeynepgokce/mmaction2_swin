# Eval config: Swin-Small / ASLCitizen100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      64×64 actual source → bilinear resize to 256×256 (matches training)
_base_ = ['../swin_aslcitizen100_train64_resize256.py']
# No overrides needed.
