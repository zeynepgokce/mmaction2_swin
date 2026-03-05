# Eval config: UniFormerV2-Base / ASLCitizen100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      Simulated 64×64 → bilinear 256 (matches training)
_base_ = ['../uniformer_aslcitizen100_train64_resize256.py']
# No overrides needed.
