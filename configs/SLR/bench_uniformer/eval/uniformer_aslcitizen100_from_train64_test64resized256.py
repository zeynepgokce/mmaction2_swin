# Eval config: UniFormerV2-Base / ASLCitizen100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      64×64 actual source → bilinear resize to 256×256 → CenterCrop 224
_base_ = ['../uniformer_aslcitizen100_train64_resize256.py']
# No overrides needed.
