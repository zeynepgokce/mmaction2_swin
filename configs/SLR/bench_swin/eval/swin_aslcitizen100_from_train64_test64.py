# Eval config: Swin-Base / ASLCitizen100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      Simulated 64×64 → bilinear 256 (matches training distribution)
_base_ = ['../swin_aslcitizen100_train64_resize256.py']
# No overrides needed; train64 config's test section already uses 64×64 pipeline.
