# Eval config: Swin-Base / ASLCitizen100
#   Checkpoint from: Train_256 workdir
#   Test input:      256×256 (native resolution, no artificial downscale)
_base_ = ['../swin_aslcitizen100_train256.py']
# No overrides needed.
