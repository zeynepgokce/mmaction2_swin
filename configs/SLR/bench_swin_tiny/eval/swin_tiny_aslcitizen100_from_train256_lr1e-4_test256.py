# Eval config: Swin-Tiny / ASLCitizen100 (lr=1e-4)
#   Checkpoint from: Train_256_lr1e-4 workdir
#   Test input:      256×256 source (native resolution, same pipeline as training)
_base_ = ['../swin_tiny_aslcitizen100_train256_lr1e-4.py']
# No overrides needed; train config's test_pipeline and test_dataloader are correct.
