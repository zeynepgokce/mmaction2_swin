# Eval config: Swin-Small / WLASL100
#   Checkpoint from: Train_256 workdir
#   Test input:      256×256 source (native resolution, same pipeline as training)
_base_ = ['../swin_small_wlasl100_train256.py']
# No overrides needed; train config's test_pipeline and test_dataloader are correct.
