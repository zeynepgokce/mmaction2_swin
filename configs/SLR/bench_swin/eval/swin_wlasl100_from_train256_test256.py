# Eval config: Swin-Base / WLASL100
#   Checkpoint from: Train_256 workdir
#   Test input:      256×256 source (native resolution, same pipeline as training)
# Usage: tools/test.py <this_config> <best_ckpt_from_train256>
_base_ = ['../swin_wlasl100_train256.py']
# No overrides needed; train config's test_pipeline and test_dataloader are correct.
