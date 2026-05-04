# Eval config: Swin-Tiny / WLASL100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      64×64 source → bilinear resize 256 → CenterCrop 224
_base_ = ['../swin_tiny_wlasl100_train64.py']
# No overrides needed; train64 config's test_pipeline and test_dataloader
# already use 64×64 source data with bilinear 64→256 resize.
