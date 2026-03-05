# Eval config: Swin-Base / WLASL100
#   Checkpoint from: Train_64_resize256 workdir
#   Test input:      64×64 source → bilinear resize 256 → CenterCrop 224
# Usage: tools/test.py <this_config> <best_ckpt_from_train64>
_base_ = ['../swin_wlasl100_train64_resize256.py']
# No overrides needed; train64 config's test_pipeline and test_dataloader
# already use 64×64 source data with bilinear 64→256 resize.
