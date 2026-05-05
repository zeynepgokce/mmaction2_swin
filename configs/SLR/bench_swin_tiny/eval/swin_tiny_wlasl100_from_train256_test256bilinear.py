# Eval config: Swin-Tiny / WLASL100
#   Checkpoint from: Train_256 workdir
#   Test input:      64×64 source → bilinear resize to 256×256 → CenterCrop 224
_base_ = ['../swin_tiny_wlasl100_train256.py']

# Override test pipeline: bilinear-upscaled source
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Override test dataloader: point to bilinear-upscaled WLASL dataset
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/media/zeynep/SSD/phd/datasets/WLASL/wlasl100_videos_256x256_bilinear/test_wlasl100_mm2.txt',
        data_prefix=dict(
            video='/media/zeynep/SSD/phd/datasets/WLASL/wlasl100_videos_256x256_bilinear/test'),
        pipeline=test_pipeline,
        test_mode=True))
