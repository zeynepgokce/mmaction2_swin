# Eval config: Swin-Base / WLASL100
#   Checkpoint from: Train_256 workdir
#   Test input:      64×64 source → bilinear resize to 256×256 → CenterCrop 224
_base_ = ['../swin_wlasl100_train256.py']

# Override test pipeline: 64×64 actual videos → bilinear 64→256
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),  # 64→256 bilinear
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Override test dataloader: point to 64×64 WLASL dataset
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/arf/scratch/zgokce/data/wlasl100_videos_64x64/test_wlasl100_mm2.txt',
        data_prefix=dict(
            video='/arf/scratch/zgokce/data/wlasl100_videos_64x64/test'),
        pipeline=test_pipeline,
        test_mode=True))
