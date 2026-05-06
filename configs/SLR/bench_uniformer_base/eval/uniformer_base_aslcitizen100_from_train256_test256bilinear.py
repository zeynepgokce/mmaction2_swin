_base_ = ['../uniformer_base_aslcitizen100_train256.py']

# Override test pipeline: bilinear-upscaled 64->256 input
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Override test dataloader: bilinear-upscaled dataset
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256_bilinear/test_aslcitizen100_mm2.txt',
        data_prefix=dict(video='/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256_bilinear/test'),
        pipeline=test_pipeline,
        test_mode=True))
