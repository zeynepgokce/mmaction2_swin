# Eval config: Swin-Tiny / ASLCitizen100
#   Checkpoint from: Train_256 workdir
#   Test input:      64×64 actual source → bilinear resize to 256×256 → CenterCrop 224
_base_ = ['../swin_tiny_aslcitizen100_train256.py']

# Override test pipeline: 64→256 bilinear (force square, consistent with ASL source)
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# Override test dataloader: point to bilinear-upscaled ASLCitizen100 dataset
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/media/zeynep/SSD/phd/datasets/ASL_Citizen/subsets/ASLCitizen100_videos_256x256_bilinear/test_aslcitizen100_mm2.txt',  # noqa: E501
        data_prefix=dict(
            video='/media/zeynep/SSD/phd/datasets/ASL_Citizen/subsets/ASLCitizen100_videos_256x256_bilinear/test'),
        pipeline=test_pipeline,
        test_mode=True))
