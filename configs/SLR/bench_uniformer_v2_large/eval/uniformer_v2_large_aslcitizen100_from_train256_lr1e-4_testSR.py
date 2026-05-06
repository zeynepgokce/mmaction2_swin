_base_ = ['../uniformer_v2_large_aslcitizen100_train256_lr1e-4.py']

# Override test dataloader: SR input
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256_SR_flashvsr/test_aslcitizen100_mm2.txt',
        data_prefix=dict(video='/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256_SR_flashvsr/test'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(type='UniformSample', clip_len=8, num_clips=1,
                 test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
