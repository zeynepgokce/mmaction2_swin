_base_ = ['../uniformer_base_aslcitizen100_train256.py']

test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/media/zeynep/SSD/phd/datasets/ASL_Citizen/subsets/ASLCitizen100_videos_256x256/test_aslcitizen100_mm2.txt',
        data_prefix=dict(video='/media/zeynep/SSD/phd/datasets/ASL_Citizen/subsets/ASLCitizen100_videos_256x256/test'),
        pipeline=test_pipeline,
        test_mode=True))
