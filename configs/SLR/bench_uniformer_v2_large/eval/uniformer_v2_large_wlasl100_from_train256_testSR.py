_base_ = ['../uniformer_v2_large_wlasl100_train256.py']

# Override test dataloader: SR input
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/media/zeynep/SSD/phd/datasets/WLASL/wlasl100_videos_256x256_SR_flashvsr/test_wlasl100_mm2.txt',
        data_prefix=dict(video='/media/zeynep/SSD/phd/datasets/WLASL/wlasl100_videos_256x256_SR_flashvsr/test'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(type='UniformSample', clip_len=16, num_clips=1,
                 test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(256, 256), keep_ratio=False),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
