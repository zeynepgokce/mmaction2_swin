# Eval config: Swin-Base / WLASL100
#   Checkpoint from: Train_256 workdir
#   Test input:      SR 256×256 (super-resolved from 64×64 using Vimeo90K-BD)
#                    SR videos are already 256×256 → same pipeline as test256
_base_ = ['../swin_wlasl100_train256.py']

# Override test dataloader only: SR source, same test_pipeline as train256
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='/arf/scratch/zgokce/data/wlasl100_videos_64x64_SR_vimeo90k_bd/test_wlasl100_mm2.txt',  # noqa: E501
        data_prefix=dict(
            video='/arf/scratch/zgokce/data/wlasl100_videos_64x64_SR_vimeo90k_bd/test'),
        pipeline=[
            dict(type='DecordInit', io_backend='disk'),
            dict(type='UniformSample', clip_len=16, num_clips=1,
                 test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
