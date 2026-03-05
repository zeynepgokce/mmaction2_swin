# Eval config: Swin-Base / ASLCitizen100
#   Checkpoint from: Train_256 workdir
#   Test input:      Simulated 64×64 (native video → Resize(64) → bilinear 256)
_base_ = ['../swin_aslcitizen100_train256.py']

# Override test pipeline to simulate 64×64 source
test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=16, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),   # simulate 64×64
    dict(type='Resize', scale=(256, 256), keep_ratio=False), # bilinear 64→256
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
        ann_file='/arf/scratch/zgokce/data/ASL_Citizen/test_aslcitizen100_mm2.txt',
        data_prefix=dict(video='/arf/scratch/zgokce/data/ASL_Citizen/test'),
        pipeline=test_pipeline,
        test_mode=True))
