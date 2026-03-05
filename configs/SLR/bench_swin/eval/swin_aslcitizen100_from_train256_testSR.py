# Eval config: Swin-Base / ASLCitizen100
#   Checkpoint from: Train_256 workdir
#   Test input:      SR 256×256 (super-resolved dataset)
# NOTE: Set SR data path below when ASLCitizen SR dataset is available.
#       Placeholder: /arf/scratch/zgokce/data/ASL_Citizen_SR
_base_ = ['../swin_aslcitizen100_train256.py']

_SR_ROOT = '/arf/scratch/zgokce/data/ASL_Citizen_SR'  # TODO: update when SR data is ready

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=_SR_ROOT + '/test_aslcitizen100_mm2.txt',
        data_prefix=dict(video=_SR_ROOT + '/test'),
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
