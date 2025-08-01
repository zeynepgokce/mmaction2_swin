_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[20, 21, 22, 23],
        n_layers=4,
        n_dim=1024,
        n_head=16,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5]),
    cls_head=dict(
        type='TimeSformerHead',
        dropout_ratio=0.5,
        num_classes=100,
        in_channels=1024,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'RawframeDataset'
dataset_root ="/arf/scratch/zgokce/data"
data_root = dataset_root +'/wlasl100_64x64_640x480_PIL/train'
data_root_val = dataset_root +'/wlasl100_64x64_640x480_PIL/val'
data_root_test = dataset_root +'/wlasl100_64x64_640x480_PIL/test'
ann_file_train = dataset_root +'/wlasl100_64x64_640x480_PIL/train_mm2.txt'
ann_file_val = dataset_root +'/wlasl100_64x64_640x480_PIL/val_mm2.txt'
ann_file_test = dataset_root +'/wlasl100_64x64_640x480_PIL/test_mm2.txt'


test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        delimiter=','))

test_evaluator = dict(type='AccMetric')
test_cfg = dict(type='TestLoop')
