_base_ = ['../../_base_/default_runtime.py']

# ── Model ──────────────────────────────────────────────────────────────────
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormer',
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        drop_path_rate=0.3),
    cls_head=dict(
        type='I3DHead',
        dropout_ratio=0.,
        num_classes=100,
        in_channels=512,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# ── Dataset ────────────────────────────────────────────────────────────────
dataset_type = 'VideoDataset'
data_root = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/train'
data_root_val = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/val'
data_root_test = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/test'
ann_file_train = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/train_aslcitizen100_mm2.txt'
ann_file_val = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/val_aslcitizen100_mm2.txt'
ann_file_test = '/arf/scratch/zgokce/data/ASLCitizen100_videos_64x64/test_aslcitizen100_mm2.txt'

# ── Pipelines ──────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

# ── Dataloaders ────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# ── Training loop ──────────────────────────────────────────────────────────
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Optimizer ──────────────────────────────────────────────────────────────
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5),
    logger=dict(interval=100))

auto_scale_lr = dict(enable=False, base_batch_size=2)

load_from = '/arf/home/zgokce/code/mmaction2_swin/ckpt/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-157c2e66.pth'
