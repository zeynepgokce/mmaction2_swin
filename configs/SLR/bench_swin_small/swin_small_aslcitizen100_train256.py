_base_ = [
    '../../_base_/models/swin_tiny.py', '../../_base_/default_runtime.py'
]


# ── Model ──────────────────────────────────────────────────────────────────
# Swin-Small, 16 frames, K400 pretrained
num_frames = 16
model = dict(
    backbone=dict(
        arch='small',
        drop_path_rate=0.2,
        pretrained2d=False),
    cls_head=dict(num_classes=100))


# ── Dataset ────────────────────────────────────────────────────────────────
# BENCH: Train_256 — source videos are 256×256 (actual 256 dataset)
dataset_type = 'VideoDataset'
data_root = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/train'
data_root_val = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/val'
data_root_test = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/test'
ann_file_train = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/train_aslcitizen100_mm2.txt'
ann_file_val = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/val_aslcitizen100_mm2.txt'
ann_file_test = '/arf/scratch/zgokce/data/ASLCitizen100_videos_256x256/test_aslcitizen100_mm2.txt'

# ── Pipelines ──────────────────────────────────────────────────────────────
# Source: 256×256 → short-side to 256 → RandomResizedCrop → 224
train_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
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
    dict(type='Resize', scale=(-1, 256)),
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

# ── Optimizer ─────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

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

load_from = './ckpt/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth'
