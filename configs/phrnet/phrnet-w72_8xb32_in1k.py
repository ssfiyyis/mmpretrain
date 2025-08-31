_base_ = [
    '../_base_/models/phrnet/phrnet-w72.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# # NOTE: `auto_scale_lr` is for automatically scaling LR
# # based on the actual training batch size.
# # base_batch_size = (8 GPUs) x (32 samples per GPU)
# auto_scale_lr = dict(base_batch_size=256)
# # 256 is the default setting in imagenet_bs256_coslr.py, no need to redefine it

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=120, by_epoch=True, begin=0, end=120)
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

dataset_type = 'ImageNet'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=448, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=512, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/ubuntu/workspace/ILSVRC/Data/CLS-LOC',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/ubuntu/workspace/ILSVRC/Data/CLS-LOC',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
