# project_bev/cfgs/bev_selfsup.py
_base_ = [
    "mmselfsup::_base_/default_runtime.py",
]

model = dict(
    type="DINO",
    backbone=dict(type='CustomBEVBackbone', out_indices=(0,),  # register path
                  init_cfg=None),
    neck=dict(type="DINOHead",
              in_channels=256, hidden_dim=256, bottleneck_dim=256),
    head=dict(type="DINOLoss", out_dim=65536))

train_pipeline = [
    dict(type='LoadBEVImage', img_prefix='dataset_bev/images'),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(type='BEVCoco',
               ann_file='dataset_bev/coco_bbox.json',
               pipeline=train_pipeline))

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
fp16 = dict(loss_scale='dynamic')
total_epochs = 100
