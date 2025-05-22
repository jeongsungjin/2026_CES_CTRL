# project_bev/cfgs/bev_ft.py
_base_ = [
    "mmdetection::_base_/default_runtime.py",
]

model = dict(
    type='BEVDetector',
    backbone=dict(
        type='CustomBEVBackbone',
        init_cfg=dict(type='Pretrained', checkpoint='work_dirs/selfsup/epoch_100.pth')
    ),
    neck=None,
    bbox_head=dict(type='YOLOXHead', num_classes=2, in_channels=256),
    seg_head=dict(type='LaneSegHead', num_classes=2, in_channels=256))

train_pipeline = [
    dict(type='LoadBEVImage', img_prefix='dataset_bev/images'),
    dict(type='LoadBEVAnnotations',
         ann_file='dataset_bev/coco_bbox.json',
         lane_mask_dir='dataset_bev/lane_mask'),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img','gt_bboxes_3d','gt_labels_3d',
                               'gt_inds','gt_lane_masks']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(type='BEVCoco',
               ann_file='dataset_bev/coco_bbox.json',
               pipeline=train_pipeline))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'backbone':dict(lr_mult=0.1)}))
fp16 = dict(loss_scale='dynamic')
total_epochs = 20
