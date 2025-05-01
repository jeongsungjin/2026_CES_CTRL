import sys
print("EXEC:", sys.executable)
print("PATH:")
for p in sys.path:
    print("  ", p)
import torch
import torch.nn as nn
import torch.optim as optim
from mmdet3d.models import build_detector
from mmdet3d_plugin.uniad.detectors.uniad_e2e import UniAD

def test_bev_input():
    # 모델 설정
    model_cfg = dict(
        type='UniAD',
        use_bev_input=True,
        bev_h=200,
        bev_w=200,
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        neck=dict(
            type='FPN',
            in_channels=[2048],
            out_channels=256,
            num_outs=1
        ),
        track_head=dict(
            type='TrackHead',
            in_channels=256,
            num_classes=10,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')
                    )
                )
            )
        ),
        motion_head=dict(
            type='MotionHead',
            in_channels=256,
            num_frames=3,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')
                    )
                )
            )
        ),
        occ_head=dict(
            type='OccHead',
            in_channels=256,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')
                    )
                )
            )
        ),
        planning_head=dict(
            type='PlanningHead',
            in_channels=256,
            num_frames=3,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')
                    )
                )
            )
        )
    )

    # 모델 생성
    model = build_detector(model_cfg)
    model = model.cuda()
    model.train()

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 이렇게 수정    # 더미 데이터 생성
    batch_size = 2
    bev_h = 200
    bev_w = 200
    num_frames = 3

    # BEV 이미지 생성 (더미 데이터)
    bev_images = torch.randn(batch_size, num_frames, 3, bev_h, bev_w).cuda()  # [B, T, C, H, W]

    # 더미 메타데이터 생성
    img_metas = [dict(
        img_shape=(bev_h, bev_w, 3),
        scale_factor=1.0,
        pad_shape=(bev_h, bev_w, 3),
        sample_idx=0,
        timestamp=0.0,
        scene_token='dummy_scene',
        can_bus=torch.zeros(18).cuda(),  # 18차원의 CAN 버스 데이터
        lidar2img=torch.eye(4).cuda(),  # 4x4 변환 행렬
        lidar2ego=torch.eye(4).cuda(),  # 4x4 변환 행렬
        ego2global=torch.eye(4).cuda(),  # 4x4 변환 행렬
        box_type_3d='LiDAR',
        box_mode_3d='xyzwhlr'
    ) for _ in range(batch_size)]

    # 더미 GT 데이터 생성
    gt_bboxes_3d = [torch.randn(5, 9).cuda() for _ in range(batch_size)]  # 5개의 객체, 9개의 속성
    gt_labels_3d = [torch.randint(0, 10, (5,)).cuda() for _ in range(batch_size)]
    gt_past_traj = [torch.randn(5, 3, 2).cuda() for _ in range(batch_size)]  # 5개 객체, 3프레임, 2차원
    gt_past_traj_mask = [torch.ones(5, 3).bool().cuda() for _ in range(batch_size)]
    gt_inds = [torch.arange(5).cuda() for _ in range(batch_size)]
    gt_sdc_bbox = [torch.randn(1, 9).cuda() for _ in range(batch_size)]
    gt_sdc_label = [torch.randint(0, 10, (1,)).cuda() for _ in range(batch_size)]

    # 학습 루프
    num_epochs = 5
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        losses = model.forward_train(
            img=bev_images,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_inds=gt_inds,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label
        )
        
        # Loss 계산
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.4f}')
        for k, v in losses.items():
            print(f'  {k}: {v.item():.4f}')

if __name__ == '__main__':
    test_bev_input() 