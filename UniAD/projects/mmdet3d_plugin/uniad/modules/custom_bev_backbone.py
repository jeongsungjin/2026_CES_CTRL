import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS

@MODELS.register_module()
class BEVBackbone(BaseModule):
    def __init__(self,
                 pretrained=True,
                 backbone='resnet50',
                 in_channels=3,
                 out_channels=256,
                 bev_size=(200, 200),
                 output_size=(25, 25)):
        super().__init__()
        
        # ResNet 백본 초기화 (사전학습 가중치 사용)
        self.backbone = getattr(torchvision.models, backbone)(pretrained=pretrained)
        
        # 출력 크기 조정을 위한 추가 레이어
        self.neck = nn.Sequential(
            nn.Conv2d(2048, out_channels, 1),  # ResNet의 출력 채널을 256으로 조정
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 이미지 정규화를 위한 파라미터
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        # 입력 정규화
        x = self.normalize(x)
        
        # 특징 추출
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # 출력 크기 조정
        x = self.neck(x)
        x = F.interpolate(x, size=(25, 25), mode='bilinear', align_corners=True)
        
        return x  # [B, 256, 25, 25]

@MODELS.register_module()
class CustomBEVModel(BaseModule):
    def __init__(self,
                 pretrained=True,
                 backbone='resnet50',
                 in_channels=3,
                 out_channels=256,
                 bev_size=(200, 200),
                 output_size=(25, 25),
                 num_encoder_layers=3):
        super().__init__()
        
        # BEV 백본
        self.backbone = BEVBackbone(
            pretrained=pretrained,
            backbone=backbone,
            in_channels=in_channels,
            out_channels=out_channels,
            bev_size=bev_size,
            output_size=output_size
        )
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
    def forward(self, x):
        # 백본을 통한 특징 추출
        x = self.backbone(x)  # [B, 256, 25, 25]
        
        # 특징 reshape for transformer
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # 인코더를 통한 특징 강화
        x = self.encoder(x)  # [H*W, B, C]
        
        # 원래 형태로 복원
        x = x.permute(1, 2, 0).reshape(B, C, H, W)  # [B, 256, 25, 25]
        
        return x 