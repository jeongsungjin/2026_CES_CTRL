import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS

# --- SE Block 정의 (간단 채널 어텐션) ---
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

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

        # [1] Stem 개선: Conv3x3 x2 (downsampling 완화)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False), # 100x100
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), # 100x100
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # [2] ResNet: FPN 형태로 여러 피처 사용
        resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.layer1 = resnet.layer1  # 256ch, 100x100
        self.layer2 = resnet.layer2  # 512ch, 50x50
        self.layer3 = resnet.layer3  # 1024ch, 25x25
        self.layer4 = resnet.layer4  # 2048ch, 13x13

        # FPN 업샘플용 conv
        self.conv_c2 = nn.Conv2d(512, out_channels, 1)
        self.conv_c3 = nn.Conv2d(1024, out_channels, 1)
        self.conv_c4 = nn.Conv2d(2048, out_channels, 1)

        # [3] SE Block (채널 어텐션)
        self.se = SEBlock(out_channels)

        # [4] 최종 neck (out_channels로 맞추기)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.normalize(x)
        x = self.stem(x)
        c1 = self.layer1(x)   # 256ch, 100x100
        c2 = self.layer2(c1)  # 512ch, 50x50
        c3 = self.layer3(c2)  # 1024ch, 25x25
        c4 = self.layer4(c3)  # 2048ch, 13x13

        # FPN 구조 (최종 해상도에 모두 맞추기, 여기선 25x25 기준)
        p2 = F.interpolate(self.conv_c2(c2), size=(25, 25), mode='bilinear', align_corners=False)
        p3 = self.conv_c3(c3)  # 이미 25x25
        p4 = F.interpolate(self.conv_c4(c4), size=(25, 25), mode='bilinear', align_corners=False)

        x = p2 + p3 + p4      # multi-scale 합성
        x = self.se(x)        # SE 블록
        x = self.out_conv(x)  # 최종 정제
        return x  # [B, 256, 25, 25]
