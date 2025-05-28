# project_bev/models/custom_bev_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .bev_encoder import BEVEncoder

class CustomBEVBackbone(nn.Module):
    """ResNet-34 + FPN + BEV Encoder → 256×50×50 BEV Feature."""
    def __init__(self, out_ch: int = 256):
        super().__init__()
        res = torchvision.models.resnet34(weights='IMAGENET1K_V1')

        # ── Stem (stride 1) ───────────────────────────────
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = res.bn1
        self.relu  = nn.ReLU(True)

        # ── ResNet blocks ────────────────────────────────
        self.layer1, self.layer2 = res.layer1, res.layer2   # stride 1
        self.layer3, self.layer4 = res.layer3, res.layer4   # stride 2

        # ── FPN 1×1 convs ────────────────────────────────
        self.l2_256 = nn.Conv2d(128, out_ch, 1)
        self.l3_256 = nn.Conv2d(256, out_ch, 1)
        self.l4_256 = nn.Conv2d(512, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        # learnable BEV positional embedding
        self.pos = nn.Parameter(torch.randn(1, out_ch, 50, 50))
        
        # BEV Encoder 추가
        self.bev_encoder = BEVEncoder(embed_dims=out_ch, num_layers=3)
        
        # 이전 프레임 BEV 특징 저장용
        self.prev_bev = None

    def forward(self, x):                     # [B,3,600,800]
        B = x.size(0)  # 현재 배치 크기
        
        # 입력 크기 조정 (메모리 사용량 감소를 위해 크기 축소)
        x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=True)
        
        # 백본 특징 추출
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.layer1(x)                   # 300×300
        c2 = self.layer2(x)                   # 150×150
        c3 = self.layer3(c2)                  # 75×75
        c4 = self.layer4(c3)                  # 38×38

        # FPN으로 초기 BEV 특징 생성
        p2 = F.interpolate(self.l2_256(c2), (50, 50))
        p3 = F.interpolate(self.l3_256(c3), (50, 50))
        p4 = F.interpolate(self.l4_256(c4), (50, 50))
        bev = self.out_conv(p2 + p3 + p4)     # [B,256,50,50]

        # BEV Encoder로 특징 강화
        # 백본 특징을 BEV 크기로 조정하고 채널 수 맞추기
        backbone_feat = F.interpolate(self.l4_256(c4), (50, 50))  # [B,256,50,50]
        
        # 이전 프레임 BEV가 있고 배치 크기가 다르면 조정
        if self.prev_bev is not None and self.prev_bev.size(0) != B:
            # 배치의 첫 B개만 사용하거나 반복하여 맞춤
            if self.prev_bev.size(0) > B:
                self.prev_bev = self.prev_bev[:B]
            else:
                # 부족한 만큼 반복
                repeat_times = B // self.prev_bev.size(0)
                remainder = B % self.prev_bev.size(0)
                self.prev_bev = torch.cat([
                    self.prev_bev.repeat(repeat_times, 1, 1, 1),
                    self.prev_bev[:remainder]
                ]) if remainder > 0 else self.prev_bev.repeat(repeat_times, 1, 1, 1)
        
        bev = self.bev_encoder(bev, self.prev_bev, backbone_feat)
        
        # 현재 BEV를 다음 프레임을 위해 저장
        if self.training:
            self.prev_bev = bev.detach()  # gradient 전파 방지
        
        # positional embedding을 배치 크기에 맞게 확장
        pos = self.pos.expand(B, -1, -1, -1)  # [B,256,50,50]
        return bev, pos
        
    def reset_states(self):
        """이전 프레임 정보 초기화"""
        self.prev_bev = None
