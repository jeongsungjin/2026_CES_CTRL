import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES

from .custom_bev_backbone import CustomBEVBackbone

@BACKBONES.register_module()
class UniADBEVAdapter(nn.Module):
    """
    우리의 BEV 백본을 UniAD에서 사용할 수 있도록 하는 어댑터 클래스
    """
    def __init__(self,
                 out_channels=256,
                 frozen_stages=-1,
                 norm_eval=False,
                 pretrained=None):
        super(UniADBEVAdapter, self).__init__()
        
        # 우리가 개발한 BEV 백본 모델 로드
        self.bev_backbone = CustomBEVBackbone(out_ch=out_channels)
        
        # FPN과 유사한 방식으로 다중 스케일 특징 출력을 위한 추가 레이어
        self.multi_scale_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),  # 1x
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 2x
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 4x
        ])
        
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # 사전 학습된 가중치 로드
        if pretrained:
            self.init_weights(pretrained)
        
    def init_weights(self, pretrained=None):
        """가중치 초기화"""
        if isinstance(pretrained, str):
            try:
                self.bev_backbone.load_state_dict(torch.load(pretrained))
                print(f"Successfully loaded pretrained model from {pretrained}")
            except:
                print(f"Failed to load pretrained model from {pretrained}")
    
    def forward(self, x):
        """
        입력: 다중 뷰 이미지 (B, N, C, H, W)
        출력: 다중 스케일 특징 맵 리스트 (FPN 스타일)
        """
        B, N, C, H, W = x.size()
        
        # 다중 뷰 이미지를 배치 차원으로 변환
        x = x.view(B*N, C, H, W)
        
        # 우리 BEV 백본을 통과
        features, _ = self.bev_backbone(x)
        
        # 다시 원래 배치 크기로 복원
        _, C_out, H_out, W_out = features.size()
        features = features.view(B, N, C_out, H_out, W_out)
        
        # 다중 뷰 특징을 평균하여 단일 BEV 특징 맵 생성
        bev_features = features.mean(dim=1)  # (B, C, H, W)
        
        # 다중 스케일 특징 생성
        multi_scale_features = [bev_features]
        
        # 추가 스케일 특징 생성
        x = bev_features
        for layer in self.multi_scale_layers:
            x = layer(x)
            multi_scale_features.append(x)
        
        return tuple(multi_scale_features)
    
    def train(self, mode=True):
        """학습 모드 설정 및 특정 레이어 고정"""
        super(UniADBEVAdapter, self).train(mode)
        
        if mode and self.frozen_stages >= 0:
            # 특정 스테이지까지 고정
            for param in self.bev_backbone.parameters():
                param.requires_grad = False
            
            # 정규화 레이어의 상태 설정
            if self.norm_eval:
                for m in self.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        m.eval() 