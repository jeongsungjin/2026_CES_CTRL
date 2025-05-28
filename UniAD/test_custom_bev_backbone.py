#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 커스텀 BEV 백본 모델 단독 테스트를 위한 스크립트

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 커스텀 BEV 백본 모델 및 어댑터 임포트
from projects.mmdet3d_plugin.models.backbones.custom_bev_backbone import CustomBEVBackbone
from projects.mmdet3d_plugin.models.backbones.uniad_bev_adapter import UniADBEVAdapter

def test_custom_bev_backbone():
    """커스텀 BEV 백본 모델 테스트"""
    print("커스텀 BEV 백본 모델 테스트를 시작합니다...")
    
    # 모델 파라미터 설정
    out_channels = 256  # UniAD에서 사용하는 채널 수
    
    # 모델 생성
    print("1. BEV 백본 모델 생성...")
    bev_backbone = CustomBEVBackbone(
        out_ch=out_channels
    )
    
    # 어댑터 생성
    print("2. BEV 어댑터 생성...")
    bev_adapter = UniADBEVAdapter(
        out_channels=out_channels,
        pretrained="/home/students/2026_CES_CTRL/my_bev/carla_bev_backbone_best.pth"
    )
    
    # GPU 사용 가능하면 모델을 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"3. 사용 장치: {device}")
    
    bev_backbone = bev_backbone.to(device)
    bev_adapter = bev_adapter.to(device)
    
    # 테스트 모드로 설정
    bev_backbone.eval()
    bev_adapter.eval()
    
    # 더미 입력 생성 (배치 크기 1, 채널 3, 높이 200, 너비 200)
    print("4. 더미 입력 데이터 생성...")
    dummy_input = torch.randn(1, 3, 200, 200).to(device)
    
    # 다중 뷰 이미지 형태의 입력(어댑터용)
    dummy_multi_view = torch.randn(1, 6, 3, 200, 200).to(device)  # [B, N, C, H, W]
    
    # 입력 데이터 크기 출력
    print(f"백본 입력 데이터 크기: {dummy_input.shape}")
    print(f"어댑터 입력 데이터 크기: {dummy_multi_view.shape}")
    
    with torch.no_grad():
        # 백본 모델을 통해 특징 추출
        print("5. BEV 백본 모델을 통한 추론...")
        try:
            features = bev_backbone(dummy_input)
            print("BEV 백본 모델 추론 성공!")
            if isinstance(features, tuple):
                for i, feat in enumerate(features):
                    print(f"특징 맵 {i} 크기: {feat.shape}")
            elif isinstance(features, dict):
                for k, v in features.items():
                    print(f"특징 맵 {k} 크기: {v.shape}")
            else:
                print(f"특징 맵 크기: {features.shape}")
                
            # 어댑터를 통한 추론
            print("6. BEV 어댑터를 통한 추론...")
            adapter_features = bev_adapter(dummy_multi_view)
            print("BEV 어댑터 추론 성공!")
            
            if isinstance(adapter_features, list):
                for i, feat in enumerate(adapter_features):
                    print(f"어댑터 출력 특징 맵 {i} 크기: {feat.shape}")
            else:
                print(f"어댑터 출력 특징 맵 크기: {adapter_features.shape}")
                
            print("\n커스텀 BEV 백본 모델 테스트가 성공적으로 완료되었습니다!")
            print("해당 모델을 UniAD Stage1에서 사용할 수 있습니다.")
            
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_custom_bev_backbone() 