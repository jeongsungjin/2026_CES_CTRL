import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# BEV 백본 모델 임포트 시도
try:
    from my_bev.models.custom_bev_backbone import CustomBEVBackbone
    print("BEV 백본 모델 임포트 성공!")
except Exception as e:
    print(f"BEV 백본 모델 임포트 실패: {e}")
    sys.exit(1)

def preprocess_image(image_path, target_size=(200, 200)):
    """이미지 전처리 함수"""
    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 크기 조정
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    
    # 정규화 및 텐서 변환
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # H,W,C -> C,H,W
    
    return img

def visualize_feature_maps(feature_map, pos_embed, save_path='my_bev/bev_features.png'):
    """특징 맵 시각화 함수"""
    # 특징 맵에서 일부 채널 선택 (4개)
    num_channels_to_show = min(4, feature_map.shape[1])
    fig, axes = plt.subplots(1, num_channels_to_show, figsize=(15, 4))
    
    for i in range(num_channels_to_show):
        if num_channels_to_show == 1:
            ax = axes
        else:
            ax = axes[i]
        
        # 특징 맵 시각화
        channel_data = feature_map[0, i].detach().numpy()
        im = ax.imshow(channel_data, cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    # 위치 임베딩 시각화
    fig, ax = plt.subplots(figsize=(6, 6))
    pos_data = torch.norm(pos_embed, dim=1)[0].detach().numpy()
    im = ax.imshow(pos_data, cmap='plasma')
    ax.set_title('Position Embedding (Norm)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('my_bev/bev_pos_embedding.png')
    plt.close(fig)

def test_backbone_with_dummy_data():
    """더미 데이터로 백본 테스트"""
    print("BEV 백본 테스트 시작...")
    
    # 더미 데이터 경로 확인
    data_dir = 'my_bev/dataset_bev/images'
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return
    
    # 첫 번째 이미지 파일 찾기
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    if not image_files:
        print(f"이미지 파일이 없습니다: {data_dir}")
        return
    
    image_path = os.path.join(data_dir, image_files[0])
    print(f"테스트 이미지: {image_path}")
    
    # 이미지 전처리
    img = preprocess_image(image_path)
    img_batch = img.unsqueeze(0)  # 배치 차원 추가 [1, C, H, W]
    print(f"입력 이미지 크기: {img_batch.shape}")
    
    # 모델 초기화
    model = CustomBEVBackbone(out_ch=256)
    model.eval()  # 평가 모드로 설정
    
    # 추론
    with torch.no_grad():
        features, pos_embed = model(img_batch)
    
    print(f"BEV 특징 맵 크기: {features.shape}")
    print(f"위치 임베딩 크기: {pos_embed.shape}")
    
    # 특징 맵 시각화
    visualize_feature_maps(features, pos_embed)
    print("특징 맵 시각화 완료: my_bev/bev_features.png")
    print("위치 임베딩 시각화 완료: my_bev/bev_pos_embedding.png")
    
    return "BEV 백본 테스트 완료"

if __name__ == "__main__":
    result = test_backbone_with_dummy_data()
    print(result) 