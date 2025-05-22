import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import os
import sys
import seaborn as sns

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from projects.mmdet3d_plugin.uniad.modules.custom_bev_backbone import BEVBackbone

def analyze_feature_statistics(features):
    """특징맵의 통계적 특성을 분석하는 함수"""
    # 기본 통계
    mean_per_channel = features.mean(dim=(2, 3))
    std_per_channel = features.std(dim=(2, 3))
    
    # 활성화 분석 (ReLU 이후이므로 0이 아닌 값들의 비율)
    active_ratio = (features > 0).float().mean(dim=(2, 3))
    
    # Dynamic range
    max_vals = features.amax(dim=(2, 3))
    min_vals = features.amin(dim=(2, 3))
    dynamic_range = max_vals - min_vals
    
    # Spatial sparsity (0에 가까운 값들의 비율)
    spatial_sparsity = (features.abs() < 1e-5).float().mean(dim=(2, 3))
    
    # 채널 간 상관관계
    features_flat = features.squeeze().view(features.size(1), -1)
    correlation = torch.corrcoef(features_flat)
    avg_correlation = correlation.abs().mean()
    
    return {
        'mean_per_channel': mean_per_channel,
        'std_per_channel': std_per_channel,
        'active_ratio': active_ratio,
        'dynamic_range': dynamic_range,
        'spatial_sparsity': spatial_sparsity,
        'avg_correlation': avg_correlation,
        'correlation_matrix': correlation
    }

def print_feature_analysis(stats):
    """분석 결과를 출력하는 함수"""
    print("\n=== 특징맵 분석 결과 ===")
    print(f"채널 평균 범위: {stats['mean_per_channel'].min():.3f} ~ {stats['mean_per_channel'].max():.3f}")
    print(f"채널 표준편차 범위: {stats['std_per_channel'].min():.3f} ~ {stats['std_per_channel'].max():.3f}")
    print(f"평균 활성화 비율: {stats['active_ratio'].mean():.3f}")
    print(f"Dynamic Range 평균: {stats['dynamic_range'].mean():.3f}")
    print(f"공간적 희소성 평균: {stats['spatial_sparsity'].mean():.3f}")
    print(f"채널 간 평균 상관계수: {stats['avg_correlation']:.3f}")
    print("========================\n")

def visualize_correlation(correlation, save_dir):
    """채널 간 상관관계를 시각화하는 함수"""
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation.detach().numpy(), 
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1)
    plt.title('Channel Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'channel_correlation.png'))
    plt.close()

def visualize_features(image_path):
    # 모델 초기화
    model = BEVBackbone(
        pretrained=True,
        backbone='resnet50',
        in_channels=3,
        out_channels=256,
        bev_size=(200, 200),
        output_size=(25, 25)
    )
    model.eval()

    print(f"Loading image from: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
    # 이미지 로드 및 전처리
    img = Image.open(image_path)
    # RGBA to RGB 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    transform = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 200, 200]

    print(f"Input tensor shape: {img_tensor.shape}")

    # 특징 추출
    with torch.no_grad():
        features = model(img_tensor)  # [1, 256, 25, 25]
    
    print(f"Feature tensor shape: {features.shape}")
    
    # 특징맵 분석
    stats = analyze_feature_statistics(features)
    print_feature_analysis(stats)
    
    # 결과 저장 디렉토리 생성
    save_dir = os.path.join(os.path.dirname(image_path), 'feature_visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    # 상관관계 시각화
    visualize_correlation(stats['correlation_matrix'], save_dir)
    
    # 채널 활성화 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.hist(stats['active_ratio'].numpy(), bins=50)
    plt.title('Channel Activation Distribution')
    plt.xlabel('Activation Ratio')
    plt.ylabel('Count')
    
    plt.subplot(122)
    plt.hist(stats['dynamic_range'].numpy(), bins=50)
    plt.title('Dynamic Range Distribution')
    plt.xlabel('Dynamic Range')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_statistics.png'))
    plt.close()

    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(131)
    plt.title('Original BEV Image')
    plt.imshow(img)
    plt.axis('off')
    
    # 특징맵 평균
    feature_mean = features.mean(dim=1).squeeze().numpy()
    plt.subplot(132)
    plt.title('Average Feature Map')
    plt.imshow(feature_mean, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    
    # 특징맵 채널별 분산
    feature_var = features.var(dim=1).squeeze().numpy()
    plt.subplot(133)
    plt.title('Feature Variance Map')
    plt.imshow(feature_var, cmap='plasma')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_summary.png'))
    plt.close()

    # 채널별 특징맵 시각화 (상위 16개 채널)
    plt.figure(figsize=(20, 20))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.title(f'Channel {i}')
        plt.imshow(features[0, i].numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_channels.png'))
    plt.close()

    print(f"Visualization results saved in: {save_dir}")

if __name__ == "__main__":
    image_path = "/home/students/2026_CES_CTRL/UniAD/CARLA-simulation-platform-a-CARLA-vehicle-camera-view-with-pedestrian-in-frame-b-CARLA.ppm"
    visualize_features(image_path) 