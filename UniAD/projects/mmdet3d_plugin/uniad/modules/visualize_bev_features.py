import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from projects.mmdet3d_plugin.uniad.modules.custom_bev_backbone import CustomBEVModel

def visualize_features(image_path):
    # 모델 초기화
    model = CustomBEVModel(
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

    # 결과 저장 디렉토리 생성
    save_dir = os.path.join(os.path.dirname(image_path), 'feature_visualization')
    os.makedirs(save_dir, exist_ok=True)

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
    image_path = "/Users/seongjinjeong/2026_CES_CTRL/UniAD/CARLA-simulation-platform-a-CARLA-vehicle-camera-view-with-pedestrian-in-frame-b-CARLA.ppm"
    visualize_features(image_path) 