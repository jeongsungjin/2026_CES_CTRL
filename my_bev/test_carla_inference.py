import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path
import json

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# BEV 백본 모델 임포트
try:
    from my_bev.models.custom_bev_backbone import CustomBEVBackbone
    print("BEV 백본 모델 임포트 성공!")
except Exception as e:
    print(f"BEV 백본 모델 임포트 실패: {e}")
    sys.exit(1)

def visualize_feature_maps(features, save_path='my_bev/bev_inference_features.png', num_channels=4):
    """특징 맵 시각화 함수"""
    # 특징 맵에서 일부 채널 선택
    num_channels_to_show = min(num_channels, features.shape[1])
    fig, axes = plt.subplots(2, num_channels_to_show//2, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_channels_to_show):
        # 특징 맵 시각화
        channel_data = features[0, i].detach().cpu().numpy()
        
        # 정규화 (시각화를 위해)
        min_val, max_val = channel_data.min(), channel_data.max()
        if max_val > min_val:
            channel_data = (channel_data - min_val) / (max_val - min_val)
        
        im = axes[i].imshow(channel_data, cmap='viridis')
        axes[i].set_title(f'Feature Channel {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"특징 맵 시각화 저장: {save_path}")

def overlay_features_on_image(image, features, save_path='my_bev/bev_inference_overlay.png', channel=0, alpha=0.7):
    """특징 맵을 원본 이미지 위에 오버레이"""
    # 이미지가 텐서인 경우 넘파이 배열로 변환
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    # 특징 맵 크기를 이미지 크기로 조정
    feature_map = features[0, channel].detach().cpu().numpy()
    feature_map_resized = cv2.resize(feature_map, (image.shape[1], image.shape[0]))
    
    # 정규화 (시각화를 위해)
    min_val, max_val = feature_map_resized.min(), feature_map_resized.max()
    if max_val > min_val:
        feature_map_resized = (feature_map_resized - min_val) / (max_val - min_val)
    
    # 히트맵 생성
    heatmap = cv2.applyColorMap((feature_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 이미지와 히트맵 합성
    overlay = image * (1-alpha) + heatmap * alpha
    
    # 정규화 및 저장
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f'Feature Map (Channel {channel}) Overlay')
    plt.savefig(save_path)
    plt.close()
    print(f"오버레이 시각화 저장: {save_path}")

def visualize_multi_channel_overlay(image, features, save_path='my_bev/bev_inference_multi_channel.png'):
    """여러 채널의 특징 맵 오버레이 시각화"""
    channels_to_show = [0, 1, 4, 8]  # 다양한 채널 선택
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # 이미지가 텐서인 경우 넘파이 배열로 변환
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    for i, channel in enumerate(channels_to_show):
        feature_map = features[0, channel].detach().cpu().numpy()
        feature_map_resized = cv2.resize(feature_map, (image.shape[1], image.shape[0]))
        
        # 정규화
        min_val, max_val = feature_map_resized.min(), feature_map_resized.max()
        if max_val > min_val:
            feature_map_resized = (feature_map_resized - min_val) / (max_val - min_val)
        
        # 히트맵 생성 및 오버레이
        heatmap = cv2.applyColorMap((feature_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        overlay = image * 0.5 + heatmap * 0.5
        overlay = np.clip(overlay, 0, 1)
        
        axes[i].imshow(overlay)
        axes[i].set_title(f'Channel {channel}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"다중 채널 오버레이 시각화 저장: {save_path}")

def visualize_lane_masks(mask_path, features, save_path='my_bev/bev_lane_visualization.png'):
    """차선 마스크와 특징맵 비교 시각화"""
    
    # 차선 마스크 로드
    lane_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if lane_mask is None:
        print(f"차선 마스크를 로드할 수 없음: {mask_path}")
        return
    
    # 특징 맵 합산 (차선 관련 특징이 여러 채널에 분산되어 있을 수 있음)
    # 여러 채널을 살펴보기 위해 채널 합산
    feature_sum = features[0].sum(dim=0).detach().cpu().numpy()
    
    # 정규화
    feature_sum = (feature_sum - feature_sum.min()) / (feature_sum.max() - feature_sum.min() + 1e-9)
    
    # 특징 맵 크기를 마스크 크기로 조정
    feature_resized = cv2.resize(feature_sum, (lane_mask.shape[1], lane_mask.shape[0]))
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 원본 차선 마스크
    axes[0].imshow(lane_mask, cmap='gray')
    axes[0].set_title('Lane Mask')
    axes[0].axis('off')
    
    # 특징맵 합산
    axes[1].imshow(feature_resized, cmap='viridis')
    axes[1].set_title('Feature Map (Channel Sum)')
    axes[1].axis('off')
    
    # 오버레이
    # 특징맵을 히트맵으로 변환
    heatmap = cv2.applyColorMap((feature_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 차선 마스크를 3채널로 변환하고 타입 일치시키기
    lane_mask_rgb = cv2.cvtColor(lane_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # 타입 확인 및 변환
    if lane_mask_rgb.dtype != heatmap.dtype:
        lane_mask_rgb = lane_mask_rgb.astype(heatmap.dtype)
    
    # 마스크와 특징맵 오버레이
    overlay = cv2.addWeighted(lane_mask_rgb, 0.7, heatmap, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay: Lane Mask + Features')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"차선 마스크 시각화 저장: {save_path}")

def search_lane_specific_channels(features, lane_mask, save_path='my_bev/bev_lane_channels.png', top_k=4):
    """차선과 가장 관련 있는 채널을 찾아 시각화"""
    
    # 차선 마스크가 텐서가 아닌 경우 텐서로 변환
    if not isinstance(lane_mask, torch.Tensor):
        lane_mask = torch.from_numpy(lane_mask.astype(np.float32) / 255.0)
    
    # 특징 맵의 크기를 마스크 크기로 조정
    num_channels = features.shape[1]
    resized_features = []
    
    for i in range(num_channels):
        feature = features[0, i].detach().cpu().numpy()
        resized = cv2.resize(feature, (lane_mask.shape[1], lane_mask.shape[0]))
        resized_features.append(resized)
    
    resized_features = np.stack(resized_features)
    
    # 각 채널과 차선 마스크 간의 상관관계 계산
    correlations = []
    for i in range(num_channels):
        # 정규화
        norm_feature = (resized_features[i] - resized_features[i].min()) / (resized_features[i].max() - resized_features[i].min() + 1e-9)
        norm_mask = lane_mask.numpy() if isinstance(lane_mask, torch.Tensor) else lane_mask / 255.0
        
        # 상관계수 계산 (절대값 사용: 음의 상관관계도 중요할 수 있음)
        corr = np.abs(np.corrcoef(norm_feature.flatten(), norm_mask.flatten())[0, 1])
        if np.isnan(corr):
            corr = 0
        correlations.append((i, corr))
    
    # 상관관계가 높은 순으로 정렬
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 K개 채널 시각화
    top_channels = correlations[:top_k]
    
    fig, axes = plt.subplots(2, top_k, figsize=(20, 8))
    
    # 첫 번째 행: 특징 맵
    for i, (channel_idx, corr) in enumerate(top_channels):
        feature = resized_features[channel_idx]
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-9)
        
        axes[0, i].imshow(feature, cmap='viridis')
        axes[0, i].set_title(f'Channel {channel_idx} (corr: {corr:.3f})')
        axes[0, i].axis('off')
    
    # 두 번째 행: 특징 맵과 차선 마스크 오버레이
    for i, (channel_idx, corr) in enumerate(top_channels):
        feature = resized_features[channel_idx]
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-9)
        
        # 히트맵 생성
        heatmap = cv2.applyColorMap((feature * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 차선 마스크를 3채널로 변환하고 타입 일치시키기
        if isinstance(lane_mask, torch.Tensor):
            lane_mask_np = (lane_mask.numpy() * 255).astype(np.uint8)
        else:
            lane_mask_np = lane_mask.astype(np.uint8)
        
        lane_mask_rgb = cv2.cvtColor(lane_mask_np, cv2.COLOR_GRAY2RGB)
        
        # 타입 확인 및 변환
        if lane_mask_rgb.dtype != heatmap.dtype:
            lane_mask_rgb = lane_mask_rgb.astype(heatmap.dtype)
        
        # 오버레이 (타입이 일치하도록 함)
        overlay = cv2.addWeighted(lane_mask_rgb, 0.7, heatmap, 0.3, 0)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Overlay: Lane + Channel {channel_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"차선 관련 채널 시각화 저장: {save_path}")

def test_inference():
    """학습된 모델의 추론 결과 시각화"""
    print("학습된 모델 추론 테스트 시작...")
    
    # 데이터셋 경로
    dataset_dir = Path("my_bev/dataset_bev")
    ann_file = dataset_dir / "coco_bbox.json"
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "lane_mask"  # 차선 마스크 디렉토리
    
    # 학습된 모델 경로 (없으면 기본 모델 사용)
    model_path = Path("my_bev/carla_bev_backbone_trained.pth")
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"추론 장치: {device}")
    
    # 모델 초기화
    model = CustomBEVBackbone(out_ch=256).to(device)
    
    # 학습된 모델 로드 (있는 경우)
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(str(model_path)))
            print(f"학습된 모델 로드 성공: {model_path}")
        except Exception as e:
            print(f"모델 로드 실패, 기본 모델 사용: {e}")
    else:
        print(f"학습된 모델 파일이 없음, 기본 모델 사용: {model_path}")
    
    model.eval()  # 평가 모드로 설정
    
    # 테스트 이미지 선택 (여러 이미지 테스트)
    test_indices = [0, 10, 20, 30, 40]  # 다양한 이미지 인덱스
    
    # 어노테이션 로드
    try:
        with open(ann_file, 'r') as f:
            coco_ann = json.load(f)
    except Exception as e:
        print(f"어노테이션 로드 실패: {e}")
        return
    
    # 테스트 이미지마다 처리
    for test_idx in test_indices:
        if test_idx >= len(coco_ann['images']):
            continue
            
        img_info = coco_ann['images'][test_idx]
        img_path = img_dir / img_info['file_name']
        mask_path = mask_dir / img_info['file_name']  # 해당 이미지의 차선 마스크 경로
        
        # 차선 마스크 확인
        if not mask_path.exists():
            print(f"차선 마스크가 존재하지 않음: {mask_path}")
            continue
        
        # 이미지 로드 및 전처리
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"이미지를 로드할 수 없음: {img_path}")
            continue
        
        # 차선 마스크 로드
        lane_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if lane_mask is None:
            print(f"차선 마스크를 로드할 수 없음: {mask_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 모델 추론
        with torch.no_grad():
            features, pos_embed = model(img_tensor)
        
        print(f"이미지 {img_info['file_name']} 처리 중...")
        print(f"특징 맵 크기: {features.shape}")
        
        # 바운딩 박스 정보 수집
        boxes = []
        for ann in coco_ann['annotations']:
            if ann['image_id'] == img_info['id']:
                boxes.append({
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                })
        
        # 저장 디렉토리
        output_dir = Path("my_bev/inference_results")
        output_dir.mkdir(exist_ok=True)
        
        # 파일명에서 확장자 제거
        filename_base = os.path.splitext(img_info['file_name'])[0]
        
        # 원본 이미지와 차선 마스크 함께 시각화
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 원본 이미지에 바운딩 박스 표시
        axes[0].imshow(img / 255.0)
        for box in boxes:
            x, y, w, h = box['bbox']
            category_id = box['category_id']
            color = 'r' if category_id == 1 else 'b'
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)
        
        axes[0].set_title(f"Input Image: {img_info['file_name']}")
        axes[0].axis('off')
        
        # 차선 마스크 표시
        axes[1].imshow(lane_mask, cmap='gray')
        axes[1].set_title(f"Lane Mask: {img_info['file_name']}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_base}_input_with_mask.png")
        plt.close(fig)
        
        # 특징 맵 시각화
        visualize_feature_maps(
            features, 
            save_path=str(output_dir / f"{filename_base}_features.png")
        )
        
        # 특징 맵 오버레이
        overlay_features_on_image(
            img_tensor[0], 
            features, 
            save_path=str(output_dir / f"{filename_base}_overlay.png")
        )
        
        # 다중 채널 오버레이
        visualize_multi_channel_overlay(
            img_tensor[0], 
            features, 
            save_path=str(output_dir / f"{filename_base}_multi_channel.png")
        )
        
        # 차선 마스크와 특징맵 비교 시각화
        visualize_lane_masks(
            mask_path,
            features,
            save_path=str(output_dir / f"{filename_base}_lane_features.png")
        )
        
        # 차선 관련 채널 찾기 및 시각화
        search_lane_specific_channels(
            features,
            lane_mask,
            save_path=str(output_dir / f"{filename_base}_lane_specific_channels.png")
        )
    
    return "모델 추론 및 시각화 완료"

if __name__ == "__main__":
    result = test_inference()
    print(result) 