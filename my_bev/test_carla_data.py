import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path

def visualize_bev_dataset():
    """CARLA 형식 BEV 데이터셋 시각화"""
    
    # 데이터셋 경로 설정
    dataset_dir = Path("my_bev/dataset_bev")
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "lane_mask"
    coco_file = dataset_dir / "coco_bbox.json"
    
    # 어노테이션 파일 로드
    try:
        with open(coco_file, 'r') as f:
            coco_ann = json.load(f)
        print(f"어노테이션 로드 성공: {len(coco_ann['images'])} 이미지, {len(coco_ann['annotations'])} 주석")
    except Exception as e:
        print(f"어노테이션 로드 실패: {e}")
        return
    
    # 카테고리 정보 출력
    print("카테고리 정보:")
    for cat in coco_ann['categories']:
        print(f"  - ID {cat['id']}: {cat['name']}")
    
    # 일부 이미지 시각화 (처음 5개)
    num_samples = min(5, len(coco_ann['images']))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    for i in range(num_samples):
        img_info = coco_ann['images'][i]
        img_id = img_info['id']
        img_file = img_info['file_name']
        
        # 이미지 로드
        img_path = image_dir / img_file
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 마스크 로드
        mask_path = mask_dir / img_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 이미지에 해당하는 바운딩 박스 가져오기
        boxes = []
        for ann in coco_ann['annotations']:
            if ann['image_id'] == img_id:
                boxes.append({
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id'],
                    'track_id': ann['track_id']
                })
        
        # 이미지 시각화
        ax = axes[i, 0]
        ax.imshow(img)
        ax.set_title(f"Image {img_id}")
        
        # 바운딩 박스 그리기
        for box in boxes:
            x, y, w, h = box['bbox']
            category_id = box['category_id']
            track_id = box['track_id']
            
            color = 'r' if category_id == 1 else 'b'  # 빨간색: 차량, 파란색: 보행자
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 객체 ID 표시
            ax.text(x, y-5, f"ID:{track_id}", fontsize=8, color=color)
        
        # 마스크 시각화
        ax = axes[i, 1]
        ax.imshow(mask, cmap='gray')
        ax.set_title(f"Lane Mask {img_id}")
        
        # 축 설정
        for ax in axes[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig("my_bev/carla_data_samples.png")
    print(f"시각화 결과 저장: my_bev/carla_data_samples.png")

if __name__ == "__main__":
    visualize_bev_dataset() 