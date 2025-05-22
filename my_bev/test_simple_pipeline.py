import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Rectangle

# 디렉토리 확인
print(f"현재 디렉토리: {os.getcwd()}")
print(f"dataset_bev 디렉토리 존재: {os.path.exists('my_bev/dataset_bev')}")
print(f"이미지 디렉토리 존재: {os.path.exists('my_bev/dataset_bev/images')}")
print(f"마스크 디렉토리 존재: {os.path.exists('my_bev/dataset_bev/lane_mask')}")
print(f"어노테이션 파일 존재: {os.path.exists('my_bev/dataset_bev/coco_bbox.json')}")

# 어노테이션 파일 로드
try:
    with open('my_bev/dataset_bev/coco_bbox.json', 'r') as f:
        coco_ann = json.load(f)
    print(f"어노테이션 로드 성공: {len(coco_ann['images'])} 이미지, {len(coco_ann['annotations'])} 주석")
except Exception as e:
    print(f"어노테이션 로드 실패: {e}")
    sys.exit(1)

# 첫 번째 이미지 정보 가져오기
if len(coco_ann['images']) > 0:
    img_info = coco_ann['images'][0]
    img_id = img_info['id']
    img_filename = img_info['file_name']
    print(f"이미지 정보: id={img_id}, filename={img_filename}")
    
    # 이미지 로드
    img_path = f"my_bev/dataset_bev/images/{img_filename}"
    if os.path.exists(img_path):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"이미지 로드 성공: 크기={img.shape}")
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            img = None
    else:
        print(f"이미지 파일이 존재하지 않음: {img_path}")
        img = None
    
    # 마스크 로드
    mask_path = f"my_bev/dataset_bev/lane_mask/{img_filename}"
    if os.path.exists(mask_path):
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"마스크 로드 성공: 크기={mask.shape}")
        except Exception as e:
            print(f"마스크 로드 실패: {e}")
            mask = None
    else:
        print(f"마스크 파일이 존재하지 않음: {mask_path}")
        mask = None
    
    # 해당 이미지의 바운딩 박스 찾기
    bboxes = []
    labels = []
    for ann in coco_ann['annotations']:
        if ann['image_id'] == img_id:
            bboxes.append(ann['bbox'])
            labels.append(ann['category_id'])
    
    print(f"바운딩 박스 {len(bboxes)}개 발견")
    
    # 결과 시각화
    if img is not None:
        fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(12, 6))
        
        # 이미지와 바운딩 박스
        if mask is not None:
            ax = axes[0]
        else:
            ax = axes
        
        ax.imshow(img)
        
        # 바운딩 박스 그리기
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            x, y, w, h = bbox
            color = 'r' if label == 0 else 'b'  # 빨간색: 차량, 파란색: 보행자
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, f'{"Car" if label == 0 else "Ped"}', color=color)
        
        ax.set_title('BEV Image with Objects')
        
        # 마스크
        if mask is not None:
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Lane Mask')
        
        plt.tight_layout()
        plt.savefig('my_bev/bev_test_simple.png')
        print(f"시각화 결과 저장: my_bev/bev_test_simple.png")
    else:
        print("이미지가 없어 시각화를 수행할 수 없습니다.")
else:
    print("어노테이션 파일에 이미지 정보가 없습니다.") 