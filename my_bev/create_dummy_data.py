import os
import json
import numpy as np
import cv2
from PIL import Image

# 디렉토리 생성
os.makedirs('my_bev/dataset_bev/images', exist_ok=True)
os.makedirs('my_bev/dataset_bev/lane_mask', exist_ok=True)

# COCO 형식의 더미 어노테이션 파일 생성
coco_ann = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "car"},
        {"id": 1, "name": "pedestrian"}
    ]
}

# 더미 이미지 및 어노테이션 생성
num_images = 5
image_size = (200, 200)  # BEV 이미지 크기

for img_id in range(num_images):
    # 이미지 파일명
    img_filename = f"bev_{img_id:04d}.png"
    
    # 이미지 정보 추가
    coco_ann["images"].append({
        "id": img_id,
        "file_name": img_filename,
        "width": image_size[0],
        "height": image_size[1]
    })
    
    # 더미 RGB 이미지 생성 (200x200)
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    # 무작위 배경 색상
    img[:, :] = np.random.randint(100, 200, (3,), dtype=np.uint8)
    
    # 더미 객체 추가 (사각형으로 표현)
    num_objects = np.random.randint(1, 4)  # 1~3개의 객체
    
    for obj_id in range(num_objects):
        # 무작위 바운딩 박스 생성
        x = np.random.randint(10, image_size[0] - 50)
        y = np.random.randint(10, image_size[1] - 50)
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        
        # 객체 카테고리 (0: car, 1: pedestrian)
        category_id = np.random.randint(0, 2)
        
        # 객체 그리기
        color = (0, 0, 255) if category_id == 0 else (255, 0, 0)  # 빨간색: 차량, 파란색: 보행자
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        
        # 어노테이션 추가
        coco_ann["annotations"].append({
            "id": len(coco_ann["annotations"]),
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "track_id": obj_id
        })
    
    # 더미 차선 마스크 생성 (흰색 선으로 표현)
    lane_mask = np.zeros(image_size, dtype=np.uint8)
    
    # 수평 차선
    for i in range(1, 3):
        y_pos = int(image_size[1] * i / 3)
        lane_mask[y_pos-2:y_pos+2, :] = 255
    
    # 수직 차선
    for i in range(1, 3):
        x_pos = int(image_size[0] * i / 3)
        lane_mask[:, x_pos-2:x_pos+2] = 255
    
    # 이미지 저장
    cv2.imwrite(f"my_bev/dataset_bev/images/{img_filename}", img)
    cv2.imwrite(f"my_bev/dataset_bev/lane_mask/{img_filename}", lane_mask)

# COCO 어노테이션 파일 저장
with open('my_bev/dataset_bev/coco_bbox.json', 'w') as f:
    json.dump(coco_ann, f, indent=2)

print(f"생성된 더미 데이터셋:")
print(f"- 이미지: {num_images}개")
print(f"- 어노테이션: {len(coco_ann['annotations'])}개")
print(f"- 카테고리: {len(coco_ann['categories'])}개 ({[c['name'] for c in coco_ann['categories']]})") 