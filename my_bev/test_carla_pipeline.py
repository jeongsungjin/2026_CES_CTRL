import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.patches import Rectangle

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# BEV 백본 모델 임포트
try:
    from my_bev.models.custom_bev_backbone import CustomBEVBackbone
    print("BEV 백본 모델 임포트 성공!")
except Exception as e:
    print(f"BEV 백본 모델 임포트 실패: {e}")
    sys.exit(1)

def create_carla_dataset_structure():
    """CARLA 데이터셋 구조 예시 생성"""
    print("CARLA 데이터셋 구조 예시 생성 중...")
    
    # 디렉토리 생성
    os.makedirs('my_bev/carla_dataset/images', exist_ok=True)
    os.makedirs('my_bev/carla_dataset/bev_masks', exist_ok=True)
    
    # 더미 이미지 생성 (실제로는 CARLA에서 수집한 이미지를 사용)
    # 여기서는 예시로 2개의 이미지만 생성
    num_images = 2
    image_size = (200, 200)
    
    # COCO 형식 어노테이션 파일 생성
    coco_ann = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "vehicle"},
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "lane_marking"}
        ]
    }
    
    for img_id in range(num_images):
        img_filename = f"carla_bev_{img_id:04d}.png"
        
        # 이미지 정보 추가
        coco_ann["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": image_size[0],
            "height": image_size[1],
            "carla_frame_id": 10000 + img_id  # CARLA 프레임 ID 예시
        })
        
        # 더미 이미지 생성 - CARLA BEV 이미지 형태로 구성
        # 실제로는 CARLA 시뮬레이터에서 렌더링된 탑뷰 이미지 사용
        img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        img[:, :] = np.random.randint(100, 150, (3,), dtype=np.uint8)  # 도로 배경색
        
        # 도로 영역 (중앙에 사각형 영역)
        road_color = (120, 120, 120)  # 회색
        cv2.rectangle(img, (50, 0), (150, 200), road_color, -1)
        
        # 차량 객체 생성
        num_vehicles = np.random.randint(1, 4)
        for veh_id in range(num_vehicles):
            # 차량 위치 (도로 위에)
            x = np.random.randint(60, 140)
            y = np.random.randint(10, 190)
            w, h = 20, 30  # 차량 크기
            
            # 차량 색상
            veh_color = (0, 0, 255)  # 빨간색
            cv2.rectangle(img, (x, y), (x + w, y + h), veh_color, -1)
            
            # 어노테이션 추가
            coco_ann["annotations"].append({
                "id": len(coco_ann["annotations"]),
                "image_id": img_id,
                "category_id": 0,  # 차량
                "bbox": [x, y, w, h],
                "carla_object_id": 1000 + veh_id  # CARLA 객체 ID 예시
            })
        
        # 차선 마스크 생성
        lane_mask = np.zeros(image_size, dtype=np.uint8)
        
        # 차선 추가 (도로 중앙에 점선)
        for i in range(0, 200, 20):
            y_start = i
            y_end = min(i + 10, 200)
            lane_mask[y_start:y_end, 98:102] = 255
        
        # 이미지 및 마스크 저장
        cv2.imwrite(f"my_bev/carla_dataset/images/{img_filename}", img)
        cv2.imwrite(f"my_bev/carla_dataset/bev_masks/{img_filename}", lane_mask)
    
    # COCO 어노테이션 파일 저장
    with open('my_bev/carla_dataset/carla_bev_annotations.json', 'w') as f:
        json.dump(coco_ann, f, indent=2)
    
    print(f"CARLA 데이터셋 구조 생성 완료:")
    print(f"- 이미지: {num_images}개")
    print(f"- 어노테이션: {len(coco_ann['annotations'])}개")
    
    return True

def test_carla_bev_pipeline():
    """CARLA BEV 데이터셋으로 파이프라인 테스트"""
    print("CARLA BEV 파이프라인 테스트 시작...")
    
    # 1. 데이터셋 구조 생성
    if not os.path.exists('my_bev/carla_dataset'):
        create_carla_dataset_structure()
    
    # 2. 어노테이션 파일 로드
    try:
        with open('my_bev/carla_dataset/carla_bev_annotations.json', 'r') as f:
            carla_ann = json.load(f)
        print(f"CARLA 어노테이션 로드 성공: {len(carla_ann['images'])} 이미지, {len(carla_ann['annotations'])} 주석")
    except Exception as e:
        print(f"CARLA 어노테이션 로드 실패: {e}")
        return False
    
    # 3. 첫 번째 이미지 정보 가져오기
    if len(carla_ann['images']) > 0:
        img_info = carla_ann['images'][0]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        print(f"이미지 정보: id={img_id}, filename={img_filename}, carla_frame_id={img_info.get('carla_frame_id')}")
        
        # 4. 이미지 로드
        img_path = f"my_bev/carla_dataset/images/{img_filename}"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"이미지 로드 성공: 크기={img.shape}")
            
            # 5. 마스크 로드
            mask_path = f"my_bev/carla_dataset/bev_masks/{img_filename}"
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                print(f"마스크 로드 성공: 크기={mask.shape}")
                
                # 6. 어노테이션 정보 가져오기
                bboxes = []
                labels = []
                for ann in carla_ann['annotations']:
                    if ann['image_id'] == img_id:
                        bboxes.append(ann['bbox'])
                        labels.append(ann['category_id'])
                
                print(f"바운딩 박스 {len(bboxes)}개 발견")
                
                # 7. 시각화
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 이미지와 바운딩 박스
                axes[0].imshow(img)
                
                # 바운딩 박스 그리기
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    x, y, w, h = bbox
                    color = 'r' if label == 0 else 'b'  # 빨간색: 차량, 파란색: 보행자
                    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
                    axes[0].add_patch(rect)
                    axes[0].text(x, y-5, f'{"Veh" if label == 0 else "Ped"}', color=color)
                
                axes[0].set_title('CARLA BEV Image with Objects')
                
                # 마스크 시각화
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('CARLA Lane Mask')
                
                plt.tight_layout()
                plt.savefig('my_bev/carla_bev_test.png')
                print(f"시각화 결과 저장: my_bev/carla_bev_test.png")
                
                # 8. BEV 백본에 입력
                # 이미지 전처리
                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                
                # 모델 초기화 및 추론
                try:
                    model = CustomBEVBackbone(out_ch=256)
                    model.eval()
                    
                    with torch.no_grad():
                        features, pos_embed = model(img_tensor)
                    
                    print(f"BEV 특징 맵 크기: {features.shape}")
                    
                    # 특징 맵 시각화
                    plt.figure(figsize=(8, 8))
                    plt.imshow(features[0, 0].numpy(), cmap='viridis')
                    plt.title('CARLA BEV Feature Map (First Channel)')
                    plt.colorbar()
                    plt.savefig('my_bev/carla_bev_feature_map.png')
                    print(f"CARLA 특징 맵 시각화 완료: my_bev/carla_bev_feature_map.png")
                    
                    return True
                except Exception as e:
                    print(f"BEV 백본 모델 추론 실패: {e}")
                    return False
            else:
                print(f"마스크 파일이 존재하지 않음: {mask_path}")
                return False
        else:
            print(f"이미지 파일이 존재하지 않음: {img_path}")
            return False
    else:
        print("어노테이션 파일에 이미지 정보가 없습니다.")
        return False

if __name__ == "__main__":
    if test_carla_bev_pipeline():
        print("CARLA BEV 파이프라인 테스트 완료: 성공")
    else:
        print("CARLA BEV 파이프라인 테스트 실패") 