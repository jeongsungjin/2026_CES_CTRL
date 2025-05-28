"""
Generate a toy BEV dataset for pipeline smoke-testing
 ├─ my_bev/dataset_bev/
 │   ├─ images/       200×200 RGB png
 │   ├─ lane_mask/    200×200 1-ch png (0/255)
 │   └─ coco_bbox.json   COCO + track_id
"""

import cv2, json, os, random
import numpy as np
from tqdm import trange
from pathlib import Path

# ------------------------------------------------------------------
N_IMG      = 500          # 더 많은 데이터 생성 (100 -> 500)
OUT_ROOT   = Path("my_bev/dataset_bev")
IMG_DIR    = OUT_ROOT / "images"
MASK_DIR   = OUT_ROOT / "lane_mask"
IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

W = H = 200               # pixel size

coco = dict(
    images       = [],
    annotations  = [],
    categories   = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"},
    ]
)
ann_id = 1

rng = np.random.default_rng(0xC0DE)

def random_color():
    return rng.integers(30, 225, size=3).tolist()

# 차선 패턴 생성 함수 추가
def generate_lane_pattern(mask, pattern_type="random"):
    """다양한 차선 패턴 생성"""
    h, w = mask.shape
    
    if pattern_type == "parallel":
        # 평행 차선
        n_lane = random.randint(2, 4)
        lane_width = w // (n_lane + 1)
        
        for i in range(1, n_lane + 1):
            x = i * lane_width
            pts = np.stack([
                np.ones(5) * x + rng.integers(-10, 10, size=5),
                np.linspace(0, h-1, 5, dtype=int)
            ], axis=1)
            cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thickness=2)
            
    elif pattern_type == "curved":
        # 곡선 차선
        n_lane = random.randint(2, 3)
        
        for _ in range(n_lane):
            # 곡선 형태의 제어점 생성
            x_points = np.linspace(0, w-1, 5)
            y_points = np.linspace(0, h-1, 5)
            
            # 중간 점들에 랜덤 변위 추가
            x_points[1:4] += rng.integers(-30, 30, size=3)
            
            # 제어점 합치기
            pts = np.stack([x_points, y_points], axis=1).astype(np.int32)
            
            # 부드러운 곡선 그리기
            cv2.polylines(mask, [pts], False, 255, thickness=2)
    
    elif pattern_type == "intersection":
        # 교차로 패턴
        # 수평선
        y1 = h // 3
        y2 = h * 2 // 3
        cv2.line(mask, (0, y1), (w-1, y1), 255, thickness=2)
        cv2.line(mask, (0, y2), (w-1, y2), 255, thickness=2)
        
        # 수직선
        x1 = w // 3
        x2 = w * 2 // 3
        cv2.line(mask, (x1, 0), (x1, h-1), 255, thickness=2)
        cv2.line(mask, (x2, 0), (x2, h-1), 255, thickness=2)
    
    elif pattern_type == "crosswalk":
        # 횡단보도 패턴
        y = rng.integers(h//4, 3*h//4)
        n_stripes = random.randint(4, 8)
        stripe_width = 5
        gap = 8
        
        for i in range(n_stripes):
            y_pos = y + i * (stripe_width + gap)
            if y_pos < h:
                cv2.line(mask, (0, y_pos), (w-1, y_pos), 255, thickness=stripe_width)
    
    else:  # random (기본)
        # 랜덤 차선
        n_lane = random.randint(2, 4)
        for _ in range(n_lane):
            # 랜덤 polyline left→right
            pts = np.stack([
                rng.integers(0, w, size=5),
                np.linspace(0, h-1, 5, dtype=int)
            ], axis=1).astype(np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness=2)
    
    return mask

# ------------------------------------------------------------------
for img_id in trange(N_IMG, desc="dummy BEV"):
    fname = f"{img_id:06d}.png"

    # ---- BEV RGB ----
    img = np.full((H, W, 3), 50, np.uint8)        # dark-gray asphalt
    
    # 간혹 도로 색상과 텍스처 변경
    if random.random() < 0.2:
        # 다른 도로 색상
        road_color = rng.integers(30, 80, size=3).tolist()
        img = np.full((H, W, 3), road_color, np.uint8)
        
        # 텍스처 추가
        if random.random() < 0.5:
            for _ in range(100):
                x, y = rng.integers(0, W), rng.integers(0, H)
                radius = rng.integers(1, 3)
                color_var = rng.integers(-10, 10, size=3).tolist()
                color = np.clip(np.array(road_color) + color_var, 0, 255).tolist()
                cv2.circle(img, (x, y), radius, color, -1)
    
    # add random rectangles (cars)
    n_car = random.randint(2, 8)  # 더 많은 차량 생성
    for _ in range(n_car):
        cx, cy = rng.integers(20, W-20, size=2)
        w, l   = rng.integers(8, 20, size=2)
        yaw    = rng.random()*np.pi*2
        center_point = (float(cx), float(cy))
        rot    = cv2.getRotationMatrix2D(center_point, np.degrees(yaw), 1.0)
        box_pts = np.array([[-w/2,-l/2],[ w/2,-l/2],[ w/2, l/2],[-w/2, l/2]],np.float32)
        box_pts = cv2.transform(np.array([box_pts]), rot)[0].astype(int)
        cv2.fillPoly(img, [box_pts], random_color(), lineType=cv2.LINE_AA)

        # COCO bbox (axis-aligned) & annotation
        x, y, w_box, l_box = cv2.boundingRect(box_pts)
        
        # numpy 타입을 Python 내장 타입으로 변환
        x = int(x)
        y = int(y)
        w_box = int(w_box)
        l_box = int(l_box)
        
        coco["annotations"].append(dict(
            id           = ann_id,
            image_id     = img_id,
            category_id  = 1,
            bbox         = [x, y, w_box, l_box],
            area         = w_box*l_box,
            iscrowd      = 0,
            track_id     = ann_id           # simple unique id
        ))
        ann_id += 1
    
    # 가끔 보행자 추가 (원으로 표현)
    if random.random() < 0.3:
        n_ped = random.randint(1, 3)
        for _ in range(n_ped):
            cx, cy = rng.integers(20, W-20, size=2)
            radius = rng.integers(3, 7)
            color = random_color()
            
            # 원으로 보행자 표현
            cv2.circle(img, (int(cx), int(cy)), int(radius), color, -1, lineType=cv2.LINE_AA)
            
            # 보행자 어노테이션 추가
            ped_size = int(radius * 2)
            cx, cy, radius = int(cx), int(cy), int(radius)  # numpy 타입을 Python 내장 타입으로 변환
            
            coco["annotations"].append(dict(
                id           = ann_id,
                image_id     = img_id,
                category_id  = 2,  # 보행자
                bbox         = [cx-radius, cy-radius, ped_size, ped_size],
                area         = ped_size*ped_size,
                iscrowd      = 0,
                track_id     = ann_id
            ))
            ann_id += 1

    cv2.imwrite(str(IMG_DIR / fname), img)

    # ---- lane mask ----
    mask = np.zeros((H, W), np.uint8)
    
    # 다양한 차선 패턴 생성
    pattern_type = random.choice(["random", "parallel", "curved", "intersection", "crosswalk"])
    generate_lane_pattern(mask, pattern_type)
    
    # 가끔 차선 패턴 두 개 결합 
    if random.random() < 0.2:
        second_pattern = random.choice(["random", "parallel", "curved"])
        generate_lane_pattern(mask, second_pattern)
    
    cv2.imwrite(str(MASK_DIR / fname), mask)

    # ---- COCO image entry ----
    coco["images"].append(dict(
        id        = int(img_id),  # numpy 타입을 Python 내장 타입으로 변환
        file_name = fname,
        width     = int(W),
        height    = int(H)
    ))

# NumPy int64를 일반 int로 변환하는 함수
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

# NumPy 타입을 Python 내장 타입으로 변환
coco = convert_numpy_types(coco)

# ------------------------------------------------------------------
with open(OUT_ROOT / "coco_bbox.json", "w") as f:
    json.dump(coco, f, indent=2)
print(f"\nCreated {N_IMG} dummy frames in {OUT_ROOT.resolve()}")
