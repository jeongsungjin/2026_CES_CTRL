import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import json
from tqdm import tqdm

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# BEV 백본 모델 임포트
try:
    from my_bev.models.custom_bev_backbone import CustomBEVBackbone
    print("BEV 백본 모델 임포트 성공!")
except Exception as e:
    print(f"BEV 백본 모델 임포트 실패: {e}")
    sys.exit(1)

# 간단한 BEV 데이터셋 클래스 정의
class SimpleBEVDataset(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.img_info = self.annotations['images']
    
    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        # 이미지 정보
        img_info = self.img_info[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 이미지 로드 및 전처리
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        
        # 단순화를 위해 타겟은 더미 텐서로 반환
        dummy_target = torch.zeros(1)
        
        return img, dummy_target

# 간단한 학습 루프 함수
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # 간단한 학습 목적 함수: BEV 특징 맵과 위치 임베딩 간의 MSE 손실
    criterion = nn.MSELoss()
    
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        
        # 모델 순전파
        features, pos_embed = model(images)
        
        # 간단한 자기지도학습 손실: 특징 맵과 위치 임베딩 간의 일관성
        loss = criterion(features, pos_embed)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def test_training():
    """BEV 백본 모델 학습 테스트"""
    print("BEV 백본 모델 학습 테스트 시작...")
    
    # 데이터셋 경로
    ann_file = 'my_bev/dataset_bev/coco_bbox.json'
    img_dir = 'my_bev/dataset_bev/images'
    
    # 경로 확인
    if not os.path.exists(ann_file) or not os.path.exists(img_dir):
        print(f"데이터셋 파일이 존재하지 않습니다: {ann_file} 또는 {img_dir}")
        return
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 장치: {device}")
    
    # 데이터셋 및 데이터로더 생성
    dataset = SimpleBEVDataset(ann_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 모델 초기화
    model = CustomBEVBackbone(out_ch=256).to(device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 에포크 수 설정 (테스트용으로 작게)
    num_epochs = 2
    
    # 학습 루프
    for epoch in range(num_epochs):
        print(f"에포크 {epoch+1}/{num_epochs} 시작")
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"에포크 {epoch+1} 손실: {loss:.6f}")
    
    # 학습된 모델 저장
    torch.save(model.state_dict(), 'my_bev/bev_backbone_test.pth')
    print(f"학습된 모델 저장 완료: my_bev/bev_backbone_test.pth")
    
    return "BEV 백본 모델 학습 테스트 완료"

if __name__ == "__main__":
    result = test_training()
    print(result) 