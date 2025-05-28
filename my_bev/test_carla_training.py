import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import time

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# BEV 백본 모델 임포트
try:
    from my_bev.models.custom_bev_backbone import CustomBEVBackbone
    print("BEV 백본 모델 임포트 성공!")
except Exception as e:
    print(f"BEV 백본 모델 임포트 실패: {e}")
    sys.exit(1)

# 커스텀 배치 컬레이터 함수
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch]) if batch[0][1] is not None else None
    
    # 타겟은 간단하게 처리 (이미지 ID만 필요)
    image_ids = [item[2]['image_id'] for item in batch]
    
    return images, masks, image_ids

# CARLA BEV 데이터셋 클래스
class CARLABEVDataset(Dataset):
    def __init__(self, ann_file, img_dir, mask_dir=None, transform=None):
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_info = self.annotations['images']
    
    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        # 이미지 정보
        img_info = self.img_info[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 이미지 로드 및 전처리
        img = cv2.imread(img_path)
        if img is None:
            # 이미지를 로드할 수 없는 경우 더미 이미지 반환
            print(f"경고: 이미지를 로드할 수 없습니다: {img_path}")
            img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        
        # 차선 마스크 로드 (있는 경우)
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_info['file_name'])
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0  # 정규화
                mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        # 간단한 타겟 반환 (학습 테스트 목적)
        target = {
            'image_id': img_id
        }
        
        return img, mask, target

# 향상된 손실 함수 클래스 정의
class EnhancedBEVLoss(nn.Module):
    def __init__(self, lambda_lane=1.0, lambda_consistency=0.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_lane = lambda_lane
        self.lambda_consistency = lambda_consistency
        
        # 차선 예측을 위한 컨볼루션 레이어
        self.lane_predictor = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    
    def forward(self, features, pos_embed, lane_masks=None):
        batch_size = features.shape[0]
        
        # 1. 자기지도학습 일관성 손실: 특징 맵과 위치 임베딩 간의 일관성
        consistency_loss = self.mse_loss(features, pos_embed)
        
        # 2. 차선 감지 손실 (차선 마스크가 있는 경우)
        lane_loss = 0.0
        if lane_masks is not None:
            # 특징 맵에서 차선 예측
            lane_pred = self.lane_predictor(features)
            
            # 예측 크기를 마스크 크기로 조정
            pred_size = lane_pred.shape[2:]
            mask_size = lane_masks.shape[2:]
            
            if pred_size != mask_size:
                lane_pred = nn.functional.interpolate(
                    lane_pred, 
                    size=mask_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 차선 손실 계산
            lane_loss = self.bce_loss(lane_pred, lane_masks)
        
        # 총 손실
        total_loss = consistency_loss * self.lambda_consistency + lane_loss * self.lambda_lane
        
        # 각 손실 요소 반환 (모니터링용)
        return total_loss, {
            'consistency_loss': consistency_loss.item(),
            'lane_loss': lane_loss.item() if isinstance(lane_loss, torch.Tensor) else lane_loss
        }

# 모델 성능 평가 함수
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_lane_loss = 0
    total_consistency_loss = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device) if masks is not None else None
            
            # 모델 순전파
            features, pos_embed = model(images)
            
            # 손실 계산
            loss, loss_components = criterion(features, pos_embed, masks)
            
            total_loss += loss.item()
            total_consistency_loss += loss_components['consistency_loss']
            total_lane_loss += loss_components['lane_loss']
    
    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    avg_consistency_loss = total_consistency_loss / len(dataloader)
    avg_lane_loss = total_lane_loss / len(dataloader)
    
    return avg_loss, avg_consistency_loss, avg_lane_loss

# 학습 루프 함수
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_lane_loss = 0
    total_consistency_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for images, masks, _ in progress_bar:
        images = images.to(device)
        masks = masks.to(device) if masks is not None else None
        
        # 모델 순전파
        features, pos_embed = model(images)
        
        # 손실 계산
        loss, loss_components = criterion(features, pos_embed, masks)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실 기록
        total_loss += loss.item()
        total_consistency_loss += loss_components['consistency_loss']
        total_lane_loss += loss_components['lane_loss']
        
        # 진행 표시줄 업데이트
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lane_loss': f"{loss_components['lane_loss']:.4f}",
            'con_loss': f"{loss_components['consistency_loss']:.4f}"
        })
    
    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    avg_consistency_loss = total_consistency_loss / len(dataloader)
    avg_lane_loss = total_lane_loss / len(dataloader)
    
    return avg_loss, avg_consistency_loss, avg_lane_loss

# 손실 시각화 함수
def plot_losses(train_losses, val_losses, lane_losses, consistency_losses, save_path):
    plt.figure(figsize=(15, 10))
    
    # 학습 및 검증 손실 그래프
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 구성요소별 손실 그래프
    plt.subplot(2, 1, 2)
    plt.plot(lane_losses, 'g-', label='Lane Loss')
    plt.plot(consistency_losses, 'y-', label='Consistency Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Component Losses')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_carla_training():
    """CARLA BEV 데이터로 학습 테스트"""
    print("CARLA BEV 데이터셋으로 강화된 학습 테스트 시작...")
    
    # 데이터셋 경로
    dataset_dir = Path("my_bev/dataset_bev")
    ann_file = str(dataset_dir / "coco_bbox.json")
    img_dir = str(dataset_dir / "images")
    mask_dir = str(dataset_dir / "lane_mask")
    
    # 경로 확인
    if not os.path.exists(ann_file) or not os.path.exists(img_dir):
        print(f"데이터셋 파일이 존재하지 않습니다: {ann_file} 또는 {img_dir}")
        return
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 장치: {device}")
    
    # CUDA 메모리 관리
    if device.type == 'cuda':
        # 사용하지 않는 CUDA 캐시 정리
        torch.cuda.empty_cache()
        # 메모리 상태 확인
        print(f"CUDA 메모리 상태: {torch.cuda.memory_allocated() / 1024**3:.2f}GB 사용 중, "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB 예약됨")
    
    # 학습 파라미터 설정 (배치 크기 축소)
    batch_size = 4  # 16 -> 4로 감소
    num_epochs = 30  # 50 -> 30으로 감소
    val_split = 0.2  # 검증 세트 비율
    
    # 데이터셋 및 데이터로더 생성
    full_dataset = CARLABEVDataset(ann_file, img_dir, mask_dir)
    print(f"전체 데이터셋 크기: {len(full_dataset)}")
    
    # 데이터셋 분할 (학습/검증)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"학습 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # 4 -> 2로 감소
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # 4 -> 2로 감소
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # 모델 초기화
    model = CustomBEVBackbone(out_ch=256).to(device)
    
    # 이전 학습된 모델 로드 (있는 경우)
    model_path = Path("my_bev/carla_bev_backbone_trained.pth")
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(str(model_path)))
            print(f"이전 학습 모델 로드 성공: {model_path}")
        except Exception as e:
            print(f"이전 모델 로드 실패, 새로 초기화: {e}")
    
    # 손실 함수 및 옵티마이저 설정
    criterion = EnhancedBEVLoss(lambda_lane=2.0, lambda_consistency=0.5).to(device)
    
    # 학습률 스케줄링과 함께 최적화 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # 학습률 조정 2e-4 -> 1e-4
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 손실 기록용 리스트
    train_losses = []
    val_losses = []
    train_lane_losses = []
    train_consistency_losses = []
    
    # 최상의 모델 저장 변수
    best_val_loss = float('inf')
    best_model_path = 'my_bev/carla_bev_backbone_best.pth'
    
    # 학습 시작 시간
    start_time = time.time()
    
    # 중간 중단 시 재개할 수 있도록 체크포인트 저장
    checkpoint_interval = 5  # 5 에포크마다 체크포인트 저장
    
    try:
        # 학습 루프
        for epoch in range(num_epochs):
            # 각 에포크 시작 시 CUDA 캐시 정리
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 학습
            avg_loss, avg_consistency_loss, avg_lane_loss = train_one_epoch(
                model, train_dataloader, optimizer, criterion, device, epoch
            )
            
            train_losses.append(avg_loss)
            train_lane_losses.append(avg_lane_loss)
            train_consistency_losses.append(avg_consistency_loss)
            
            # 검증
            val_loss, val_consistency_loss, val_lane_loss = evaluate_model(
                model, val_dataloader, criterion, device
            )
            val_losses.append(val_loss)
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 진행 상황 출력
            print(f"에포크 {epoch+1}/{num_epochs}")
            print(f"  학습 손실: {avg_loss:.6f} (차선: {avg_lane_loss:.6f}, 일관성: {avg_consistency_loss:.6f})")
            print(f"  검증 손실: {val_loss:.6f} (차선: {val_lane_loss:.6f}, 일관성: {val_consistency_loss:.6f})")
            
            # 메모리 상태 출력
            if device.type == 'cuda':
                print(f"  CUDA 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f}GB 사용 중")
            
            # 최상의 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  새로운 최상 모델 저장: {best_model_path}")
            
            # 체크포인트 저장
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = f'my_bev/carla_checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_lane_losses': train_lane_losses,
                    'train_consistency_losses': train_consistency_losses,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"  체크포인트 저장: {checkpoint_path}")
            
            # 10 에포크마다 중간 손실 그래프 저장
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                plot_losses(
                    train_losses, val_losses, 
                    train_lane_losses, train_consistency_losses,
                    f"my_bev/carla_training_loss_epoch_{epoch+1}.png"
                )
    
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n학습 중 오류 발생: {e}")
    finally:
        # 학습 완료 시간 측정
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"학습 완료! 총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
        
        # 최종 손실 그래프 저장
        if train_losses:
            plot_losses(
                train_losses, val_losses, 
                train_lane_losses, train_consistency_losses,
                "my_bev/carla_training_loss_final.png"
            )
        
        # 최종 학습된 모델 저장
        torch.save(model.state_dict(), 'my_bev/carla_bev_backbone_trained.pth')
        print(f"최종 모델 저장 완료: my_bev/carla_bev_backbone_trained.pth")
        
        if os.path.exists(best_model_path):
            print(f"최상 모델 저장 완료: {best_model_path} (검증 손실: {best_val_loss:.6f})")
            
            # 최상의 모델 로드
            model.load_state_dict(torch.load(best_model_path))
            print("최상의 모델 로드 완료, 특징 맵 시각화 중...")
    
    # 학습된 모델로 특징 맵 추출 및 시각화
    model.eval()
    
    # 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 검증 세트에서 샘플 이미지 선택
    val_samples = []
    for i in range(min(5, len(val_dataset))):
        val_samples.append(val_dataset[i])
    
    # 특징 맵 시각화
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    
    for i, (img, mask, _) in enumerate(val_samples):
        if i >= 5:  # 최대 5개 샘플만 시각화
            break
            
        img_tensor = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features, _ = model(img_tensor)
            
            # 특징 맵에서 차선 예측 (손실 함수의 레이어 사용)
            lane_pred = criterion.lane_predictor(features)
            
            # 원래 크기로 복원
            lane_pred = nn.functional.interpolate(
                lane_pred, size=(200, 200), mode='bilinear', align_corners=False
            )
            lane_pred = torch.sigmoid(lane_pred)
        
        # 원본 이미지
        axes[i, 0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title(f"Sample {i+1} - Input Image")
        axes[i, 0].axis('off')
        
        # 원본 차선 마스크
        if mask is not None:
            axes[i, 1].imshow(mask[0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title(f"Sample {i+1} - Ground Truth Lane")
            axes[i, 1].axis('off')
        
        # 예측된 차선 마스크
        axes[i, 2].imshow(lane_pred[0, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title(f"Sample {i+1} - Predicted Lane")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("my_bev/carla_trained_predictions.png")
    print(f"예측 시각화 저장 완료: my_bev/carla_trained_predictions.png")
    
    return "CARLA BEV 데이터셋 강화 학습 테스트 완료"

if __name__ == "__main__":
    result = test_carla_training()
    print(result) 