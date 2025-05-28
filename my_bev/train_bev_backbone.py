import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.append('.')

# 모델과 데이터셋 임포트
from models.custom_bev_backbone import CustomBEVBackbone
from datasets.bev_dataset import BEVDataset

def custom_collate_fn(batch):
    """커스텀 배치 콜레이터"""
    # 이미지와 마스크는 모두 같은 크기이므로 스택 가능
    images = torch.stack([item['img'] for item in batch])
    masks = torch.stack([item['gt_lane_masks'] for item in batch])
    
    # 바운딩 박스는 패딩하여 최대 크기로 맞춤
    max_boxes = max(item['gt_bboxes_3d'].shape[0] for item in batch)
    boxes_padded = []
    for item in batch:
        boxes = item['gt_bboxes_3d']
        if boxes.shape[0] < max_boxes:
            padding = torch.zeros(max_boxes - boxes.shape[0], 4, dtype=boxes.dtype)
            boxes_padded.append(torch.cat([boxes, padding], dim=0))
        else:
            boxes_padded.append(boxes)
    boxes = torch.stack(boxes_padded)
    
    # 레이블도 같은 방식으로 패딩
    labels_padded = []
    for item in batch:
        labels = item['gt_labels_3d']
        if labels.shape[0] < max_boxes:
            padding = torch.zeros(max_boxes - labels.shape[0], dtype=labels.dtype)
            labels_padded.append(torch.cat([labels, padding]))
        else:
            labels_padded.append(labels)
    labels = torch.stack(labels_padded)
    
    # 객체 ID도 패딩
    ids_padded = []
    for item in batch:
        ids = item['gt_inds']
        if ids.shape[0] < max_boxes:
            padding = torch.zeros(max_boxes - ids.shape[0], dtype=ids.dtype)
            ids_padded.append(torch.cat([ids, padding]))
        else:
            ids_padded.append(ids)
    ids = torch.stack(ids_padded)
    
    # 유효한 박스 수를 저장
    valid_boxes = torch.tensor([item['gt_bboxes_3d'].shape[0] for item in batch])
    
    return {
        'img': images,
        'gt_lane_masks': masks,
        'gt_bboxes_3d': boxes,
        'gt_labels_3d': labels,
        'gt_inds': ids,
        'valid_boxes': valid_boxes
    }

class BEVLoss(nn.Module):
    """BEV 백본을 위한 복합 손실 함수"""
    def __init__(self, lane_weight, obj_weight, feat_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.lane_weight = lane_weight
        self.obj_weight = obj_weight
        self.feat_weight = feat_weight
        
    def forward(self, bev_features, pos_embed, gt_masks, gt_boxes, valid_boxes):
        # 마스크 크기 조정
        gt_masks = F.interpolate(gt_masks.unsqueeze(1).float(), 
                               size=bev_features.shape[-2:], 
                               mode='bilinear', 
                               align_corners=True)
        
        # 1. 차선 분할 손실 (BEV 특징 → 차선 마스크)
        lane_pred = bev_features[:, :1]  # 첫 번째 채널을 차선 예측으로 사용
        lane_loss = self.bce(lane_pred, gt_masks)
        
        # 2. 객체 검출 손실 (BEV 특징 → 객체 히트맵)
        obj_pred = bev_features[:, 1:2]  # 두 번째 채널을 객체 히트맵으로 사용
        obj_heatmap = self.create_heatmap(gt_boxes, valid_boxes, bev_features.shape[-2:])
        obj_loss = self.bce(obj_pred, obj_heatmap)
        
        # 3. 특징-위치 일관성 손실
        feat_loss = self.mse(bev_features, pos_embed)
        
        # 손실 가중치 설정
        total_loss = self.lane_weight * lane_loss + self.obj_weight * obj_loss + self.feat_weight * feat_loss
        
        return total_loss, {
            'lane_loss': lane_loss.item(),
            'obj_loss': obj_loss.item(),
            'feat_loss': feat_loss.item()
        }
    
    @staticmethod
    def create_heatmap(boxes, valid_boxes, size):
        """바운딩 박스로부터 가우시안 히트맵 생성"""
        heatmap = torch.zeros((boxes.shape[0], 1, size[0], size[1]), device=boxes.device)
        
        for idx, (bbox, num_valid) in enumerate(zip(boxes, valid_boxes)):
            for i in range(num_valid):
                box = bbox[i]
                x, y, w, h = box
                x, y = int(x * size[0] / 600), int(y * size[1] / 600)  # 좌표 스케일 조정
                w, h = int(w * size[0] / 600), int(h * size[1] / 600)
                
                # 가우시안 커널 생성
                sigma = min(w, h) / 6
                for i in range(max(0, x-w//2), min(size[0], x+w//2)):
                    for j in range(max(0, y-h//2), min(size[1], y+h//2)):
                        d2 = ((i-x)/sigma)**2 + ((j-y)/sigma)**2
                        heatmap[idx, 0, j, i] = max(heatmap[idx, 0, j, i], np.exp(-d2/2))
                        
        return heatmap

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    loss_dict_sum = {'lane_loss': 0, 'obj_loss': 0, 'feat_loss': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch in pbar:
        # 데이터 준비
        images = batch['img'].to(device)
        masks = batch['gt_lane_masks'].to(device)
        boxes = batch['gt_bboxes_3d'].to(device)
        valid_boxes = batch['valid_boxes'].to(device)
        
        # 이전 프레임 데이터가 있으면 처리
        prev_images = batch.get('prev_img', None)
        if prev_images is not None:
            prev_images = prev_images.to(device)
            model.prev_bev = None  # 새로운 시퀀스 시작
            with torch.no_grad():
                _, _ = model(prev_images)  # 이전 프레임 처리하여 prev_bev 설정
        
        # 순전파
        bev_features, pos_embed = model(images)
        
        # 손실 계산
        loss, loss_dict = criterion(bev_features, pos_embed, masks, boxes, valid_boxes)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # GPU 캐시 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 손실 기록
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] += v
            
        # 진행바 업데이트
        pbar.set_postfix({k: v/len(dataloader) for k, v in loss_dict_sum.items()})
    
    return total_loss / len(dataloader), loss_dict_sum

def validate(model, dataloader, criterion, device):
    """검증 및 시각화"""
    model.eval()
    total_loss = 0
    loss_dict_sum = {'lane_loss': 0, 'obj_loss': 0, 'feat_loss': 0}
    
    # 시각화를 위한 샘플 저장
    vis_samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 데이터 준비
            images = batch['img'].to(device)
            masks = batch['gt_lane_masks'].to(device)
            boxes = batch['gt_bboxes_3d'].to(device)
            valid_boxes = batch['valid_boxes'].to(device)
            
            # 이전 프레임 데이터가 있으면 처리
            prev_images = batch.get('prev_img', None)
            if prev_images is not None:
                prev_images = prev_images.to(device)
                model.prev_bev = None  # 새로운 시퀀스 시작
                _, _ = model(prev_images)  # 이전 프레임 처리
            
            # 순전파
            bev_features, pos_embed = model(images)
            loss, loss_dict = criterion(bev_features, pos_embed, masks, boxes, valid_boxes)
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
            
            # 처음 4개 배치만 시각화용으로 저장
            if i < 4:
                vis_samples.append({
                    'img': images.cpu(),
                    'gt_mask': masks.cpu(),
                    'pred_lane': torch.sigmoid(bev_features[:, :1]).cpu(),
                    'pred_obj': torch.sigmoid(bev_features[:, 1:2]).cpu(),
                    'gt_boxes': boxes.cpu(),
                    'valid_boxes': valid_boxes.cpu(),
                    'bev_features': bev_features.cpu(),  # 전체 BEV 특징 텐서 저장
                    'pos_embed': pos_embed.cpu()  # positional embedding 저장
                })
    
    return total_loss / len(dataloader), loss_dict_sum, vis_samples

def main():
    # 설정
    data_root = '_out_bev_all_outputs'
    batch_size = 8  # 메모리 허용시 증가
    num_epochs = 50  # 더 긴 학습
    lr = 2e-4  # 초기 학습률 증가
    min_lr = 1e-6  # 최소 학습률 설정
    weight_decay = 0.05  # 정규화 강화
    save_dir = 'checkpoints'
    vis_dir = 'visualizations'
    max_grad_norm = 5.0  # gradient clipping 값 증가
    
    # 디렉토리 생성
    Path(save_dir).mkdir(exist_ok=True)
    Path(vis_dir).mkdir(exist_ok=True)
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 및 데이터로더 설정
    dataset = BEVDataset(data_root, seq_len=2)
    train_size = int(0.9 * len(dataset))  # 학습 데이터 비율 증가
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,  # 데이터 로딩 워커 증가
        pin_memory=True,  # GPU 전송 최적화
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')
    
    # 모델 설정
    model = CustomBEVBackbone(out_ch=256).to(device)
    
    # 손실 함수 설정 (가중치 조정)
    criterion = BEVLoss(lane_weight=1.0, obj_weight=1.0, feat_weight=0.1)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)  # Adam 파라미터 조정
    )
    
    # 학습률 스케줄러 설정
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # warm-up 비율
        anneal_strategy='cos',
        final_div_factor=lr/min_lr,
        div_factor=10.0
    )
    
    # AMP (Automatic Mixed Precision) 설정
    scaler = torch.cuda.amp.GradScaler()
    
    # 학습 곡선 기록용
    train_losses = []
    val_losses = []
    train_loss_details = {'lane_loss': [], 'obj_loss': [], 'feat_loss': []}
    val_loss_details = {'lane_loss': [], 'obj_loss': [], 'feat_loss': []}
    
    # 학습 루프
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 학습
        model.train()
        total_loss = 0
        loss_dict_sum = {'lane_loss': 0, 'obj_loss': 0, 'feat_loss': 0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            # 데이터 준비
            images = batch['img'].to(device)
            masks = batch['gt_lane_masks'].to(device)
            boxes = batch['gt_bboxes_3d'].to(device)
            valid_boxes = batch['valid_boxes'].to(device)
            
            # 이전 프레임 처리
            prev_images = batch.get('prev_img', None)
            if prev_images is not None:
                prev_images = prev_images.to(device)
                model.prev_bev = None
                with torch.no_grad():
                    _, _ = model(prev_images)
            
            # AMP 순전파
            with torch.cuda.amp.autocast():
                bev_features, pos_embed = model(images)
                loss, loss_dict = criterion(bev_features, pos_embed, masks, boxes, valid_boxes)
            
            # AMP 역전파
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # 학습률 업데이트
            scheduler.step()
            
            # 손실 기록
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
            
            # 진행바 업데이트
            pbar.set_postfix({
                'lr': scheduler.get_last_lr()[0],
                **{k: v/len(train_loader) for k, v in loss_dict_sum.items()}
            })
        
        # 검증
        val_loss, val_loss_dict, vis_samples = validate(model, val_loader, criterion, device)
        
        # 학습 곡선 기록
        train_losses.append(total_loss/len(train_loader))
        val_losses.append(val_loss)
        for k, v in loss_dict_sum.items():
            train_loss_details[k].append(v/len(train_loader))
        for k, v in val_loss_dict.items():
            val_loss_details[k].append(v/len(val_loader))
        
        # 로깅
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}')
        print('Train Loss Details:')
        for k, v in loss_dict_sum.items():
            print(f'  - {k}: {v/len(train_loader):.4f}')
        
        print(f'Val Loss: {val_loss:.4f}')
        print('Val Loss Details:')
        for k, v in val_loss_dict.items():
            print(f'  - {k}: {v/len(val_loader):.4f}')
        
        # 학습 곡선 저장 (매 에포크마다)
        if len(train_losses) > 0:  # 데이터가 있을 때만 그리기
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            epochs = range(1, epoch + 2)  # 1부터 시작하는 에포크 번호
            plt.plot(epochs, train_losses, label='Train')
            plt.plot(epochs, val_losses, label='Val')
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            for k in ['lane_loss', 'obj_loss', 'feat_loss']:
                plt.plot(epochs, train_loss_details[k], label=f'Train {k}')
                plt.plot(epochs, val_loss_details[k], '--', label=f'Val {k}')
            plt.title('Loss Components')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'learning_curves.png'), bbox_inches='tight')
            plt.close()
        
        # 시각화 저장 (5 에포크마다)
        if (epoch + 1) % 5 == 0 and vis_samples:
            save_visualizations(vis_samples, epoch + 1, vis_dir)
        
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'{save_dir}/best_model.pth')
        
        # 주기적 체크포인트 저장 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
            }, f'{save_dir}/checkpoint_epoch{epoch+1}.pth')
        
        # 에포크 종료 시 상태 초기화
        model.reset_states()

def save_visualizations(samples, epoch, vis_dir):
    """결과 시각화 및 저장"""
    os.makedirs(vis_dir, exist_ok=True)
    print(f"\nSaving visualizations for epoch {epoch} to {vis_dir}")
    
    # 예측값 분포 시각화
    plt.figure(figsize=(12, 4))
    
    # 차선 예측 분포
    plt.subplot(1, 2, 1)
    all_lane_preds = []
    for batch in samples:
        all_lane_preds.extend(batch['pred_lane'].flatten().numpy())
    plt.hist(all_lane_preds, bins=50, range=(0, 1))
    plt.title('Lane Prediction Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    
    # 객체 예측 분포
    plt.subplot(1, 2, 2)
    all_obj_preds = []
    for batch in samples:
        all_obj_preds.extend(batch['pred_obj'].flatten().numpy())
    plt.hist(all_obj_preds, bins=50, range=(0, 1))
    plt.title('Object Prediction Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'pred_distribution_epoch{epoch}.png'))
    plt.close()
    
    # 기존 시각화 코드
    for batch_idx, batch in enumerate(samples):
        for sample_idx in range(len(batch['img'])):
            plt.switch_backend('Agg')
            
            # 1. 기본 시각화 (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # 원본 이미지
            img = batch['img'][sample_idx].permute(1,2,0).numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)  # denormalize
            axes[0,0].imshow(img)
            axes[0,0].set_title('Input Image')
            
            # GT 차선 마스크와 예측 오버레이
            gt_mask = batch['gt_mask'][sample_idx].numpy()
            pred_lane = batch['pred_lane'][sample_idx,0].numpy()
            
            # 크기 맞추기
            if gt_mask.shape != pred_lane.shape:
                print(f"Resizing masks - GT: {gt_mask.shape}, Pred: {pred_lane.shape}")
                pred_lane = F.interpolate(
                    batch['pred_lane'][sample_idx:sample_idx+1], 
                    size=gt_mask.shape,
                    mode='bilinear',
                    align_corners=True
                )[0,0].numpy()
            
            # 차선 예측과 GT를 RGB로 표시 (R: GT, G: 예측, B: 0)
            lane_viz = np.stack([gt_mask, pred_lane, np.zeros_like(gt_mask)], axis=-1)
            axes[0,1].imshow(lane_viz)
            axes[0,1].set_title('Lane GT(R) vs Pred(G)')
            
            # 예측 차선 (threshold 적용)
            pred_thresh = (pred_lane > 0.5).astype(float)
            axes[1,0].imshow(pred_thresh, cmap='gray')
            axes[1,0].set_title('Predicted Lane (thresh=0.5)')
            
            # 예측 객체 히트맵
            obj_pred = batch['pred_obj'][sample_idx,0].numpy()
            if obj_pred.shape != gt_mask.shape:
                obj_pred = F.interpolate(
                    batch['pred_obj'][sample_idx:sample_idx+1], 
                    size=gt_mask.shape,
                    mode='bilinear',
                    align_corners=True
                )[0,0].numpy()
            
            axes[1,1].imshow(obj_pred, cmap='jet')
            axes[1,1].set_title('Object Heatmap')
            
            # GT 박스 그리기
            valid_boxes = batch['valid_boxes'][sample_idx].int().item()
            boxes = batch['gt_boxes'][sample_idx][:valid_boxes]
            
            for box in boxes:
                x, y, w, h = box.numpy()
                rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, color='r')
                axes[1,1].add_patch(rect)
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'epoch{epoch}_batch{batch_idx}_sample{sample_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 2. BEV 특징 텐서 시각화 (8x8 그리드)
            plt.figure(figsize=(20, 20))
            features = batch['bev_features'][sample_idx]  # [C,H,W]
            
            # 처음 64개 채널만 시각화
            num_channels = min(64, features.shape[0])
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = rows
            
            for i in range(num_channels):
                plt.subplot(rows, cols, i + 1)
                # 각 특징 맵을 -1 ~ 1 범위로 정규화
                feat = features[i].numpy()
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
                plt.imshow(feat, cmap='viridis')
                plt.axis('off')
                plt.title(f'Ch {i}')
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'features_epoch{epoch}_batch{batch_idx}_sample{sample_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 3. Positional Embedding 시각화
            plt.figure(figsize=(20, 20))
            pos = batch['pos_embed'][sample_idx]  # [C,H,W]
            
            # 처음 64개 채널만 시각화
            num_channels = min(64, pos.shape[0])
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = rows
            
            for i in range(num_channels):
                plt.subplot(rows, cols, i + 1)
                # 각 특징 맵을 -1 ~ 1 범위로 정규화
                p = pos[i].numpy()
                p = (p - p.min()) / (p.max() - p.min() + 1e-6)
                plt.imshow(p, cmap='viridis')
                plt.axis('off')
                plt.title(f'Pos {i}')
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'pos_embed_epoch{epoch}_batch{batch_idx}_sample{sample_idx}.png')
            plt.savefig(save_path)
            plt.close()
            
            # 4. 채널별 통계 시각화
            plt.figure(figsize=(15, 5))
            
            # 특징 텐서의 채널별 통계
            plt.subplot(1, 2, 1)
            channel_means = features.mean(dim=(1,2)).numpy()
            channel_stds = features.std(dim=(1,2)).numpy()
            plt.errorbar(range(len(channel_means)), channel_means, 
                        yerr=channel_stds, fmt='o', markersize=2, 
                        capsize=2, alpha=0.5)
            plt.title('Feature Channel Statistics')
            plt.xlabel('Channel')
            plt.ylabel('Mean ± Std')
            
            # Positional Embedding의 채널별 통계
            plt.subplot(1, 2, 2)
            pos_means = pos.mean(dim=(1,2)).numpy()
            pos_stds = pos.std(dim=(1,2)).numpy()
            plt.errorbar(range(len(pos_means)), pos_means, 
                        yerr=pos_stds, fmt='o', markersize=2, 
                        capsize=2, alpha=0.5)
            plt.title('Positional Embedding Statistics')
            plt.xlabel('Channel')
            plt.ylabel('Mean ± Std')
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'stats_epoch{epoch}_batch{batch_idx}_sample{sample_idx}.png')
            plt.savefig(save_path)
            plt.close()

if __name__ == '__main__':
    main() 