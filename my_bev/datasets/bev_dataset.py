import os
import glob
import torch
from torch.utils.data import Dataset
from datasets.loader_bev_image import LoadBEVImage
from datasets.loader_bev_ann import LoadBEVAnnotations

class BEVDataset(Dataset):
    """BEV Dataset for training backbone."""
    def __init__(self, data_root, seq_len=2):
        super().__init__()
        self.data_root = data_root
        self.seq_len = seq_len  # 시퀀스 길이 (현재 + 이전 프레임)
        
        # Setup paths
        self.img_dir = os.path.join(data_root, 'bev_camera_images')
        self.ann_dir = os.path.join(data_root, 'object_json_data')
        self.lane_dir = os.path.join(data_root, 'lane_only_visualizations')
        
        # Get frame IDs (remove 'f' prefix)
        self.frame_ids = []
        for img_path in sorted(glob.glob(os.path.join(self.img_dir, '*.png'))):
            frame_id = os.path.basename(img_path).split('.')[0].split('_')[-1]
            if frame_id.startswith('f'):
                frame_id = frame_id[1:]  # Remove 'f' prefix
            self.frame_ids.append(int(frame_id))
            
        # 프레임 ID 정렬
        self.frame_ids.sort()
        
        # Setup loaders
        self.img_loader = LoadBEVImage(img_prefix=self.img_dir)
        self.ann_loader = LoadBEVAnnotations(
            ann_file=os.path.join(self.ann_dir, f'objects_f{self.frame_ids[0]}.json'),
            lane_mask_dir=self.lane_dir
        )
        
    def __len__(self):
        # 시퀀스 길이를 고려한 데이터셋 크기
        return max(0, len(self.frame_ids) - self.seq_len + 1)
        
    def __getitem__(self, idx):
        # 현재 프레임과 이전 프레임들의 ID
        curr_idx = idx + self.seq_len - 1
        seq_frame_ids = self.frame_ids[curr_idx - self.seq_len + 1:curr_idx + 1]
        
        # 각 프레임의 데이터 로드
        seq_data = []
        for frame_id in seq_frame_ids:
            # Prepare data dict
            results = {
                'img_filename': f'bev_cam_f{frame_id}.png',
                'img_id': frame_id,
            }
            
            # Load image and annotations
            results = self.img_loader(results)
            results = self.ann_loader(results)
            seq_data.append(results)
        
        # 현재 프레임을 기준으로 데이터 구성
        curr_data = seq_data[-1]
        if len(seq_data) > 1:
            curr_data['prev_img'] = seq_data[-2]['img']
            curr_data['prev_gt_lane_masks'] = seq_data[-2]['gt_lane_masks']
            curr_data['prev_gt_bboxes_3d'] = seq_data[-2]['gt_bboxes_3d']
            curr_data['prev_gt_labels_3d'] = seq_data[-2]['gt_labels_3d']
            curr_data['prev_gt_inds'] = seq_data[-2]['gt_inds']
        
        return curr_data

if __name__ == '__main__':
    # Test dataset
    dataset = BEVDataset('/home/students/2026_CES_CTRL/my_bev/_out_bev_all_outputs')
    print(f'Dataset size: {len(dataset)}')
    
    # Test loading
    sample = dataset[0]
    print('\nSample contents:')
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f'{k}: {v.shape} {v.dtype}')
        else:
            print(f'{k}: {type(v)}') 