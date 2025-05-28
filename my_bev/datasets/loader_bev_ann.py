import cv2
import json
import numpy as np
import torch
import os


class LoadBEVAnnotations:
    """Load BEV annotations from JSON files and lane masks."""
    def __init__(self, ann_file, lane_mask_dir):
        self.lane_dir = lane_mask_dir
        self.ann_dir = os.path.dirname(ann_file)
        
    def __call__(self, results):
        img_id = results['img_id']
        ann_file = os.path.join(self.ann_dir, f'objects_f{img_id}.json')
        
        # Load annotations for this frame
        with open(ann_file, 'r') as f:
            anns = json.load(f)
            
        # Extract bounding boxes and labels
        boxes, labels, track_ids = [], [], []
        for ann in anns:
            # Convert polygon points to bounding box
            points = np.array(ann['bev_pixels'])
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            w, h = x2 - x1, y2 - y1
            boxes.append([x1, y1, w, h])
            
            # Convert category to label index
            category = ann['category']
            label = 1 if category == 'vehicle' else 2  # 1: vehicle, 2: pedestrian
            labels.append(label)
            
            # Track ID
            track_ids.append(ann['actor_id'])
            
        # Convert to tensors
        if boxes:
            results['gt_bboxes_3d'] = torch.tensor(boxes, dtype=torch.float32)
            results['gt_labels_3d'] = torch.tensor(labels, dtype=torch.long)
            results['gt_inds'] = torch.tensor(track_ids, dtype=torch.long)
        else:
            # Empty tensors if no objects
            results['gt_bboxes_3d'] = torch.zeros((0, 4), dtype=torch.float32)
            results['gt_labels_3d'] = torch.zeros(0, dtype=torch.long)
            results['gt_inds'] = torch.zeros(0, dtype=torch.long)

        # Load lane mask
        mask_file = os.path.join(self.lane_dir, f'lane_only_f{img_id}.png')
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        results['gt_lane_masks'] = torch.tensor(mask > 127, dtype=torch.uint8)
        
        return results
