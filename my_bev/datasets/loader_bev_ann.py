import cv2
import json
import numpy as np
import torch


class LoadBEVAnnotations:
    def __init__(self, ann_file, lane_mask_dir):
        self.anns = json.load(open(ann_file))
        # index by image_id → [anns]
        self.idx = {}
        for a in self.anns['annotations']:
            self.idx.setdefault(a['image_id'], []).append(a)
        self.lane_dir = lane_mask_dir
        self.images = {i['id']: i for i in self.anns['images']}

    def __call__(self, results):
        img_id = results['img_id']
        # ① bbox
        boxes, labels, track_ids = [], [], []
        for a in self.idx[img_id]:
            boxes.append(a['bbox'])          # x,y,w,l in pixels
            labels.append(a['category_id'])
            track_ids.append(a['track_id'])
        results['gt_bboxes_3d'] = torch.tensor(boxes, dtype=torch.float32)
        results['gt_labels_3d'] = torch.tensor(labels, dtype=torch.long)
        results['gt_inds']      = torch.tensor(track_ids, dtype=torch.long)

        # ② lane mask (H,W) 0/255 → 0/1
        img_name = self.images[img_id]['file_name']
        mask = cv2.imread(f"{self.lane_dir}/{img_name}", cv2.IMREAD_GRAYSCALE)
        results['gt_lane_masks'] = torch.tensor(mask // 255, dtype=torch.uint8)
        return results
