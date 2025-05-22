from mmcv import imread
import numpy as np
import torch


class LoadBEVImage:
    """Read BEV PNG → float32 Tensor [C,H,W] in [0,1]."""
    def __init__(self, img_prefix):
        self.img_prefix = img_prefix

    def __call__(self, results):
        fname = f"{self.img_prefix}/{results['img_filename']}"
        img = imread(fname, flag='color')           # H,W,3 BGR uint8
        img = img[..., ::-1].astype(np.float32) / 255.  # RGB 0-1
        img = torch.from_numpy(img).permute(2, 0, 1)    # C,H,W
        results['img'] = img
        results['img_shape'] = img.shape[1:]
        return results
