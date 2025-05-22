from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class BEVCoco(CustomDataset):
    CLASSES = ('car', 'pedestrian')
    PALETTE = None

    def load_annotations(self, ann_file):
        import json, os
        anns = json.load(open(ann_file))
        self.img_infos = []
        for img in anns['images']:
            self.img_infos.append(
                dict(id=img['id'],
                     filename=img['file_name'],
                     width=img['width'],
                     height=img['height']))
        self.ann_dict = anns      # raw json (for loader)
        return self.img_infos
