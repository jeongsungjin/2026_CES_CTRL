#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from os import path as osp
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .eval_utils.nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.tracking.evaluate import TrackingEval
from .eval_utils.nuscenes_eval_motion import MotionEval
from nuscenes.eval.common.config import config_factory
import tempfile
from mmcv.parallel import DataContainer as DC
import random
import pickle
from prettytable import PrettyTable

from nuscenes import NuScenes
from projects.mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap
from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj
from .data_utils.data_utils import lidar_nusc_box_to_global, obtain_map_info, output_to_nusc_box, output_to_nusc_box_det
from nuscenes.prediction import convert_local_coords_to_global


@DATASETS.register_module()
class NuScenesE2EDataset(NuScenesDataset):
    """NuScenes E2E Dataset.

    This dataset adds support for BEV images and sequential data processing.
    """

    def __init__(self,
                 queue_length=4,
                 bev_size=(200, 200),
                 patch_size=(16, 16),
                 canvas_size=(200, 200),
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.queue = []

    def prepare_train_data(self, index):
        """Prepare training data.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length+1, index+1))
        index_list = [i for i in index_list if i >= 0]
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if 'bev_path' in info:
            input_dict.update(
                dict(
                    img_prefix=None,
                    img_info=dict(filename=info['bev_path']),
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def union2one(self, queue):
        """Convert queue to one sample.

        Args:
            queue (list): List of samples.

        Returns:
            dict: Unified sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args['backend'] == 'disk':
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file.name))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        elif self.file_client_args['backend'] == 'petrel':
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(
                sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[::self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'
        return data_infos

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if 'l2g' in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): .
                - gt_fut_traj_mask (np.ndarray): .
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_inds'][mask]

        sample = self.nusc.get('sample', info['token'])
        ann_tokens = np.array(sample['anns'])[mask]
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = self.traj_api.get_traj_label(
            info['token'], ann_tokens)

        sdc_vel = self.traj_api.sdc_vel_info[info['token']]
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(
            info['token'])

        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info['token'])

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
        )
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        nusc_map_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if 'map' in self.eval_mod:
                map_annos = {}
                for key, value in det['ret_iou'].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            if 'boxes_3d' not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in ['car', 'truck', 'bus', 'trailer', 'motorcycle',
                                'bicycle', 'pedestrian', ]:
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center
                if 'traj' in det:
                    traj_local = det['traj'][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det['traj_scores'][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot)

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': nusc_map_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if det is None:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box_det(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                                       mapped_class_names,
                                                       self.eval_detection_configs,
                                                       self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc_det.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results_det')
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 planning_evaluation_strategy="uniad"):
        """Evaluation in nuScenes protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(results, dict):
            if 'occ_results_computed' in results.keys():
                occ_results_computed = results['occ_results_computed']
                out_metrics = ['iou']

                # pan_eval
                if occ_results_computed.get('pq', None) is not None:
                    out_metrics = ['iou', 'pq', 'sq', 'rq']

                print("Occ-flow Val Results:")
                for panoptic_key in out_metrics:
                    print(panoptic_key)
                    # HERE!! connect
                    print(' & '.join(
                        [f'{x:.1f}' for x in occ_results_computed[panoptic_key]]))

                if 'num_occ' in occ_results_computed.keys() and 'ratio_occ' in occ_results_computed.keys():
                    print(
                        f"num occ evaluated:{occ_results_computed['num_occ']}")
                    print(
                        f"ratio occ evaluated: {occ_results_computed['ratio_occ'] * 100:.1f}%")
            if 'planning_results_computed' in results.keys():
                planning_results_computed = results['planning_results_computed']
                planning_tab = PrettyTable()
                planning_tab.title = f"{planning_evaluation_strategy}'s definition planning metrics"
                planning_tab.field_names = [
                    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        if planning_evaluation_strategy == "stp3":
                            row_value.append("%.4f" % float(value[: i + 1].mean()))
                        elif planning_evaluation_strategy == "uniad":
                            row_value.append("%.4f" % float(value[i]))
                        else:
                            raise ValueError(
                                "planning_evaluation_strategy should be uniad or spt3"
                            )
                    planning_tab.add_row(row_value)
                print(planning_tab)
            results = results['bbox_results']  # get bbox_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_files_det, tmp_dir = self.format_results_det(
            results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], result_files_det[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(
                result_files, result_files_det)

        if 'map' in self.eval_mod:
            drivable_intersection = 0
            drivable_union = 0
            lanes_intersection = 0
            lanes_union = 0
            divider_intersection = 0
            divider_union = 0
            crossing_intersection = 0
            crossing_union = 0
            contour_intersection = 0
            contour_union = 0
            for i in range(len(results)):
                drivable_intersection += results[i]['ret_iou']['drivable_intersection']
                drivable_union += results[i]['ret_iou']['drivable_union']
                lanes_intersection += results[i]['ret_iou']['lanes_intersection']
                lanes_union += results[i]['ret_iou']['lanes_union']
                divider_intersection += results[i]['ret_iou']['divider_intersection']
                divider_union += results[i]['ret_iou']['divider_union']
                crossing_intersection += results[i]['ret_iou']['crossing_intersection']
                crossing_union += results[i]['ret_iou']['crossing_union']
                contour_intersection += results[i]['ret_iou']['contour_intersection']
                contour_union += results[i]['ret_iou']['contour_union']
            results_dict.update({'drivable_iou': float(drivable_intersection / drivable_union),
                                 'lanes_iou': float(lanes_intersection / lanes_union),
                                 'divider_iou': float(divider_intersection / divider_union),
                                 'crossing_iou': float(crossing_intersection / crossing_union),
                                 'contour_iou': float(contour_intersection / contour_union)})

            print(results_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_single(self,
                         result_path,
                         result_path_det,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        # TODO: fix the evaluation pipelines

        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, 'det')
        output_dir_track = osp.join(output_dir, 'track')
        output_dir_motion = osp.join(output_dir, 'motion')
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)
        mmcv.mkdir_or_exist(output_dir_motion)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        detail = dict()

        if 'det' in self.eval_mod:
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path_det,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)
            # record metrics
            metrics = mmcv.load(
                osp.join(
                    output_dir_det,
                    'metrics_summary.json'))
            metric_prefix = f'{result_name}_NuScenes'
            for name in self.CLASSES:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(
                        metric_prefix, name, k)] = val
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix,
                                          self.ErrNameMapping[k])] = val
            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']

        if 'track' in self.eval_mod:
            cfg = config_factory("tracking_nips_2019")
            self.nusc_eval_track = TrackingEval(
                config=cfg,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root
            )
            self.nusc_eval_track.main()
            # record metrics
            metrics = mmcv.load(
                osp.join(
                    output_dir_track,
                    'metrics_summary.json'))
            keys = ['amota', 'amotp', 'recall', 'motar',
                    'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                    'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
            for key in keys:
                detail['{}/{}'.format(metric_prefix, key)] = metrics[key]

        # if 'map' in self.eval_mod:
        #     for i, ret_iou in enumerate(ret_ious):
        #         detail['iou_{}'.format(i)] = ret_iou

        if 'motion' in self.eval_mod:
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='motion_category'
            )
            print('-'*50)
            print(
                'Evaluate on motion category, merge class for vehicles and pedestrians...')
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='standard')
            print('evaluate motion mAP-minFDE metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='motion_map')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='epa')
            print('-'*50)
            print('Evaluate on detection category...')
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type='detection_category'
            )
            print('evaluate standard motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='standard')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='motion_map')
            print('evaluate EPA motion metrics...')
            self.nusc_eval_motion.main(
                plot_examples=0,
                render_curves=False,
                eval_mode='epa')

        return detail
