"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class IntermediateFusionDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self). \
            __init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

    def data_reformat(self, observations, gt_info):
        """
        Receive data from Scene Generator and convert the right format
        for cooperative perception models.

        Parameters
        ----------
        observations : dict
            The dictionary that contains all cavs' info including lidar pose
            and lidar observations.

        gt_info : dict
            groundtruth information for all objects in the scene (not
            neccessary all in the valid range).

        Returns
        -------
        A dictionary that contains the data with the right format that
        detection model needs.
        """

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in observations.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['lidar_pose']
                break

        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        # convert the objects coordinates under ego(infra)
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center(gt_info,
                                                       ego_lidar_pose)

        processed_features = []
        # this is just used to make the format the same as previously
        object_stack = [object_bbx_center]
        object_id_stack = object_ids

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in observations.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base[
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed, void_lidar = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            if void_lidar:
                continue

            processed_features.append(
                selected_cav_processed['processed_features'])

            infra.append(1 if int(cav_id) < 0 else 0)

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        # pad dv, dt, infra to max_cav
        # velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        # time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        # spatial_correction_matrix = np.stack(spatial_correction_matrix)
        # padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
        #     spatial_correction_matrix), 1, 1))
        # spatial_correction_matrix = np.concatenate(
        #     [spatial_correction_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             # 'velocity': velocity,
             # 'time_delay': time_delay,
             'infra': infra,
             # 'spatial_correction_matrix': spatial_correction_matrix
             })

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix,
        # no delay or loc error is considered under this setting.
        transformation_matrix = \
            x1_to_x2(selected_cav_base['lidar_pose'],
                     ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # Check if filtered LiDAR points are not void
        void_lidar = True if lidar_np.shape[0] < 1 else False

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        # velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        # velocity = velocity / 30

        selected_cav_processed.update(
            {'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             # 'velocity': velocity
             })

        return selected_cav_processed, void_lidar

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding
        velocity = []
        time_delay = []
        infra = []

        # used for correcting the spatial transformation
        # between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])

            # velocity.append(ego_dict['velocity'])
            # time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            # spatial_correction_matrix_list.append(
            #     ego_dict['spatial_correction_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M]
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        # velocity = torch.from_numpy(np.array(velocity))
        # time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        # spatial_correction_matrix_list = torch.from_numpy(
        #     np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([infra], dim=-1).float()

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   # 'spatial_correction_matrix':
                                   # spatial_correction_matrix_list
                                   })

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor


if __name__ == '__main__':
    params = load_yaml('./opencood/hypes_yaml/point_pillar_transformer.yaml')

    opencda_dataset = IntermediateFusionDataset(params,
                                                train=True,
                                                visualize=True)
    data_loader = DataLoader(opencda_dataset, batch_size=2, num_workers=10,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)
    import time

    t1 = time.time()
    for j, batch_data in enumerate(data_loader):
        print(f'{j} -- time: {time.time() - t1}')
